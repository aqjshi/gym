import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2
from gym.envs.box2d import CarRacing
import time
from torch.utils.tensorboard import SummaryWriter
import os 

# Action Map for Discrete Actions
ACTION_MAP = {
    0: [-1.0, 0.0, 0.0],  # Steer left
    1: [1.0, 0.0, 0.0],   # Steer right
    2: [0.0, 1.0, 0.0],   # Gas
    3: [0.0, 0.0, 0.5],   # Brake
    4: [0.0, 0.0, 0.0],   # Do nothing
}

# Preprocess frames for the CNN
def preprocess_frame(frame):
    frame = cv2.resize(frame, (32, 32)).astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))  # Channels first
    return frame

# Normalize sensor data
def get_sensor_data(env):
    sensor_data = torch.tensor([
        env.get_speed() / 100,  # Normalize speed
        env.get_abs_sensors_0(),
        env.get_abs_sensors_1(),
        env.get_abs_sensors_2(),
        env.get_abs_sensors_3(),
        float(env.get_on_grass())
    ], dtype=torch.float32, device=device)
    return sensor_data

# Q-Network with adjusted CNN for 32x32 images
class QNetwork(nn.Module):
    def __init__(self, sensor_dim, num_actions):
        super(QNetwork, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Reduced channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Reduced channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_image = torch.zeros((1, 3, 32, 32))
            cnn_output_size = self.cnn_layers(dummy_image).shape[1]

        self.sensor_fc = nn.Sequential(
            nn.Linear(sensor_dim, 16),  # Reduced size
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(cnn_output_size + 16, 64),  # Reduced size
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x_image, x_sensor):
        x_image = self.cnn_layers(x_image)
        x_sensor = self.sensor_fc(x_sensor)
        x = torch.cat((x_image, x_sensor), dim=1)
        return self.fc_layers(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_image, state_sensor, action, reward,
             next_state_image, next_state_sensor, done):
        # Store images as float16 for reduced memory usage
        state_image = state_image.astype(np.float16)
        next_state_image = next_state_image.astype(np.float16)
        self.buffer.append((state_image, state_sensor, action, reward,
                            next_state_image, next_state_sensor, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        (state_images, state_sensors, actions, rewards,
         next_state_images, next_state_sensors, dones) = zip(*batch)
        return (
            np.array(state_images).astype(np.float32),  # Convert back to float32
            np.array(state_sensors),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_state_images).astype(np.float32),
            np.array(next_state_sensors),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

# Epsilon-Greedy Action Selection
def select_action(state_image, state_sensor, q_net, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        with torch.no_grad():
            state_image_tensor = torch.tensor(
                state_image, dtype=torch.float32, device=device
            ).unsqueeze(0)
            state_sensor_tensor = torch.tensor(
                state_sensor, dtype=torch.float32, device=device
            ).unsqueeze(0)
            q_values = q_net(state_image_tensor, state_sensor_tensor)
            return q_values.argmax().item()

# DQN Training Function
scaler = torch.amp.GradScaler()

def train_dqn(q_net, target_q_net, replay_buffer, optimizer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return
    (state_images, state_sensors, actions, rewards,
     next_state_images, next_state_sensors, dones) = replay_buffer.sample(batch_size)

    state_images = torch.tensor(state_images, dtype=torch.float32).to(device)
    state_sensors = torch.tensor(state_sensors, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(1)
    next_state_images = torch.tensor(next_state_images, dtype=torch.float32).to(device)
    next_state_sensors = torch.tensor(next_state_sensors, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device).unsqueeze(1)

    with torch.cuda.amp.autocast():
        q_values = q_net(state_images, state_sensors).gather(1, actions)
        with torch.no_grad():
            next_q_values = target_q_net(next_state_images, next_state_sensors).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * gamma * next_q_values
        loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Function to save video frames
def save_video_incremental(frame, video_writer):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame)

# Main Training Loop
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    env = CarRacing(render_mode='rgb_array')
    input_shape = (3, 32, 32)
    sensor_dim = 6
    num_actions = len(ACTION_MAP)

    replay_capacity = 5000
    batch_size = 128
    gamma = 0.99
    learning_rate = 1e-4
    epsilon_start = 0.99
    epsilon_end = 0.01
    decay_rate = 100
    num_episodes = 500
    frame_skip = 2
    off_track_threshold = 20
    low_speed_threshold = 0.1
    low_speed_count_threshold = 50

    q_net = QNetwork(sensor_dim, num_actions).to(device)
    target_q_net = QNetwork(sensor_dim, num_actions).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        q_net = nn.DataParallel(q_net)
        target_q_net = nn.DataParallel(target_q_net)

    target_q_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(replay_capacity)

    epsilon = epsilon_start

    best_true_reward = -float('inf')
    best_video_path = "best_episode.mp4"

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}")
        action_counts = {i: 0 for i in range(num_actions)}
        start_time = time.time()
        obs, _ = env.reset()
        state_image = preprocess_frame(env.render())
        state_sensor = get_sensor_data(env)
        total_reward = 0

        steps = 0
        off_track_count = 0
        low_speed_count = 0
        true_reward = 0
        done = False

        current_video_path = "current_episode.mp4"
        height, width, _ = env.render().shape
        current_video_writer = cv2.VideoWriter(
            current_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,
            (width, height)
        )

        while not done:
            action_idx = select_action(state_image, state_sensor, q_net, epsilon, num_actions)
            action = ACTION_MAP[action_idx]
            action_counts[action_idx] += 1

            for _ in range(frame_skip):
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state_image = preprocess_frame(env.render())
                next_state_sensor = get_sensor_data(env)

                true_reward += reward
                if env.get_on_grass():
                    off_track_count += 1
                else:
                    off_track_count = 0
                if env.get_speed() < low_speed_threshold:
                    low_speed_count += 1
                else:
                    low_speed_count = 0

                reward += env.get_speed() / 10

                if off_track_count > off_track_threshold or low_speed_count > low_speed_count_threshold:
                    reward -= 20
                    print(f"Episode terminated: Off track {off_track_count}, Low speed {low_speed_count}")
                    done = True
                    break

                replay_buffer.push(state_image, state_sensor.cpu().numpy(), action_idx, reward,
                                   next_state_image, next_state_sensor.cpu().numpy(),
                                   terminated or truncated)

                save_video_incremental(env.render(), current_video_writer)

                state_image, state_sensor = next_state_image, next_state_sensor
                total_reward += reward
                steps += 1
                if terminated or truncated:
                    done = True
                    break

                train_dqn(q_net, target_q_net, replay_buffer, optimizer, batch_size, gamma)

        current_video_writer.release()
        if true_reward > best_true_reward:
            best_true_reward = true_reward
            if os.path.exists(best_video_path):
                os.remove(best_video_path)
            os.rename(current_video_path, best_video_path)
            print(f"New best video saved with True Reward: {best_true_reward:.2f}")
        else:
            os.remove(current_video_path)

        if episode % 10 == 0:
            target_q_net.load_state_dict(q_net.state_dict())
        print(f"True Reward: {true_reward}")

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / decay_rate)

        end_time = time.time()
        episode_duration = end_time - start_time
        print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Steps: {steps}, "
              f"Epsilon: {epsilon:.2f}, Duration: {episode_duration:.2f} seconds")
        print(f"Action counts: {action_counts}")

    env.close()
