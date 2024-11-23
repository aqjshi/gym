import gym
import numpy as np
import cv2
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# Frame Preprocessing
def preprocess_observation(observation):
    observation = cv2.resize(observation, (96, 96))  # Resize to 96x96
    observation = observation / 255.0  # Normalize to [0, 1]
    observation = np.transpose(observation, (2, 0, 1))  # Rearrange to [channels, height, width]
    return observation

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_image, state_sensor, action, reward, next_state_image, next_state_sensor, done):
        self.buffer.append((state_image, state_sensor, action, reward, next_state_image, next_state_sensor, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_images, state_sensors, actions, rewards, next_state_images, next_state_sensors, dones = map(np.stack, zip(*batch))
        return state_images, state_sensors, actions, rewards, next_state_images, next_state_sensors, dones

    def __len__(self):
        return len(self.buffer)

# Policy Network for SAC
class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, sensor_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # CNN for image processing
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_size = self._get_conv_out(input_shape)

        # Fully connected for sensor data
        self.sensor_fc = nn.Linear(sensor_dim, 64)

        # Combine CNN output and sensor data
        self.fc1 = nn.Linear(conv_out_size + 64, 512)
        self.fc_mean = nn.Linear(512, action_dim)
        self.fc_log_std = nn.Linear(512, action_dim)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv3(self.conv2(self.conv1(o)))
        return int(np.prod(o.size()))

    def forward(self, image, sensor_data):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        sensor_x = F.relu(self.sensor_fc(sensor_data))

        x = torch.cat((x, sensor_x), dim=1)
        x = F.relu(self.fc1(x))

        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # To ensure numerical stability

        return mean, log_std

# Q Network for SAC
class QNetwork(nn.Module):
    def __init__(self, input_shape, sensor_dim, action_dim):
        super(QNetwork, self).__init__()
        # CNN for image processing
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_size = self._get_conv_out(input_shape)

        # Fully connected for sensor data
        self.sensor_fc = nn.Linear(sensor_dim, 64)

        # Combine CNN output, sensor data, and action
        self.fc1 = nn.Linear(conv_out_size + 64 + action_dim, 512)
        self.fc2 = nn.Linear(512, 1)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv3(self.conv2(self.conv1(o)))
        return int(np.prod(o.size()))

    def forward(self, image, sensor_data, action):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        sensor_x = F.relu(self.sensor_fc(sensor_data))

        x = torch.cat((x, sensor_x, action), dim=1)
        x = F.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value

# Function to sample actions using the policy network
def sample_action(policy_net, image, sensor_data):
    mean, log_std = policy_net(image, sensor_data)
    std = log_std.exp()
    normal = torch.distributions.Normal(mean, std)
    z = normal.rsample()
    action = torch.tanh(z)
    log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
    log_prob = log_prob.sum(1, keepdim=True)
    return action, log_prob

# Training Setup
if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 3e-4
    gamma = 0.99
    tau = 0.005  # For soft updates
    alpha = 0.2  # Entropy coefficient
    batch_size = 64
    replay_capacity = 100000
    num_episodes = 1000

    # Initialize Environment
    env = gym.make('CarRacing-v2', continuous=True)

    print("created env")
    # Initialize Networks and Optimizers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (3, 96, 96)
    sensor_dim = 6  # Number of sensor data points
    action_dim = 3  # Steering, Gas, Brake

    policy_net = PolicyNetwork(input_shape, sensor_dim, action_dim).to(device)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    print("created Policy")
    q_net1 = QNetwork(input_shape, sensor_dim, action_dim).to(device)
    q_optimizer1 = optim.Adam(q_net1.parameters(), lr=learning_rate)

    q_net2 = QNetwork(input_shape, sensor_dim, action_dim).to(device)
    q_optimizer2 = optim.Adam(q_net2.parameters(), lr=learning_rate)

    # Target Q networks
    q_net1_target = QNetwork(input_shape, sensor_dim, action_dim).to(device)
    q_net2_target = QNetwork(input_shape, sensor_dim, action_dim).to(device)
    q_net1_target.load_state_dict(q_net1.state_dict())
    q_net2_target.load_state_dict(q_net2.state_dict())

    replay_buffer = ReplayBuffer(replay_capacity)

    for episode in range(num_episodes):
        obs = env.reset()[0]
        state_image = preprocess_observation(obs)
        state_sensor = np.array([
            env.get_speed(),
            env.get_abs_sensors_0(),
            env.get_abs_sensors_1(),
            env.get_abs_sensors_2(),
            env.get_abs_sensors_3(),
            env.get_on_grass()
        ], dtype=np.float32)

        total_reward = 0
        done = False

        while not done:
            state_image_tensor = torch.tensor(state_image, dtype=torch.float32).unsqueeze(0).to(device)
            state_sensor_tensor = torch.tensor(state_sensor, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action, _ = sample_action(policy_net, state_image_tensor, state_sensor_tensor)
                action = action.cpu().numpy()[0]

            # Execute Action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_image = preprocess_observation(next_obs)
            next_state_sensor = np.array([
                env.get_speed(),
                env.get_abs_sensors_0(),
                env.get_abs_sensors_1(),
                env.get_abs_sensors_2(),
                env.get_abs_sensors_3(),
                env.get_on_grass()
            ], dtype=np.float32)

            # Store Transition
            replay_buffer.push(state_image, state_sensor, action, reward, next_state_image, next_state_sensor, done)

            state_image = next_state_image
            state_sensor = next_state_sensor
            total_reward += reward

            # Train the Networks
            if len(replay_buffer) > batch_size:
                # Sample a batch
                state_images, state_sensors, actions, rewards, next_state_images, next_state_sensors, dones = replay_buffer.sample(batch_size)

                # Convert to tensors
                state_images = torch.tensor(state_images, dtype=torch.float32).to(device)
                state_sensors = torch.tensor(state_sensors, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.float32).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
                next_state_images = torch.tensor(next_state_images, dtype=torch.float32).to(device)
                next_state_sensors = torch.tensor(next_state_sensors, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

                # Update Q networks
                with torch.no_grad():
                    next_actions, next_log_probs = sample_action(policy_net, next_state_images, next_state_sensors)
                    q1_next = q_net1_target(next_state_images, next_state_sensors, next_actions)
                    q2_next = q_net2_target(next_state_images, next_state_sensors, next_actions)
                    q_min = torch.min(q1_next, q2_next) - alpha * next_log_probs
                    q_target = rewards + (1 - dones) * gamma * q_min

                q1_pred = q_net1(state_images, state_sensors, actions)
                q2_pred = q_net2(state_images, state_sensors, actions)

                q1_loss = F.mse_loss(q1_pred, q_target)
                q2_loss = F.mse_loss(q2_pred, q_target)

                q_optimizer1.zero_grad()
                q1_loss.backward()
                q_optimizer1.step()

                q_optimizer2.zero_grad()
                q2_loss.backward()
                q_optimizer2.step()

                # Update policy network
                actions_new, log_probs = sample_action(policy_net, state_images, state_sensors)
                q1_new = q_net1(state_images, state_sensors, actions_new)
                q2_new = q_net2(state_images, state_sensors, actions_new)
                q_new = torch.min(q1_new, q2_new)

                policy_loss = (alpha * log_probs - q_new).mean()

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # Soft update target networks
                for param, target_param in zip(q_net1.parameters(), q_net1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(q_net2.parameters(), q_net2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        print(f"Episode {episode}, Total Reward: {total_reward}")
