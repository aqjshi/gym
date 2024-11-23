from gym.envs.box2d.car_racing import CarRacing
import numpy as np
import gym
from datetime import datetime
import pygame


env = gym.make('CarRacing-v2')

from gymnasium.envs.registration import register

gym.register(
    id='CustomCarRacing_v0',
    entry_point='gym.envs.box2d.car_racing:CarRacing',
)


env.reset()

from collections import deque

# Stack frames to provide temporal information
def preprocess_observation(observation):
    observation = cv2.resize(observation, (96, 96))  # Resize to 96x96
    observation = observation / 255.0  # Normalize to [0, 1]
    return observation

# Maintain a stack of 4 frames
def stack_frames(frame_stack, frame, stack_size=4):
    frame_stack.append(frame)
    return np.stack(frame_stack, axis=0)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_size = self._get_conv_out(input_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv3(self.conv2(self.conv1(o)))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)



import torch.optim as optim
if __name__ == "__main__":
 

    # Hyperparameters
    learning_rate = 1e-4
    gamma = 0.99
    batch_size = 64
    replay_capacity = 100000
    stack_size = 4

    # Initialize DQN and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_actions = env.action_space.n  # Number of discrete actions
    input_shape = (stack_size, 96, 96)
    dqn = DQN(input_shape, num_actions).to(device)
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(replay_capacity)

    # Training loop
    num_episodes = 1000
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995

    frame_stack = deque(maxlen=stack_size)
    for episode in range(num_episodes):
        obs = env.reset()[0]
        state = preprocess_observation(obs)
        frame_stack.extend([state] * stack_size)
        state = stack_frames(frame_stack, state)

        total_reward = 0
        done = False

        while not done:
            # Select action (epsilon-greedy)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = dqn(state_tensor)
                action = torch.argmax(q_values).item()

            # Execute action in environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess_observation(next_obs)
            frame_stack.append(next_state)
            next_state = stack_frames(frame_stack, next_state)

            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # Train the DQN
            if len(replay_buffer) > batch_size:
                transitions = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*transitions)

                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                # Compute Q-values
                q_values = dqn(states).gather(1, actions)
                next_q_values = dqn(next_states).max(1)[0].detach()
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                # Loss
                loss = F.mse_loss(q_values, target_q_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode}, Total Reward: {total_reward}")
