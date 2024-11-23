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
if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])
    env =gym.make('CustomCarRacing_v0')

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, terminated, truncated, info = env.step(a) #, on_grass, sensor_readings
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()

    
