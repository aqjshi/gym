from gym.envs.box2d.car_racing import CarRacing
import numpy as np
import gym
from datetime import datetime
import pygame

if __name__ == "__main__":
    env = CarRacing(render_mode="human")
    a = np.array([0.0, 0.0, 0.0])  # Action array: [steering, gas, brake]
    quit = False

    def autonomous_policy(observation):
        """
        Example heuristic to control the car based on observation.
        Args:
            observation: Current state/observation from the environment.
        Returns:
            Action array [steering, gas, brake].
        """
        steering = 0.0
        gas = 0.2
        brake = 0.0

        # Example heuristic: Adjust steering based on center of the track
        track_data = observation[80:96, :, 0]  # Assume track is in specific rows
        track_center = np.mean(np.nonzero(track_data))
        if track_center < observation.shape[1] // 2:
            steering = -0.5  # Turn left
        elif track_center > observation.shape[1] // 2:
            steering = 0.5  # Turn right

        return np.array([steering, gas, brake])

    while not quit:
        s, info = env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            # Use the autonomous policy to decide the action
            a = autonomous_policy(s)

            # Perform the action in the environment
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r

            # Logging for debugging and monitoring
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")

            steps += 1
            if terminated or truncated or restart or quit:
                break
        print("Episode finished. Total reward:", total_reward)
        env.close()
