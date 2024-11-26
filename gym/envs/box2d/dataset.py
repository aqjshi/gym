import os
import cv2
import numpy as np
import pygame
from datetime import datetime
from gym.envs.box2d.car_racing import CarRacing

# Set directories for storing data
OUTPUT_CSV = "output.csv"
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Define action mapping for dataset compatibility
 # do nothing, left, right, gas, brake
ACTION_MAP = [
    [0.0, 0.0, 0.0],  # Do nothing
    [-1.0, 0.0, 0.0],  # Steer left
    [1.0, 0.0, 0.0],  # Steer right
    [0.0, 1.0, 0.0],  # Gas
    [0.0, 0.0, 0.8],  # Brake
]

# Helper to map continuous action to the closest discrete action
def get_action_index(action):
    return np.argmin([np.linalg.norm(np.array(action) - np.array(a)) for a in ACTION_MAP])

# Initialize the environment
env = CarRacing(render_mode="human")
pygame.init()

# Function to handle keyboard input
def register_input(action):
    restart = False
    quit = False
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action[0] = -1.0  # Steer left
            if event.key == pygame.K_RIGHT:
                action[0] = 1.0   # Steer right
            if event.key == pygame.K_UP:
                action[1] = 1.0   # Gas
            if event.key == pygame.K_DOWN:
                action[2] = 0.8   # Brake
            if event.key == pygame.K_RETURN:
                restart = True
            if event.key == pygame.K_ESCAPE:
                quit = True
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                action[0] = 0.0
            if event.key == pygame.K_UP:
                action[1] = 0.0
            if event.key == pygame.K_DOWN:
                action[2] = 0.0
        if event.type == pygame.QUIT:
            quit = True
    return restart, quit

# Function to log sensor and action data
def collect_data(output_file, frame, reward, action, sensors, step, unique_id):
    # Save the current frame
    image_filename = os.path.join(IMAGE_DIR, f"{unique_id}_{step}.png")
    cv2.imwrite(image_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Get action index
    action_index = get_action_index(action)

    # Log data to CSV
    output_file.write(
        f"{image_filename},{reward:.2f},{sensors['true_speed']:.2f},"
        f"{sensors['abs_0']:.2f},{sensors['abs_1']:.2f},{sensors['abs_2']:.2f},"
        f"{sensors['abs_3']:.2f},{sensors['on_grass']},{action_index}\n"
    )

# Main loop for data collection
if __name__ == "__main__":
    # Open CSV file for writing
    with open(OUTPUT_CSV, "a") as output_file:
        if os.stat(OUTPUT_CSV).st_size == 0:  # Write header if file is empty
            output_file.write(
                "image_path,reward,true_speed,abs_0,abs_1,abs_2,abs_3,"
                "on_grass,action_index\n"
            )

        quit = False
        while not quit:
            env.reset()
            total_reward = 0.0
            step = 0
            restart = False
            unique_id = datetime.now().strftime("%Y%m%d%H%M%S")  # Unique ID for images
            action = np.array([0.0, 0.0, 0.0])  # Initial action array

            while True:
                # Handle user input
                restart, quit = register_input(action)

                # Step through the environment
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                # Collect sensor data
                sensors = {
                    "true_speed": env.get_speed(),
                    "abs_0": env.get_abs_sensors_0(),
                    "abs_1": env.get_abs_sensors_1(),
                    "abs_2": env.get_abs_sensors_2(),
                    "abs_3": env.get_abs_sensors_3(),
                    "on_grass": int(env.get_on_grass()),
                }

                # Collect and save data
                collect_data(output_file, obs, reward, action, sensors, step, unique_id)

                # Increment step count
                step += 1

                # End the episode if game is terminated or user quits/restarts
                if terminated or truncated or restart or quit:
                    print(f"Episode ended. Total Reward: {total_reward:.2f}")
                    break

    env.close()
