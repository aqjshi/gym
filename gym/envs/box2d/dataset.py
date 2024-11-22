__credits__ = ["Qingjain Shi Luke He"]
#Luke's version hello

import math
from typing import Optional, Union

import numpy as np
import os
import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.error import DependencyNotInstalled, InvalidAction
from gym.utils import EzPickle
from gym.envs.box2d.car_racing import CarRacing

import Box2D
from Box2D.b2 import contactListener, fixtureDef, polygonShape
import pygame
from pygame import gfxdraw


import cv2
import numpy as np

pygame.init()

from datetime import datetime 
if __name__ == "__main__":
    # Initialize action array
    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True
    
    output_csv = "output.csv"
    image_dir = "images"
    
    

    # Initialize environment
    env = CarRacing(render_mode="human")
    quit = False

 
    # Open log files for writing
    with open(output_csv, "a") as output:

        unique_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        if os.stat(output_csv).st_size == 0:
            output.write("unique_id,step,reward,steer,gas,brake,true_speed,abs_0,abs_1,abs_2,abs_3,on_grass,sensor_0,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5,sensor_6,sensor_7\n")
        while not quit:
            #generate unique id
            env.reset()
            total_reward = 0.0
            steps = 0
            restart = False
            
            while True:
                action_space = env.action_space  # Get the current action space dynamically
                register_input()  # Log keyboard input with action space
                s, r, terminated, truncated, info, on_grass, sensor_readings = env.step(a)
                total_reward += r
                output.write(f"{unique_id},{steps},{r},{a[0]},{a[1]},{a[2]},{env.get_speed():.2f},{env.get_abs_sensors_0():.2f},{env.get_abs_sensors_1():.2f},{env.get_abs_sensors_2():.2f},{env.get_abs_sensors_3():.2f}, {on_grass}, {sensor_readings[0]}, {sensor_readings[1]}, {sensor_readings[2]}, {sensor_readings[3]}, {sensor_readings[4]}, {sensor_readings[5]}, {sensor_readings[6]}, {sensor_readings[7]}\n")


                if (steps % 10 == 0 and steps > 30) or terminated:
                    image_filename = os.path.join(image_dir, f"{unique_id}_{steps}.png")
                    env.save_image(image_filename, quality=30, resolution=(96, 96))
                    # print(f"\naction: {a[0]:+0.2f} {a[1]:+0.2f} {a[2]:+0.2f}")
                    # print(f"step {steps} total_reward {total_reward:+0.2f}")
                steps += 1
                if terminated or restart or quit:
                    break
            env.close()
