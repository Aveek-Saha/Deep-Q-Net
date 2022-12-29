import gym
import tensorflow as tf
import numpy as np
import random

class Environment():
    def __init__(self, env_name, no_op_steps=10, agent_history_length=4):
        self.env = gym.make(env_name)
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.agent_history_length = agent_history_length

    def preprocess(self, frame, height, width):
        processed_frame = tf.image.rgb_to_grayscale(frame)
        processed_frame = tf.image.crop_to_bounding_box(processed_frame, 34, 0, 160, 160)
        processed_frame = tf.image.resize(processed_frame, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        processed_frame = tf.reshape(processed_frame, (height, width, 1))

        return processed_frame

    def reset(self, evaluation=False):
        self.frame = self.env.reset()[0]
        self.last_lives = 0
        
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.env.step(1)

        self.state = np.repeat(self.preprocess(self.frame, 84, 84), self.agent_history_length, axis=2)

    def step(self, action):
        new_frame, reward, terminal, ignore, info = self.env.step(action)

        if info['lives'] < self.last_lives:
            life_lost = True
        else:
            life_lost = terminal
        self.last_lives = info['lives']

        processed_frame = self.preprocess(new_frame, 84, 84)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2)

        return processed_frame, new_frame, reward, terminal, life_lost