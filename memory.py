import numpy as np
from collections import deque
import random
import os

class ReplayMemory():
    def __init__(self, input_shape=(84, 84), size=1000000, agent_history_length=4, batch_size=32):
        
        self.size = size
        self.input_shape = input_shape
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        self.rewards = np.empty(self.size, dtype=np.float32)
        self.actions = np.empty(self.size, dtype=np.int32)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.frames = np.empty((self.size, self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        
        self.indices = np.empty(self.batch_size, dtype=np.int32)
        self.states = np.empty((self.batch_size, self.agent_history_length, 
                                self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length, 
                                    self.input_shape[0], self.input_shape[1]), dtype=np.uint8)

    def add_experience(self, action, frame, reward, terminal, clip_reward=True):
        if frame.shape != (self.input_shape[0], self.input_shape[1]):
            raise ValueError('Wrong Dimension!')
        
        if clip_reward:
            reward = np.sign(reward)

        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size
             
    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("No replay memory")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be at least 3")
        return self.frames[index-self.agent_history_length+1:index+1, ...]
        
    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index
            
    def get_minibatch(self):
        if self.count < self.agent_history_length:
            raise ValueError('Not enough for mini batch')
        
        self._get_valid_indices()
            
        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
        
        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]

    def save(self, folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

        np.save(folder + '/actions.npy', self.actions)
        np.save(folder + '/terminal_flags.npy', self.terminal_flags)
        np.save(folder + '/rewards.npy', self.rewards)
        np.save(folder + '/frames.npy', self.frames)

    def load(self, folder):
        self.actions = np.load(folder + '/actions.npy')
        self.terminal_flags = np.load(folder + '/terminal_flags.npy')
        self.rewards = np.load(folder + '/rewards.npy')
        self.frames = np.load(folder + '/frames.npy')