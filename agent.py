from dqn import *
from collections import deque
import random
import numpy as np
import os
import json

class Agent():

    def __init__(self,
                 dqn,
                 target_dqn,
                 replay_buffer,
                 num_actions,
                 input_shape=(84, 84),
                 batch_size=32,
                 agent_history_length=4,
                 epsilon_init=1,
                 epsilon_final=0.1,
                 epsilon_final_frame=0.01,
                 epsilon_eval=0.0,
                 epsilon_annealing=1000000,
                 replay_buffer_init_size=50000,
                 max_frames=25000000):
        
        self.num_actions = num_actions
        self.input_shape = input_shape
        self.agent_history_length = agent_history_length

        self.max_frames = max_frames
        self.batch_size = batch_size
        self.replay_buffer_init_size = replay_buffer_init_size
        self.replay_buffer = replay_buffer

        self.epsilon_annealing = epsilon_annealing
        self.epsilon_eval = epsilon_eval
        self.epsilon_final = epsilon_final
        self.epsilon_init = epsilon_init
        self.epsilon_final_frame = epsilon_final_frame

        self.slope = -(self.epsilon_init - self.epsilon_final) / self.epsilon_annealing
        self.intercept = self.epsilon_init - self.slope*self.replay_buffer_init_size
        self.slope_2 = -(self.epsilon_final - self.epsilon_final_frame) / (self.max_frames - self.epsilon_annealing - self.replay_buffer_init_size)
        self.intercept_2 = self.epsilon_final_frame - self.slope_2*self.max_frames

        self.DQN = dqn
        self.target_dqn = target_dqn

    def calc_epsilon(self, frame_number, evaluation=False):
        if evaluation:
            return self.epsilon_eval
        elif frame_number < self.replay_buffer_init_size:
            return self.epsilon_init
        elif frame_number >= self.replay_buffer_init_size and frame_number < self.replay_buffer_init_size + self.epsilon_annealing:
            return self.slope*frame_number + self.intercept
        elif frame_number >= self.replay_buffer_init_size + self.epsilon_annealing:
            return self.slope_2*frame_number + self.intercept_2

    def get_action(self, frame_number, state, evaluation=False):
        eps = self.calc_epsilon(frame_number, evaluation)

        if np.random.rand(1) < eps:
            return np.random.randint(0, self.num_actions)

        q_vals = self.DQN.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.agent_history_length)))[0]
        return q_vals.argmax()

    def get_intermediate_representation(self, state, layer_names=None, stack_state=True):
        if isinstance(layer_names, list) or isinstance(layer_names, tuple):
            layers = [self.DQN.get_layer(name=layer_name).output for layer_name in layer_names]
        else:
            layers = self.DQN.get_layer(name=layer_names).output

        temp_model = tf.keras.Model(self.DQN.inputs, layers)

        if stack_state:
            if len(state.shape) == 2:
                state = state[:, :, np.newaxis]
            state = np.repeat(state, self.agent_history_length, axis=2)

        return temp_model.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.agent_history_length)))

    def add_experience(self, action, frame, reward, terminal, clip_reward=True):
        self.replay_buffer.add_experience(action, frame, reward, terminal, clip_reward)

    def update_target_network(self):
        self.target_dqn.set_weights(self.DQN.get_weights())

    def learn(self, batch_size, gamma):
        states, actions, rewards, new_states, terminal_flags = self.replay_buffer.get_minibatch()

        arg_q_max = self.DQN.predict(new_states).argmax(axis=1)

        future_q_vals = self.target_dqn.predict(new_states)
        double_q = future_q_vals[range(batch_size), arg_q_max]
        target_q = rewards + (gamma*double_q * (1-terminal_flags))

        with tf.GradientTape() as tape:
            q_values = self.DQN(states)

            one_hot_actions = tf.keras.utils.to_categorical(actions, self.num_actions, dtype=np.float32)
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            error = Q - target_q
            loss = tf.keras.losses.Huber()(target_q, Q)

        model_gradients = tape.gradient(loss, self.DQN.trainable_variables)
        self.DQN.optimizer.apply_gradients(zip(model_gradients, self.DQN.trainable_variables))

        return float(loss.numpy()), error

    def save(self, folder_name, **kwargs):
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        self.DQN.save(folder_name + '/dqn.h5')
        self.target_dqn.save(folder_name + '/target_dqn.h5')

        self.replay_buffer.save(folder_name + '/replay-buffer')

        with open(folder_name + '/meta.json', 'w+') as f:
            f.write(json.dumps({**{'buff_count': self.replay_buffer.count, 'buff_curr': self.replay_buffer.current}, **kwargs})) 

    def load(self, folder_name, load_replay_buffer=True):
        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        self.DQN = tf.keras.models.load_model(folder_name + '/dqn.h5')
        self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5')
        self.optimizer = self.DQN.optimizer

        if load_replay_buffer:
            self.replay_buffer.load(folder_name + '/replay-buffer')

        with open(folder_name + '/meta.json', 'r') as f:
            meta = json.load(f)

        if load_replay_buffer:
            self.replay_buffer.count = meta['buff_count']
            self.replay_buffer.current = meta['buff_curr']

        del meta['buff_count'], meta['buff_curr']
        return meta


