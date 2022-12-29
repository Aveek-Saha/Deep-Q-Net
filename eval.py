ENV_NAME = 'BreakoutDeterministic-v4'

LOAD_FROM = 'breakout-saves/save-dqn'
# LOAD_FROM = None
SAVE_PATH = 'breakout-saves'
LOAD_REPLAY_BUFFER = True

TOTAL_FRAMES = 300000
MAX_EPISODE_LENGTH = 3600
FRAMES_BETWEEN_EVAL = 10000
EVAL_LENGTH = 10000
UPDATE_FREQ = 1000

DISCOUNT_FACTOR = 0.99
MIN_REPLAY_BUFFER_SIZE = 5000
MEM_SIZE = 10000

MAX_NOOP_STEPS = 10
UPDATE_FREQ = 4

INPUT_SHAPE = (84, 84)
FRAME_HEIGHT = 84
FRAME_WIDTH = 84
BATCH_SIZE = 32
LEARNING_RATE = 0.00001

import imageio
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dqn import *
from memory import *
from agent import *
from environment import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


if LOAD_FROM is None:
    raise UserWarning('No saves')

game_wrapper = Environment(ENV_NAME, MAX_NOOP_STEPS)

MAIN_DQN = DeepQNetwork(LEARNING_RATE, game_wrapper.env.action_space.n, FRAME_HEIGHT, FRAME_WIDTH, 4)
TARGET_DQN = DeepQNetwork(LEARNING_RATE, game_wrapper.env.action_space.n, FRAME_HEIGHT, FRAME_WIDTH, 4)
# MAIN_DQN = DuellingDQN(LEARNING_RATE, game_wrapper.env.action_space.n, FRAME_HEIGHT, FRAME_WIDTH, 4)
# TARGET_DQN = DuellingDQN(LEARNING_RATE, game_wrapper.env.action_space.n, FRAME_HEIGHT, FRAME_WIDTH, 4)
TARGET_DQN.set_weights(MAIN_DQN.get_weights())

replay_buffer = ReplayMemory(size=MEM_SIZE, input_shape=INPUT_SHAPE)
agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE)


def generate_gif(frame_number, frames_for_gif, reward, path):
    for idx, frame_idx in enumerate(frames_for_gif): 
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3), 
                                     preserve_range=True, order=0).astype(np.uint8)
        
    imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}', 
                    frames_for_gif, duration=1/30)


agent.load(LOAD_FROM)

gif_path = "GIF/"
os.makedirs(gif_path, exist_ok=True)

terminal = True
eval_rewards = []
frames_for_gif = []

for frame in range(EVAL_LENGTH):
    if terminal:
        game_wrapper.reset(evaluation=True)
        life_lost = True
        episode_reward_sum = 0
        terminal = False
        
    action = 1 if life_lost else agent.get_action(0, game_wrapper.state, evaluation=True)

    _, new_frame, reward, terminal, life_lost = game_wrapper.step(action)
    frames_for_gif.append(new_frame)
    

    episode_reward_sum += reward

    if terminal:
        print(f'Game ended, reward: {episode_reward_sum}, frame: {frame}/{EVAL_LENGTH}')
        eval_rewards.append(episode_reward_sum)
        generate_gif(0, frames_for_gif, episode_reward_sum, gif_path)
        frames_for_gif = []

print('Average reward:', np.mean(eval_rewards) if len(eval_rewards) > 0 else episode_reward_sum)