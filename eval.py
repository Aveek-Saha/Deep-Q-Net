# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'BreakoutDeterministic-v4'

# Loading and saving information.
# If LOAD_FROM is None, it will train a new agent.
# If SAVE_PATH is None, it will not save the agent
LOAD_FROM = 'breakout-saves/save-3'
# LOAD_FROM = None
SAVE_PATH = 'breakout-saves'
LOAD_REPLAY_BUFFER = True

WRITE_TENSORBOARD = True
TENSORBOARD_DIR = 'tensorboard/'

CLIP_REWARD = True                # Any positive reward is +1, and negative reward is -1, 0 is unchanged


TOTAL_FRAMES = 300000           # Total number of frames to train for
MAX_EPISODE_LENGTH = 3600        # Maximum length of an episode (in frames).  18000 frames / 60 fps = 5 minutes
FRAMES_BETWEEN_EVAL = 10000      # Number of frames between evaluations
EVAL_LENGTH = 10000               # Number of frames to evaluate for
UPDATE_FREQ = 1000               # Number of actions chosen between updating the target network

DISCOUNT_FACTOR = 0.99            # Gamma, how much to discount future rewards
MIN_REPLAY_BUFFER_SIZE = 5000    # The minimum size the replay buffer must be before we start to update the agent
MEM_SIZE = 10000                # The maximum size of the replay buffer

MAX_NOOP_STEPS = 10               # Randomly perform this number of actions before every evaluation to give it an element of randomness
UPDATE_FREQ = 4                   # Number of actions between gradient descent steps

INPUT_SHAPE = (84, 84)            # Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
FRAME_HEIGHT = 84
FRAME_WIDTH = 84
BATCH_SIZE = 32                   # Number of samples the agent learns from at once
LEARNING_RATE = 0.00001

import imageio
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage,
                                  TextArea)
from sklearn.decomposition import PCA

from dqn import *
from memory import *
from agent import *
from environment import *

# This will usually fix any issues involving the GPU and cuDNN
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

# Change this to the path of the model you would like to visualize
# RESTORE_PATH = 'breakout-saves/save-dueling'
# RESTORE_PATH = 'breakout-saves/save-3'

if LOAD_FROM is None:
    raise UserWarning('Please change the variable `RESTORE_PATH` to where you would like to load the model from. If you haven\'t trained a model, try \'example-save\'')

ENV_NAME = 'BreakoutDeterministic-v4'
FRAMES_TO_VISUALIZE = 750
FRAMES_TO_ANNOTATE = 0

# Create environment
game_wrapper = Environment(ENV_NAME, MAX_NOOP_STEPS)
print("The environment has the following {} actions: {}".format(game_wrapper.env.action_space.n, game_wrapper.env.unwrapped.get_action_meanings()))

# Create agent

MAIN_DQN = DeepQNetwork(LEARNING_RATE, game_wrapper.env.action_space.n, FRAME_HEIGHT, FRAME_WIDTH, 4)
TARGET_DQN = DeepQNetwork(LEARNING_RATE, game_wrapper.env.action_space.n, FRAME_HEIGHT, FRAME_WIDTH, 4)
# MAIN_DQN = DuellingDQN(LEARNING_RATE, game_wrapper.env.action_space.n, FRAME_HEIGHT, FRAME_WIDTH, 4)
# TARGET_DQN = DuellingDQN(LEARNING_RATE, game_wrapper.env.action_space.n, FRAME_HEIGHT, FRAME_WIDTH, 4)
# MAIN_DQN.compile(tf.keras.optimizers.Adam(LEARNING_RATE), loss=tf.keras.losses.Huber())
# TARGET_DQN.compile(tf.keras.optimizers.Adam(LEARNING_RATE), loss=tf.keras.losses.Huber())
TARGET_DQN.set_weights(MAIN_DQN.get_weights())

replay_buffer = ReplayMemory(size=MEM_SIZE, input_shape=INPUT_SHAPE)
agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE)


# print('Loading agent...')
# agent.load(LOAD_FROM)

# print('Generating embeddings...')
# embeddings = []
# values = []

# frame_indices = np.random.choice(agent.replay_buffer.count, size=FRAMES_TO_VISUALIZE)

# for frame in agent.replay_buffer.frames[frame_indices]:
#     # TODO: combine things into one
#     embeddings.append(agent.get_intermediate_representation(frame, 'flatten_1')[0])
#     values.append(agent.get_intermediate_representation(frame, 'dense')[0])

# print('Fitting PCA...')
# pca = PCA(2)
# pca_embeddings = pca.fit_transform(embeddings)

# print('Displaying...')
# fig, ax = plt.subplots()
# indices = np.random.choice(100, FRAMES_TO_ANNOTATE)
# for i, frame in enumerate(agent.replay_buffer.frames[frame_indices]):
#     if i in indices:
#         im = OffsetImage(frame, zoom=2, cmap='gray')
#         im.image.axes = ax
#         ab = AnnotationBbox(im, pca_embeddings[i],
#                             xybox=(-120., 120.),
#                             xycoords='data',
#                             boxcoords="offset points",
#                             pad=0.3,
#                             arrowprops=dict(arrowstyle="->"))

#         ax.add_artist(ab)

# plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=values, cmap='jet')
# plt.colorbar().set_label('Q-Value')
# # plt.show()
# plt.savefig("replay_DuellingDQN.png", dpi=300, bbox_inches = "tight")

def generate_gif(frame_number, frames_for_gif, reward, path):
    """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """
    for idx, frame_idx in enumerate(frames_for_gif): 
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3), 
                                     preserve_range=True, order=0).astype(np.uint8)
        
    imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}', 
                    frames_for_gif, duration=1/30)


print('Loading model...')
agent.load(LOAD_FROM)
print('Loaded')

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

    # Breakout require a "fire" action (action #1) to start the
    # game each time a life is lost.
    # Otherwise, the agent would sit around doing nothing.
    action = 1 if life_lost else agent.get_action(0, game_wrapper.state, evaluation=True)

    # Step action
    _, new_frame, reward, terminal, life_lost = game_wrapper.step(action)
    frames_for_gif.append(new_frame)
    

    episode_reward_sum += reward

    # On game-over
    if terminal:
        print(f'Game over, reward: {episode_reward_sum}, frame: {frame}/{EVAL_LENGTH}')
        eval_rewards.append(episode_reward_sum)
        generate_gif(0, frames_for_gif, episode_reward_sum, gif_path)
        frames_for_gif = []

print('Average reward:', np.mean(eval_rewards) if len(eval_rewards) > 0 else episode_reward_sum)