ENV_NAME = 'BreakoutDeterministic-v4'
LOAD_FROM = None
SAVE_PATH = None
# SAVE_PATH = 'breakout-saves'
LOAD_REPLAY_BUFFER = True

CLIP_REWARD = True


TOTAL_FRAMES = 300000
MAX_EPISODE_LENGTH = 3600
FRAMES_BETWEEN_EVAL = 10000
EVAL_LENGTH = 1000
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

from environment import *
from dqn import *
from memory import *
from agent import *
import time

game_wrapper = Environment(ENV_NAME, MAX_NOOP_STEPS)
print("The environment has the following {} actions: {}".format(game_wrapper.env.action_space.n, game_wrapper.env.unwrapped.get_action_meanings()))


MAIN_DQN = DeepQNetwork(LEARNING_RATE, game_wrapper.env.action_space.n, FRAME_HEIGHT, FRAME_WIDTH, 4)
TARGET_DQN = DeepQNetwork(LEARNING_RATE, game_wrapper.env.action_space.n, FRAME_HEIGHT, FRAME_WIDTH, 4)

# MAIN_DQN = DuellingDQN(LEARNING_RATE, game_wrapper.env.action_space.n, FRAME_HEIGHT, FRAME_WIDTH, 4)
# TARGET_DQN = DuellingDQN(LEARNING_RATE, game_wrapper.env.action_space.n, FRAME_HEIGHT, FRAME_WIDTH, 4)
TARGET_DQN.set_weights(MAIN_DQN.get_weights())

replay_buffer = ReplayMemory(size=MEM_SIZE, input_shape=INPUT_SHAPE)
agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE)

if LOAD_FROM is None:
    frame_number = 0
    rewards = []
    loss_list = []
else:
    print('Loading from', LOAD_FROM)
    meta = agent.load(LOAD_FROM, LOAD_REPLAY_BUFFER)

    frame_number = meta['frame_number']
    rewards = meta['rewards']
    loss_list = meta['loss_list']


while frame_number < TOTAL_FRAMES:

    epoch_frame = 0
    while epoch_frame < FRAMES_BETWEEN_EVAL:
        start_time = time.time()
        game_wrapper.reset()
        life_lost = True
        episode_reward_sum = 0
        for _ in range(MAX_EPISODE_LENGTH):
            action = agent.get_action(frame_number, game_wrapper.state)

            processed_frame, new_frame, reward, terminal, life_lost = game_wrapper.step(action)
            frame_number += 1
            epoch_frame += 1
            episode_reward_sum += reward

            agent.add_experience(action,
                                processed_frame[:, :, 0],
                                reward, 
                                life_lost,
                                clip_reward=CLIP_REWARD)

            if frame_number % UPDATE_FREQ == 0 and agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                loss, _ = agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR)
                loss_list.append(loss)
            if frame_number % UPDATE_FREQ == 0 and frame_number > MIN_REPLAY_BUFFER_SIZE:
                agent.update_target_network()

            if terminal:
                terminal = False
                break

        rewards.append(episode_reward_sum)
    terminal = True
    eval_rewards = []
    evaluate_frame_number = 0

    for _ in range(EVAL_LENGTH):
        if terminal:
            game_wrapper.reset(evaluation=True)
            life_lost = True
            episode_reward_sum = 0
            terminal = False

        action = 1 if life_lost else agent.get_action(frame_number, game_wrapper.state, evaluation=True)

        _, new_frame, reward, terminal, life_lost = game_wrapper.step(action)
        evaluate_frame_number += 1
        episode_reward_sum += reward

        if terminal:
            eval_rewards.append(episode_reward_sum)

    if len(eval_rewards) > 0:
        final_score = np.mean(eval_rewards)
    else:
        final_score = episode_reward_sum
    print('Evaluation score:', final_score)

    if len(rewards) > 100 and SAVE_PATH is not None:
        agent.save(f'{SAVE_PATH}/save-dqn', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
