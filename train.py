from agent import *
import gym

EPISODES = 1000

env = gym.make("Breakout-v4", obs_type= "grayscale")
state_size = (210,160)
# state_size = env.observation_space.shape
action_size = env.action_space.n
agent = Agent(state_size, action_size)

done = False
batch_size = 32

for e in range(EPISODES):
    state, info = env.reset()
    # state = np.reshape(state, [1, state_size])
    for time in range(500):
        # env.render()
        action = agent.move(state)
        next_state, reward, done, trunc, info = env.step(action)
        reward = reward if not done else -10
        # next_state = np.reshape(next_state, [1, state_size])
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                    .format(e, EPISODES, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay_memory(batch_size)