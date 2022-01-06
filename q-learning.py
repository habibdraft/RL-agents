import gym
import agents

env = gym.make('Taxi-v3')
agent = agents.QLearnAgent(actions=env.action_space.n)

episodes = 10000
totalSteps = 0
totalReward = 0

for episode in range(episodes):
    
    state = env.reset()
    reward = 0
    steps = 0
    done = False
    
    while not done:
        action = agent.act(state) 
        next_state, reward, done, info = env.step(action)
        agent.learn(state, action, reward, next_state, _)
        state = next_state
        
        steps += 1
        totalReward += reward

    totalSteps += steps
    
print("Average timesteps taken: {}".format(totalSteps/episodes))
print("Average reward: {}".format(totalReward/episodes))
print("Total reward: {}".format(totalReward))
