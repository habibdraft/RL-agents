import gym
import agents

env = gym.make('Taxi-v3')
agent = agents.SarsaAgent(actions=env.action_space.n)

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
        if agent.lastAction is not None:
            agent.learn(agent.lastState, agent.lastAction, agent.lastReward, state, action)
            
        agent.lastAction = action
        agent.lastState = state
        
        state, reward, done, info = env.step(action)
        agent.lastReward = reward
        
        steps += 1
        totalReward += reward

    totalSteps += steps
    
print("Average timesteps taken: {}".format(totalSteps/episodes))
print("Average reward: {}".format(totalReward/episodes))
print("Total reward: {}".format(totalReward))
