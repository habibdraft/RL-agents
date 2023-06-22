import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

env = gym.make('Taxi-v3')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

class Agent:
    def __init__(self, states, actions, epsilon=0.7, alpha=0.7, gamma=0.7, epsilon_decay=0.9):
        self.q = {}
        self.actions = [i for i in range(actions)]
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.last_state = None
        self.last_action = None
        self.last_reward = None
    
    def act(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            state_q = [self.q.get(state, a) for a in self.actions]
            max_q = max(state_q)
            count = state_q.count(max_q)
            if count > 1:
                max_actions = [a for a in self.actions if state_q[a] == max_q] 
                index = random.choice(max_actions)
            else:
                index = state_q.index(max_q)

            action = self.actions[index]
            self.epsilon = self.epsilon * self.epsilon_decay
            
        return action
    
    def learn(self, state, action, reward, state2, _):
        max_q = max([self.q.get(state2, a) for a in self.actions])
        self.update(state, action, reward, reward + self.gamma * max_q)
    
    def update(self, state, action, reward, new_value):
        current_q = self.q.get(state, action)
        if current_q == 0:
            self.q[state][action] = reward 
        else:
            self.q[state][action] = current_q + self.alpha * (new_value - current_q)

n_actions = env.action_space.n
state, info = env.reset()
n_observations = env.observation_space.n

agent = Agent(states=n_observations, actions=n_actions)

episodes = 10000
total_steps = 0
total_reward = 0

for episode in range(episodes):
    
    state = env.reset()
    reward = 0
    steps = 0
    done = False
    
    while not done:
        action = agent.act(state) 
        next_state, reward, terminated, truncated, info = env.step(action.item())
        agent.learn(state, action, reward, next_state, _)
        state = next_state
        
        steps += 1
        total_reward += reward

    total_steps += steps
    
print("Average timesteps taken: {}".format(total_steps/episodes))
print("Average reward: {}".format(total_reward/episodes))
print("Total reward: {}".format(total_reward))
