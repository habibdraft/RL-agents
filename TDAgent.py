import numpy as np
import random

class TDAgent:
    def __init__(self, actions, epsilon=0.7, alpha=0.1, gamma=0.1, epsilonDecay=0.9):
        self.Q = {}
        self.actions = [i for i in range(actions)]
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilonDecay = epsilonDecay
        self.lastState = None
        self.lastAction = None
        self.lastReward = None
    
    def update(self, state, action, reward, newValue):
        currentQ = self.getQValue(state, action)
        if currentQ == 0:
            self.Q[(state, action)] = reward 
        else:
            self.Q[(state, action)] = currentQ + self.alpha * (newValue - currentQ)
            
    def learn(self, state, action, reward, state2, action2):
        pass
        
    def act(self, state):
        if np.random.rand() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQValue(state, a) for a in self.actions] 
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [a for a in self.actions if q[a] == maxQ] 
                index = random.choice(best)
            else:
                index = q.index(maxQ)

            action = self.actions[index]
            self.epsilon = self.epsilon * self.epsilonDecay
            
        return action
    
    def getQValue(self, state, action):
        return self.q.get((state, action), 0.0)

class SarsaAgent(TDAgent):
    
    def learn(self, state, action, reward, state2, action2):
        nextQ = self.getQValue(state2, action2)
        self.update(state, action, reward, reward + self.gamma * nextQ)

class QLearnAgent(TDAgent):
    
    def learn(self, state, action, reward, state2, _):
        maxQ = max([self.getQValue(state2, a) for a in self.actions])
        self.update(state, action, reward, reward + self.gamma * maxQ)

class ExpectedSarsaAgent(TDAgent):
    
    def learn(self, state, action, reward, state2):
        meanQ = mean([self.getQValue(state2, a) for a in self.actions])
        self.updateQ(state, action, reward, reward + self.gamma * meanQ)

