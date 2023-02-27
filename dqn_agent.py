class Agent:
  def __init__(self, states, actions, epsilon=0.7, alpha=0.7, gamma=0.7, epsilon_decay=0.9):
          self.q = [[0]*actions]*states
          self.actions = [i for i in range(actions)]
          self.epsilon = epsilon
          self.alpha = alpha
          self.gamma = gamma
          self.epsilon_decay = epsilon_decay
          self.last_state = None
          self.last_action = None
          self.last_reward = None
