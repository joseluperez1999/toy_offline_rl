import numpy as np

class QLearningAgent():
    
    def __init__(self,env,Q_shape,alpha,gamma):
        self.obs_shape = env.observation_space.n
        self.action_shape = env.action_space.n
        self.Q_table = np.zeros(Q_shape)
        
        self.alpha = alpha
        self.gamma = gamma
        
    def get_action(self, state, eps):
        if np.random.random() < eps:
            return np.random.choice([a for a in range(self.action_shape)])

        max_q = self.Q_table[state].max()
        max_q_actions = np.argwhere(self.Q_table[state] == max_q).flatten()
        return np.random.choice(max_q_actions)
        
    def update_q_table(self, state, action, reward, next_state):
        self.Q_table[state][action] += self.alpha * \
        (reward + self.gamma * np.max(self.Q_table[next_state]) - self.Q_table[state][action])