import numpy as np
from gymnasium.spaces import Box

class QLearningAgent():
    
    def __init__(self,env,Q_shape):
        if isinstance(env.observation_space,Box):
            self.discrete_env = False
            
            self.obs_shape = env.observation_space.shape[0]
            self.action_shape = env.action_space.n
            self.Q_table = np.zeros(Q_shape)
            
            
            self.bins = np.linspace(env.observation_space.low, env.observation_space.high, Q_shape[0]).T
        else:
            self.discrete_env =  True
            
            self.obs_shape = env.observation_space.n
            self.action_shape = env.action_space.n
            self.Q_table = np.zeros(Q_shape)
            
    def discretize(self, obs):
        index = []
        for i, o in enumerate(obs):
            index.append(np.digitize(o, self.bins[i]) -1) #Función suelo
        return tuple(index)
        
    def get_action(self, state, eps):
        if not self.discrete_env:
            state = self.discretize(state)
            
        if np.random.random() < eps:
            return np.random.choice([a for a in range(self.action_shape)])

        max_q = self.Q_table[state].max()
        max_q_actions = np.argwhere(self.Q_table[state] == max_q).flatten()
        return np.random.choice(max_q_actions)
    
        
    def update_q_table(self, state, action, reward, next_state, alpha, gamma):
        if not self.discrete_env:
            state = self.discretize(state)
            next_state = self.discretize(next_state)
        
        self.Q_table[state][action] += alpha * \
        (reward + gamma * np.max(self.Q_table[next_state]) - self.Q_table[state][action])
        
    def set_policy(self, policy):
        self.Q_table = policy