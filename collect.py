import gymnasium as gym
import joblib
from QLearningAgent import QLearningAgent
import os

def collect(env, agent, datasets_path, n_episodes, exploration_rate = 0):
    dataset = []
    for i in range(n_episodes):
        episode = []
        done = False
        state = env.reset()[0]
        while not done:
            action = agent.get_action(state, exploration_rate)
            next_state, reward, done,_ ,_ = env.step(action)
            
            transition = (state,action,next_state,reward,done)
            episode.append(transition)
            
            state = next_state
        dataset.append(episode)
    
    datasets_path = datasets_path + "Frozen_Lake/"
    
    if not os.path.exists(datasets_path):
        print("Generating datasets directory for this env")
        os.makedirs(datasets_path)
    joblib.dump(dataset,f'{datasets_path}dataset_{n_episodes}_{exploration_rate}.pkl', compress=1) #Mejorar nomenclatura datasets

if __name__ == '__main__':
    #Meter argumentos de entorno, agente e hiperparámetros por argumento
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False)
    policy = joblib.load("policies/Frozen_Lake/policy_episode_2000.pkl") #Hardcodeado para probar
    agent = QLearningAgent(env,policy.shape)
    agent.set_policy(policy)
    
    path =  "datasets/"
    if not os.path.exists(path):
        os.mkdir(path)
    collect(env, agent, path, 1000, 0)
    print("End")