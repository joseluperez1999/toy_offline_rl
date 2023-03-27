import argparse
import os

import gymnasium as gym
import joblib

from agents.QLearningAgent import QLearningAgent

ALPHA = 0.7
GAMMA = 0.95
EPSILON = 0.2
NUM_BINS = 30

TRAIN_EPISODES = 100000
VALIDATION_STEPS = 10
VALIDATION_EPISODES = 100

POLICY_SAVES = 4
            

def run_episode(agent, env, validation = False, render = False):
    done = False
    state = env.reset()[0]
    accumulated_reward = 0
    while not done:
        
        if not validation:
            action = agent.get_action(state,EPSILON)
        else:
            action = agent.get_action(state,0)
            
        next_state, reward, done,_ ,_ = env.step(action)
        
        if not validation:
            agent.update_q_table(state, action, reward, next_state, ALPHA, GAMMA)

        state = next_state
        accumulated_reward += reward
        
        if render:
            env.render()
            
    return accumulated_reward

def save_policy(path,agent,episode):
    if not os.path.exists(path):
        print("Generating policies directory for this env")
        os.makedirs(path)
    joblib.dump(agent.Q_table,f'{path}policy_episode_{episode}.pkl', compress=1)
        
def train(env,agent,policies_path):
    env_path = policies_path + env.spec.id + "/"
    for episode in range(TRAIN_EPISODES):
        if episode == 0:
            print(f"Saving random policy, path: {env_path}")
            save_policy(env_path,agent,episode)
        
        run_episode(agent,env)
        
        if (episode + 1) % (TRAIN_EPISODES // VALIDATION_STEPS) == 0:
            validation_reward = 0
            for i in range(VALIDATION_EPISODES):
                accumulated_reward = run_episode(agent, env, True)
                validation_reward += accumulated_reward
            validation_reward /= VALIDATION_EPISODES
            print(f"{episode+1} episodes: validation_reward = {validation_reward}")
            
        if (episode + 1) % (TRAIN_EPISODES // POLICY_SAVES) == 0:
            print(f"Saving policy at episode: {episode + 1}, path: {env_path}")
            save_policy(env_path,agent,episode + 1)
        
if __name__ == '__main__':
    #Meter argumentos de entorno, agente e hiperparámetros
    parser = argparse.ArgumentParser(description='Generate policies given an environment')
    parser.add_argument('--env', '-e', type=str, required=True, help='Environment name')
    args = parser.parse_args()
    
    match args.env:
        case "Frozen-Lake":
            env = gym.make("FrozenLake-v1", desc=None, map_name="8x8", is_slippery=False)
            Q_shape = (env.observation_space.n,env.action_space.n) # type: ignore
        case "Mountain-Car":
            env =gym.make('MountainCar-v0')
            Q_shape = [NUM_BINS] * env.observation_space.shape[0] + [env.action_space.n]
        case _:
            raise Exception("Environment not registred")
    
    policies_path = "policies/"
    if not os.path.exists(policies_path):
        os.mkdir(policies_path)
        
    agent = QLearningAgent(env,Q_shape)
    train(env,agent,policies_path)
    
    print('End')
    
    