import argparse
import os
from tqdm import tqdm

import gymnasium as gym
import joblib

from agents.QLearningAgent import QLearningAgent

TRAIN_EPISODES = 10000
VALIDATION_STEPS = 50
VALIDATION_EPISODES = 100

ALPHA = 0.1
GAMMA = 0.99

EPSILON = 0.2
EPSILON_DECAY = 0

NUM_BINS = 40
# 1400, 2900
POLICY_SAVES = 50
            
def run_episode(agent, env, eps, validation = False, render = False):
    done = False
    truncated = False
    state = env.reset()[0]
    accumulated_reward = 0
    
    while not (done or truncated):
        
        action = agent.get_action(state,eps)
            
        next_state, reward, done, truncated ,_ = env.step(action)
        
        if not validation:
            agent.update_q_table(state, action, reward, next_state, ALPHA, GAMMA)

        state = next_state
        accumulated_reward += reward
        
        if render:
            env.render()
    
    #print(env.desc[state//8][state%8])
                    
    return accumulated_reward

def save_policy(path, agent, episode, stochastic):
    if not os.path.exists(path):
        print("Generating policies directory for this env")
        os.makedirs(path)
    if stochastic:
        joblib.dump(agent.Q_table,f'{path}policy_episode_{episode}_s.pkl', compress=1)
    else:
        joblib.dump(agent.Q_table,f'{path}policy_episode_{episode}_s.pkl', compress=1)
        
def train(env, agent, policies_path, eps, stochastic):
    env_path = policies_path + env.spec.id + "/"
    for episode in tqdm(range(TRAIN_EPISODES)):
        if episode == 0:
            print(f"Saving random policy, path: {env_path}")
            save_policy(env_path,agent,episode, stochastic)
        
        run_episode(agent,env,eps)
        
        if bool(EPSILON_DECAY) and eps > 0.005:
            eps -= EPSILON_DECAY    
            
        if (episode + 1) % (TRAIN_EPISODES // VALIDATION_STEPS) == 0:
            validation_reward = 0
            for i in range(VALIDATION_EPISODES):
                accumulated_reward = run_episode(agent, env, 0, True)
                validation_reward += accumulated_reward
            validation_reward /= VALIDATION_EPISODES
            print(f"{episode+1} episodes: validation_reward = {validation_reward}, epsilon = {eps}")
            
        if (episode + 1) % (TRAIN_EPISODES // POLICY_SAVES) == 0:
            print(f"Saving policy at episode: {episode + 1}, path: {env_path}")
            save_policy(env_path,agent,episode + 1, stochastic)
        
if __name__ == '__main__':
    #Meter argumentos de entorno, agente e hiperparámetros
    parser = argparse.ArgumentParser(description='Generate policies given an environment')
    parser.add_argument('--env', '-e', type=str, required=True, help='Environment name')
    parser.add_argument("--stochastic", "-s", action=argparse.BooleanOptionalAction, help="Set stochastic behaviour in case it exists for environment")
    args = parser.parse_args()
    
    match args.env:
        case "Frozen-Lake":
            env = gym.make("FrozenLake-v1", desc=None, map_name="8x8", is_slippery=args.stochastic)
            Q_shape = (env.observation_space.n,env.action_space.n) # type: ignore
        case "Mountain-Car":
            if args.stochastic:
                raise Exception("Environment has no stochastic version")
            env =gym.make('MountainCar-v0')
            Q_shape = [NUM_BINS] * env.observation_space.shape[0] + [env.action_space.n] # type: ignore
        case _:
            raise Exception("Environment not registred")
    
    policies_path = "policies/"
    if not os.path.exists(policies_path):
        os.mkdir(policies_path)
        
    agent = QLearningAgent(env,Q_shape)
    train(env, agent, policies_path, EPSILON, args.stochastic)
    
    print('End')
    
    