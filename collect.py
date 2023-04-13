import argparse
import os

import gymnasium as gym
import joblib

from agents.QLearningAgent import QLearningAgent

from utils.quality import tq, saco


def collect(env, agent, datasets_path, n_episodes, expertise, stochastic, exploration_rate = 0):
    dataset = []
    for i in range(n_episodes):
        episode = []
        done = False
        truncated = False
        state = env.reset()[0]
        while not (done or truncated):
            action = agent.get_action(state, exploration_rate)
            next_state, reward, done, truncated, _ = env.step(action)
            
            transition = (state, action, next_state, reward, done, truncated)
            episode.append(transition)
            
            state = next_state
        dataset.append(episode)
    
    datasets_path = datasets_path + env.spec.id + "/"
    
    if not os.path.exists(datasets_path):
        print("Generating datasets directory for this env")
        os.makedirs(datasets_path)
        
    #Mejorar nomenclatura datasets
    if stochastic:
        joblib.dump(dataset,f'{datasets_path}dataset_{n_episodes}_{expertise}_{str(exploration_rate).replace(".", "-")}_s.pkl', compress=1)
    else:
        joblib.dump(dataset,f'{datasets_path}dataset_{n_episodes}_{expertise}_{str(exploration_rate).replace(".", "-")}.pkl', compress=1)

    return dataset

if __name__ == '__main__':
    #Meter argumentos de entorno, agente e hiperparámetros por argumento
    parser = argparse.ArgumentParser(description='Generate policies given an environment')
    parser.add_argument('--env', '-e', type=str, required=True, help='Environment name')
    parser.add_argument('--amount', '-a', type=int, required=True, help='Number of episodes to collect')
    parser.add_argument('--level', '-l', type=int, required=True, help='Expertise level of policy')
    parser.add_argument('--randomness', '-r', type=float, required=True, help='Exploration rate')
    parser.add_argument("--stochastic", "-s", action=argparse.BooleanOptionalAction, help="Set stochastic behaviour in case it exists for environment")
    args = parser.parse_args()
    
    match args.env:
        case "Frozen-Lake":
            env = gym.make("FrozenLake-v1", desc=None, map_name="8x8", is_slippery=args.stochastic)
        case "Mountain-Car":
            if args.stochastic:
                raise Exception("Environment has no stochastic version")
            env =gym.make('MountainCar-v0')
        case _:
            raise Exception("Environment not registred")
    
    if args.stochastic:
        policy = joblib.load(f"policies/{env.spec.id}/policy_episode_{args.level}_s.pkl") # type: ignore
    else:
        policy = joblib.load(f"policies/{env.spec.id}/policy_episode_{args.level}.pkl") # type: ignore
    agent = QLearningAgent(env,policy.shape)
    agent.set_policy(policy)
    
    path =  "datasets/"
    if not os.path.exists(path):
        os.mkdir(path)
        
    dataset = collect(env, agent, path, args.amount, args.level, args.stochastic, args.randomness)
    
    tq = tq(dataset)
    saco = saco(env,dataset)
    
    #TO:DO - Report module
    
    print("End")