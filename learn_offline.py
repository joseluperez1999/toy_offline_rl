import argparse
import os
from time import time

import gymnasium as gym
import joblib

from agents.QLearningAgent import QLearningAgent

TRAIN_EPISODES = 50
VALIDATION_EPISODES = 10

ALPHA = 0.2
GAMMA = 0.99


def dataset_train(agent, dataset):
    for episode in dataset:
        for tup in episode:
            agent.update_q_table(tup[0], tup[1], tup[3], tup[2], ALPHA, GAMMA)


def run_episode(agent, env):
    done = False
    obs = env.reset()[0]
    total_reward = 0
    current_action = 0
    max_actions = 50
    while not done and current_action < max_actions:
        action = agent.get_action(obs, 0)
        next_obs, reward, done, _, _ = env.step(action)

        obs = next_obs
        total_reward += reward
        current_action += 1
    
    return total_reward


def save_data(path, data, save_type, dataset_info):
    if not os.path.exists(path):
        print("Generating policies directory for this env")
        os.makedirs(path)
    joblib.dump(data,f'{path}{save_type}_{dataset_info[0]}_{dataset_info[1]}_{dataset_info[2]}.pkl', compress=1)


def train(env, agent, dataset, results_path, dataset_info, verbose):
    acc_val_rewards = []
    for episode in range(TRAIN_EPISODES):
        if verbose:
            print("Train with dataset")
        dataset_train(agent, dataset)

        if verbose:
            print("Validate with environment")
        validation_reward = 0
        for i in range(VALIDATION_EPISODES):
            accumulated_reward = run_episode(agent, env)
            validation_reward += accumulated_reward
        validation_reward /= VALIDATION_EPISODES
        acc_val_rewards.append(validation_reward)
        if verbose:
            print(f"{episode+1} episodes: validation_reward = {validation_reward}")
    
    if verbose:
        print("Save validation rewards and Q-table")
    save_data(results_path + "validations/", acc_val_rewards, "validations", dataset_info)
    save_data(results_path + "q_tables/", agent.Q_table, "q_table", dataset_info)


if __name__ == '__main__':
    #Meter argumentos de entorno, agente e hiperparámetros
    parser = argparse.ArgumentParser(description='Train offline environments given datasets')
    parser.add_argument('--env', '-e', type=str, required=True, help='Environment name')
    parser.add_argument("--trayectories", "-t", type=int, required=True, help="Dataset trayectories to use")
    parser.add_argument("--level", "-l", type=int, required=True, help="Dataset expertise level to use")
    parser.add_argument("--randomness", "-r", type=float, required=True, help="Dataset randomness to use")
    parser.add_argument("--verbose", "-v", type=bool, default=True, help="See traces or not")
    args = parser.parse_args()
    
    match args.env:
        case "Frozen-Lake":
            env = gym.make("FrozenLake-v1", desc=None, map_name="8x8", is_slippery=False)
            Q_shape = (env.observation_space.n,env.action_space.n) # type: ignore
            print(f"Loaded {env.spec.id} environment with Q-table shape of {Q_shape}")
        case "Mountain-Car":
            raise Exception("Developing")
        case _:
            raise Exception("Environment not registred")
        
    try:
        randomness = str(args.randomness).replace(".", "-")
        dataset = joblib.load(f"datasets/{env.spec.id}/dataset_{args.trayectories}_{args.level}_{randomness}.pkl")
        print(f"Loaded dataset with {args.trayectories} episodes, {args.level} policy's level and {randomness} exploration rate")
    
    except FileNotFoundError:
        raise Exception("Specified dataset does not exist")
    
    results_path = "results/"
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        
    agent = QLearningAgent(env,Q_shape)
    dataset_info = (args.trayectories, args.level, randomness)
    print("Start training")
    start = time()
    train(env,agent,dataset, results_path + env.spec.id + "/", dataset_info, args.verbose)
    end = time()
    
    print(f"Finish training. Time spent: {round(end - start, 2)}s")

    with open(results_path + env.spec.id + "/times.txt", "a+") as times_file:
        times_file.write(f"{dataset_info}: {round(end - start, 2)}s")
        times_file.write("\n")
