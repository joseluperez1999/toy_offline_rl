import numpy as np
from gymnasium.spaces import Box


def tq(dataset):
    acc_rewards = []
    for episode in dataset:
        episode_reward = 0
        for transition in episode:
            episode_reward += transition[3]
        acc_rewards.append(episode_reward)
        
    tq = np.mean(acc_rewards)
    print(f"[QUALITY]: TQ metric for generated data is: {tq}")
     
    return tq
    
def saco(env,dataset):
    saco = 0
    pairs = []
    for episode in dataset:
        for transition in episode:
            state_action = (transition[0],transition[1])
            pairs.append(state_action)
    
    if isinstance(env.observation_space,Box):
        print("SACo Continuous case, developing...")
    else:
        unique = set(pairs)
        saco = len(np.unique(pairs))
        print(f"[QUALITY]: SACo metric for generated data is: {saco}")
        
    return saco 