Generate policies:

    python generate_policy.py --env {env name}

Collect datasets:

    python collect.py --env {env name} --level {policy expertise} --amount {number of episodes to collect} --randomness {exploration rate}

Learn offline: 

    python learn_offline.py --env {env name} --trayectories {number of trayectories} --level {policy expertise} --randomness {exploration rate} --verbose {traces visible or not}

Results obtained from learning offline are:

    - Validation mean reward after every training episode
    - Q-table of specified configuration
    - Time needed to learn

Datasets nomenclature:

    dataset_{amount}_{level}_{exp rate}

Envs allowed:

    Frozen-Lake
    Mountain-Car
