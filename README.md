Generate policies:

    python generate_policy.py --env {env name}

Collect datasets:

    python collect.py --env {env name} --level {policy expertise} --amount {number of episodes to collect} --randomness {exploration rate}

Datasets nomenclature:

    dataset_{amount}_{level}_{exp rate}

Envs allowed:

    Frozen-Lake
