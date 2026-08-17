# Flappy Bird DQN

A Deep Q-Network (DQN) agent that learns to play Flappy Bird using the [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium) environment.

## How it works

The agent uses a simple DQN setup:

- **`dqn.py`** — the neural network itself. A small fully connected network (state -> hidden -> action) that outputs Q-values for each action.
- **`experience_replay.py`** — a FIFO replay buffer (`deque`) that stores past experiences `(state, action, next_state, reward, done)` and lets the agent sample random mini-batches from it during training, instead of learning only from the most recent step.
- **`agent.py`** — the main training/testing loop. It:
  - loads hyperparameters from `parameters.yaml`
  - runs episodes in the environment
  - uses an epsilon-greedy policy during training (random action with probability `epsilon`, otherwise the network's best guess)
  - stores experiences in the replay buffer and samples mini-batches to update the network
  - keeps a separate target network that gets synced with the policy network every `network_sync_rate` steps, for more stable training
  - saves the model whenever a new best episode reward is reached

## Project structure

```
flappy_bird/
├── agent.py              # training / testing loop
├── dqn.py                 # the Q-network
├── experience_replay.py   # replay memory buffer
├── parameters.yaml        # hyperparameters for each run/config
└── runs/                  # saved models + logs (created automatically)
    ├── flappybirdv0.pt
    └── flappybirdv0.log
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Train the agent (uses the `flappybirdv0` config from `parameters.yaml`):

```bash
python agent.py flappybirdv0 --train
```

Watch the trained agent play (loads the saved model and renders the game):

```bash
python agent.py flappybirdv0
```

Training runs until you stop it manually (Ctrl+C). While training, the best model so far is saved to `runs/flappybirdv0.pt`, and each new best reward is logged to `runs/flappybirdv0.log`.

## Hyperparameters

All hyperparameters live in `parameters.yaml`, keyed by config name (e.g. `flappybirdv0`), so you can add more configs without touching the code:

| Parameter | Meaning |
|---|---|
| `epsilon_init` / `epsilon_min` / `epsilon_decay` | controls how much the agent explores vs exploits over time |
| `replay_memory_size` | max number of experiences kept in the replay buffer |
| `mini_batch_size` | number of experiences sampled per training step |
| `network_sync_rate` | how often (in steps) the target network is updated |
| `alpha` | learning rate for the Adam optimizer |
| `gamma` | discount factor for future rewards |
| `reward_threshold` | cuts an episode short once this reward is reached |
