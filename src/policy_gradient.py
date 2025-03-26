# %%
# Import library
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer

gamma = 0.99

# SETUP device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Setup Gym for CartPole-v1
env = gym.make("CartPole-v1")

# %%
# We are going to define a Policy Network
# In policy-gradient method, we don't explicitely calculate the Q-value
class PolicyNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.linear1 = nn.Linear(4, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        # self.out = nn.Linear(2 * hidden_dim, 2)
        #
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, state_):
        x = self.linear1(state_)
        x = F.relu(x)
        # x = self.linear2(x)
        # x = F.relu(x)
        logits = self.out(x)
        return logits

policy_model = PolicyNet().to(device)

def pick_action(state_, policy_model_):
    """
    Get the action distribution for input state
    Since the output is logits, we should manually convert it into probability.
    """

    with torch.no_grad():
        # NP doesn't have unsqueeze
        # state_batch = state_.unsqueeze(dim=0)
        state_batch = np.expand_dims(state_, axis=0)

        # shape of state_batch: (1, 4)
        state_batch = torch.tensor(state_batch, dtype=torch.float).to(device)

        # get logits for the state using policy-network
        # shape of logits: (1, 2)
        logits = policy_model_(state_batch)

        # since we are dealing with single state, we squeeze the logits
        logits = logits.squeeze(dim=0)

        probs = F.softmax(logits, dim=-1)

        a = torch.multinomial(probs, num_samples=1)

        return a.tolist()[0]


def train_step(optimizer_: Optimizer):
    done = False
    states = []
    actions = []
    rewards = []

    curr_state, _ = env.reset()

    # get one episode using policy_model
    while not done:
        states.append(curr_state.tolist())
        action = pick_action(curr_state, policy_model)

        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        curr_state = next_state
        actions.append(action)
        rewards.append(reward)


    # convert states, actions, rewards into tensor
    expected_total_rewards = np.zeros_like(rewards)

    # calculate the state-value for each step
    # We use Monte-Carlo method, which accumulates rewards until the termination
    for i in reversed(range(len(rewards))):
        expected_total_rewards[i] = rewards[i] + (gamma * expected_total_rewards[i+1] if i+1 < len(rewards) else 0)

    states = torch.tensor(states, dtype=torch.float).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)

    expected_total_rewards = torch.tensor(expected_total_rewards, dtype=torch.float).to(device)

    optimizer_.zero_grad()

    logits = policy_model(states)

    # we wrote as this form for understanding.
    probs = F.softmax(logits, dim=-1)

    # log probability of all actions
    log_probs_ = torch.log(probs)

    # get the one-hot actions
    actions_one_hot = F.one_hot(actions, env.action_space.n).float()

    # log probability of action we took during the episode
    log_probs = torch.sum(log_probs_ * actions_one_hot, dim=1)

    # we should maximize
    # Sum of G_t * log_probs
    loss_ = -log_probs * expected_total_rewards
    loss = loss_.sum()

    loss.backward()

    optimizer_.step()

    return np.sum(rewards)



# %%
# Actual training happens here

# Use AdamW optimizer.
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=0.001)


episode_rewards = []
for epoch in range(10000):

    episode_reward = train_step(optimizer)

    episode_rewards.append(episode_reward)

    if epoch % 100 == 1:
        print(f"EPOCH {epoch} | episode total reward: {episode_reward:.3f}")
        print(f"EPOCH {epoch} | episode average reward: {np.mean(episode_rewards[-100:]):.3f}")

# %%
# Visualize the train result

avg_rewards = []

for i in range(len(episode_rewards)):
    if i < 50:
        avg_window = episode_rewards[:i+1]
    else:
        avg_window = episode_rewards[i-49:i+1]

    avg_rewards.append(np.average(avg_window))

plt.plot(episode_rewards)
plt.plot(avg_rewards)
