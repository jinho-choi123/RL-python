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

# SETUP device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Setup Gym for CartPole-v1
env = gym.make("CartPole-v1")

gamma = 0.99

# %%
# Define Actor Network and Critic Network
class ActorNet(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()

        self.linear1 = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, state_):
        x = self.linear1(state_)
        x = F.relu(x)
        logits = self.output(x)
        return logits

class CriticNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.linear1 = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, state_):
        x = self.linear1(state_)
        x = F.relu(x)
        value = self.output(x)
        return value

actor_net = ActorNet().to(device)
critic_net = CriticNet().to(device)

# %%
# define helper function that pick action
# It also returns the discrete distribution of each action probability

def pick_action(state_, actor_model_):
    with torch.no_grad():
        # NP doesn't have unsqueeze
        # state_batch = state_.unsqueeze(dim=0)
        state_batch = np.expand_dims(state_, axis=0)

        # shape of state_batch: (1, 4)
        state_batch = torch.tensor(state_batch, dtype=torch.float).to(device)

        logits_batch = actor_model_(state_batch)

        logits = logits_batch.squeeze(dim=0)

        probs = F.softmax(logits, dim=-1)

        actions = torch.multinomial(probs, num_samples=1)
        action = actions.tolist()[0]

        log_probs = torch.log(probs)

        return action, logits, log_probs

# %%
# Define train_step helper function
# For PPO, we train actor model and critic model at the same time
# With single optimizer!!

KL_FACTOR = 0.3
VF_LOSS_FACTOR=0.5

def train_step(optimizer_: Optimizer):
    done = False
    states = []
    actions = []
    rewards = []

    logits = []
    log_probs = []

    curr_state, _ = env.reset()

    while not done:
        states.append(curr_state.tolist())
        action, logit, log_prob = pick_action(curr_state, actor_net)

        next_state, reward, term, trunc, _ = env.step(action)

        done = term or trunc
        curr_state = next_state

        actions.append(action)
        rewards.append(reward)
        logits.append(logit)
        log_probs.append(log_prob)

    states = torch.tensor(states, dtype=torch.float).to(device)
    next_states = torch.roll(states, -1, 0)
    next_states[-1] = 0


    actions = torch.tensor(actions, dtype=torch.long).to(device)
    logits = torch.stack(logits).to(device)
    log_probs = torch.stack(log_probs).to(device)

    ## CALCULATE CRITIC LOSS
    # critic values of each state
    # shape: (len(states), )

    # state values
    critic_values = critic_net(states)

    # logits
    actor_values  = actor_net(states)

    cummulative_rewards = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        cummulative_rewards[i] = rewards[i] + gamma * (cummulative_rewards[i+1] if i+1 < len(rewards) else 0)

    cummulative_rewards = torch.tensor(cummulative_rewards, dtype=torch.float).unsqueeze(dim=-1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(dim=-1).to(device)

    critic_loss = F.mse_loss(critic_values, cummulative_rewards, reduction="none").squeeze(dim=-1)


    ## CALCULATE ACTOR LOSS

    # calculate advantages
    advantages = cummulative_rewards - critic_values
    # normalize advantages
    advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)
    advantages = advantages.squeeze(dim=-1)


    # calculate the kl-divergence
    # KL divergance between grad-tracking logits and no-grad-tracking logits
    new_logits = actor_values
    old_logits = logits



    kl_divergances = []
    prob_ratios = []
    for state_id in range(new_logits.size(0)):
        new_dist = torch.distributions.Categorical(logits=new_logits[state_id])
        old_dist = torch.distributions.Categorical(logits=old_logits[state_id])

        new_prob = F.softmax(new_logits[state_id], dim=-1)
        old_prob = F.softmax(old_logits[state_id], dim=-1)
        taken_action = actions[state_id]
        prob_ratios.append(new_prob[taken_action] / old_prob[taken_action])

        kl_divergances.append(torch.distributions.kl_divergence(old_dist, new_dist))

    kl_divergances = torch.stack(kl_divergances)
    prob_ratios = torch.stack(prob_ratios)

    loss = -advantages * prob_ratios + KL_FACTOR * kl_divergances + VF_LOSS_FACTOR * critic_loss
    optimizer.zero_grad()
    loss = torch.sum(loss).backward()
    optimizer.step()


    return torch.sum(rewards)


# %%
# Actual Training happens here!

optimizer = torch.optim.AdamW(list(actor_net.parameters()) + list(critic_net.parameters()), lr=0.001)

episode_rewards = []

for epoch in range(2000):
    episode_reward = train_step(optimizer)
    episode_rewards.append(episode_reward.cpu())

    if epoch % 100 == 1:
        print(f"EPOCH {epoch} | episode total reward: {episode_reward:.3f}")
        print(f"EPOCH {epoch} | episode average reward: {np.mean(episode_rewards[-100:]):.3f}")

# %%
# Visualization
avg_rewards = []

for i in range(len(episode_rewards)):
    if i < 50:
        avg_window = episode_rewards[:i+1]
    else:
        avg_window = episode_rewards[i-49:i+1]

    avg_rewards.append(np.average(avg_window))

plt.plot(episode_rewards)
plt.plot(avg_rewards)
