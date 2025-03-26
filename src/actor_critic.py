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

def pick_action(state_, actor_model_):
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
        logits = actor_model_(state_batch)

        # since we are dealing with single state, we squeeze the logits
        logits = logits.squeeze(dim=0)

        probs = F.softmax(logits, dim=-1)

        a = torch.multinomial(probs, num_samples=1)

        return a.tolist()[0]

# define helper function for training

# We are going to train critic model with Monte-Carlo method
# And actor model using Monte-Carlo method
# You can change the method to TD(lambda) or TD(0).
# But for this specific example, Monte-Carlo works much better
def train_step(actor_optimizer_: Optimizer, critic_optimizer_: Optimizer):
    done = False
    states = []
    actions = []
    rewards = []

    curr_state, _ = env.reset()

    while not done:
        states.append(curr_state.tolist())
        action = pick_action(curr_state, actor_net)

        next_state, reward, term, trunc, _ = env.step(action)

        done = term or trunc
        curr_state = next_state

        actions.append(action)
        rewards.append(reward)

    states = torch.tensor(states, dtype=torch.float).to(device)
    next_states = torch.roll(states, -1, 0)
    next_states[-1] = 0

    actions = torch.tensor(actions, dtype=torch.long).to(device)

    # critic values of each state
    # shape: (len(states), )
    critic_values = critic_net(states)

    cummulative_rewards = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        cummulative_rewards[i] = rewards[i] + gamma * (cummulative_rewards[i+1] if i+1 < len(rewards) else 0)

    cummulative_rewards = torch.tensor(cummulative_rewards, dtype=torch.float).unsqueeze(dim=-1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(dim=-1).to(device)

    # we use advantages to train critic net
    critic_loss = F.mse_loss(critic_values, cummulative_rewards)

    critic_optimizer_.zero_grad()
    critic_loss.backward()
    critic_optimizer_.step()


    # Then we train the actor
    # Recalculate the critic values with updated critic_net
    with torch.no_grad():
        critic_values = critic_net(states)

        # TD(0)
        # For this case, Monte-Carlo works much much better...
        # next_state_critic_values = critic_net(next_states)
        # advantages = rewards + gamma * next_state_critic_values - critic_values

        # Monte-Carlo
        advantages = cummulative_rewards - critic_values

    logits = actor_net(states)

    # we wrote as this form for understanding.
    probs = F.softmax(logits, dim=-1)

    # log probability of all actions
    log_probs_ = torch.log(probs)

    # get the one-hot actions
    actions_one_hot = F.one_hot(actions, env.action_space.n).float()

    # log probability of action we took during the episode
    log_probs = torch.sum(log_probs_ * actions_one_hot, dim=1)

    # we should maximize
    # Sum of Advantage * log_probs
    actor_loss_ = -log_probs * advantages
    actor_loss = actor_loss_.sum()

    actor_optimizer_.zero_grad()
    actor_loss.backward()
    actor_optimizer_.step()

    return torch.sum(rewards)

# %%
# Actual training happens here
actor_optimizer = torch.optim.AdamW(actor_net.parameters(), lr=0.001)
critic_optimizer = torch.optim.AdamW(critic_net.parameters(), lr=0.001)

episode_rewards = []
for epoch in range(2000):
    episode_reward = train_step(actor_optimizer, critic_optimizer)
    episode_rewards.append(episode_reward.cpu())

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
