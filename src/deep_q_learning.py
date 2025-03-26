# %%
# Import library
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

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
# Setup Q-Net
# Action-Value Function(Q) Approximation
class QNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.linear1 = nn.Linear(4, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.output = nn.Linear(2 * hidden_dim, 2)

    def forward(self, s):
        x = self.linear1(s)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.output(x)
        return x

q_model = QNet().to(device)
q_target_model = QNet().to(device)

# copy paramter of q_model to q_target_model
def copy_param_to_target_mode(q_model_: QNet, q_target_model_: QNet):
    q_target_model_.load_state_dict(q_model_.state_dict())
    _ = q_target_model_.requires_grad_(False)

copy_param_to_target_mode(q_model, q_target_model)

# %%
# Setup Replay Memory
# We are going to store previous episodes, and keep learning from it

class ReplayMemory:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, item):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(item)

    def sample(self, sample_size):
        items = random.sample(self.buffer, sample_size)

        # divide each columns
        states   = [i[0] for i in items]
        actions  = [i[1] for i in items]
        rewards  = [i[2] for i in items]
        n_states = [i[3] for i in items]
        dones    = [i[4] for i in items]

        # convert it into tensor
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        n_states = torch.tensor(n_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        return states, actions, rewards, n_states, dones

    def length(self):
        return len(self.buffer)

memory = ReplayMemory(buffer_size=10000)

# %%
# Define one train_step
gamma = 0.99
optimizer = torch.optim.Adam(q_model.parameters(), lr=0.0005)

def train_step(states, actions, rewards, next_states, dones):
    with torch.no_grad():

        # compute Q(a, s_{t+1}) for all a's
        # shape: [batch_size, 2]
        target_vals_for_all_actions = q_target_model(next_states)

        # compute argmax_a Q(a, s_{t+1})
        # shape: [batch_size, ]
        target_actions = torch.argmax(target_vals_for_all_actions, 1)

        # convert the target_actions into one_hot
        # shape: [batch_size, env.action_space.n(which is 2)]
        target_actions_one_hot = F.one_hot(target_actions, env.action_space.n)

        # compute the max Q(s_{t+1})
        # shape: [batch_size, ]
        target_vals = torch.sum(target_vals_for_all_actions * target_actions_one_hot, 1)

        # get new q_vals
        # This is derived by TD(0)
        target_td0_q_vals = rewards + gamma * (1.0 - dones) * target_vals
        target_td0_q_vals = target_td0_q_vals.detach()

    optimizer.zero_grad()

    # compute td0_q_vals2 for q_model
    actions_one_hot = F.one_hot(actions, env.action_space.n).float()
    td0_q_vals = torch.sum(q_model(states) * actions_one_hot, 1)

    loss = F.mse_loss(target_td0_q_vals, td0_q_vals, reduction="mean")

    loss.backward()
    optimizer.step()


# %%
# Define helper function for training
# pick_action: use epsilon-greedy policy to determine the next action
# evaluate: evaluate the current q_model with epsilon-greedy(epsilon=0.0) policy
batch_size = 64


epsilon = 1.0
epsilon_decay = epsilon / 3000
epsilon_final = 0.1

# pick next action using epsilon-greedy
# s is not a batch. it is a single state
def pick_action(s, epsilon, q_model_):
    with torch.no_grad():
        if np.random.random() > epsilon:
            state_ = torch.tensor(s, dtype=torch.float).to(device)

            # state_'s batch_size is 1
            state_ = state_.unsqueeze(dim=0)

            q_vals_for_all_actions = q_model_(state_)

            a = torch.argmax(q_vals_for_all_actions, 1)
            a = a.squeeze(dim=0)

            # convert a tensor of single element into integer
            a = a.tolist()
        else:
            a = np.random.randint(0, env.action_space.n)
        return a

# evaluate current agent with no exploration
def evaluate(iter_num = 500):
    rewards = []
    for _ in range(iter_num):
        with torch.no_grad():
            s, _ = env.reset()
            done = False
            total = 0

            while not done:
                a = pick_action(s, 0.0, q_model)
                s_next, r, term, trunc, _ = env.step(a)
                done = term or trunc

                total += float(r)
                s = s_next

            rewards.append(total)

    return np.mean(rewards)

# %%
# Actual training happens here

eval_rewards = []

for epoch in range(100000):

    # Initialize the memory by filling episodes
    done = True
    curr_state, _ = env.reset()
    cummulative_rewards = 0

    for _ in range(500):
        if done:
            curr_state, _ = env.reset()
            done = False
            cummulative_rewards = 0

        action = pick_action(curr_state, epsilon, q_model)

        next_state, r, term, trunc, _ = env.step(action)
        done = term or trunc

        memory.add([curr_state.tolist(), action, r, next_state.tolist(), float(term)])

        cummulative_rewards += float(r)

        curr_state = next_state

    # keep fill memory until we get 2000 samples
    if memory.length() < 2000:
        continue

    for _ in range(30):
        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        # we should reshape the states
        states = torch.reshape(states, (batch_size, 4))
        actions = torch.reshape(actions, (batch_size, ))
        rewards = torch.reshape(rewards, (batch_size, ))
        next_states = torch.reshape(next_states, (batch_size, 4))
        dones = torch.reshape(dones, (batch_size, ))

        # optimize the q_model for sampled memory
        train_step(states, actions, rewards, next_states, dones)

    eval_rewards.append(evaluate(1))

    if epoch % 200 == 1:
        copy_param_to_target_mode(q_model, q_target_model)
        print(f"Run iteration {epoch} | last evaluated rewards {eval_rewards[-1]:.3f}")
        print(f"Run iteration {epoch} | average evaluated rewards {np.mean(eval_rewards[-200:]):.3f}")
        print("\n")


    if epsilon - epsilon_decay >= epsilon_final:
        epsilon -= epsilon_decay

    if len(eval_rewards) > 200 and np.mean(eval_rewards[-200:]) > 495.0:
        break

env.close()

# %%
# Plot the train results
avg_rewards = []
for idx in range(len(eval_rewards)):
    if idx < 100:
        avg_window = eval_rewards[:idx+1]
    else:
        avg_window = eval_rewards[idx - 100:]

    avg_rewards.append(np.average(avg_window))

plt.plot(eval_rewards)
plt.plot(avg_rewards)
