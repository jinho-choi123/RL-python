# %%
# Import libraries
import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt

# setup gym CartPole-v1
env = gym.make("CartPole-v1")

# The CartPole-v1 has 4 fields of continuous state. To use q-learning, we should convert the continuous state into discrete state
# (continuous_state, continuous_state, continuous_state, continuous_state) -> (discrete_state, discrete_state, discrete_state, discrete_state)
discrete_state_shape = (20, 20, 20, 20)

discrete_states = []

for i in range(4):
    discrete_state = np.linspace(
        env.observation_space.low[i] if (i == 0 or i == 2) else -4,
        env.observation_space.high[i] if (i == 0 or i == 2) else 4,
        num=discrete_state_shape[i],
        endpoint=False,
    )
    discrete_state = np.delete(discrete_state, 0)
    discrete_states.append(discrete_state)

print("Discrete State:")
print(discrete_states)

def get_discrete_state(s):
    """
    convert continuous states into discrete states
    :param s: continuous state that we want to convert
    """
    new_s = []
    for i in range(4):
        new_s.append(np.digitize(s[i], discrete_states[i]))
    return new_s

# %%
#Define a Q Table
# Q table shape will be (20, 20, 20, 20, 2)
q_table = np.zeros(discrete_state_shape + (env.action_space.n, ))

print("Q-Table Shape:")
print(q_table.shape)
#%%
# We are going to use TD(0) method for Q value function approximation

gamma = 0.999 # discount factor per timestep
alpha = 0.1 # learning rate
epsilon = 1 # exploration rate for epsilon-greedy policy improvement
epsilon_decay = epsilon / 4000

def pick_sample(s, epsilon):
    """
    pick a action to take for given state.
    Since we are using epsilon-greedy policy improvement, it has epsilon chance of doing random exploration
    param s: current state
    param epsilon: rate of random exploration
    """

    if np.random.random() > epsilon:
        a = np.argmax(q_table[tuple(s)])
    else:
        a = np.random.randint(0, env.action_space.n)

    return a

# improve the policy by doing
# TD(0) value function approximation
# epsilon-greedy policy iteration

rewards = []

for i in range(6000):
    done = False
    total_reward = 0
    s, _ = env.reset()
    s_dis = get_discrete_state(s)

    while not done:
        a = pick_sample(s_dis, epsilon)
        s, r, term, trunc, _ = env.step(a)

        done = term or trunc

        s_dis_next = get_discrete_state(s)

        # update Q-Table
        s_next_q_value_max = np.max(q_table[tuple(s_dis_next)])

        # new_q_table_entry = (1-alpha) * old_table_entry + alpha * (reward + gamma * next_state_q_value_max)
        # By iterating this, the q-value will converge
        q_table[tuple(s_dis)][a] = (1 - alpha) * q_table[tuple(s_dis)][a] + alpha * (float(r) + gamma * s_next_q_value_max)

        s_dis = s_dis_next
        total_reward += float(r)

    # Update epsilon for each episode
    if epsilon - epsilon_decay >= 0:
        epsilon -= epsilon_decay

    print(f"Run episode {i} with rewards {total_reward}")
    rewards.append(total_reward)

print("\n\nDone")
env.close()

# %%
# Plot the rewards
average_reward = []

for idx in range(len(rewards)):
    avg_list = np.empty(shape=(1,))
    if idx < 50:
        avg_list = rewards[:idx+1]
    else:
        avg_list = rewards[idx-49:idx+1]
    average_reward.append(np.average(avg_list))

plt.plot(rewards)
plt.plot(average_reward)
