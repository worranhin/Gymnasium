import gymnasium as gym
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class BlackjackAgent:
    def __init__(self, env:gym.Env, 
                 learning_rate:float,
                initial_epsilon: float,
                epsilon_decay: float,
                final_epsilon: float,
                discount_factor: float = 0.95,
                ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # For plotting metrics
        self.training_error = []
        self.epsilon_history = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))
            # print(np.argmax(self.q_values[obs]))
        
    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        temporal_difference = reward + self.discount_factor * np.max(
            self.q_values[next_obs]
        )
        self.q_values[obs][action] = (1-self.lr) * self.q_values[obs][action] + self.lr * temporal_difference
        # future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        # temporal_difference = (
        #     reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        # )

        # self.q_values[obs][action] = (
        #     self.q_values[obs][action] + self.lr * temporal_difference
        # )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        
        self.epsilon_history.append(self.epsilon)
        

# hyperparameters
learning_rate = 0.001 #0.01
n_episodes = 1_000_000 #100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1
game_history = []


env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

        if done:
            game_history.append(reward)

    agent.decay_epsilon()

# plt.subplot(1,2,1)
# plt.plot(agent.training_error)
# plt.title("Training error")
# plt.xlabel("Episode")
# plt.ylabel("Error")

total_game_count = 0
win_game_count = 0
win_game_rate = []
for result in game_history:
    total_game_count += 1
    if result == 1:
        win_game_count += 1
    win_game_rate.append(win_game_count / total_game_count)

plt.subplot(1,2,1)
plt.plot(win_game_rate)
plt.title("Win Game Rate")
plt.xlabel("Episode")
plt.ylabel("rate")

plt.subplot(1,2,2)
plt.plot(agent.epsilon_history)
plt.title("Epsilon history")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.show()

print(agent.q_values)

# env = gym.make("Blackjack-v1", render_mode="human")

# for _ in range(1):
#     observation, info = env.reset()

#     episode_over = False
#     while not episode_over:
#         action = env.action_space.sample()  # agent policy that uses the observation and info
#         observation, reward, terminated, truncated, info = env.step(action)
#         print(observation, reward, terminated, truncated, info)
#         episode_over = terminated or truncated  # 结束条件或超时

# env.close()