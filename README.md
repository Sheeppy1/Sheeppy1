- 👋 Hi, I’m @Sheeppy1
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
import gym
import numpy as np

# 初始化小车爬坡问题环境
env = gym.make("MountainCar-v0")
env.reset()

# 定义 SARSA 算法代理
class SarsaAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.action_space = env.action_space.n

        # 将状态空间离散化为 Q-table 的形状
        self.num_buckets = (20, 20)  # 指定状态空间的离散化分桶数

        # 初始化 Q-table
        self.q_table = np.zeros(self.num_buckets + (self.action_space,))

    def discretize_state(self, state, low, high):
        # 将连续状态空间离散化为 Q-table 的索引
        discretized_state = []
        for i in range(len(state)):
            ratio = (state[i] + abs(low[i])) / (high[i] - low[i])
            discretized_state.append(int(np.round((self.num_buckets[i] - 1) * ratio)))
        return tuple(discretized_state)

    def choose_action(self, state):
        discretized_state = self.discretize_state(state, self.env.observation_space.low, self.env.observation_space.high)

        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_space)  # 随机选择动作
        else:
            return np.argmax(self.q_table[discretized_state])

    def update_q_table(self, state, action, reward, next_state, next_action):
        discretized_state = self.discretize_state(state, self.env.observation_space.low, self.env.observation_space.high)
        discretized_next_state = self.discretize_state(next_state, self.env.observation_space.low, self.env.observation_space.high)

        td_target = reward + self.discount_factor * self.q_table[discretized_next_state][next_action]
        td_error = td_target - self.q_table[discretized_state][action]
        self.q_table[discretized_state][action] += self.learning_rate * td_error


# 调用 SARSA 算法求解小车爬坡问题
agent = SarsaAgent(env)

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    action = agent.choose_action(state)

    done = False
    while not done:
        next_state, reward, done, _ = env.step(action)
        next_action = agent.choose_action(next_state)
        agent.update_q_table(state, action, reward, next_state, next_action)

        state = next_state
        action = next_action

# 使用贪婪策略展示处理结果
greedy_policy = np.argmax(agent.q_table, axis=-1)
print("Greedy Policy:")
print(greedy_policy)

<!---
Sheeppy1/Sheeppy1 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
