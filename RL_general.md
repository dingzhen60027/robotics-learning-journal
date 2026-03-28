# 一、总分类

强化学习算法通常可分为两大类：

## 1. Model-Based Reinforcement Learning

核心思想是：**先学习或已知环境模型，再利用模型做规划或决策。**

这里的“模型”通常指环境动力学：

* 状态转移概率：P(s′∣s,a)P(s'|s,a)
* 奖励函数：R(s,a)R(s,a) 或 R(s,a,s′)R(s,a,s')

也就是智能体知道，或者能够学到：

“执行动作 aa 后，环境会怎么变，奖励是多少”。

### 特点

* 样本效率高
* 可以做规划（planning）
* 训练时更省真实交互数据
* 但模型不准会带来误差累积（model bias）

---

## 2. Model-Free Reinforcement Learning

核心思想是：**不显式学习环境转移模型，直接从交互中学习价值函数或策略。**

它不关心“环境为什么这样变化”，只关心：

“这个动作值不值”“这条策略好不好”。

### 特点

* 思路直接，使用广泛
* 对复杂环境更灵活
* 但通常样本效率较低
* 往往需要大量交互数据

---

# 二、Model-Free 再细分

Model-Free 通常又分成两大类：

## 1. Value-Based

学习价值函数，再由价值函数导出策略。

常见：

* Q-Learning
* SARSA
* DQN
* Double DQN
* Dueling DQN
* Rainbow

适用特点：

* 多用于 **离散动作空间**
* 通过估计 Q(s,a)Q(s,a) 选动作

---

## 2. Policy-Based

直接学习策略 π(a∣s)\pi(a|s)。

常见：

* REINFORCE
* PPO
* TRPO

适用特点：

* 更适合 **连续动作空间**
* 可直接优化随机策略或确定性策略

---

## 3. Actor-Critic

结合 Value-Based 和 Policy-Based：

* Actor 负责输出策略
* Critic 负责评估策略或动作价值

常见：

* A2C / A3C
* DDPG
* TD3
* SAC
* PPO（也常归到 Actor-Critic 框架）

---

# 三、Model-Based 算法整理

## 1. 已知环境模型 + 规划

如果环境模型已知，可以直接做动态规划或搜索。

### 代表方法

* Value Iteration
* Policy Iteration
* MPC（Model Predictive Control）
* MCTS（Monte Carlo Tree Search）

### 说明

这类方法严格来说很多属于“规划”范畴，但通常也放进 Model-Based RL 的框架里。

---

## 2. 学习环境模型 + 规划/控制

先用数据学一个环境模型，再用这个模型生成轨迹、模拟未来、优化策略。

### 代表方法

* Dyna-Q
* PILCO
* PETS
* MBPO
* Dreamer
* MuZero（更偏高级、结合隐空间模型与规划）

### 说明

这是现代 Model-Based RL 的主流。

---

# 四、按算法放入分类框架

下面给你一个更实用的归类表。

## A. Model-Free：Value-Based

### 1. SARSA

* on-policy
* 学习 Q(s,a)Q(s,a)
* 更新时使用当前策略实际选出的动作

### 2. Q-Learning

* off-policy
* 学习最优 Q(s,a)Q(s,a)
* 经典公式里对下一状态取 max

### 3. DQN

* Q-Learning + 神经网络
* 用于高维状态、离散动作
* 两个关键技术：
  * Experience Replay
  * Target Network

### 4. Double DQN

* 缓解 DQN 的 Q 值高估问题

### 5. Dueling DQN

* 将 Q 分解成 Value 和 Advantage

### 6. Rainbow

* 综合多种 DQN 改进：
  * Double
  * Dueling
  * Prioritized Replay
  * Noisy Net
  * Distributional RL
  * N-step learning

---

## B. Model-Free：Policy-Based

### 1. REINFORCE

* 最基础策略梯度算法
* 直接对策略参数求梯度
* 方差大，收敛慢

### 2. TRPO

* 限制策略更新步长
* 提高训练稳定性
* 理论较强，实现复杂

### 3. PPO

* TRPO 的实用简化版
* 当前最常用策略优化算法之一
* 训练稳定、实现相对简单

---

## C. Model-Free：Actor-Critic

### 1. A2C / A3C

* Actor-Critic 经典方法
* A3C 是异步并行版本
* A2C 是同步版本

### 2. DDPG

* 面向连续动作空间
* 确定性策略梯度
* 可看作 DQN + Actor-Critic 的结合

### 3. TD3

* 对 DDPG 的改进
* 缓解 Q 值高估与训练不稳定

### 4. SAC

* Soft Actor-Critic
* 引入最大熵思想
* 样本效率高，训练稳定
* 连续控制里很常用

---

## D. Model-Based

### 1. Dynamic Programming

* Value Iteration
* Policy Iteration
* 前提：已知模型，状态空间通常较小

### 2. Dyna-Q

* 真实经验学习 + 模型模拟学习
* 是 Model-Based 和 Model-Free 的桥梁型算法

### 3. PILCO

* 基于概率模型的经典方法
* 样本效率很高，但扩展性一般

### 4. PETS

* 概率集成模型 + MPC
* 现代经典 Model-Based RL 方法

### 5. MBPO

* 用短模型 rollout 辅助策略学习
* 结合 SAC 等 Model-Free 算法

### 6. Dreamer / PlaNet

* 学习潜在空间世界模型
* 在潜在空间中做预测和策略优化

### 7. MuZero

* 不显式学习原始环境转移
* 学习“隐式模型”并结合树搜索
* AlphaZero 风格方法的延伸

---

# 五、再按几个常见维度交叉整理

## 1. 按是否 on-policy / off-policy

### On-policy

只能用当前策略产生的数据更新：

* SARSA
* REINFORCE
* A2C / A3C
* PPO
* TRPO

特点：

* 更稳定
* 但样本利用率低

### Off-policy

可复用旧数据、经验回放：

* Q-Learning
* DQN 系列
* DDPG
* TD3
* SAC
* Dyna-Q
* 很多 Model-Based 方法

特点：

* 样本效率高
* 训练实现更复杂

---

## 2. 按动作空间

### 离散动作空间更常见

* Q-Learning
* SARSA
* DQN
* Double DQN
* Dueling DQN
* Rainbow
* PPO（也可做离散）

### 连续动作空间更常见

* REINFORCE
* PPO
* TRPO
* DDPG
* TD3
* SAC
* Dreamer
* MBPO

---

## 3. 按是否需要环境模型

### 需要 / 使用环境模型

* Value Iteration
* Policy Iteration
* Dyna-Q
* PILCO
* PETS
* MBPO
* Dreamer
* MuZero

### 不使用显式环境模型

* SARSA
* Q-Learning
* DQN 系列
* REINFORCE
* A2C/A3C
* PPO
* DDPG
* TD3
* SAC

---

# 六、最推荐你记忆的知识框架

可以直接记成这棵树：

```text
强化学习
├── Model-Based
│   ├── 已知模型
│   │   ├── Value Iteration
│   │   └── Policy Iteration
│   └── 学习模型
│       ├── Dyna-Q
│       ├── PILCO
│       ├── PETS
│       ├── MBPO
│       ├── Dreamer
│       └── MuZero
│
└── Model-Free
    ├── Value-Based
    │   ├── SARSA
    │   ├── Q-Learning
    │   ├── DQN
    │   ├── Double DQN
    │   ├── Dueling DQN
    │   └── Rainbow
    │
    ├── Policy-Based
    │   ├── REINFORCE
    │   ├── TRPO
    │   └── PPO
    │
    └── Actor-Critic
        ├── A2C / A3C
        ├── DDPG
        ├── TD3
        └── SAC
```

---

# 七、怎么理解两者区别

可以用一句话区分：

* **Model-Based** ：先学“世界怎么运作”，再做决策
* **Model-Free** ：不管世界怎么运作，直接学“怎么做更好”

---

# 八、面试/考试版总结

如果你要写成简答题，可以直接写：

**强化学习算法按是否显式使用环境模型，可分为 Model-Based 和 Model-Free。**

* **Model-Based RL** ：已知或学习环境转移模型与奖励函数，并利用模型进行规划和控制。代表算法有 Value Iteration、Policy Iteration、Dyna-Q、PETS、MBPO、Dreamer、MuZero 等。其优点是样本效率高，缺点是模型误差会影响性能。
* **Model-Free RL** ：不显式学习环境模型，而是直接从交互数据中学习价值函数或策略。可进一步分为：
* **Value-Based** ：如 SARSA、Q-Learning、DQN；
* **Policy-Based** ：如 REINFORCE、TRPO、PPO；
* **Actor-Critic** ：如 A2C、DDPG、TD3、SAC。

  其优点是适用性广，缺点是样本效率通常较低。

---

# 九、学习顺序建议

如果你是为了系统学习，建议顺序：

1. MDP、Bellman Equation
2. Dynamic Programming
3. SARSA / Q-Learning
4. DQN
5. Policy Gradient
6. Actor-Critic
7. PPO / SAC
8. Dyna-Q
9. Dreamer / MuZero / MBPO

---

我也可以继续帮你整理成一版 **表格版** 或  **思维导图版** 。
