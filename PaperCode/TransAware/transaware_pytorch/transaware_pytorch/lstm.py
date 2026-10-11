"""TransAware 的 PPO 策略生成 / 强化学习模型 —— PyTorch 移植版。

对应原 TF 实现 ``LSTM.py``（3.4 章节近端策略优化）。数学公式与原实现
逐行对应；只把 TF 图执行换成 PyTorch 张量运算。

注意：原实现里 ``PPOStrategyGenerator`` 被重复定义了两次（内容完全相同），
本移植只保留一份。
"""

import torch
import torch.nn as nn


class PPOStrategyGenerator(nn.Module):
    """PPO 策略生成网络。

    原 TF 结构：``LSTM(hidden_dim, return_sequences=True)`` + ``Dense(device_count)``
    + ``Softmax``（策略头）+ ``Dense(1)``（价值头）。
    TF Keras LSTM 默认 ``batch_first=False``，输入 ``(1, node_num, hidden)``
    解释为 ``(timesteps=1, batch=node_num, feat)``；PyTorch ``nn.LSTM`` 输入
    约定相同，可直接对齐。
    """

    def __init__(self, hidden_dim, device_count):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=False)
        # 策略网络
        self.policy_dense = nn.Linear(hidden_dim, device_count)
        self.policy_softmax = nn.Softmax(dim=-1)
        # 价值网络
        self.value_dense = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        x, _ = self.lstm(inputs)  # (seq=1, batch=node_num, hidden)
        policy_logits = self.policy_dense(x)
        policy = self.policy_softmax(policy_logits)
        value = self.value_dense(x)
        device_indices = policy.argmax(dim=-1)[0]
        return policy, value, device_indices

    def predict_value(self, inputs):
        """预测给定状态的价值。"""
        _, value, _ = self(inputs)
        return value


def ppo_loss(advantages, rewards, logits, values, old_policy, beta=0.01):
    """近端策略优化损失（对应 3.4 章节）。

    与原实现公式逐行对应：clip 代理目标 + 0.5 * 价值损失 - beta * 熵损失。
    """
    # 计算策略损失
    probability_ratios = torch.exp(logits - torch.log(old_policy))
    clipped_ratios = torch.clamp(probability_ratios, 1 - 0.2, 1 + 0.2)
    policy_loss = -torch.mean(
        torch.min(advantages * probability_ratios,
                  advantages * clipped_ratios))

    # 计算价值损失
    value_loss = torch.mean((values - rewards) ** 2).float()

    # 计算熵损失以鼓励探索
    entropy_loss = -torch.mean(-old_policy * torch.log(old_policy))

    total_loss = policy_loss + 0.5 * value_loss - beta * entropy_loss
    return total_loss


def compute_gae(rewards, values, next_values, gamma=0.99, lam=0.95):
    """广义优势估计（GAE）。"""
    deltas = rewards + gamma * next_values - values
    advantages = torch.zeros_like(deltas)

    # 反向循环累计优势，保持时间顺序
    advantage = 0
    for t in reversed(range(deltas.shape[0])):
        advantage = deltas[t] + gamma * lam * advantage
        advantages[t] = advantage

    advantages = advantages.transpose(0, 1)
    return advantages


def td_update(reward, value, alpha=0.1, gamma=0.95):
    """时序差分更新 V(s)。"""
    # TD目标
    td_target = reward + gamma * value
    # TD误差
    td_error = td_target - value
    # 更新V(s)
    new_value = value + alpha * td_error

    return new_value


class ReinforceModel:
    """强化学习模型，用于生成分布式并行策略（对应 3.4 章节）。"""

    def __init__(self, hidden_dim, n_devs):
        self.lstm_network = PPOStrategyGenerator(hidden_dim, n_devs)
        # 忠实保留原学习率（极小，原文如此）
        self.optimizer = torch.optim.Adam(self.lstm_network.parameters(),
                                          lr=1e-8)

    def train_step_forward(self, input_data):
        """前向生成策略，返回 ``(policy, value, device_indices)`` 张量。"""
        policy, value, device_indices = self.lstm_network(input_data)
        return policy, value, device_indices

    def train_step_backward(self, input_data, reward, old_policy):
        """用 PPO 损失更新参数。"""
        # compute_rewards 返回的 reward 是形如 [float] 的列表，归一化为标量
        if isinstance(reward, (list, tuple)):
            reward = float(reward[0])
        else:
            reward = float(reward)

        policy, values, device_indices = self.lstm_network(input_data)

        next_values = td_update(reward, values)
        advantages = compute_gae(reward, values, next_values)
        loss = ppo_loss(advantages, reward, policy, values, old_policy)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
