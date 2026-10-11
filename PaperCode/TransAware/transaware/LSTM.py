import tensorflow as tf

class PPOStrategyGenerator(tf.keras.Model):
    def __init__(self, hidden_dim, device_count):
        super(PPOStrategyGenerator, self).__init__()
        self.lstm = tf.keras.layers.LSTM(hidden_dim, return_sequences=True)
        # 策略网络
        self.policy_dense = tf.keras.layers.Dense(device_count)
        self.policy_softmax = tf.keras.layers.Softmax()
        # 价值网络
        self.value_dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        policy_logits = self.policy_dense(x)
        policy = self.policy_softmax(policy_logits)
        value = self.value_dense(x)
        device_indices = tf.argmax(policy, axis=-1)[0]
        return policy, value, device_indices

    def predict_value(self, inputs):
        """预测给定状态的价值"""
        _, value, _ = self(inputs)  # 调用模型得到策略和价值输出
        return value

# 计算PPO LOSS，用于更新LSTM的参数，对应3.4章节中的近端策略优化
def ppo_loss(advantages, rewards, logits, values, old_policy, beta=0.01):
    # 计算策略损失
    probability_ratios = tf.exp(logits - tf.math.log(old_policy))
    clipped_ratios = tf.clip_by_value(probability_ratios, 1 - 0.2, 1 + 0.2)
    policy_loss = -tf.reduce_mean(tf.minimum(advantages * probability_ratios, advantages * clipped_ratios))
    
    # 计算价值损失
    value_loss = tf.reduce_mean((values - rewards) ** 2)
    value_loss = tf.cast(value_loss, tf.float32)
    
    # 计算熵损失以鼓励探索
    entropy_loss = -tf.reduce_mean(-old_policy * tf.math.log(old_policy))
    
    total_loss = policy_loss + 0.5 * value_loss - beta * entropy_loss
    return total_loss

def compute_gae(rewards, values, next_values, gamma=0.99, lam=0.95):
    """计算Generalized Advantage Estimation（GAE）"""
    deltas = rewards + gamma * next_values - values
    # deltas = tf.squeeze(deltas, axis=0)
    advantages = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    # 初始化累积优势
    advantage = 0
    # 由于 deltas 是二维的 [num_steps, 1]，我们需要沿时间步长进行反向循环
    for t in reversed(range(deltas.shape[0])):
        advantage = deltas[t] + gamma * lam * advantage
        # 将计算的优势插入到 TensorArray 的前端，以保持正确的时间顺序
        advantages = advantages.write(t, advantage)

    # 将 TensorArray 转换为 Tensor
    advantages = advantages.stack()
    advantages = tf.transpose(advantages)
    return advantages

def td_update(reward, value, alpha=0.1, gamma=0.95):
    # 这部分简写，缺少下一步状态的估计
    # TD目标
    td_target = reward + gamma * value
    # TD误差
    td_error = td_target - value
    # 更新V(s)
    new_value = value + alpha * td_error

    return new_value

# 近端策略优化生成，对应3.4章节的近端策略优化
class PPOStrategyGenerator(tf.keras.Model):
    def __init__(self, hidden_dim, device_count):
        super(PPOStrategyGenerator, self).__init__()
        self.lstm = tf.keras.layers.LSTM(hidden_dim, return_sequences=True)
        # 策略网络
        self.policy_dense = tf.keras.layers.Dense(device_count)
        self.policy_softmax = tf.keras.layers.Softmax()
        # 价值网络
        self.value_dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        policy_logits = self.policy_dense(x)
        policy = self.policy_softmax(policy_logits)
        value = self.value_dense(x)
        device_indices = tf.argmax(policy, axis=-1)[0]
        return policy, value, device_indices

    def predict_value(self, inputs):
        """预测给定状态的价值"""
        _, value, _ = self(inputs)  # 调用模型得到策略和价值输出
        return value

# 强化学习模型，用于生成分布式并行策略，对应3.4章节
class ReinforceModel:
    def __init__(self, hidden_dim, n_devs):
        # 初始化你的模型组件，例如LSTM网络等
        self.lstm_network = PPOStrategyGenerator(hidden_dim, n_devs)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-8)

    def train_step_forward(self, input_data):
        # 假设input_data是经过第一阶段处理后的输入数据
        policy, value, device_indices = self.lstm_network(input_data)

        return policy, value, device_indices

    def train_step_backward(self, input_data, reward, old_policy):
        # 假设input_data是经过第一阶段处理后的输入数据
        policy, values, device_indices = self.lstm_network(input_data)

        next_values = td_update(reward, values)
        advantages = compute_gae(reward, values, next_values)
        loss = ppo_loss(advantages, reward, policy, values, old_policy) 
        # 利用ppo生成的loss参数更新
        gradients = tf.gradients(loss, self.lstm_network.trainable_variables)
        train_op = self.optimizer.apply_gradients(zip(gradients, self.lstm_network.trainable_variables))
        # train_op = self.optimizer.minimize(loss)

        return train_op