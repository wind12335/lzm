        def train_step(x):
            results = self.lstm_network(x, training=True)
            return results
        
        results = train_step(x)

        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 初始化变量
            res = sess.run(results)
            self.policy, self.next_values, device_indices = res
            
        device_list = device_indices.tolist()
        placement = {}
        i = 0
        for node in self.progressive_graph.node_traversal_order:
            placement[node] = device_list[i]
            i += 1
        run_time, _, mem_util = self.eval_placement(placement)

        # 计算奖励值
        self.run_times.append(run_time)
        self.memory_utils.append(mem_util)
        self.reward = self.compute_rewards(self.run_times, self.memory_utils)
        self.next_values = td_update(self.reward, self.values)

        # 根据奖励值，利用近端策略优化生成loss
        advantages = compute_gae(self.reward, self.values, self.next_values)
        loss = ppo_loss(advantages, self.reward, self.policy, self.values, self.old_policy)
        self.old_policy = self.policy

        # 利用ppo生成的loss参数更新
        gradients = tf.gradients(loss, self.lstm_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.lstm_network.trainable_variables))






        

            
            


    #     @tf.function
    #     def train_step(inputs):
    #         with tf.GradientTape() as tape:
    #             policy, values, device_indices = self.lstm_network(inputs, training=True)
    #             device_indices = device_indices.numpy()

    #             # 将原先的tensor打散，并且转换成placement
    #             tensor_list = tf.split(device_indices, x.shape[1], axis=0)

    #             device_dict = {}
    #             for i in range(self.config_params['n_devs']):
    #                 device_tensor = tf.constant([i])
    #                 device_dict[device_tensor] = i

    #             print(type(tensor_list))
    #             print(len(tensor_list))
    #             # 根据获取的placement，使用模拟执行引擎计算
    #             placement = {}
    #             i = 0
    #             for node in self.progressive_graph.node_traversal_order:
    #                 placement[node] = device_dict[tensor_list[i]]
    #                 i += 1
    #             run_time, _, mem_util = self.eval_placement(placement)

    #             # 计算奖励值
    #             self.run_times.append(run_time)
    #             self.memory_utils.append(mem_util)
    #             self.reward = self.compute_rewards(run_times, memory_utils)

    #             # 根据奖励值，利用近端策略优化生成loss
    #             self.next_values = self.lstm_network.predict_value(inputs)
    #             advantages = compute_gae(self.reward, values, self.next_values)
    #             loss = ppo_loss(advantages, self.reward, self.value, policy, values, self.old_policy)
            
    #         gradients = tape.gradient(loss, self.lstm_network.trainable_variables)
    #         self.optimizer.apply_gradients(zip(gradients, self.lstm_network.trainable_variables))
    #         return loss
        
    #     for epoch in range(30):
    #         loss = train_step(x)


    # # def train_step(self, sess, inputs, progressive_graph):
    # #     # 定义梯度计算的图
    # #     policy, values = self.lstm_network(inputs, training=True)
    # #     max_device_indices = tf.argmax(policy, axis=-1)[0]

    # #     # 注意：这里需要基于TensorFlow 1.x的方式调整你的 eval_placement, compute_rewards, compute_gae 和 ppo_loss 方法

    # #     # 计算奖励和loss（假设这些函数已经适配为TensorFlow 1.x版本）
    # #     # 注意：你可能需要根据实际情况调整这里的代码，以确保它们在TensorFlow 1.x环境下正确运行
    # #     # 这里略去具体的计算细节...

    # #     loss = ...  # 使用你的ppo_loss方法计算loss

    # #     # 计算梯度并应用
    # #     gradients = tf.gradients(loss, self.lstm_network.trainable_variables)
    # #     train_op = self.optimizer.apply_gradients(zip(gradients, self.lstm_network.trainable_variables))

    # #     # 执行图
    # #     _, loss_val = sess.run([train_op, loss], feed_dict={...})  # 确保正确设置feed_dict

    # #     return loss_val

    # #     # 实际使用
    # #     with tf.Session() as sess:
    # #         sess.run(tf.global_variables_initializer())  # 初始化变量

    # #         for epoch in range(30):
    # #             # 假设 x 是你的输入数据，progressive_graph 是你的计算图结构
    # #             loss_val = model.train_step(sess, x, progressive_graph)
    # #             print(f"Epoch {epoch}, Loss: {loss_val}")

    #     @tf.function
    #     def train_step(inputs):
    #         with tf.GradientTape() as tape:
    #             policy, values = self.lstm_network(inputs, training=True)
    #             # 获取placement
    #             # with tf.Session() as sess:
    #             #     max_device_indices = sess.run(tf.argmax(policy, axis=-1)[0])

    #         max_device_indices = tf.argmax(policy, axis=-1)

    #         # 根据获取的placement，使用模拟执行引擎计算
    #         placement = {}
    #         i = 0
    #         for node in self.progressive_graph.node_traversal_order:
    #             placement[node] = max_device_indices[i]
    #             i += 1
    #         run_time, _, mem_util = self.eval_placement(placement)

    #         # 计算奖励值
    #         self.run_times.append(run_time)
    #         self.memory_utils.append(mem_util)
    #         self.reward = self.compute_rewards(run_times, memory_utils)

    #         # 根据奖励值，利用近端策略优化生成loss
    #         self.next_values = self.lstm_network.predict_value(inputs)
    #         advantages = compute_gae(self.reward, values, self.next_values)
    #         loss = ppo_loss(advantages, self.reward, self.value, policy, values, self.old_policy)
            
            
    #         gradients = tape.gradient(loss, self.lstm_network.trainable_variables)
    #         self.optimizer.apply_gradients(zip(gradients, self.lstm_network.trainable_variables))
    #         return loss
        
    #     for epoch in range(30):
    #         loss = train_step(x)