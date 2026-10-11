import tensorflow as tf
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
import pickle

# 随机生成数据作为输入
x_train = np.random.random((100, 224, 224, 3))  # 假设有100个样本
y_train = np.random.randint(10, size=(100, 1))  # 假设有10个类别
y_train = tf.keras.utils.to_categorical(y_train, 10)

x_test = np.random.random((20, 224, 224, 3))  # 测试集
y_test = np.random.randint(10, size=(20, 1))
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 加载预训练的NASNetMobile模型
base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 冻结基模型
base_model.trainable = False

# 构建新的模型
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=10, epochs=1)

# 预测
predictions = model.predict(x_test)

# 保存预测结果到pickle文件
with open('pickle/nasnet.pkl', 'wb') as f:
    pickle.dump(predictions, f)
