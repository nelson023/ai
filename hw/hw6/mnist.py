from engine import Tensor  # 引入自定義的 Tensor 類

from keras.datasets import mnist
import keras
import numpy as np

# 加載 MNIST 數據集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 預處理圖像數據：標準化並重塑為一維向量
train_images = np.asarray(x_train, dtype=np.float32) / 255.0
test_images = np.asarray(x_test, dtype=np.float32) / 255.0
train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)
# 轉換標籤為獨熱編碼
y_train = keras.utils.to_categorical(y_train)

def forward(X, Y, W):
    """
    定義前向傳播過程
    :param X: 輸入數據
    :param Y: 標籤
    :param W: 權重
    :return: 損失
    """
    y_pred = X.matmul(W)
    probs = y_pred.softmax()
    loss = probs.cross_entropy(Y)
    return loss

batch_size = 32
steps = 20000

# 初始化數據和權重
X = Tensor(train_images)
Y = Tensor(y_train)
W = Tensor(np.random.randn(784, 10) * 0.01)  # 使用較小的隨機值初始化權重

for step in range(steps):
    # 隨機抽取一個批次的數據
    ri = np.random.permutation(train_images.shape[0])[:batch_size]
    Xb, Yb = Tensor(train_images[ri]), Tensor(y_train[ri])
    
    # 前向傳播和反向傳播
    lossb = forward(Xb, Yb, W)
    lossb.backward()
    
    # 每 1000 步輸出一次全量數據集的平均損失
    if step % 1000 == 0 or step == steps - 1:
        with np.no_grad():  # 確保在評估模式下不計算梯度
            loss = forward(X, Y, W).data / X.data.shape[0]
            print(f'Loss at step {step}: {loss}')

    # 權重更新
    W.data -= 0.01 * W.grad  # 使用固定的學習率更新權重
    W.grad = np.zeros_like(W.grad)  # 重置梯度

