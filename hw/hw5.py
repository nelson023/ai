class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None
        self.label = label

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

def gradient_descent(parameters, learning_rate):
    for p in parameters:
        p.data -= learning_rate * p.grad
        p.grad = 0.0  # 清除梯度以準備下一輪更新

# 創建變量
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

# 預測值
f = a * b + c

# 損失函數：目標是讓 f 接近 0
loss = (f - Value(0.0)) * (f - Value(0.0))

# 設定學習率
learning_rate = 0.01

# 訓練過程
for epoch in range(100):
    loss.backward()  # 反向傳播計算梯度
    gradient_descent([a, b, c], learning_rate)  # 梯度下降更新參數
    
    # 前向傳播以更新損失
    f = a * b + c
    loss = (f - Value(0.0)) * (f - Value(0.0))
    
    print(f'Epoch {epoch}, Loss: {loss.data}')
