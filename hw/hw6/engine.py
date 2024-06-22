import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)  # 確保資料型態為 float32 以避免精度問題
        self.grad = np.zeros_like(self.data)  # 使用 zeros_like 確保梯度和資料形狀一致
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Exponents must be int or float"
        out = Tensor(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), (self, other), 'matmul')
        def _backward():
            self.grad += np.dot(out.grad, other.data.T)
            other.grad += np.dot(self.data.T, out.grad)
        out._backward = _backward
        return out

    def softmax(self):
        exps = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
        softmax = exps / np.sum(exps, axis=1, keepdims=True)
        out = Tensor(softmax, (self,), 'softmax')
        def _backward():
            s = np.sum(out.grad * softmax, axis=1, keepdims=True)
            self.grad += (out.grad - s) * softmax
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')
        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backward():
            if keepdims:
                self.grad += out.grad
            else:
                self.grad += np.reshape(out.grad, [d if i == axis else 1 for i, d in enumerate(self.shape)])
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
        self.grad = np.array(1.0, dtype=np.float32)
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)
