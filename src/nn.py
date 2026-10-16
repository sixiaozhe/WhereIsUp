"""手写神经网络层（仅用 numpy），每层提供 forward/backward。

内部一律 NCHW。所有参数为 float32。
"""
import numpy as np


class Conv2d:
    def __init__(self, in_ch, out_ch, k=3, stride=1, pad=1, rng=None):
        rng = rng or np.random.default_rng(0)
        self.k, self.stride, self.pad = k, stride, pad
        scale = np.sqrt(2.0 / (in_ch * k * k))          # He 初始化
        self.W = (rng.standard_normal((out_ch, in_ch, k, k)) * scale).astype(np.float32)
        self.b = np.zeros(out_ch, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.cols = None
        self.x_shape = None

    def _im2col(self, x):
        N, C, H, W = x.shape
        k, s, p = self.k, self.stride, self.pad
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        oh = (H + 2 * p - k) // s + 1
        ow = (W + 2 * p - k) // s + 1
        cols = np.empty((N, C, k, k, oh, ow), dtype=x.dtype)
        for i in range(k):
            for j in range(k):
                cols[:, :, i, j, :, :] = xp[:, :, i:i + s * oh:s, j:j + s * ow:s]
        return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * oh * ow, -1), oh, ow

    def forward(self, x):
        self.x_shape = x.shape
        N = x.shape[0]
        cols, oh, ow = self._im2col(x)
        self.cols = cols
        out = cols @ self.W.reshape(self.W.shape[0], -1).T + self.b
        return out.reshape(N, oh, ow, -1).transpose(0, 3, 1, 2)

    def backward(self, dout):
        N, OC, oh, ow = dout.shape
        _, C, H, W = self.x_shape
        k, s, p = self.k, self.stride, self.pad
        Wf = self.W.reshape(OC, -1)
        dflat = dout.transpose(0, 2, 3, 1).reshape(N * oh * ow, OC)
        self.dW = (dflat.T @ self.cols).reshape(self.W.shape)
        self.db = dflat.sum(axis=0)
        dcols = (dflat @ Wf).reshape(N, oh, ow, C, k, k).transpose(0, 3, 4, 5, 1, 2)
        dxp = np.zeros((N, C, H + 2 * p, W + 2 * p), dtype=dout.dtype)
        for i in range(k):
            for j in range(k):
                dxp[:, :, i:i + s * oh:s, j:j + s * ow:s] += dcols[:, :, i, j, :, :]
        return dxp[:, :, p:p + H, p:p + W] if p > 0 else dxp


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2d:
    def __init__(self, size=2):
        self.size = size

    def forward(self, x):
        N, C, H, W = x.shape
        s = self.size
        oh, ow = H // s, W // s
        xr = x[:, :, :oh * s, :ow * s].reshape(N, C, oh, s, ow, s)
        xr = xr.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, oh, ow, s * s)
        self.argmax = xr.argmax(axis=-1)
        self.x_shape = (N, C, H, W)
        self.ohow = (oh, ow)
        return xr.max(axis=-1)

    def backward(self, dout):
        N, C, H, W = self.x_shape
        oh, ow = self.ohow
        s = self.size
        dxr = np.zeros((N, C, oh, ow, s * s), dtype=dout.dtype)
        np.put_along_axis(dxr, self.argmax[..., None], dout[..., None], axis=-1)
        dxr = dxr.reshape(N, C, oh, ow, s, s).transpose(0, 1, 2, 4, 3, 5)
        dxr = dxr.reshape(N, C, oh * s, ow * s)
        dx = np.zeros((N, C, H, W), dtype=dout.dtype)
        dx[:, :, :oh * s, :ow * s] = dxr
        return dx


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class Linear:
    def __init__(self, in_f, out_f, rng=None):
        rng = rng or np.random.default_rng(0)
        scale = np.sqrt(2.0 / in_f)
        self.W = (rng.standard_normal((out_f, in_f)) * scale).astype(np.float32)
        self.b = np.zeros(out_f, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, dout):
        self.dW = dout.T @ self.x
        self.db = dout.sum(axis=0)
        return dout @ self.W


class SoftmaxCE:
    """Softmax 与交叉熵合并，数值稳定，反向即 (p - onehot) / N。"""

    def forward(self, logits, y):
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        self.p, self.y = p, y
        n = y.shape[0]
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
        return float(loss), p

    def backward(self):
        n = self.y.shape[0]
        d = self.p.copy()
        d[np.arange(n), self.y] -= 1.0
        return (d / n).astype(np.float32)


class Net:
    """固定结构：conv-relu-pool-conv-relu-pool-flatten-fc。"""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.conv1 = Conv2d(3, 8, rng=rng)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(2)
        self.conv2 = Conv2d(8, 16, rng=rng)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(2)
        self.flatten = Flatten()
        self.fc = Linear(16 * 16 * 16, 4, rng=rng)
        self.loss = SoftmaxCE()
        self.layers = [self.conv1, self.relu1, self.pool1,
                       self.conv2, self.relu2, self.pool2,
                       self.flatten, self.fc]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def params_and_grads(self):
        for layer in self.layers:
            if isinstance(layer, (Conv2d, Linear)):
                yield layer.W, layer.dW
                yield layer.b, layer.db

    def save(self, path):
        d = {}
        for name in ["conv1", "conv2", "fc"]:
            layer = getattr(self, name)
            d[name + ".W"] = layer.W
            d[name + ".b"] = layer.b
        np.savez(path, **d)

    def load(self, path):
        d = np.load(path)
        for name in ["conv1", "conv2", "fc"]:
            layer = getattr(self, name)
            layer.W = d[name + ".W"].astype(np.float32)
            layer.b = d[name + ".b"].astype(np.float32)
