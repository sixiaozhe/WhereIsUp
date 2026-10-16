import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import nn


def _num_grad(f, x, eps=1e-3):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; fp = f()
        x[i] = old - eps; fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


class TestNN(unittest.TestCase):
    def test_conv_forward_against_naive(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 3, 6, 6)).astype(np.float32)
        layer = nn.Conv2d(3, 4, k=3, stride=1, pad=1, rng=rng)
        out = layer.forward(x)
        self.assertEqual(out.shape, (2, 4, 6, 6))
        # 逐点手算第 0 个样本第 0 个通道中心像素
        oc, yy, xx = 0, 2, 3
        acc = layer.b[oc]
        for c in range(3):
            for i in range(3):
                for j in range(3):
                    acc += x[0, c, yy + i - 1, xx + j - 1] * layer.W[oc, c, i, j]
        self.assertTrue(np.allclose(out[0, oc, yy, xx], acc, atol=1e-4))

    def test_conv_backward_gradcheck(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((2, 3, 5, 5)).astype(np.float32)
        layer = nn.Conv2d(3, 2, k=3, stride=1, pad=1, rng=rng)
        layer.forward(x)
        dout = rng.standard_normal((2, 2, 5, 5)).astype(np.float32)
        dx = layer.backward(dout)
        f = lambda: float((layer.forward(x) * dout).sum())
        self.assertTrue(np.allclose(dx, _num_grad(f, x), atol=1e-2))

    def test_maxpool_backward_gradcheck(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
        layer = nn.MaxPool2d(2)
        layer.forward(x)
        dout = rng.standard_normal((1, 2, 2, 2)).astype(np.float32)
        dx = layer.backward(dout)
        f = lambda: float((layer.forward(x) * dout).sum())
        self.assertTrue(np.allclose(dx, _num_grad(f, x), atol=1e-2))

    def test_linear_backward_gradcheck(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((4, 5)).astype(np.float32)
        layer = nn.Linear(5, 3, rng=rng)
        layer.forward(x)
        dout = rng.standard_normal((4, 3)).astype(np.float32)
        dx = layer.backward(dout)
        f = lambda: float((layer.forward(x) * dout).sum())
        self.assertTrue(np.allclose(dx, _num_grad(f, x), atol=1e-2))

    def test_softmax_ce_onehot_equals_neglogp(self):
        logits = np.array([[2.0, 0.0, -1.0, 0.5]], dtype=np.float32)
        y = np.array([0])
        loss, p = nn.SoftmaxCE().forward(logits, y)
        self.assertTrue(np.allclose(loss, -np.log(p[0, 0]), atol=1e-6))
        self.assertTrue(np.allclose(p.sum(axis=1), 1.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
