import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import train


class TestTrain(unittest.TestCase):
    def test_train_steps_updates_parameters(self):
        rng = np.random.default_rng(0)
        imgs = rng.integers(0, 255, size=(16, 64, 64, 3), dtype=np.uint8)
        y = np.array([i % 4 for i in range(16)])
        net = train.Net(seed=0)
        before = net.conv1.W.copy()
        losses = train.train_steps(net, imgs, y, epochs=1, batch=8, lr=0.05, seed=0)
        self.assertEqual(len(losses), 1)
        self.assertTrue(np.isfinite(losses[0]))
        self.assertFalse(np.allclose(before, net.conv1.W))

    def test_eval_tta_constant_net_is_chance(self):
        # 常量 logits 时四类无法区分，准确率应恰为 1/4，且混淆矩阵形状正确。
        class ConstNet:
            def forward(self, x):
                return np.zeros((x.shape[0], 4), dtype=np.float32)

        imgs = np.zeros((4, 64, 64, 3), dtype=np.uint8)
        acc, cm, _ = train.eval_tta(ConstNet(), imgs,
                                    np.zeros(3, np.float32), np.ones(3, np.float32))
        self.assertAlmostEqual(acc, 0.25, places=6)
        self.assertEqual(cm.shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
