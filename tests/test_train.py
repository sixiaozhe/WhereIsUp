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


if __name__ == "__main__":
    unittest.main()
