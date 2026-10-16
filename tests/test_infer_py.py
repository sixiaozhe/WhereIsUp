import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import infer_py


class TestInferPy(unittest.TestCase):
    def test_forward_matches_manual_softmax_sum(self):
        if not os.path.exists("data/best.npz"):
            self.skipTest("尚未训练 (data/best.npz 不存在)")
        net, mean, std = infer_py.load("data/best.npz", "data/norm.npz")
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        logits, probs = infer_py.infer_image(net, mean, std, img)
        self.assertEqual(logits.shape, (4,))
        self.assertLess(abs(probs.sum() - 1.0), 1e-5)


if __name__ == "__main__":
    unittest.main()
