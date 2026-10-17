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

    def test_tta_average_undoes_rotation(self):
        # 若真实类别是 k，把输入顺时针转 r 后网络应预测 (k+r)%4。
        # TTA 必须把每个 r 的 logits 反向平移后求和，才能得到 k。
        k = 1
        per_rotation = []
        for r in range(4):
            v = np.zeros(4, dtype=np.float32)
            v[(k + r) % 4] = 1.0
            per_rotation.append(v)
        avg = infer_py.tta_average(per_rotation)
        self.assertEqual(int(avg.argmax()), k)
        self.assertTrue(np.allclose(avg.sum(), 1.0, atol=1e-6))

    def test_no_tta_matches_single_forward(self):
        if not os.path.exists("data/best.npz"):
            self.skipTest("尚未训练 (data/best.npz 不存在)")
        net, mean, std = infer_py.load("data/best.npz", "data/norm.npz")
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        x = infer_py.D.normalize(img[None], mean, std)
        expected = net.forward(x)[0]
        got, _ = infer_py.infer_image(net, mean, std, img, tta=False)
        self.assertTrue(np.allclose(got, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
