import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import data as D


def _img():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    a[0, :, 0] = 10      # 顶行
    a[:, 0, 1] = 20      # 左列
    return a


class TestData(unittest.TestCase):
    def test_rotate_cw_matches_expected(self):
        a = _img()
        cw1 = D.rotate_cw(a, 1)
        expected = np.rot90(a, -1, axes=(0, 1))
        self.assertTrue(np.array_equal(cw1, expected))

    def test_up_side_names(self):
        self.assertEqual(D.up_side(0), "up")
        self.assertEqual(D.up_side(1), "right")
        self.assertEqual(D.up_side(2), "down")
        self.assertEqual(D.up_side(3), "left")

    def test_correct_undoes_rotation(self):
        a = _img()
        for k in range(4):
            disp = D.rotate_cw(a, k)
            self.assertTrue(np.array_equal(D.correct(disp, k), a))

    def test_split_by_source_no_leak(self):
        ids = [f"{i:04d}" for i in range(100)]
        tr, va, te = D.split_ids(ids, seed=0)
        self.assertTrue(set(tr).isdisjoint(va))
        self.assertTrue(set(va).isdisjoint(te))
        self.assertTrue(set(tr).isdisjoint(te))
        self.assertEqual(len(tr) + len(va) + len(te), 100)

    def test_normalization_stats_shape(self):
        imgs = np.random.randint(0, 255, size=(5, 64, 64, 3), dtype=np.uint8)
        mean, std = D.norm_stats(imgs)
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(std.shape, (3,))
        self.assertTrue(np.all(std > 0))

    def test_augment_preserves_shape_and_label_range(self):
        imgs = np.random.randint(0, 255, size=(3, 64, 64, 3), dtype=np.uint8)
        rng = np.random.default_rng(0)
        for _ in range(5):
            x, y = D.augment(imgs, rng)
            self.assertEqual(x.shape, (3, 64, 64, 3))
            self.assertEqual(x.dtype, np.uint8)
            self.assertEqual(y.shape, (3,))
            self.assertTrue(set(np.unique(y)).issubset({0, 1, 2, 3}))


if __name__ == "__main__":
    unittest.main()
