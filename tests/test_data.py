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

    def test_up_cue_detects_bright_top(self):
        imgs = np.zeros((2, 64, 64, 3), dtype=np.uint8)
        imgs[0, :16] = 200   # 顶部亮
        imgs[1, -16:] = 200  # 底部亮
        cue = D.up_cue(imgs)
        self.assertGreater(cue[0], 100)
        self.assertLess(cue[1], -100)

    def test_curate_ids_filters_by_cue(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            bright = np.zeros((64, 64, 3), np.uint8)
            bright[:16] = 200
            flat = np.zeros((64, 64, 3), np.uint8)
            flat[-16:] = 200
            for name, img in [("0000", bright), ("0001", flat)]:
                with open(os.path.join(td, f"{name}.ppm"), "wb") as f:
                    f.write(b"P6\n64 64\n255\n")
                    f.write(img.tobytes())
            kept = D.curate_ids(["0000", "0001"], min_cue=50.0, ppm_dir=td)
            self.assertEqual(kept, ["0000"])

    def test_augment_label_matches_rotation(self):
        rng = np.random.default_rng(1)
        imgs = np.random.randint(0, 255, size=(8, 64, 64, 3), dtype=np.uint8)
        # 不翻转、不抖动：按标签纠正后应还原为原图
        x, y = D.augment(imgs, rng, flip_p=0.0, jitter=0.0)
        for i in range(8):
            self.assertTrue(np.array_equal(D.correct(x[i], int(y[i])), imgs[i]))
        # 必翻转、不抖动：纠正后应等于水平翻转的原图（翻转先于旋转）
        x, y = D.augment(imgs, rng, flip_p=1.0, jitter=0.0)
        for i in range(8):
            self.assertTrue(np.array_equal(D.correct(x[i], int(y[i])), imgs[i][:, ::-1, :]))


if __name__ == "__main__":
    unittest.main()
