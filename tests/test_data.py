import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import data as D

def _img():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    a[0, :, 0] = 10      # 顶行
    a[:, 0, 1] = 20      # 左列
    return a

def test_rotate_cw_matches_expected():
    a = _img()
    cw1 = D.rotate_cw(a, 1)
    expected = np.rot90(a, -1, axes=(0, 1))
    assert np.array_equal(cw1, expected)

def test_up_side_names():
    assert D.up_side(0) == "up"
    assert D.up_side(1) == "right"
    assert D.up_side(2) == "down"
    assert D.up_side(3) == "left"

def test_correct_undoes_rotation():
    a = _img()
    for k in range(4):
        disp = D.rotate_cw(a, k)
        assert np.array_equal(D.correct(disp, k), a)

def test_split_by_source_no_leak():
    ids = [f"{i:04d}" for i in range(100)]
    tr, va, te = D.split_ids(ids, seed=0)
    assert set(tr).isdisjoint(va) and set(va).isdisjoint(te) and set(tr).isdisjoint(te)
    assert len(tr) + len(va) + len(te) == 100

def test_normalization_stats_shape():
    imgs = np.random.randint(0, 255, size=(5, 64, 64, 3), dtype=np.uint8)
    mean, std = D.norm_stats(imgs)
    assert mean.shape == (3,) and std.shape == (3,)
    assert np.all(std > 0)

def test_augment_preserves_shape_and_label_range():
    imgs = np.random.randint(0, 255, size=(3, 64, 64, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x, y = D.augment(imgs, rng)
        assert x.shape == (3, 64, 64, 3) and x.dtype == np.uint8
        assert y.shape == (3,) and set(np.unique(y)).issubset({0, 1, 2, 3})


def load_tests(loader, standard_tests, pattern):
    # 让 stdlib unittest 能收集本模块的模块级 test_* 函数
    # （unittest 默认只收集 TestCase 子类）。
    import unittest
    cases = [
        unittest.FunctionTestCase(fn, description=name)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    return unittest.TestSuite(cases)
