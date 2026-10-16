"""WhereIsUp 数据管线。

- 读取 data/ppm/*.ppm（P6, 64x64 RGB，由 c/img2ppm 生成）
- 按“源照片”切分 train/val/test（防数据泄漏）
- 训练增强：随机 90 度旋转（即标签）、水平翻转、亮度/对比度抖动
- 归一化常数在训练集上统计
"""
import os
import glob
import numpy as np

IMG = 64
RAW = "data/raw"
PPM = "data/ppm"

DIRS = ["up", "right", "down", "left"]


def read_ppm(path):
    """读取二进制 P6 PPM，返回 HWC uint8。"""
    with open(path, "rb") as f:
        data = f.read()
    # 解析头部：P6 <w> <h> <maxval>\n
    fields, idx = [], 0
    assert data[:2] == b"P6"
    idx = 2
    while len(fields) < 3:
        while data[idx:idx + 1].isspace():
            idx += 1
        if data[idx:idx + 1] == b"#":
            while data[idx:idx + 1] != b"\n":
                idx += 1
            continue
        num = b""
        while data[idx:idx + 1] and not data[idx:idx + 1].isspace():
            num += data[idx:idx + 1]
            idx += 1
        fields.append(int(num))
    idx += 1
    w, h, _ = fields
    buf = np.frombuffer(data, dtype=np.uint8, count=w * h * 3, offset=idx)
    return buf.reshape(h, w, 3).copy()


def rotate_cw(img, k):
    """把 HWC 图像顺时针旋转 k*90 度。"""
    return np.rot90(img, -k, axes=(0, 1))


def up_side(k):
    """类别 k 时，真实上方在显示图的哪一侧。"""
    return DIRS[k]


def correct(img, k):
    """把显示图纠正回正立（逆时针 k*90 度）。"""
    return np.rot90(img, k, axes=(0, 1))


def list_ids(ppm_dir=PPM):
    files = sorted(glob.glob(os.path.join(ppm_dir, "*.ppm")))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def split_ids(ids, seed=0, ratios=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ids))
    n = len(ids)
    n_tr = int(round(ratios[0] * n))
    n_va = int(round(ratios[1] * n))
    tr = [ids[i] for i in idx[:n_tr]]
    va = [ids[i] for i in idx[n_tr:n_tr + n_va]]
    te = [ids[i] for i in idx[n_tr + n_va:]]
    return tr, va, te


def load_images(ids, ppm_dir=PPM):
    return np.stack([read_ppm(os.path.join(ppm_dir, f"{i}.ppm")) for i in ids])


def norm_stats(imgs):
    """imgs: uint8 NHWC。返回每通道 mean/std（在 [0,1] 尺度上）。"""
    x = imgs.astype(np.float32) / 255.0
    mean = x.reshape(-1, 3).mean(axis=0)
    std = x.reshape(-1, 3).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def augment(imgs, rng, flip_p=0.5, jitter=0.10):
    """对一批正立图像做增强，返回 (增强图, 标签)。

    旋转在本函数内施加，旋转角即标签。
    """
    n = imgs.shape[0]
    out = np.empty_like(imgs)
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        k = int(rng.integers(0, 4))
        img = imgs[i]
        if rng.random() < flip_p:
            img = img[:, ::-1, :]        # 先翻转（翻转不改变“上”），再旋转
        img = rotate_cw(img, k)
        img = img.astype(np.float32) / 255.0
        gain = 1.0 + rng.uniform(-jitter, jitter)
        bias = rng.uniform(-jitter / 2, jitter / 2)
        img = np.clip(img * gain + bias, 0.0, 1.0)
        out[i] = (img * 255.0 + 0.5).astype(np.uint8)
        labels[i] = k
    return out, labels


def normalize(imgs, mean, std):
    """uint8 NHWC → float32 NCHW 归一化。"""
    x = imgs.astype(np.float32) / 255.0
    x = (x - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
    return np.transpose(x, (0, 3, 1, 2))


def rotate_cw_batch(imgs, k):
    """对一批 NHWC 图像施加同一个顺时针旋转角 k。"""
    return np.rot90(imgs, -k, axes=(1, 2))
