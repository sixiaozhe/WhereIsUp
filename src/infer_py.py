"""numpy 推理，用于与 C 端做数值对拍。

用法:
  PYTHONPATH=src python3 src/infer_py.py <file.ppm> [--logits] [--no-tta]
"""
import argparse
import numpy as np

import data as D
from nn import Net


def load(npz="data/best.npz", norm="data/norm.npz"):
    net = Net()
    net.load(npz)
    n = np.load(norm)
    return net, n["mean"], n["std"]


def tta_average(per_rotation):
    """把 4 个旋转视角的 logits 对齐到原图类别后求平均。

    per_rotation[r] 是输入顺时针旋转 r*90° 后的 logits；此时真实类别
    从 k 变成 (k+r)%4，因此第 j 个输出对应原图类别 (j-r)%4。
    """
    acc = np.zeros_like(np.asarray(per_rotation[0], dtype=np.float32))
    for r, logits in enumerate(per_rotation):
        for j in range(4):
            acc[..., (j - r) % 4] += logits[..., j]
    return acc / 4.0


def _softmax(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def infer_image(net, mean, std, img_hwc, tta=True):
    if tta:
        per_rotation = []
        for r in range(4):
            x = D.normalize(D.rotate_cw(img_hwc, r)[None], mean, std)
            per_rotation.append(net.forward(x)[0])
        logits = tta_average(per_rotation)
    else:
        x = D.normalize(img_hwc[None], mean, std)
        logits = net.forward(x)[0]
    return logits, _softmax(logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ppm")
    ap.add_argument("--npz", default="data/best.npz")
    ap.add_argument("--norm", default="data/norm.npz")
    ap.add_argument("--logits", action="store_true")
    ap.add_argument("--no-tta", dest="tta", action="store_false",
                    help="关闭四方向测试时增强")
    ap.set_defaults(tta=True)
    args = ap.parse_args()
    net, mean, std = load(args.npz, args.norm)
    img = D.read_ppm(args.ppm)
    logits, probs = infer_image(net, mean, std, img, tta=args.tta)
    if args.logits:
        print(" ".join(f"{v:.6f}" for v in logits))
    else:
        k = int(probs.argmax())
        print("probs:", " ".join(f"{v:.4f}" for v in probs))
        print("up is on the:", D.up_side(k), f"(k={k})")


if __name__ == "__main__":
    main()
