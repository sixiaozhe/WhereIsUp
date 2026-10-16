"""numpy 推理，用于与 C 端做数值对拍。

用法:
  PYTHONPATH=src python3 src/infer_py.py <file.ppm> [--logits]
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


def infer_image(net, mean, std, img_hwc):
    x = D.normalize(img_hwc[None], mean, std)
    logits = net.forward(x)[0]
    z = logits - logits.max()
    e = np.exp(z)
    probs = e / e.sum()
    return logits, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ppm")
    ap.add_argument("--npz", default="data/best.npz")
    ap.add_argument("--norm", default="data/norm.npz")
    ap.add_argument("--logits", action="store_true")
    args = ap.parse_args()
    net, mean, std = load(args.npz, args.norm)
    img = D.read_ppm(args.ppm)
    logits, probs = infer_image(net, mean, std, img)
    if args.logits:
        print(" ".join(f"{v:.6f}" for v in logits))
    else:
        k = int(probs.argmax())
        print("probs:", " ".join(f"{v:.4f}" for v in probs))
        print("up is on the:", D.up_side(k), f"(k={k})")


if __name__ == "__main__":
    main()
