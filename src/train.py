"""WhereIsUp 训练脚本。

用法:
  PYTHONPATH=src python3 src/train.py [--epochs 60] [--batch 64] [--lr 0.006] [--min-cue 60]
"""
import argparse
import csv
import os
import numpy as np

import data as D
from nn import Net


def cosine_lr(lr0, step, total):
    return lr0 * 0.5 * (1.0 + np.cos(np.pi * step / max(1, total)))


def train_steps(net, imgs, y, epochs, batch, lr, seed=0, momentum=0.9,
                mean=None, std=None, verbose=False):
    """在给定（正立）图像上训练，内部做旋转增强。返回每 epoch 的平均 loss。

    本调用内使用恒定 lr；跨 epoch 的学习率调度由调用方通过 lr 参数负责。
    """
    rng = np.random.default_rng(seed)
    if mean is None:
        mean = np.zeros(3, dtype=np.float32)
    if std is None:
        std = np.ones(3, dtype=np.float32)
    vel = {id(p): np.zeros_like(p) for p, _ in net.params_and_grads()}
    n = imgs.shape[0]
    hist = []
    for ep in range(epochs):
        order = rng.permutation(n)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            aug, labels = D.augment(imgs[idx], rng)
            x = D.normalize(aug, mean, std)
            logits = net.forward(x)
            loss, _ = net.loss.forward(logits, labels)
            d = net.loss.backward()
            for layer in reversed(net.layers):
                d = layer.backward(d)
            for p, g in net.params_and_grads():
                v = vel[id(p)]
                v *= momentum
                v -= lr * g
                p += v
            ep_loss += loss
            nb += 1
        hist.append(ep_loss / max(1, nb))
        if verbose:
            print(f"epoch {ep + 1}/{epochs} loss={hist[-1]:.4f} lr={lr:.5f}")
    return hist


def evaluate(net, imgs, labels, mean, std, batch=128):
    """在未增强数据上评估准确率与混淆矩阵。labels 为 0（不旋转）。"""
    preds = []
    for i in range(0, imgs.shape[0], batch):
        x = D.normalize(imgs[i:i + batch], mean, std)
        logits = net.forward(x)
        preds.append(logits.argmax(axis=1))
    pred = np.concatenate(preds)
    acc = float((pred == labels).mean())
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(labels, pred):
        cm[t, p] += 1
    return acc, cm, pred


def eval_rotated(net, imgs, mean, std, rng, batch=128):
    """对所有 4 个旋转角评估（每个源图都会出现在 4 个类别里）。"""
    X, Y = [], []
    for k in range(4):
        X.append(D.rotate_cw_batch(imgs, k))
        Y.append(np.full(imgs.shape[0], k, dtype=np.int64))
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    return evaluate(net, X, Y, mean, std, batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.006)
    ap.add_argument("--min-cue", type=float, default=60.0,
                    help="数据策展阈值：只保留顶部亮度-底部亮度 >= 该值的照片（-1e9 表示不筛）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=D.PPM)
    args = ap.parse_args()

    ids = D.list_ids(args.data)
    if len(ids) < 20:
        raise SystemExit(f"not enough images in {args.data}; run scripts/download.sh && scripts/prepare.sh")
    ids = D.curate_ids(ids, args.min_cue, args.data)
    print(f"curated: kept {len(ids)} images with up-cue >= {args.min_cue}")
    if len(ids) < 20:
        raise SystemExit("too few images after curation; lower --min-cue")
    tr_ids, va_ids, te_ids = D.split_ids(ids, seed=args.seed)
    tr = D.load_images(tr_ids, args.data)
    va = D.load_images(va_ids, args.data)
    te = D.load_images(te_ids, args.data)
    mean, std = D.norm_stats(tr)
    print(f"images: train={len(tr_ids)} val={len(va_ids)} test={len(te_ids)}")
    print(f"norm mean={mean} std={std}")

    net = Net(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    best_acc = -1.0
    os.makedirs("data", exist_ok=True)
    curve = []
    for ep in range(args.epochs):
        lr_ep = cosine_lr(args.lr, ep, args.epochs)
        loss_hist = train_steps(net, tr, np.zeros(len(tr), dtype=np.int64),
                                epochs=1, batch=args.batch, lr=lr_ep,
                                seed=args.seed + ep, mean=mean, std=std)
        acc, cm, _ = eval_rotated(net, va, mean, std, rng)
        curve.append((ep + 1, loss_hist[0], acc))
        print(f"epoch {ep + 1}/{args.epochs} train_loss={loss_hist[0]:.4f} lr={lr_ep:.5f} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            net.save("data/best.npz")
            np.savez("data/norm.npz", mean=mean, std=std)
    with open("data/training_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_acc"])
        w.writerows(curve)

    net.load("data/best.npz")
    acc, cm, _ = eval_rotated(net, te, mean, std, rng)
    print(f"BEST val_acc={best_acc:.4f}  TEST acc={acc:.4f}")
    print("confusion matrix (rows=true k, cols=pred k):")
    print(cm)


if __name__ == "__main__":
    main()
