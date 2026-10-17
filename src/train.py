"""WhereIsUp 训练脚本。

用法:
  PYTHONPATH=src python3 src/train.py [--epochs 60] [--batch 64] [--lr 0.006] [--min-cue 60]
"""
import argparse
import csv
import os
import shutil

import numpy as np

import data as D
from infer_py import tta_average
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


def _forward_batch(net, imgs, mean, std, batch=128):
    """在给定 NHWC 正立图上前向，返回 logits [N,4]。"""
    out = []
    for i in range(0, imgs.shape[0], batch):
        x = D.normalize(imgs[i:i + batch], mean, std)
        out.append(net.forward(x))
    return np.concatenate(out)


def eval_tta(net, imgs, mean, std, batch=128):
    """四方向 TTA 评估：每张源图等概率出现在 4 个类别里。"""
    n = imgs.shape[0]
    logs = [_forward_batch(net, D.rotate_cw_batch(imgs, r), mean, std, batch) for r in range(4)]
    correct = 0
    cm = np.zeros((4, 4), dtype=int)
    for k in range(4):
        per_rot = [logs[(k + r) % 4] for r in range(4)]  # 旋转 r 后类别应为 (k+r)%4
        pred = tta_average(per_rot).argmax(axis=1)
        for t, p in zip(np.full(n, k), pred):
            cm[t, p] += 1
        correct += int((pred == k).sum())
    return correct / (4 * n), cm, None


def train_run(args, seed, out_dir):
    """单个随机种子：切分、训练、按验证集保存最优、返回指标。"""
    ids = D.curate_ids(D.list_ids(args.data), args.min_cue, args.data)
    if len(ids) < 20:
        raise SystemExit("too few images after curation; lower --min-cue")
    tr_ids, va_ids, te_ids = D.split_ids(ids, seed=seed)
    tr = D.load_images(tr_ids, args.data)
    va = D.load_images(va_ids, args.data)
    te = D.load_images(te_ids, args.data)
    mean, std = D.norm_stats(tr)
    print(f"[seed {seed}] curated={len(ids)} train={len(tr_ids)} "
          f"val={len(va_ids)} test={len(te_ids)}")

    net = Net(seed=seed)
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best.npz")
    norm_path = os.path.join(out_dir, "norm.npz")
    best_acc = -1.0
    curve = []
    for ep in range(args.epochs):
        lr_ep = cosine_lr(args.lr, ep, args.epochs)
        loss_hist = train_steps(net, tr, np.zeros(len(tr), dtype=np.int64),
                                epochs=1, batch=args.batch, lr=lr_ep,
                                seed=seed + ep, mean=mean, std=std)
        acc, cm, _ = eval_rotated(net, va, mean, std, rng)
        curve.append((ep + 1, loss_hist[0], acc))
        print(f"[seed {seed}] epoch {ep + 1}/{args.epochs} "
              f"train_loss={loss_hist[0]:.4f} lr={lr_ep:.5f} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            net.save(best_path)
            np.savez(norm_path, mean=mean, std=std)
    with open(os.path.join(out_dir, "training_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_acc"])
        w.writerows(curve)

    net.load(best_path)
    acc, cm, _ = eval_rotated(net, te, mean, std, rng)
    tta_val, _, _ = eval_tta(net, va, mean, std)
    tta_test, _, _ = eval_tta(net, te, mean, std)
    print(f"[seed {seed}] BEST val={best_acc:.4f} test={acc:.4f} | "
          f"TTA val={tta_val:.4f} test={tta_test:.4f}")
    print(f"[seed {seed}] confusion (rows=true k, cols=pred k):\n{cm}")
    return {"seed": seed, "out_dir": out_dir, "n_train": len(tr_ids),
            "n_val": len(va_ids), "n_test": len(te_ids), "best_val": best_acc,
            "test": acc, "tta_val": tta_val, "tta_test": tta_test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.006)
    ap.add_argument("--min-cue", type=float, default=60.0,
                    help="数据策展阈值：只保留顶部亮度-底部亮度 >= 该值的照片（-1e9 表示不筛）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None,
                    help="逗号分隔的多个随机种子；给出时逐个训练并按 TTA 验证集选最优")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--data", default=D.PPM)
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        runs = [train_run(args, s, os.path.join(args.out_dir, f"seed{s}")) for s in seeds]
        best = max(runs, key=lambda r: (r["tta_val"], r["best_val"]))
        for r in runs:
            for name in ("best.npz", "norm.npz", "training_curve.csv"):
                shutil.copyfile(os.path.join(r["out_dir"], name),
                                os.path.join(args.out_dir, name))
        with open(os.path.join(args.out_dir, "seed_summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "best_val", "test", "tta_val", "tta_test"])
            for r in runs:
                w.writerow([r["seed"], f"{r['best_val']:.4f}", f"{r['test']:.4f}",
                            f"{r['tta_val']:.4f}", f"{r['tta_test']:.4f}"])
        print(f"best seed = {best['seed']} (TTA val={best['tta_val']:.4f}); "
              f"copied to {args.out_dir}/best.npz")
    else:
        train_run(args, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
