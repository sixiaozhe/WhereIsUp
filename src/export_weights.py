"""把 data/best.npz 导出成 c/weights.h（float32 C 数组）。

用法: PYTHONPATH=src python3 src/export_weights.py [--npz data/best.npz] [--out c/weights.h]
"""
import argparse
import hashlib
import numpy as np


def _fmt_array(name, arr, per_line=8):
    flat = arr.astype(np.float32).reshape(-1)
    lines = [f"static const float {name}[{flat.size}] = {{"]
    row = []
    for i, v in enumerate(flat):
        row.append(f"{v:.8g}f")
        if len(row) == per_line:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    lines.append("};")
    return "\n".join(lines)


def write_header(npz_path, mean, std, out_path, dims=None):
    d = np.load(npz_path)
    with open(npz_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    dims = dims or {"C": 3, "H": 64, "W": 64, "OC1": 8, "OC2": 16, "CLASSES": 4}
    parts = [
        "/* 自动生成，请勿手改。",
        f" * 源: {npz_path}  md5: {md5}",
        " * 卷积权重布局: W[oc][ic][kh][kw] 行主序展开。",
        " * 全连接权重布局: W[out][in] 行主序展开。",
        " */",
        "#ifndef WEIGHTS_H",
        "#define WEIGHTS_H",
        _fmt_array("NORM_MEAN", mean),
        _fmt_array("NORM_STD", std),
        _fmt_array("W1", d["conv1.W"]),
        _fmt_array("B1", d["conv1.b"]),
        _fmt_array("W2", d["conv2.W"]),
        _fmt_array("B2", d["conv2.b"]),
        _fmt_array("W3", d["fc.W"]),
        _fmt_array("B3", d["fc.b"]),
        "#endif",
        "",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {out_path} (6 weight arrays, md5={md5})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/best.npz")
    ap.add_argument("--norm", default="data/norm.npz")
    ap.add_argument("--out", default="c/weights.h")
    args = ap.parse_args()
    n = np.load(args.norm)
    write_header(args.npz, n["mean"], n["std"], args.out)


if __name__ == "__main__":
    main()
