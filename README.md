# WhereIsUp —— 判断图片的“上边”在哪

输入一张被旋转的照片，判断“上边”在 上/右/下/左 哪个方向，并输出纠正后的图片。
权重是 C 数组，推理只需 gcc；训练用 Python3 + numpy 手写（无深度学习框架）。

## 快速开始

```bash
# 1) 取数据（picsum 可用 id 上限约 1085，实际约 1000 张；可先小规模验证）
bash scripts/download.sh 2000 400
bash scripts/prepare.sh 64

# 2) 训练（默认自动做数据策展：只保留有明确上下线索的照片；输出 data/best.npz 等）
PYTHONPATH=src python3 src/train.py

# 3) 导出权重为 C 数组
PYTHONPATH=src python3 src/export_weights.py

# 4) 编译并推理
make -C c whereisup
./c/whereisup path/to/photo.jpg -o fixed.png
```

## 目录

- `src/` 训练与导出；`c/` 纯 C 推理；`docs/TUTORIAL.md` 从零教学。

## 测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
bash scripts/check_parity.sh 8
```
