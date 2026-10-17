# WhereIsUp · 照片朝哪边，一眼就知道

> 输入一张被旋转的照片，判断它的“上边”在 上 / 右 / 下 / 左 哪一侧，并输出纠正后的图片。
> **推理端纯 C，只需 gcc；训练端纯 numpy 手写，没有任何深度学习框架。**

![Python](https://img.shields.io/badge/python-3.x-blue)
![NumPy](https://img.shields.io/badge/numpy-only-013243)
![C99](https://img.shields.io/badge/C-C99-00599C)
![runtime deps](https://img.shields.io/badge/runtime%20deps-none-brightgreen)
![tests](https://img.shields.io/badge/tests-21%20passing-success)
![parity](https://img.shields.io/badge/numpy%20vs%20C-%3C1e--4-success)

<!-- 演示占位：录一段 rotate -> detect -> fix 的 GIF 放到 docs/demo.gif 后取消注释 -->
<!-- <p align="center"><img src="docs/demo.gif" alt="rotate -> detect -> fix" width="600"></p> -->

```
被旋转的照片 ──► 64×64 归一化 ──► 微型 CNN ──► 上/右/下/左 ──► 自动纠偏 PNG
        （纯 C 前向，权重是编译进程序的 C 数组）
```

## 为什么值得一看

- **推理零依赖**：没有 PyTorch、没有 ONNX、没有运行时。权重是一个 C 数组，`make` 完就能跑。
- **训练全透明**：卷积、池化、Softmax 交叉熵、反向传播、动量 SGD + 余弦退火全部手写，每层都有数值梯度校验。
- **标签免费**：把任意正常照片随机旋转 0/90/180/270°，旋转角就是标签——不需要任何人工标注。
- **有教学**：`docs/TUTORIAL.md` 用 13 章从“图像即张量”一路讲到“C 端部署与数值对拍”。
- **可验证**：numpy 与 C 的前向 logits 做数值对拍（阈值 1e-4），21 个单元测试全绿。
- **17,780 参数 / 约 69 KB**：小巧到能一眼看懂整个数据流。

## 快速开始

```bash
# 1) 取数据（picsum 可用 id 上限约 1085，实际约 1000 张；可先小规模验证）
bash scripts/download.sh 2000 400
bash scripts/prepare.sh 64

# 2) 训练（多随机种子，按验证集选最优；默认自动做数据策展）
#    单种子：PYTHONPATH=src python3 src/train.py
PYTHONPATH=src python3 src/train.py --seeds 0,1,2,3,4
cat data/seed_summary.csv   # 各随机种子的 val/test（含 TTA）

# 3) 导出权重为 C 数组
PYTHONPATH=src python3 src/export_weights.py

# 4) 编译并推理（默认开启四方向 TTA，--no-tta 可关闭）
make -C c whereisup
./c/whereisup path/to/photo.jpg -o fixed.png
```

## 效果

策展后（有明确上下线索的照片，cue ≥ 60，469 张源图）：

| 指标 | 值 |
|---|---|
| 单次前向 test acc | 100%（选模种子；5 种子均值 ≈ 97.9%） |
| 四方向 TTA test acc | 100%（5 种子均值 ≈ 98.7%） |
| numpy vs C 最大绝对差 | < 1e-4 |
| 单元测试 | 21 passed |

未策展的全部 993 张约 64%——因为近一半照片旋转后仍然“合理”，信息本身不在图里。
这不是模型不行，而是最好的教学案例：**数据质量决定上限**（见教程第 9 章）。

## 目录

```
src/    训练与导出（Python3 + numpy，零框架）
c/      纯 C 推理（gcc + 随仓库附带的 stb 单头库，权重在 weights.h）
scripts/ 下载 / 预处理 / 对拍
docs/TUTORIAL.md   从零教学（13 章）
```

## 测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
bash scripts/check_parity.sh 8
```

## License

未指定。欢迎在开源前补充（如 MIT）。
