# WhereIsUp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 训练一个微型 CNN，输入被旋转的照片，判断图片的“上边”在哪个方向（顺时针旋转 0°/90°/180°/270° 四分类），并把权重导出为 C 数组，用纯 C 完成推理与纠偏。

**Architecture:** 训练端使用 Python3 + numpy 手写各层前向/反向传播（不使用任何深度学习框架）。图片解码交给一个由 stb_image.h 编译的极简 C 工具 `img2ppm`，因此 Python 侧零图像库依赖。推理端为纯 C（仅 libc + 附带的 stb 单头库），加载导出的 `weights.h` 做前向计算，并保存纠偏后的 PNG。

**Tech Stack:** Python 3 + numpy（训练）、C99 + gcc（解码/推理）、stb_image.h / stb_image_write.h（随仓库附带）、picsum.photos（训练图片源）。

> 说明：本计划相对设计文档有一处简化——训练端不引入 Pillow，而由 C 工具 `img2ppm` 把下载的 JPEG 解码并缩放成 64×64 PPM，Python 直接用 numpy 读取。这保持了“Python 仅需 numpy、C 仅需 gcc”的极简原则，并让训练与推理的缩放逻辑使用同一份 C 代码，保证数值一致。

---

## File Structure

```
WhereIsUp/
├── .gitignore
├── README.md
├── data/                      # 运行产物（不入 git）
│   ├── raw/                   # 下载的原始 JPEG（400x400）
│   ├── ppm/                   # 64x64 PPM（img2ppm 生成）
│   ├── best.npz               # 训练得到的权重+归一化常数
│   └── training_curve.csv
├── src/
│   ├── data.py                # 枚举/切分/旋转标签/增强/归一化
│   ├── nn.py                  # Conv2d/MaxPool2d/ReLU/Flatten/Linear/SoftmaxCE + Net
│   ├── train.py               # 训练循环（动量 SGD + 余弦调度）
│   ├── infer_py.py            # numpy 推理（读 PPM，输出 logits/概率）
│   └── export_weights.py      # best.npz → c/weights.h
├── c/
│   ├── stb_image.h            # vendored（下载）
│   ├── stb_image_write.h      # vendored（下载）
│   ├── imutil.h / imutil.c    # 双线性缩放 / 旋转 / 读 PPM
│   ├── nn_inference.h / .c    # 纯 C 前向
│   ├── weights.h              # 生成产物
│   ├── img2ppm.c              # JPEG/PNG → 缩放 → PPM
│   ├── main.c                 # whereisup CLI
│   └── Makefile
├── scripts/
│   ├── download.sh            # 下载原图
│   ├── prepare.sh             # img2ppm 批量转 PPM
│   └── check_parity.sh        # numpy vs C logits 对拍
├── tests/
│   ├── test_data.py
│   ├── test_nn.py
│   ├── test_export.py
│   └── test_infer_py.py
└── docs/
    ├── TUTORIAL.md
    └── superpowers/{specs,plans}/...
```

**约定（标签与方向，贯穿全项目）：**
- 类别 `k ∈ {0,1,2,3}` = 图像被**顺时针旋转**了 `k*90°`。
- 显示图中“真实上方”所在的方向：`k=0→上, 1→右, 2→下, 3→左`。
- 纠偏：把显示图**逆时针**旋转 `k*90°`（等价于顺时针 `(4-k)%4` 次）即可复原。
- 通道顺序：网络内部一律 **CHW**；图像文件一律 **HWC**。

---

## Task 1: 项目骨架与 git 初始化

**Files:**
- Create: `.gitignore`
- Create: `README.md`

- [ ] **Step 1: 建目录**

Run:
```bash
mkdir -p src c scripts tests data docs
```

- [ ] **Step 2: 写 `.gitignore`**

```gitignore
data/
*.o
c/whereisup
c/img2ppm
*.ppm
__pycache__/
```

- [ ] **Step 3: 写 `README.md` 占位（后续 Task 10 补全）**

```markdown
# WhereIsUp

判断一张（被旋转的）照片的“上边”在哪个方向，并用纯 C 纠正方向。

- 训练：`src/`（Python3 + numpy）
- 推理：`c/`（gcc + 附带 stb 单头库，权重为 C 数组）
- 教程：`docs/TUTORIAL.md`
```

- [ ] **Step 4: 提交**

```bash
git add .gitignore README.md
git commit -m "chore: 初始化项目骨架"
```

---

## Task 2: 依赖获取 + C 解码工具 `img2ppm` + 下载脚本

**Files:**
- Create: `c/stb_image.h`, `c/stb_image_write.h`（下载）
- Create: `c/imutil.h`, `c/imutil.c`
- Create: `c/img2ppm.c`
- Create: `c/Makefile`
- Create: `scripts/download.sh`
- Create: `scripts/prepare.sh`

- [ ] **Step 1: 下载 stb 单头库**

Run:
```bash
curl -fsSL -o c/stb_image.h       "https://fastly.jsdelivr.net/gh/nothings/stb@master/stb_image.h"
curl -fsSL -o c/stb_image_write.h "https://fastly.jsdelivr.net/gh/nothings/stb@master/stb_image_write.h"
head -c 60 c/stb_image.h
```
Expected: 输出以 `/* stb_image - v2.3` 开头（两文件各约 28 万/7 万字节）。

- [ ] **Step 2: 写 `c/imutil.h`**

```c
#ifndef IMUTIL_H
#define IMUTIL_H

/* 双线性缩放: src/dst 均为 HWC, 通道数 c。dst 尺寸 dw x dh。 */
void bilinear_resize(const unsigned char *src, int sw, int sh, int c,
                     unsigned char *dst, int dw, int dh);

/* 顺时针旋转 k*90 度。dst 需调用者分配 w*h*c 字节。 */
void rotate90cw(const unsigned char *src, int w, int h, int c,
                unsigned char *dst, int k);

/* 读取二进制 P6 的 PPM。返回 malloc 的缓冲区, 失败返回 NULL。 */
unsigned char *read_ppm(const char *path, int *w, int *h, int *c);

/* 写二进制 P6 的 PPM。成功返回 0。 */
int write_ppm(const char *path, const unsigned char *data, int w, int h, int c);

#endif
```

- [ ] **Step 3: 写 `c/imutil.c`**

```c
#include "imutil.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

void bilinear_resize(const unsigned char *src, int sw, int sh, int c,
                     unsigned char *dst, int dw, int dh) {
    for (int y = 0; y < dh; y++) {
        float sy = ((y + 0.5f) * sh / dh) - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - y0;
        int y1 = y0 + 1;
        y0 = clampi(y0, 0, sh - 1); y1 = clampi(y1, 0, sh - 1);
        for (int x = 0; x < dw; x++) {
            float sx = ((x + 0.5f) * sw / dw) - 0.5f;
            int x0 = (int)floorf(sx);
            float fx = sx - x0;
            int x1 = x0 + 1;
            x0 = clampi(x0, 0, sw - 1); x1 = clampi(x1, 0, sw - 1);
            for (int ch = 0; ch < c; ch++) {
                float v00 = src[(y0 * sw + x0) * c + ch];
                float v01 = src[(y0 * sw + x1) * c + ch];
                float v10 = src[(y1 * sw + x0) * c + ch];
                float v11 = src[(y1 * sw + x1) * c + ch];
                float top = v00 + (v01 - v00) * fx;
                float bot = v10 + (v11 - v10) * fx;
                float v = top + (bot - top) * fy;
                dst[(y * dw + x) * c + ch] = (unsigned char)clampi((int)(v + 0.5f), 0, 255);
            }
        }
    }
}

void rotate90cw(const unsigned char *src, int w, int h, int c,
                unsigned char *dst, int k) {
    k = ((k % 4) + 4) % 4;
    if (k == 0) { memcpy(dst, src, (size_t)w * h * c); return; }
    unsigned char *cur = malloc((size_t)w * h * c);
    memcpy(cur, src, (size_t)w * h * c);
    int cw = w, chh = h;
    for (int t = 0; t < k; t++) {
        unsigned char *nxt = (t == k - 1) ? dst : malloc((size_t)chh * cw * c);
        for (int y = 0; y < chh; y++)
            for (int x = 0; x < cw; x++)
                for (int ch = 0; ch < c; ch++)
                    nxt[(x * chh + (chh - 1 - y)) * c + ch] = cur[(y * cw + x) * c + ch];
        free(cur);
        cur = nxt;
        int tmp = cw; cw = chh; chh = tmp;
    }
}

unsigned char *read_ppm(const char *path, int *w, int *h, int *c) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[3] = {0};
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) { fclose(f); return NULL; }
    int vals[3], got = 0;
    while (got < 3) {
        int ch = fgetc(f);
        if (ch == EOF) { fclose(f); return NULL; }
        if (ch == '#') { while ((ch = fgetc(f)) != EOF && ch != '\n') {} continue; }
        if (isspace(ch)) continue;
        ungetc(ch, f);
        if (fscanf(f, "%d", &vals[got]) != 1) { fclose(f); return NULL; }
        got++;
    }
    fgetc(f);
    int W = vals[0], H = vals[1];   /* vals[2] 是 maxval, P6 下忽略 */
    (void)vals[2];
    int CH = 3;                     /* P6 恒为 RGB 三通道 */
    unsigned char *buf = malloc((size_t)W * H * CH);
    if (fread(buf, 1, (size_t)W * H * CH, f) != (size_t)W * H * CH) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *w = W; *h = H; *c = CH;
    return buf;
}

int write_ppm(const char *path, const unsigned char *data, int w, int h, int c) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(data, 1, (size_t)w * h * c, f);
    fclose(f);
    return 0;
}
```

- [ ] **Step 4: 写 `c/img2ppm.c`**

```c
#include <stdio.h>
#include <stdlib.h>
#include "imutil.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: img2ppm <in> <out.ppm> [size=64]\n"); return 2; }
    int size = (argc >= 4) ? atoi(argv[3]) : 64;
    int w, h, c;
    unsigned char *img = stbi_load(argv[1], &w, &h, &c, 3);
    if (!img) { fprintf(stderr, "img2ppm: cannot decode %s\n", argv[1]); return 1; }
    unsigned char *dst = malloc((size_t)size * size * 3);
    bilinear_resize(img, w, h, 3, dst, size, size);
    stbi_image_free(img);
    int rc = write_ppm(argv[2], dst, size, size, 3);
    free(dst);
    if (rc) { fprintf(stderr, "img2ppm: cannot write %s\n", argv[2]); return 1; }
    return 0;
}
```

- [ ] **Step 5: 写 `c/Makefile`**

```makefile
CC ?= cc
CFLAGS ?= -std=c99 -O3 -Wall -Wextra
LDLIBS ?= -lm

TOOLS = img2ppm whereisup

all: $(TOOLS)

whereisup: main.c nn_inference.c imutil.c weights.h nn_inference.h imutil.h
	$(CC) $(CFLAGS) -o $@ main.c nn_inference.c imutil.c $(LDLIBS)

img2ppm: img2ppm.c imutil.c imutil.h
	$(CC) $(CFLAGS) -o $@ img2ppm.c imutil.c $(LDLIBS)

clean:
	rm -f $(TOOLS)
```

> 注：`whereisup` 依赖 `weights.h`，该文件在 Task 6 生成；本任务先只编译 `img2ppm`。

- [ ] **Step 6: 编译并测试 `img2ppm`**

Run:
```bash
make -C c img2ppm && printf 'test image' > /dev/null
```
Expected: 编译无警告，生成 `c/img2ppm`。

- [ ] **Step 7: 写 `scripts/download.sh`**

```bash
#!/usr/bin/env bash
# 从 picsum.photos 下载“正确朝向”的参考照片（Unsplash 授权）。
# 用法: scripts/download.sh [COUNT] [SIZE]
set -u
COUNT="${1:-2000}"
SIZE="${2:-400}"
OUT="data/raw"
mkdir -p "$OUT"
ok=0
for ((id=0; id<=1084 && ok<COUNT; id++)); do
  f="$OUT/$(printf '%04d' "$id").jpg"
  if [ -s "$f" ]; then ok=$((ok+1)); continue; fi
  url="https://picsum.photos/id/$id/$SIZE/$SIZE.jpg"
  if curl -fsSL --max-time 30 -o "$f" "$url" && [ -s "$f" ]; then
    magic=$(od -An -tx1 -N2 "$f" | tr -d ' \n')
    if [ "$magic" = "ffd8" ]; then
      ok=$((ok+1))
    else
      rm -f "$f"
    fi
  else
    rm -f "$f"
  fi
done
echo "done: $ok photos in $OUT (scanned id 0..$id)"
```

- [ ] **Step 8: 写 `scripts/prepare.sh`**

```bash
#!/usr/bin/env bash
# 把 data/raw/*.jpg 批量缩放为 64x64 PPM 到 data/ppm/
set -eu
SIZE="${1:-64}"
mkdir -p data/ppm
n=0
for f in data/raw/*.jpg; do
  [ -e "$f" ] || continue
  base="$(basename "$f" .jpg)"
  ./c/img2ppm "$f" "data/ppm/$base.ppm" "$SIZE" && n=$((n+1))
done
echo "converted $n images to data/ppm (${SIZE}x${SIZE})"
```

- [ ] **Step 9: 小规模验证下载与转换**

Run:
```bash
chmod +x scripts/*.sh
bash scripts/download.sh 20 400
bash scripts/prepare.sh 64
ls data/ppm | head
```
Expected: 下到约 20 张 JPEG；`data/ppm` 下出现 20 个 `.ppm` 文件。

- [ ] **Step 10: 提交**

```bash
git add .gitignore c scripts
git commit -m "feat: stb 依赖、img2ppm 解码工具与下载/预处理脚本"
```

---

## Task 3: `src/data.py` —— 切分、旋转标签、增强、归一化

**Files:**
- Create: `src/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: 写失败测试 `tests/test_data.py`**

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=src python3 -m unittest tests.test_data -v`
Expected: FAIL（`No module named 'data'` 或属性缺失）。

- [ ] **Step 3: 写 `src/data.py`**

```python
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
        img = rotate_cw(imgs[i], k)
        if rng.random() < flip_p:
            img = img[:, ::-1, :]
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=src python3 -m unittest tests.test_data -v`
Expected: 6 个测试全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat(data): 切分/旋转标签/增强/归一化 + 单元测试"
```

---

## Task 4: `src/nn.py` —— 手写各层与反向传播

**Files:**
- Create: `src/nn.py`
- Test: `tests/test_nn.py`

- [ ] **Step 1: 写失败测试 `tests/test_nn.py`（含数值梯度校验）**

```python
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import nn


def _num_grad(f, x, eps=1e-3):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; fp = f()
        x[i] = old - eps; fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


class TestNN(unittest.TestCase):
    def test_conv_forward_against_naive(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 3, 6, 6)).astype(np.float32)
        layer = nn.Conv2d(3, 4, k=3, stride=1, pad=1, rng=rng)
        out = layer.forward(x)
        self.assertEqual(out.shape, (2, 4, 6, 6))
        # 逐点手算第 0 个样本第 0 个通道中心像素
        oc, yy, xx = 0, 2, 3
        acc = layer.b[oc]
        for c in range(3):
            for i in range(3):
                for j in range(3):
                    acc += x[0, c, yy + i - 1, xx + j - 1] * layer.W[oc, c, i, j]
        self.assertTrue(np.allclose(out[0, oc, yy, xx], acc, atol=1e-4))

    def test_conv_backward_gradcheck(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((2, 3, 5, 5)).astype(np.float32)
        layer = nn.Conv2d(3, 2, k=3, stride=1, pad=1, rng=rng)
        layer.forward(x)
        dout = rng.standard_normal((2, 2, 5, 5)).astype(np.float32)
        dx = layer.backward(dout)
        f = lambda: float((layer.forward(x) * dout).sum())
        self.assertTrue(np.allclose(dx, _num_grad(f, x), atol=1e-2))

    def test_maxpool_backward_gradcheck(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
        layer = nn.MaxPool2d(2)
        layer.forward(x)
        dout = rng.standard_normal((1, 2, 2, 2)).astype(np.float32)
        dx = layer.backward(dout)
        f = lambda: float((layer.forward(x) * dout).sum())
        self.assertTrue(np.allclose(dx, _num_grad(f, x), atol=1e-2))

    def test_linear_backward_gradcheck(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((4, 5)).astype(np.float32)
        layer = nn.Linear(5, 3, rng=rng)
        layer.forward(x)
        dout = rng.standard_normal((4, 3)).astype(np.float32)
        dx = layer.backward(dout)
        f = lambda: float((layer.forward(x) * dout).sum())
        self.assertTrue(np.allclose(dx, _num_grad(f, x), atol=1e-2))

    def test_softmax_ce_onehot_equals_neglogp(self):
        logits = np.array([[2.0, 0.0, -1.0, 0.5]], dtype=np.float32)
        y = np.array([0])
        loss, p = nn.SoftmaxCE().forward(logits, y)
        self.assertTrue(np.allclose(loss, -np.log(p[0, 0]), atol=1e-6))
        self.assertTrue(np.allclose(p.sum(axis=1), 1.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=src python3 -m unittest tests.test_nn -v`
Expected: FAIL（`No module named 'nn'`）。

- [ ] **Step 3: 写 `src/nn.py`**

```python
"""手写神经网络层（仅用 numpy），每层提供 forward/backward。

内部一律 NCHW。所有参数为 float32。
"""
import numpy as np


class Conv2d:
    def __init__(self, in_ch, out_ch, k=3, stride=1, pad=1, rng=None):
        rng = rng or np.random.default_rng(0)
        self.k, self.stride, self.pad = k, stride, pad
        scale = np.sqrt(2.0 / (in_ch * k * k))          # He 初始化
        self.W = (rng.standard_normal((out_ch, in_ch, k, k)) * scale).astype(np.float32)
        self.b = np.zeros(out_ch, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.cols = None
        self.x_shape = None

    def _im2col(self, x):
        N, C, H, W = x.shape
        k, s, p = self.k, self.stride, self.pad
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        oh = (H + 2 * p - k) // s + 1
        ow = (W + 2 * p - k) // s + 1
        cols = np.empty((N, C, k, k, oh, ow), dtype=x.dtype)
        for i in range(k):
            for j in range(k):
                cols[:, :, i, j, :, :] = xp[:, :, i:i + s * oh:s, j:j + s * ow:s]
        return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * oh * ow, -1), oh, ow

    def forward(self, x):
        self.x_shape = x.shape
        N = x.shape[0]
        cols, oh, ow = self._im2col(x)
        self.cols = cols
        out = cols @ self.W.reshape(self.W.shape[0], -1).T + self.b
        return out.reshape(N, oh, ow, -1).transpose(0, 3, 1, 2)

    def backward(self, dout):
        N, OC, oh, ow = dout.shape
        _, C, H, W = self.x_shape
        k, s, p = self.k, self.stride, self.pad
        Wf = self.W.reshape(OC, -1)
        dflat = dout.transpose(0, 2, 3, 1).reshape(N * oh * ow, OC)
        self.dW = (dflat.T @ self.cols).reshape(self.W.shape)
        self.db = dflat.sum(axis=0)
        dcols = (dflat @ Wf).reshape(N, oh, ow, C, k, k).transpose(0, 3, 4, 5, 1, 2)
        dxp = np.zeros((N, C, H + 2 * p, W + 2 * p), dtype=dout.dtype)
        for i in range(k):
            for j in range(k):
                dxp[:, :, i:i + s * oh:s, j:j + s * ow:s] += dcols[:, :, i, j, :, :]
        return dxp[:, :, p:p + H, p:p + W] if p > 0 else dxp


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2d:
    def __init__(self, size=2):
        self.size = size

    def forward(self, x):
        N, C, H, W = x.shape
        s = self.size
        oh, ow = H // s, W // s
        xr = x[:, :, :oh * s, :ow * s].reshape(N, C, oh, s, ow, s)
        xr = xr.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, oh, ow, s * s)
        self.argmax = xr.argmax(axis=-1)
        self.x_shape = (N, C, H, W)
        self.ohow = (oh, ow)
        return xr.max(axis=-1)

    def backward(self, dout):
        N, C, H, W = self.x_shape
        oh, ow = self.ohow
        s = self.size
        dxr = np.zeros((N, C, oh, ow, s * s), dtype=dout.dtype)
        np.put_along_axis(dxr, self.argmax[..., None], dout[..., None], axis=-1)
        dxr = dxr.reshape(N, C, oh, ow, s, s).transpose(0, 1, 2, 4, 3, 5)
        dxr = dxr.reshape(N, C, oh * s, ow * s)
        dx = np.zeros((N, C, H, W), dtype=dout.dtype)
        dx[:, :, :oh * s, :ow * s] = dxr
        return dx


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class Linear:
    def __init__(self, in_f, out_f, rng=None):
        rng = rng or np.random.default_rng(0)
        scale = np.sqrt(2.0 / in_f)
        self.W = (rng.standard_normal((out_f, in_f)) * scale).astype(np.float32)
        self.b = np.zeros(out_f, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, dout):
        self.dW = dout.T @ self.x
        self.db = dout.sum(axis=0)
        return dout @ self.W


class SoftmaxCE:
    """Softmax 与交叉熵合并，数值稳定，反向即 (p - onehot) / N。"""

    def forward(self, logits, y):
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        self.p, self.y = p, y
        n = y.shape[0]
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
        return float(loss), p

    def backward(self):
        n = self.y.shape[0]
        d = self.p.copy()
        d[np.arange(n), self.y] -= 1.0
        return (d / n).astype(np.float32)


class Net:
    """固定结构：conv-relu-pool-conv-relu-pool-flatten-fc。"""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.conv1 = Conv2d(3, 8, rng=rng)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(2)
        self.conv2 = Conv2d(8, 16, rng=rng)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(2)
        self.flatten = Flatten()
        self.fc = Linear(16 * 16 * 16, 4, rng=rng)
        self.loss = SoftmaxCE()
        self.layers = [self.conv1, self.relu1, self.pool1,
                       self.conv2, self.relu2, self.pool2,
                       self.flatten, self.fc]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def params_and_grads(self):
        for layer in self.layers:
            if isinstance(layer, (Conv2d, Linear)):
                yield layer.W, layer.dW
                yield layer.b, layer.db

    def save(self, path):
        d = {}
        for name in ["conv1", "conv2", "fc"]:
            layer = getattr(self, name)
            d[name + ".W"] = layer.W
            d[name + ".b"] = layer.b
        np.savez(path, **d)

    def load(self, path):
        d = np.load(path)
        for name in ["conv1", "conv2", "fc"]:
            layer = getattr(self, name)
            layer.W = d[name + ".W"].astype(np.float32)
            layer.b = d[name + ".b"].astype(np.float32)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=src python3 -m unittest tests.test_nn -v`
Expected: 5 个测试全部 PASS（梯度校验通过）。

- [ ] **Step 5: 提交**

```bash
git add src/nn.py tests/test_nn.py
git commit -m "feat(nn): 手写 Conv/Pool/ReLU/FC/SoftmaxCE 及反向传播 + 梯度校验"
```

---

## Task 5: `src/train.py` —— 训练循环

**Files:**
- Create: `src/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: 写冒烟测试 `tests/test_train.py`**

```python
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import train


class TestTrain(unittest.TestCase):
    def test_train_steps_updates_parameters(self):
        rng = np.random.default_rng(0)
        imgs = rng.integers(0, 255, size=(16, 64, 64, 3), dtype=np.uint8)
        y = np.array([i % 4 for i in range(16)])
        net = train.Net(seed=0)
        before = net.conv1.W.copy()
        losses = train.train_steps(net, imgs, y, epochs=1, batch=8, lr=0.05, seed=0)
        self.assertEqual(len(losses), 1)
        self.assertTrue(np.isfinite(losses[0]))
        self.assertFalse(np.allclose(before, net.conv1.W))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=src python3 -m unittest tests.test_train -v`
Expected: FAIL（`No module named 'train'`）。

- [ ] **Step 3: 写 `src/train.py`**

```python
"""WhereIsUp 训练脚本。

用法:
  PYTHONPATH=src python3 src/train.py [--epochs 10] [--batch 64] [--lr 0.01] [--seed 0]
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
    """在给定（正立）图像上训练，内部做旋转增强。返回每 epoch 的平均 loss。"""
    rng = np.random.default_rng(seed)
    if mean is None:
        mean = np.zeros(3, dtype=np.float32)
    if std is None:
        std = np.ones(3, dtype=np.float32)
    vel = {id(p): np.zeros_like(p) for p, _ in net.params_and_grads()}
    n = imgs.shape[0]
    total = epochs * max(1, n // batch)
    step = 0
    hist = []
    for ep in range(epochs):
        order = rng.permutation(n)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n - batch + 1, batch):
            idx = order[i:i + batch]
            aug, labels = D.augment(imgs[idx], rng)
            x = D.normalize(aug, mean, std)
            logits = net.forward(x)
            loss, _ = net.loss.forward(logits, labels)
            d = net.loss.backward()
            for layer in reversed(net.layers):
                d = layer.backward(d)
            cur_lr = cosine_lr(lr, step, total)
            for p, g in net.params_and_grads():
                v = vel[id(p)]
                v *= momentum
                v -= cur_lr * g
                p += v
            ep_loss += loss
            nb += 1
            step += 1
        hist.append(ep_loss / max(1, nb))
        if verbose:
            print(f"epoch {ep + 1}/{epochs} loss={hist[-1]:.4f} lr={cosine_lr(lr, step, total):.5f}")
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
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=D.PPM)
    args = ap.parse_args()

    ids = D.list_ids(args.data)
    if len(ids) < 20:
        raise SystemExit(f"not enough images in {args.data}; run scripts/download.sh && scripts/prepare.sh")
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
        loss_hist = train_steps(net, tr, np.zeros(len(tr), dtype=np.int64),
                                epochs=1, batch=args.batch, lr=args.lr,
                                seed=args.seed + ep, mean=mean, std=std)
        acc, cm, _ = eval_rotated(net, va, mean, std, rng)
        curve.append((ep + 1, loss_hist[0], acc))
        print(f"epoch {ep + 1}/{args.epochs} train_loss={loss_hist[0]:.4f} val_acc={acc:.4f}")
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
```

- [ ] **Step 4: 在 `src/data.py` 追加批量旋转辅助**

在 `src/data.py` 末尾添加：

```python
def rotate_cw_batch(imgs, k):
    """对一批 NHWC 图像施加同一个顺时针旋转角 k。"""
    return np.rot90(imgs, -k, axes=(1, 2))
```

- [ ] **Step 5: 运行测试确认通过**

Run: `PYTHONPATH=src python3 -m unittest tests.test_train -v`
Expected: PASS（参数被更新、loss 为有限值）。

- [ ] **Step 6: 提交**

```bash
git add src/train.py src/data.py tests/test_train.py
git commit -m "feat(train): 训练循环（动量SGD+余弦调度）+ 冒烟测试"
```

---

## Task 6: `src/export_weights.py` —— 导出 `c/weights.h`

**Files:**
- Create: `src/export_weights.py`
- Test: `tests/test_export.py`

- [ ] **Step 1: 写测试 `tests/test_export.py`（生成后回读校验）**

```python
import os
import sys
import re
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import export_weights as E


class TestExport(unittest.TestCase):
    def test_export_roundtrip(self):
        d = {
            "conv1.W": np.arange(8 * 3 * 3 * 3, dtype=np.float32).reshape(8, 3, 3, 3) / 7.0,
            "conv1.b": np.arange(8, dtype=np.float32),
            "conv2.W": np.arange(16 * 8 * 3 * 3, dtype=np.float32).reshape(16, 8, 3, 3) / 13.0,
            "conv2.b": np.arange(16, dtype=np.float32),
            "fc.W": np.arange(4 * 4096, dtype=np.float32).reshape(4, 4096) / 100.0,
            "fc.b": np.arange(4, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as td:
            npz = os.path.join(td, "w.npz")
            np.savez(npz, **d)
            out = os.path.join(td, "weights.h")
            E.write_header(npz, np.array([0.5, 0.5, 0.5], np.float32),
                           np.array([0.25, 0.25, 0.25], np.float32), out)
            with open(out) as f:
                text = f.read()
        self.assertIn("NORM_MEAN", text)
        self.assertIn("W1", text)
        self.assertIn("B3", text)

        def parse(name):
            m = re.search(rf"{name}\[[^\]]*\]\s*=\s*\{{(.*?)\}};", text, re.S)
            return np.array([float(v) for v in re.findall(r"[-+0-9.eE]+", m.group(1))], dtype=np.float32)

        self.assertTrue(np.allclose(parse("B1"), d["conv1.b"], atol=1e-4))
        self.assertTrue(np.allclose(parse("NORM_STD"), [0.25, 0.25, 0.25], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=src python3 -m unittest tests.test_export -v`
Expected: FAIL。

- [ ] **Step 3: 写 `src/export_weights.py`**

```python
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
    md5 = hashlib.md5(open(npz_path, "rb").read()).hexdigest()
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
```

> 注意：`write_header` 也用于测试，签名含 `dims` 以便未来扩展；测试只传位置参数。

- [ ] **Step 4: 运行测试确认通过**

Run: `PYTHONPATH=src python3 -m unittest tests.test_export -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/export_weights.py tests/test_export.py
git commit -m "feat(export): best.npz → c/weights.h + 回读测试"
```

---

## Task 7: `src/infer_py.py` —— numpy 推理（对拍用）

**Files:**
- Create: `src/infer_py.py`
- Test: `tests/test_infer_py.py`

- [ ] **Step 1: 写测试 `tests/test_infer_py.py`**

```python
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import infer_py


class TestInferPy(unittest.TestCase):
    def test_forward_matches_manual_softmax_sum(self):
        if not os.path.exists("data/best.npz"):
            self.skipTest("尚未训练 (data/best.npz 不存在)")
        net, mean, std = infer_py.load("data/best.npz", "data/norm.npz")
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        logits, probs = infer_py.infer_image(net, mean, std, img)
        self.assertEqual(logits.shape, (4,))
        self.assertLess(abs(probs.sum() - 1.0), 1e-5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `PYTHONPATH=src python3 -m unittest tests.test_infer_py -v`
Expected: FAIL（`No module named 'infer_py'`）。

- [ ] **Step 3: 写 `src/infer_py.py`**

```python
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
```

- [ ] **Step 4: 运行测试**

Run: `PYTHONPATH=src python3 -m unittest tests.test_infer_py -v`
Expected: PASS（未训练时自动跳过；训练后执行断言）。

- [ ] **Step 5: 提交**

```bash
git add src/infer_py.py tests/test_infer_py.py
git commit -m "feat(infer): numpy 推理（对拍入口）"
```

---

## Task 8: C 推理端 `nn_inference` + `main` + 编译

**Files:**
- Create: `c/nn_inference.h`, `c/nn_inference.c`, `c/main.c`
- Requires: `c/weights.h`（由 Task 6 生成，或先用 Task 11 产出）

- [ ] **Step 1: 写 `c/nn_inference.h`**

```c
#ifndef NN_INFERENCE_H
#define NN_INFERENCE_H

#define IMG_C 3
#define IMG_H 64
#define IMG_W 64
#define N_CLASSES 4

/* 输入 x 为 CHW 归一化后的 float[3*64*64]。
 * logits[4] 原始分数; probs[4] softmax 概率。 */
void forward(const float *x, float *logits, float *probs);

#endif
```

- [ ] **Step 2: 写 `c/nn_inference.c`**

```c
#include "nn_inference.h"
#include "weights.h"
#include <math.h>

static void conv3x3(const float *in, int C, int H, int W,
                    const float *K, int OC, const float *B, float *out) {
    for (int oc = 0; oc < OC; oc++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float acc = B[oc];
                for (int c = 0; c < C; c++) {
                    for (int i = 0; i < 3; i++) {
                        int yy = y + i - 1;
                        if (yy < 0 || yy >= H) continue;
                        for (int j = 0; j < 3; j++) {
                            int xx = x + j - 1;
                            if (xx < 0 || xx >= W) continue;
                            acc += in[c * H * W + yy * W + xx] *
                                   K[((oc * C + c) * 3 + i) * 3 + j];
                        }
                    }
                }
                out[oc * H * W + y * W + x] = acc;
            }
        }
    }
}

static void relu_inplace(float *a, int n) {
    for (int i = 0; i < n; i++) if (a[i] < 0.0f) a[i] = 0.0f;
}

static void maxpool2(const float *in, int C, int H, int W, float *out) {
    int oh = H / 2, ow = W / 2;
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                float m = in[c * H * W + (2 * y) * W + 2 * x];
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++) {
                        float v = in[c * H * W + (2 * y + i) * W + (2 * x + j)];
                        if (v > m) m = v;
                    }
                out[c * oh * ow + y * ow + x] = m;
            }
        }
    }
}

static void linear(const float *in, int IN, const float *Wm, int OUT,
                   const float *b, float *out) {
    for (int o = 0; o < OUT; o++) {
        float acc = b[o];
        for (int i = 0; i < IN; i++) acc += in[i] * Wm[o * IN + i];
        out[o] = acc;
    }
}

static void softmax(const float *in, int n, float *out) {
    float m = in[0];
    for (int i = 1; i < n; i++) if (in[i] > m) m = in[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { out[i] = expf(in[i] - m); s += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= s;
}

void forward(const float *x, float *logits, float *probs) {
    static float c1[8 * 64 * 64];
    static float p1[8 * 32 * 32];
    static float c2[16 * 32 * 32];
    static float p2[16 * 16 * 16];
    conv3x3(x, 3, 64, 64, W1, 8, B1, c1);
    relu_inplace(c1, 8 * 64 * 64);
    maxpool2(c1, 8, 64, 64, p1);
    conv3x3(p1, 8, 32, 32, W2, 16, B2, c2);
    relu_inplace(c2, 16 * 32 * 32);
    maxpool2(c2, 16, 32, 32, p2);
    linear(p2, 16 * 16 * 16, W3, 4, B3, logits);
    softmax(logits, 4, probs);
}
```

- [ ] **Step 3: 写 `c/main.c`**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "imutil.h"
#include "nn_inference.h"
#include "weights.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static const char *DIRNAME[4] = {"上", "右", "下", "左"};
static float input_buf[3 * 64 * 64];

static void build_input(const unsigned char *hwc) {
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++)
            for (int c = 0; c < 3; c++) {
                float v = hwc[(y * 64 + x) * 3 + c] / 255.0f;
                input_buf[c * 64 * 64 + y * 64 + x] = (v - NORM_MEAN[c]) / NORM_STD[c];
            }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: whereisup <image> [-o fixed.png]\n"
                        "       whereisup --ppm <file.ppm> [--logits]\n");
        return 2;
    }
    int ppm_mode = (strcmp(argv[1], "--ppm") == 0);
    int logits_mode = 0;
    const char *out_path = NULL;
    const char *in_path = NULL;

    if (ppm_mode) {
        if (argc < 3) { fprintf(stderr, "whereisup: --ppm needs a file\n"); return 2; }
        in_path = argv[2];
        for (int i = 3; i < argc; i++) if (!strcmp(argv[i], "--logits")) logits_mode = 1;
    } else {
        in_path = argv[1];
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "-o") && i + 1 < argc) out_path = argv[++i];
        }
    }

    int ow = 0, oh = 0, oc = 0;
    unsigned char *small = NULL;      /* 64x64x3 HWC */
    unsigned char *full = NULL;       /* 原分辨率 HWC，用于纠偏保存 */

    if (ppm_mode) {
        small = read_ppm(in_path, &ow, &oh, &oc);
        if (!small || ow != 64 || oh != 64 || oc != 3) {
            fprintf(stderr, "whereisup: %s must be a 64x64 P6 PPM\n", in_path);
            free(small);
            return 1;
        }
    } else {
        int w = 0, h = 0, c = 0;
        unsigned char *img = stbi_load(in_path, &w, &h, &c, 3);
        if (!img) { fprintf(stderr, "whereisup: cannot decode %s\n", in_path); return 1; }
        if (w < 8 || h < 8) { fprintf(stderr, "whereisup: image too small (%dx%d)\n", w, h); stbi_image_free(img); return 1; }
        full = img;
        small = malloc(64 * 64 * 3);
        bilinear_resize(img, w, h, 3, small, 64, 64);
    }

    build_input(small);
    float logits[4], probs[4];
    forward(input_buf, logits, probs);

    if (logits_mode) {
        printf("%.6f %.6f %.6f %.6f\n", logits[0], logits[1], logits[2], logits[3]);
    } else {
        int k = 0;
        for (int i = 1; i < 4; i++) if (probs[i] > probs[k]) k = i;
        printf("probs: %.4f %.4f %.4f %.4f\n", probs[0], probs[1], probs[2], probs[3]);
        printf("上边在: %s (k=%d, 纠正需逆时针 %d 度)\n", DIRNAME[k], k, k * 90);
        if (!ppm_mode && out_path) {
            int w = 0, h = 0, c = 0;
            (void)c;
            stbi_info(in_path, &w, &h, &c);
            unsigned char *fixed = malloc((size_t)w * h * 3);
            rotate90cw(full, w, h, 3, fixed, (4 - k) % 4);
            if (!stbi_write_png(out_path, (k == 0 || k == 2) ? w : h,
                                (k == 0 || k == 2) ? h : w, 3, fixed,
                                ((k == 0 || k == 2) ? w : h) * 3))
                fprintf(stderr, "whereisup: cannot write %s\n", out_path);
            else
                printf("纠正图已保存: %s\n", out_path);
            free(fixed);
        }
    }

    free(small);
    if (full) stbi_image_free(full);
    return 0;
}
```

- [ ] **Step 4: 编译（先生成一个临时 weights.h 以便本任务能编译）**

Run:
```bash
make -C c img2ppm
python3 - <<'PY'
import numpy as np, sys
sys.path.insert(0, "src")
import export_weights as E
rng = np.random.default_rng(0)
d = {
 "conv1.W": rng.standard_normal((8,3,3,3)).astype(np.float32)*0.1,
 "conv1.b": np.zeros(8, np.float32),
 "conv2.W": rng.standard_normal((16,8,3,3)).astype(np.float32)*0.1,
 "conv2.b": np.zeros(16, np.float32),
 "fc.W": rng.standard_normal((4,4096)).astype(np.float32)*0.01,
 "fc.b": np.zeros(4, np.float32),
}
np.savez("/tmp/opencode/fake.npz", **d)
E.write_header("/tmp/opencode/fake.npz", np.zeros(3,np.float32), np.ones(3,np.float32), "c/weights.h")
print("temp weights.h written")
PY
make -C c whereisup
```
Expected: 编译零警告，生成 `c/whereisup`。

- [ ] **Step 5: 冒烟运行**

Run:
```bash
f=$(ls data/ppm/*.ppm | head -1)
./c/whereisup --ppm "$f"
```
Expected: 打印 4 个概率（随机权重，数值无所谓）与方向行。

- [ ] **Step 6: 提交**

```bash
git add c/nn_inference.h c/nn_inference.c c/main.c
git commit -m "feat(c): 纯 C 前向推理与 whereisup CLI"
```

---

## Task 9: 对拍脚本 `scripts/check_parity.sh`

**Files:**
- Create: `scripts/check_parity.sh`

- [ ] **Step 1: 写脚本**

```bash
#!/usr/bin/env bash
# 校验 numpy 与 C 的前向 logits 一致（最大绝对差 < 1e-4）。
set -eu
TOL="1e-4"
n="${1:-8}"
tmp="$(mktemp -d)"
pass=0
i=0
for f in data/ppm/*.ppm; do
  [ "$i" -lt "$n" ] || break
  py=$(PYTHONPATH=src python3 src/infer_py.py "$f" --logits)
  cc=$(./c/whereisup --ppm "$f" --logits)
  PY_OUT="$py" CC_OUT="$cc" TOL="$TOL" python3 - <<'PY'
import os, sys
py = [float(v) for v in os.environ["PY_OUT"].split()]
cc = [float(v) for v in os.environ["CC_OUT"].split()]
tol = float(os.environ["TOL"])
d = max(abs(a-b) for a, b in zip(py, cc))
if d >= tol:
    print(f"MISMATCH max|Δ|={d:.3e} py={py} cc={cc}")
    sys.exit(1)
print(f"ok max|Δ|={d:.3e}")
PY
  pass=$((pass+1))
  i=$((i+1))
done
rm -rf "$tmp"
echo "parity passed for $pass images (tol $TOL)"
```

- [ ] **Step 2: 运行前需有真实权重**

先完成 Task 11 的训练与导出，或临时用 Task 8 Step 4 的假权重跑一次确认脚本机制可用：

Run:
```bash
chmod +x scripts/check_parity.sh
bash scripts/check_parity.sh 4
```
Expected: `parity passed for 4 images (tol 1e-4)`。

- [ ] **Step 3: 提交**

```bash
git add scripts/check_parity.sh
git commit -m "test: numpy/C 前向数值对拍脚本"
```

---

## Task 10: 文档 `docs/TUTORIAL.md` 与 `README.md`

**Files:**
- Create: `docs/TUTORIAL.md`
- Modify: `README.md`

- [ ] **Step 1: 写 `docs/TUTORIAL.md`（13 章）**

逐章撰写，每章要求如下（正文为中文；含公式、代码引用与小实验）：

1. **任务与自标注**：旋转任务定义；说明数据集大小与标签来源；`rotate_cw/up_side/correct` 的约定。
2. **数据切分与泄漏**：为什么按源照片切；`split_ids` 代码解读。
3. **图像即张量**：HWC/CHW、/255、归一化公式 `(x-mean)/std` 与 `norm_stats`。
4. **积木**：卷积（输出尺寸公式、参数计数）、ReLU、最大池化、展平、全连接、softmax；每层配一个 3×3 手算例子。
5. **损失**：交叉熵推导 `-log p_y`；为何不用准确率当损失（不可导/无梯度信息）。
6. **反向传播**：链式法则逐层推导，含 softmax+CE 合并梯度 `(p-onehot)/N`、卷积 im2col 反向、池化 argmax 路由；与 `nn.py` 逐行对照。
7. **优化器**：SGD → 动量公式 `v=μv-ηg; p+=v` → 学习率、余弦调度 `cosine_lr`；建议实验：把 lr 改成 0.001/0.1 观察曲线。
8. **训练循环**：epoch/batch/验证集选模/检查点；解读 `train.py` 主循环与 `training_curve.csv`。
9. **过拟合与正则化**：数据增强（水平翻转合法、**垂直翻转非法**、亮度抖动）；可选 weight decay 实验。
10. **评估与错误分析**：混淆矩阵解读（哪两个方向易混）。
11. **部署**：权重导出为 C 数组（`export_weights.py` / `weights.h` 布局说明）；`forward` 的 C 实现对照。
12. **数值对拍**：为什么移植后必须对拍；`check_parity.sh` 用法与判定阈值。
13. **进阶路线**：int8 量化（对称量化公式与效果预期）、更深网络、真实旋转角度回归。

- [ ] **Step 2: 写 `README.md` 完整版**

```markdown
# WhereIsUp —— 判断图片的“上边”在哪

输入一张被旋转的照片，判断“上边”在 上/右/下/左 哪个方向，并输出纠正后的图片。
权重是 C 数组，推理只需 gcc；训练用 Python3 + numpy 手写（无深度学习框架）。

## 快速开始

```bash
# 1) 取数据（约 2000 张，可先小规模验证）
bash scripts/download.sh 2000 400
bash scripts/prepare.sh 64

# 2) 训练（输出 data/best.npz, data/norm.npz, data/training_curve.csv）
PYTHONPATH=src python3 src/train.py --epochs 10

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
```

- [ ] **Step 3: 提交**

```bash
git add docs/TUTORIAL.md README.md
git commit -m "docs: 教学长文与 README"
```

---

## Task 11: 端到端训练、导出与验收

**Files:**
- Modify: `c/weights.h`（训练后真实导出）

- [ ] **Step 1: 运行全部单元测试**

Run: `PYTHONPATH=src python3 -m unittest discover -s tests -v`
Expected: 全部 PASS。

- [ ] **Step 2: 全量下载与预处理**

Run:
```bash
bash scripts/download.sh 2000 400
bash scripts/prepare.sh 64
ls data/ppm | wc -l
```
Expected: `>= 1500`（picsum 有效 id 约 1000+，够用）。若网络受限导致数量过少，向用户说明并记录实际数量。

- [ ] **Step 3: 训练**

Run: `PYTHONPATH=src python3 src/train.py --epochs 10 --batch 64 --lr 0.01`
Expected: 末行打印 `BEST val_acc=... TEST acc=...` 且 **TEST acc ≥ 0.95**，并打印混淆矩阵。

- [ ] **Step 4: 导出真实权重并编译**

Run:
```bash
PYTHONPATH=src python3 src/export_weights.py
make -C c clean && make -C c
```
Expected: `wrote c/weights.h ...`；`whereisup` 与 `img2ppm` 均零警告编译。

- [ ] **Step 5: 对拍**

Run: `bash scripts/check_parity.sh 8`
Expected: `parity passed for 8 images (tol 1e-4)`。

- [ ] **Step 6: 真实图片纠偏演示**

Run:
```bash
f=$(ls data/raw/*.jpg | head -1)
./c/img2ppm "$f" /tmp/opencode/one.ppm 64
# 用 python 造一张旋转 90° 的图并保存为 ppm，验证 C 能判出 k=1
PYTHONPATH=src python3 - <<'PY'
import numpy as np, sys
sys.path.insert(0, "src")
import data as D
img = D.read_ppm("/tmp/opencode/one.ppm")
# 顺时针 90°，并写成 PPM
rot = D.rotate_cw(img, 1)
h, w, c = rot.shape
with open("/tmp/opencode/rot90.ppm", "wb") as f:
    f.write(b"P6\n%d %d\n255\n" % (w, h)); f.write(rot.tobytes())
print("wrote /tmp/opencode/rot90.ppm")
PY
./c/whereisup --ppm /tmp/opencode/rot90.ppm
```
Expected: 打印 `上边在: 右 (k=1, ...)`（应为 k=1）。

- [ ] **Step 7: 完整 CLI 保存纠偏图**

Run:
```bash
./c/whereisup "$f" -o /tmp/opencode/fixed.png && file /tmp/opencode/fixed.png
```
Expected: 保存 PNG 成功，`file` 显示 PNG 图片。

- [ ] **Step 8: 提交**

```bash
git add -A
git commit -m "feat: 端到端训练并导出真实权重，通过验收（test acc>=0.95, parity<1e-4）"
```

---

## Self-Review（作者自查记录）

- **Spec 覆盖**：数据下载/自标注/按源切分（Task 2/3/11）、增强与归一化（Task 3/5）、模型与手写反向传播（Task 4）、训练循环与超参（Task 5）、权重导出 C 数组（Task 6）、纯 C 推理+纠偏图（Task 8）、对拍（Task 9）、教程 13 章（Task 10）、验收标准（Task 11）——均有对应任务。
- **占位符**：无 TBD/TODO；每个代码步骤含完整代码与确切命令/预期。
- **一致性**：`rotate_cw/up_side/correct`、CHW 布局、`W[oc][ic][kh][kw]`、`W[out][in]`、`forward(x, logits, probs)`、`NORM_MEAN/STD`、类别 0..3 与方向 {上,右,下,左} 在 Python 与 C 两侧命名/语义一致。
- **已知偏差**：设计文档的“旋转在高分辨率上做”简化为对 64×64 做 90° 旋转（90° 倍数无角落损失），并记录于文首说明。
