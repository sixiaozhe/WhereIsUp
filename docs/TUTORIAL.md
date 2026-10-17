# WhereIsUp 从零教学

这份教程用一个完整、可运行的项目，把"训练一个小神经网络"所需要的绝大部分知识讲一遍：

- 输入一张可能被旋转过的照片，判断照片的**上边**在 上 / 右 / 下 / 左 哪一侧；
- 训练端只用 **Python3 + numpy 手写**前向与反向传播，不依赖任何深度学习框架；
- 推理端是**纯 C**（只依赖 gcc 与仓库自带的两个 stb 单头库），权重以 C 数组形式烧进程序。

阅读方式：手边开着源码，边读边对照。本文所有代码引用都指向仓库里的真实实现。

---

## 1. 任务与自标注

### 1.1 任务定义

给定一张照片，它可能是把某张"正立"照片**顺时针**旋转 $k\cdot 90^\circ$ 得到的，其中 $k\in\{0,1,2,3\}$。我们要判断 $k$，也就是判断"真实的上方"落在显示图的哪一侧。

约定贯穿整个仓库（见 `src/data.py:16` 的 `DIRS`）：

| 类别 $k$ | 显示图里"上"在哪一侧 |
|---|---|
| 0 | up（上） |
| 1 | right（右） |
| 2 | down（下） |
| 3 | left（左） |

三个核心函数：

- `rotate_cw(img, k)`（`src/data.py:45`）把 HWC 图像顺时针旋转 $k\cdot90^\circ$：
  ```python
  return np.rot90(img, -k, axes=(0, 1))
  ```
- `up_side(k)`（`src/data.py:50`）返回类别 $k$ 时"上"所在的方位名：`DIRS[k]`。
- `correct(img, k)`（`src/data.py:55`）把显示图**逆时针**旋转 $k\cdot90^\circ$，即复原：
  ```python
  return np.rot90(img, k, axes=(0, 1))
  ```

注意 `rotate_cw` 用 `np.rot90(..., -k)`（负号代表顺时针），`correct` 用 `np.rot90(..., k)`（正号代表逆时针），二者互为逆操作。`tests/test_data.py:30` 的 `test_correct_undoes_rotation` 就是在验证这一点：对 $k=0,1,2,3$，`correct(rotate_cw(img, k), k)` 必须还原为原图。

C 端用整数旋转表达同一件事：`correct` 等价于顺时针旋转 $(4-k)\bmod 4$ 次，所以 `c/main.c:112` 写的是 `rotate90cw(full, w, h, 3, fixed, (4 - k) % 4)`。

### 1.2 数据集大小与标签来源

- 数据来自 `scripts/download.sh`，从 `picsum.photos` 按 id（`0..1084`）下载免版权照片，默认每张 400×400，最多约 1000 张（失效 id 自动跳过），存到 `data/raw/`。
- `scripts/prepare.sh` 调用 `c/img2ppm` 把每张 JPEG 双线性缩放成 **64×64 RGB** 的 P6 PPM，存到 `data/ppm/`（默认 `SIZE=64`）。
- **标签是免费的**：`picsum.photos` 的源照片都被视为"正立"。训练时在线地对每张源图随机抽一个 $k$，顺时针转 $k\cdot90^\circ$，旋转角就是标签。我们不需要人工标注——这正是这个任务的教学价值所在：**自监督式构造标签**，把"无标签图片"变成"无限多的带标签样本"。

这一步在 `src/data.py:90` 的 `augment` 里完成（下节与第 9 章详谈）。因为每次取图都重新随机旋转，同一张源图在不同 epoch 会以不同类别出现，等于免费扩充了 4 倍数据。

---

## 2. 数据切分与数据泄漏

### 2.1 为什么必须按"源照片"切

如果先对每张源图生成 4 个旋转版本，再随机把**这些旋转版本**打散到 train/val/test，那么同一张源图的旋转副本会同时出现在训练集和测试集里。模型只要记住"这张图的纹理长什么样"，就能在测试时认出它的旋转版本，从而虚高准确率。这叫**数据泄漏**。

正确做法：**按源照片 id 切分**，一张源图的所有旋转版本只会出现在同一个集合里。

### 2.2 `split_ids` 代码解读

`src/data.py:65`：

```python
def split_ids(ids, seed=0, ratios=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ids))     # 固定种子 → 可复现
    n = len(ids)
    n_tr = int(round(ratios[0] * n))    # 80%
    n_va = int(round(ratios[1] * n))    # 10%
    tr = [ids[i] for i in idx[:n_tr]]
    va = [ids[i] for i in idx[n_tr:n_tr + n_va]]
    te = [ids[i] for i in idx[n_tr + n_va:]]
    return tr, va, te
```

- 用固定 `seed` 打乱 id 顺序，保证每次运行切分一致（可复现）。
- 按 8:1:1 切成 train/val/test。
- 返回的是**源图 id 列表**；真正加载时 `load_images(ids)`（`src/data.py:77`）才按 id 读 PPM。

不变量由 `tests/test_data.py:36` 校验：三个集合两两不相交，且总数等于输入数量。

> 关键心法：**切分单位 = 独立样本单位**。只要同一"实物"能派生出多个样本，就必须在实物层面切。人脸识别按人切、语音按说话人切、本项目按源照片切。

---

## 3. 图像即张量

### 3.1 两种内存布局

- **HWC**：图像文件的自然顺序，`height × width × channels`。读 PPM 得到的就是 HWC，`src/data.py:19` 的 `read_ppm` 最后 `reshape(h, w, 3)`。
- **CHW**：网络内部一律用 CHW（通道在前），因为卷积按通道组织更自然。`src/data.py:113` 的 `normalize` 在最后做 `np.transpose(x, (0, 3, 1, 2))` 完成 HWC→NCHW。

约定：**文件里是 HWC，网络里是 CHW**（另见设计文档 §3）。

### 3.2 缩放与归一化

像素原本是 `uint8` 的 0..255。第一步除 255 变到 $[0,1]$：

$$x' = \frac{x_{\text{pixel}}}{255}$$

第二步按**每通道**做标准化：

$$x'' = \frac{x' - \mu_c}{\sigma_c}, \qquad c\in\{R,G,B\}$$

其中 $\mu_c,\sigma_c$ 是训练集上的均值与标准差。为什么必须用训练集统计量？因为验证/测试时不能"偷看"测试集分布，否则又是一种泄漏。

`src/data.py:81` 的 `norm_stats`：

```python
def norm_stats(imgs):
    x = imgs.astype(np.float32) / 255.0
    mean = x.reshape(-1, 3).mean(axis=0)   # 每通道均值
    std  = x.reshape(-1, 3).std(axis=0)    # 每通道标准差
    std  = np.where(std < 1e-6, 1.0, std)  # 防 0 除
    return mean.astype(np.float32), std.astype(np.float32)
```

- `reshape(-1, 3)` 把所有像素摊平成 `(N*H*W, 3)`，再对每个通道求统计量。
- 标准差可能为 0（例如某通道全同色），用 `np.where` 兜底为 1，避免除零。

训练时 `mean/std` 由训练集算出（`src/train.py:121`），并随权重一起保存到 `data/norm.npz`（`src/train.py:144`），导出给 C 端。归一化的完整链条在 `src/data.py:113`：

```python
def normalize(imgs, mean, std):
    x = imgs.astype(np.float32) / 255.0
    x = (x - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
    return np.transpose(x, (0, 3, 1, 2))
```

C 端 `c/main.c:16` 的 `build_input` 是它的逐像素翻版（注意已是单张图，无 batch 维）：

```c
float v = hwc[(y * 64 + x) * 3 + c] / 255.0f;
input_buf[c * 64 * 64 + y * 64 + x] = (v - NORM_MEAN[c]) / NORM_STD[c];
```

归一化让每层输入的尺度稳定，是让训练收敛快、数值稳的最廉价手段之一。

---

## 4. 神经网络的积木

网络结构（`src/nn.py:140` 的 `Net`）：

```
输入 3×64×64
├─ Conv2d 3→8,  3×3, stride 1, pad 1   → 8×64×64
├─ ReLU
├─ MaxPool2d 2×2                       → 8×32×32
├─ Conv2d 8→16, 3×3, stride 1, pad 1   → 16×32×32
├─ ReLU
├─ MaxPool2d 2×2                       → 16×16×16
├─ Flatten                             → 4096
├─ Linear 4096→4                       → 4
└─ Softmax（推理时给出概率）
```

下面逐层给出公式和一个 3×3 规模的手算例子。

### 4.1 卷积

对输入 $x$（$C_{in}$ 个通道）、卷积核 $w$（$C_{out}$ 个输出通道，每通道 $C_{in}\times k\times k$）、偏置 $b$：

$$y[o,\,y,\,x] = b[o] + \sum_{c=0}^{C_{in}-1}\sum_{i=0}^{k-1}\sum_{j=0}^{k-1} x[c,\,y+i-p,\,x+j-p]\cdot w[o,\,c,\,i,\,j]$$

输出空间尺寸（$p$ 为 padding，$s$ 为 stride）：

$$O = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1$$

本项目 $k=3,s=1,p=1$，所以 $O=H$：卷积**不改变**分辨率。代码里 `oh, ow` 就是这样算的（`src/nn.py:24`）。

参数计数（每层）：

$$C_{out}\times C_{in}\times k\times k + C_{out}$$

本项目：conv1 $= 8\times3\times3\times3+8 = 224$；conv2 $=16\times8\times3\times3+16 = 1168$；fc $=4\times4096+4 = 16388$；**合计 17,780**，float32 约 69 KB。

**手算 3×3 卷积（单通道、pad=1、stride=1）**

输入 $x$：

```
1 2 0
0 1 2
2 0 1
```

核 $w$（记 $b=0$）：

```
0 1 0
1 2 1
0 1 0
```

先 pad 一圈 0，得 5×5：

```
0 0 0 0 0
0 1 2 0 0
0 0 1 2 0
0 2 0 1 0
0 0 0 0 0
```

滑动计算（每个输出位置取 3×3 窗口与核逐元素相乘再求和）。左上角窗口取 padded 的 0–2 行、0–2 列：

```
0*0 + 0*1 + 0*0
0*1 + 1*2 + 2*1   = 0 + 4 + 0 = 4
0*0 + 0*1 + 1*0
```

中心窗口（1–3 行、1–3 列）：

```
1*0 + 2*1 + 0*0
0*1 + 1*2 + 2*1   = 2 + 4 + 0 = 6
2*0 + 0*1 + 1*0
```

同样算完 9 个位置，输出为：

```
4 6 4
4 6 6
4 4 4
```

（加偏置 $b$ 则每个元素再加 $b$。这里 $b=0$。）

代码里卷积不是写 9 个窗口，而是用 **im2col** 把所有窗口摊成矩阵一次乘完，见 `src/nn.py:20`。第 6 章会解释它为什么等价。

### 4.2 ReLU

$$\text{ReLU}(x) = \max(0, x)$$

逐元素、无参数。例：输入 `[-2, 1.5, 0, 3, -0.5]` → 输出 `[0, 1.5, 0, 3, 0]`。

代码 `src/nn.py:56`：

```python
def forward(self, x):
    self.mask = x > 0
    return x * self.mask
```

它把负值清零、正值保留。非线性正是让多层网络能拟合复杂函数的关键；没有它，堆再多线性层也等价于一层线性变换。

### 4.3 最大池化

把特征图按 $s\times s$ 的窗口不重叠地取最大值。$2\times2$ 池化把分辨率减半。

**手算 2×2 池化**

输入 4×4：

```
1 3 2 4
5 6 1 0
2 0 7 2
1 1 3 1
```

四个 2×2 窗口分别取最大：`max{1,3,5,6}=6`、`max{2,4,1,0}=4`、`max{2,0,1,1}=2`、`max{7,2,3,1}=7`，输出：

```
6 4
2 7
```

代码 `src/nn.py:65`。池化没有可学习参数，作用是降分辨率、扩大感受野、并带来一点平移不变性。

### 4.4 展平

把 $16\times16\times16$ 的 CHW 张量按通道优先顺序拉成长度 4096 的向量。

**手算小例**：一个 $C=2,H=W=2$ 的张量，通道 0 = `[[1,2],[3,4]]`，通道 1 = `[[5,6],[7,8]]`，按 CHW 展平为 `[1,2,3,4, 5,6,7,8]`。代码 `src/nn.py:93`：`x.reshape(x.shape[0], -1)`。

### 4.5 全连接（线性）

$$y = Wx + b,\qquad W\in\mathbb{R}^{out\times in},\ b\in\mathbb{R}^{out}$$

**手算**：输入 $x=[1,2,3]$，$W=\begin{bmatrix}1&0&-1\\0.5&0.5&0.5\end{bmatrix}$，$b=[0,1]$：

- $y_0 = 1\cdot1+0\cdot2+(-1)\cdot3+0 = -2$
- $y_1 = 0.5\cdot1+0.5\cdot2+0.5\cdot3+1 = 4$

输出 `[-2, 4]`。代码 `src/nn.py:102` 的 `forward` 即 `x @ self.W.T + self.b`。

### 4.6 Softmax

把原始分数（logits）$z$ 变成概率：

$$p_c = \frac{e^{z_c}}{\sum_{c'} e^{z_{c'}}}$$

**手算**：$z=[2,1,0]$，则 $e^z=[7.389,2.718,1]$，和 $=11.107$，$p=[0.665,0.245,0.090]$。

实现里先减去最大值保证数值稳定（`src/nn.py:125`）：

```python
z = logits - logits.max(axis=1, keepdims=True)
e = np.exp(z)
p = e / e.sum(axis=1, keepdims=True)
```

减去同一个常数不改变 softmax 结果，却能避免 $e^{z}$ 溢出。C 端 `c/nn_inference.c:59` 的 `softmax` 同理。

> 为什么池化/ReLU 没有参数，却仍是"层"？因为它们也要在反向传播里把梯度正确地传给前一层（ReLU 用 mask、池化用 argmax），所以同样要实现 `backward`。

---

## 5. 损失函数：交叉熵

分类任务用**交叉熵**衡量预测分布与真实标签的距离。对一个样本，真实类别为 $y$：

$$L = -\log p_y$$

一个 batch（$N$ 个样本）取平均：

$$L = -\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i}$$

直白理解：$p_y$ 越接近 1，$-\log p_y$ 越接近 0；$p_y$ 越接近 0，损失越大。例：$p_y=0.9 \Rightarrow 0.105$；$p_y=0.1 \Rightarrow 2.303$。

实现 `src/nn.py:121` 的 `SoftmaxCE` 把 softmax 与交叉熵**合并**，一次算完（也便于求导，见第 6 章）：

```python
loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
```

`+1e-12` 是防止 $\log 0$。

### 为什么不用准确率当损失？

准确率 $=\frac{1}{N}\sum \mathbb{1}[\arg\max_c p_c = y]$ 看似直观，却不能当损失，因为：

1. **不可导**：$\arg\max$ 和指示函数都是阶梯状的，梯度几乎处处为 0，无法反向传播。
2. **没有梯度信息**：即使模型错了，准确率只告诉"错了"，不告诉"错多远、往哪个方向改"。把所有概率都压低一点，准确率不变、梯度依然为 0。
3. **不连续**：预测概率从 0.49 变到 0.51 会让准确率跳变，而交叉熵是平滑的，能给出"我现在有多不确定"的连续信号。

所以我们用交叉熵**训练**、用准确率/混淆矩阵**评估**——这是两套不同的指标，各司其职。

---

## 6. 反向传播

反向传播就是链式法则的工程化：损失对某参数的梯度 = "损失对输出的梯度 × 输出对参数的梯度"。`Net` 把所有层放进一个列表（`src/nn.py:154`），前向顺序调用 `forward`，反向用 `reversed(net.layers)` 顺序调用 `backward`（`src/train.py:46`）。

设损失为 $L$，每层收到来自上一层的 $\mathrm{d}L/\mathrm{d}\text{out}$，要算出并返回 $\mathrm{d}L/\mathrm{d}\text{in}$，同时把参数梯度存下来。

### 6.1 softmax + 交叉熵合并梯度

设 logits 为 $z$，经过 softmax 得到 $p$，标签 one-hot 为 $t$。可以证明

$$\frac{\partial L}{\partial z_c} = p_c - t_c$$

在上式对整个 batch 取平均后，实现就是：

```python
d = self.p.copy()
d[np.arange(n), self.y] -= 1.0     # 只有真实类减 1，等价于 p - onehot
return (d / n).astype(np.float32)
```

见 `src/nn.py:133`。这就是"合并"的好处：省掉了单独的 softmax 求导，形式极简。

### 6.2 全连接

前向 $y = Wx + b$。若已知 $\mathrm{d}L/\mathrm{d}y = g$，则

$$\frac{\partial L}{\partial x} = W^\top g,\qquad \frac{\partial L}{\partial W} = g\,x^\top,\qquad \frac{\partial L}{\partial b} = \sum_{\text{batch}} g$$

实现 `src/nn.py:115`：

```python
def backward(self, dout):
    self.dW = dout.T @ self.x
    self.db = dout.sum(axis=0)
    return dout @ self.W
```

（`dout` 形状 `(N, out)`，`self.x` 形状 `(N, in)`，故 `dout.T @ x` 得到 `(out, in)` 的 `dW`。）

### 6.3 ReLU

前向 `mask = x > 0`。反向就把梯度"原样通过正元素、负元素归零"：

$$\frac{\partial L}{\partial x} = g \odot \text{mask}$$

实现 `src/nn.py:61`：`return dout * self.mask`。

### 6.4 最大池化：argmax 路由

前向时记录了每个窗口最大值的位置 `self.argmax`（`src/nn.py:75`）。反向时梯度只回传给当初取到最大值的那一个位置，其余为 0：

```python
dxr = np.zeros(...)
np.put_along_axis(dxr, self.argmax[..., None], dout[..., None], axis=-1)
```

见 `src/nn.py:80`。直觉：前向只有最大元素参与了输出，反向自然只有它接受梯度。

### 6.5 卷积：im2col 前向与反向

**前向**先把每个 $k\times k$ 窗口摊成一行，做一个矩阵乘法（`src/nn.py:20` 的 `_im2col`，`src/nn.py:32` 的 `forward`）：

$$\text{out} = \text{cols}\cdot W_f^\top + b,\qquad W_f = W.reshape(C_{out}, -1)$$

`cols` 形状 `(N·oh·ow, C·k·k)`，`W_f` 形状 `(C_out, C·k·k)`，乘完 reshape 回 `(N, C_out, oh, ow)`。这样把卷积变成一次大 GEMM，快且好读。

**反向**（`src/nn.py:40`）：

- 权重梯度：`self.dW = (dflat.T @ self.cols).reshape(self.W.shape)`，即 $\mathrm{d}W = \text{dout}_{\text{flat}}^\top \cdot \text{cols}$；
- 偏置梯度：`self.db = dflat.sum(axis=0)`；
- 输入梯度：先 `dcols = dflat @ Wf` 得到每个窗口的贡献，再**散射相加**回原图位置（`src/nn.py:48-52`）：同一个输入像素会被多个窗口用到，所以要用 `+=` 累加。padding 部分最后裁掉。

### 6.6 与 `nn.py` 的逐行对照

| 公式/概念 | 代码位置 |
|---|---|
| im2col 窗口展开 | `src/nn.py:20-30` |
| 卷积前向 | `src/nn.py:32-38` |
| 卷积反向（dW/db/dx 散射） | `src/nn.py:40-53` |
| ReLU 前向/反向（mask） | `src/nn.py:56-62` |
| MaxPool 前向（记 argmax）/反向（路由） | `src/nn.py:65-90` |
| Flatten 前向/反向 | `src/nn.py:93-99` |
| Linear 前向/反向 | `src/nn.py:102-118` |
| SoftmaxCE 前向（稳定）/反向 $(p-t)/N$ | `src/nn.py:121-137` |

反向传播的正确性不是靠肉眼保证的。`tests/test_nn.py` 用**数值梯度**（中心差分）校验每一层的反向：对输入每个分量施加 $\pm\varepsilon$ 扰动、算损失差，再与解析梯度比对（`test_conv_backward_gradcheck`、`test_maxpool_backward_gradcheck`、`test_linear_backward_gradcheck`）。这是调试自定义层的标准手段：**先信数值梯度**。

---

## 7. 优化器

### 7.1 从 SGD 到动量

最朴素的随机梯度下降（SGD）：

$$p \leftarrow p - \eta\, g$$

其中 $g$ 是当前 batch 的梯度、$\eta$ 是学习率。缺点是方差大、在狭长谷底来回震荡。

**动量**引入速度 $v$，把历史梯度做指数滑动平均：

$$v \leftarrow \mu v - \eta g,\qquad p \leftarrow p + v$$

$\mu$ 通常取 0.9。动量让方向一致的分量累积、来回震荡的分量抵消，从而更稳更快。实现 `src/train.py:49`：

```python
for p, g in net.params_and_grads():
    v = vel[id(p)]
    v *= momentum          # v = μ v
    v -= lr * g            # v = μ v - η g
    p += v                 # p += v
```

速度用参数的 `id` 索引（`src/train.py` 里的 `vel` 字典）。

### 7.2 学习率与余弦调度

学习率 $\eta$ 是最重要、最该先调的超参：太大不收敛（loss 震荡甚至爆掉），太小收敛慢。经验做法是**先大后小**：前期大步快跑，后期小步细化。

本项目用**余弦退火**（`src/train.py:18`）：

$$\eta(t) = \eta_0 \cdot \tfrac12\left(1 + \cos\!\left(\pi \cdot \frac{t}{T}\right)\right)$$

$t$ 是当前 epoch，$T$ 是总 epoch 数。$t=0$ 时 $\eta=\eta_0$，$t=T$ 时 $\eta\to0$，中间平滑下降。注意调度是**跨 epoch** 计算的（`main` 里每个 epoch 用 `cosine_lr` 算出一个 `lr_ep` 再传给 `train_steps`），因此在单个 epoch 内部学习率是恒定的；这样总步数 $T$ 才是"整个训练"而不是"一个 epoch"，否则学习率会每个 epoch 都跌到 0 又重启，浪费训练。

> **建议实验**：把学习率改成 `0.001` 和 `0.05` 各训一遍，对比 `data/training_curve.csv` 里的 `train_loss` 曲线：
> ```bash
> PYTHONPATH=src python3 src/train.py --epochs 60 --lr 0.001
> PYTHONPATH=src python3 src/train.py --epochs 60 --lr 0.05
> ```
> 典型现象：`0.001` 收敛很慢；`0.05` 太大时会**发散**——本模型在 `lr≈0.01` 以上就会出现 loss 先降后升、然后卡住的现象。这会让你直观感受学习率为何是第一超参。

---

## 8. 训练循环

### 8.1 epoch / batch

- **batch**：一次前向+反向送入的样本数，本项目默认 64（`--batch`）。
- **epoch**：把训练集每个样本大致用一遍。

每个 epoch 先 `order = rng.permutation(n)` 打乱样本顺序（`src/train.py:37`），再按 batch 切段。`train_steps`（`src/train.py:22`）对每个 batch：

```python
aug, labels = D.augment(imgs[idx], rng)   # 在线旋转增强，标签由旋转角给出
x = D.normalize(aug, mean, std)           # /255 + 标准化 + HWC→NCHW
logits = net.forward(x)
loss, _ = net.loss.forward(logits, labels)
d = net.loss.backward()
for layer in reversed(net.layers):
    d = layer.backward(d)                 # 反向传播
# 动量更新参数（学习率 lr 由调用方按 epoch 传入，epoch 内恒定）
```

注意 `train_steps` 的 `y` 参数其实没被使用：**标签完全来自 `D.augment` 内部产生的旋转角**（`src/train.py:42`）。这是自标注的核心。

### 8.2 验证集选模与检查点

`train_run`（`src/train.py:112`，单个随机种子的完整流程）每个 epoch 结束后：

1. 在**验证集**上评估。验证用的是 `eval_rotated`（`src/train.py:77`）：把每张验证源图分别旋转 $k=0,1,2,3$，构造出四类样本再评估——这才是与真实使用一致的评价方式。
2. 若验证准确率创新高，保存权重到 `data/best.npz`，同时保存归一化常数到 `data/norm.npz`（`src/train.py:141-144`）——即"**按验证集选最优模型**"。
3. 追加一行到 `training_curve.csv`：`epoch, train_loss, val_acc`（`src/train.py:145`）。

训练结束后重新 `net.load("data/best.npz")`，在**测试集**上评估并打印混淆矩阵（`src/train.py:150-152`）。

### 8.3 如何读 `training_curve.csv`

文件长这样（表头 + 每 epoch 一行，下面是策展后一次真实训练的节选）：

```
epoch,train_loss,val_acc
1,2.5780,0.6755
2,0.5185,0.9149
10,0.0927,0.9787
30,0.0770,0.9840
60,0.0616,0.9840
```

健康的训练应当是 `train_loss` 总体下降、`val_acc` 总体上升。若 `train_loss` 下降而 `val_acc` 停滞或下降，多半是过拟合（第 9 章）；若两者都下不去，则是欠拟合（模型太小/训练不足/学习率不合适）。这份曲线对应的验证集只有 47 张图，因此 `val_acc` 会以 1/47≈0.021 为台阶跳动。

### 8.4 多随机种子：让选模更可靠

验证集只有几十张时，单次训练的 `best val` 带有随机性（权重初始化、样本打乱、增强顺序都受种子影响）。同一套超参换一个种子，测试准确率可能在 96%–100% 之间跳动。

`src/train.py` 的 `--seeds` 一次训练多个种子，每个种子独立切分、独立训练、各自保存到 `data/seedN/`，最后按**验证集**（不是测试集）挑最优并复制到 `data/best.npz`：

```bash
PYTHONPATH=src python3 src/train.py --seeds 0,1,2,3,4
cat data/seed_summary.csv
```

`data/seed_summary.csv` 同时给出每个种子的单次与 TTA 的 val/test 准确率。**用测试集挑种子是作弊**：测试集只能用来在流程结束后汇报一次结果，选模永远只看验证集。

---

## 9. 过拟合与正则化

过拟合：模型把训练集的噪声也背了下来，训练损失很低但验证/测试表现差。小数据集 + 大参数量的组合最容易过拟合。手段：

### 9.1 数据增强（本项目的主力度）

`src/data.py:90` 的 `augment` 每步做：

1. 随机抽旋转角 $k\in\{0,1,2,3\}$（**这是标签的来源**）；
2. 以 0.5 概率做**水平翻转**，且在**旋转之前**做；
3. 亮度/对比度抖动：`gain = 1±0.10`、`bias = ±0.05`，裁剪回 $[0,1]$。

```python
if rng.random() < flip_p:
    img = img[:, ::-1, :]   # 先翻转（翻转不改变“上”），再旋转
img = rotate_cw(img, k)
```

**为什么水平翻转合法、垂直翻转非法？**

- 水平翻转是沿**竖直轴**镜像，图像的"上"方向向量不变，所以翻转后再旋转 $k\cdot90^\circ$，"上"仍然落在第 $k$ 侧——**标签不变**。`tests/test_data.py:61` 的第二个用例验证了这一点：必定翻转时，`correct(x, y)` 应等于水平翻转后的原图。
- 垂直翻转是沿**水平轴**镜像，会把"上"变成"下"，标签被破坏。用它训练等于给模型喂错标签，属于**非法增强**。这是本任务最值得记住的知识点：**增强必须与标签语义相容**。

因为旋转标签本就是免费的，增强几乎不花额外标注成本，却能成倍提升泛化。

### 9.2 weight decay（可选实验）

权重衰减在更新时把参数往 0 拉一点，抑制过大的权重：

$$p \leftarrow p - \eta\,(g + \lambda p)$$

在 `train_steps` 的更新循环里把 `v -= cur_lr * g` 换成 `v -= cur_lr * (g + wd * p)` 即可。可以试着加 `wd=1e-4` 观察验证曲线是否更平滑。

### 9.3 数据策展：数据质量决定上限（本项目最重要的实践）

比任何正则化都有效的一步是**先把无法判断的样本剔除**。旋转任务里，随机照片中约一半根本没有上下线索（特写、俯拍、纹理、抽象图），把它们旋转 90° 后依然"看起来合理"。模型对这类图无法做出正确判断——这不是模型不够大，而是**信息本身不在图片里**。

一个简单可解释的线索是"天空在上"：正立照片顶部通常比底部更亮。`src/data.py` 的 `up_cue` 用「顶部 16 行平均亮度 − 底部 16 行平均亮度」度量它，`curate_ids` 据此过滤（由 `train.py --min-cue` 控制，默认 60）。

实测（同一模型、同一训练流程，验证/测试集每张源图都按 4 个方向评估、要求全对）：

| 数据 | 4 路全对率 |
|---|---|
| 全部 993 张 | ≈ 0.70 |
| 按线索强度分档：最弱 25% | 0.12 |
| 最强 25% | 0.92 |
| 策展后（cue ≥ 60，469 张） | **val 0.96–0.99 / test 0.97–1.00** |

结论：**数据质量与标注的可判别性，往往比模型容量更决定上限**。先问"这条信息在输入里到底有没有"，再考虑加深网络。想复现未策展结果：`--min-cue -1e9`。

> 其余正则化（Dropout、早停、减小模型）不在本项目实现，但都值得了解。

---

## 10. 评估与错误分析

只用"准确率"一个数字，会掩盖模型到底错在哪。**混淆矩阵**把"真实类别 × 预测类别"的计数摊开，`src/train.py:71` 构造、`src/train.py:156` 打印，行列含义为：**行=真实 $k$，列=预测 $k$**。

```
        pred0 pred1 pred2 pred3
true0 [  ...    .     .     . ]
true1 [   .   ...     .     . ]
true2 [   .     .   ...     . ]
true3 [   .     .     .   ... ]
```

读法：对角线是正确分类，非对角线是错误。若大量错误集中在 `true0→pred2` 和 `true2→pred0`，说明模型分不清"上"和"下"（即差了 $180^\circ$）：对很多照片，倒过来的版本纹理统计仍与正立相似，模型只学会了"天空颜色在图片上部"这类弱特征。若错误集中在 `true1↔pred3`，说明分不清"左"和"右"（镜像式的混淆）。

针对性改进很直接：如果 0↔2 混淆严重，说明模型没学到足够的方向性结构，可以加深网络或减少池化带来的平移不变性；如果只是相邻类（如 0↔1）混淆，往往是训练不足或学习率不合适。错误分析的价值就在于**把"准确率 92%"翻译成"具体哪两类在打架"**，从而指导下一步。

---

## 11. 部署：权重导出为 C 数组

### 11.1 导出 `c/weights.h`

`src/export_weights.py` 把 `data/best.npz`（以及 `data/norm.npz`）转成 C 源码。核心函数 `write_header`（`src/export_weights.py:36`）写出：

```c
static const float NORM_MEAN[3] = {...};
static const float NORM_STD[3]  = {...};
static const float W1[216]  = {...};   /* conv1 权重 8×3×3×3 */
static const float B1[8]    = {...};   /* conv1 偏置 */
static const float W2[1152] = {...};   /* conv2 权重 16×8×3×3 */
static const float B2[16]   = {...};   /* conv2 偏置 */
static const float W3[16384] = {...};  /* fc 权重 */
static const float B3[4]    = {...};   /* fc 偏置 */
```

数组是**行主序展平**的，布局约定写在文件头注释里（`src/export_weights.py:44-45`）：

- 卷积权重：`W[oc][ic][kh][kw]`；
- 全连接权重：`W[out][in]`。

`_cfloat`（`src/export_weights.py:10`）保证每个浮点都带小数点/后缀成为合法 C 字面量（`0 → 0.0f`）。文件头还写入了源 npz 的 md5，用于权重与代码的版本绑定。用法：

```bash
PYTHONPATH=src python3 src/export_weights.py
# 或指定：--npz data/best.npz --norm data/norm.npz --out c/weights.h
```

### 11.2 C 端的 `forward` 对照

`c/nn_inference.c:67` 的 `forward` 是 numpy 前向的逐层直译：

```c
conv3x3(x, 3, 64, 64, W1, 8, B1, c1);
relu_inplace(c1, 8 * 64 * 64);
maxpool2(c1, 8, 64, 64, p1);
conv3x3(p1, 8, 32, 32, W2, 16, B2, c2);
relu_inplace(c2, 16 * 32 * 32);
maxpool2(c2, 16, 32, 32, p2);
linear(p2, 16 * 16 * 16, W3, 4, B3, logits);
softmax(logits, 4, probs);
```

- `conv3x3`（`c/nn_inference.c:5`）用直接双重循环实现带 padding 的卷积，核索引 `K[((oc*C+c)*3+i)*3+j]` 正是 `[oc][ic][kh][kw]` 的行主序展开。
- `relu_inplace`、`maxpool2`、`linear`、`softmax` 一一对应第 4 章的公式。
- 单头库 `stb_image.h`/`stb_image_write.h` 负责读任意格式图片、写 PNG；`c/main.c` 用 `bilinear_resize` 缩放到 64×64、`build_input` 归一化、预测后把**原分辨率**图按 $(4-k)\bmod4$ 顺时针旋转并保存为 PNG。整个推理只依赖 libc。

编译与运行：

```bash
make -C c whereisup
./c/whereisup path/to/photo.jpg -o fixed.png
```

---

## 12. 数值对拍

### 12.1 为什么移植后必须对拍

把 Python 训练好的模型搬到 C，最容易出错的不是算法思想，而是**细节**：数组索引顺序、padding 边界、归一化常数量级、softmax 是否减最大值……这些错误往往不会让程序崩溃，只会让输出"看起来差不多但就是不对"，极难排查。

**数值对拍**的思路：让两条独立实现处理**完全相同的输入**，比较中间/最终数值。若最大绝对差小于阈值，就确认移植正确。`src/infer_py.py` 就是用 numpy 复刻推理流程、作为 C 端的参照。

对拍用 `--ppm` 模式，两边读的是**同一个 64×64 PPM**，从而绕开缩放实现差异，只考验网络前向本身是否一致。

### 12.2 `check_parity.sh` 用法与判定

`scripts/check_parity.sh`：对前 `n` 张 PPM（默认 8），分别运行

```bash
PYTHONPATH=src python3 src/infer_py.py "$f" --logits   # numpy logits
./c/whereisup --ppm "$f" --logits                      # C logits
```

再比较两者 logits 的**最大绝对差**，判定阈值 `TOL=1e-4`：小于则 `ok`，否则打印 `MISMATCH` 并以非 0 退出。

```bash
bash scripts/check_parity.sh 8
# 输出示例：ok [tta] max|Δ|=...  /  ok [single] max|Δ|=...
# parity passed for 16 checks (8 images x 2 modes, tol 1e-4)
```

阈值定在 1e-4 是合理的：float32 累加顺序不同会带来约 $10^{-6}\sim10^{-5}$ 量级的舍入差，1e-4 既能容忍正常浮点误差，又能抓住任何真正的实现错误。对拍脚本同时充当**版本回归测试**：每次改权重或改推理代码都要重跑。脚本默认会**同时跑 TTA 与 `--no-tta` 两条路径**（共 2n 次比较）。

### 12.3 测试时增强（TTA）

**TTA**（Test-Time Augmentation）是在推理阶段对同一输入做多种变换、把结果融合，以降低单次前向的方差。旋转任务有一个天然的对称性可用：把输入顺时针再转 $r\cdot90^\circ$，真实类别就从 $k$ 变成 $(k+r)\bmod4$。于是对 $r=0,1,2,3$ 各前向一次，把第 $r$ 次结果里第 $j$ 个分数**反向平移**到类别 $(j-r)\bmod4$ 上再求平均，就得到对原图的估计。这个对齐规则实现在 `src/infer_py.py:20` 的 `tta_average`：

```python
acc[..., (j - r) % 4] += logits[..., j]   # 平均后再 softmax
```

C 端在 `c/main.c:83` 用同一规则：对 64×64 小图调用 `rotate90cw` 得到 4 个视角，累加对齐后的 logits，最后 `softmax4`。默认开启，可用 `--no-tta` 关闭；两边默认行为一致，所以对拍仍然成立。实测在本项目上，多随机种子的测试集平均准确率会从约 97.9% 提升到约 98.7%（个别模型本已接近满分，提升不明显，但 TTA 显著降低了"差一张图"的方差）。

---

## 13. 进阶路线

### 13.1 int8 量化

现在权重是 float32，约 69 KB，推理用浮点。可以把权重与激活量化成 8 位整数，模型体积缩小约 4 倍，且在支持 SIMD 的 CPU 上快得多。

常用**对称量化**：对每个张量找缩放因子

$$s = \frac{\max|x|}{127},\qquad q = \operatorname{round}\!\left(\frac{x}{s}\right),\ q\in[-127,127]$$

反量化 $x \approx q\cdot s$。卷积时可把 $q_x\cdot q_w$ 用 int32 累加，最后再乘 $s_x s_w$。预期效果：精度通常只掉零点几个百分点，但对本项目这种小模型要小心——量化误差在浅层网络里占比更明显。难点是导出的 `weights.h` 要同时带 $s$（每通道或每张量一个），C 端把 float 累加换成 int32 累加。

### 13.2 更深的网络

可以加卷积层/通道数、用 BatchNorm、加残差连接等。但要记住：网络越深，手写反向传播越容易出错，务必配合 `tests/test_nn.py` 那样的数值梯度校验逐层验证。

### 13.3 真实旋转角度回归

现在是 4 分类（只认 90° 的整数倍）。真实场景里照片可能被旋转任意角度。把任务改成**回归**：网络输出连续角度 $\theta$，损失要处理角度的周期性（例如用 $\cos/\sin$ 两个分量表示，或对预测做圆形距离）。同时要处理"翻转歧义"——很多照片上下颠倒后依然自然，需要考虑是否引入翻转分支。这是从"玩具分类"走向"实用纠偏"的自然一步。

---

## 附：一键流程回顾

```bash
# 取数据（约 1000 张；可先小规模验证）
bash scripts/download.sh 2000 400
bash scripts/prepare.sh 64

# 训练（多随机种子，按验证集选最优；默认推理开启四方向 TTA）
PYTHONPATH=src python3 src/train.py --seeds 0,1,2,3,4

# 导出权重为 C 数组
PYTHONPATH=src python3 src/export_weights.py

# 编译 + 推理 + 纠偏（--no-tta 可关闭测试时增强）
make -C c whereisup
./c/whereisup path/to/photo.jpg -o fixed.png

# 单元测试 + 数值对拍
PYTHONPATH=src python3 -m unittest discover -s tests -v
bash scripts/check_parity.sh 8
```

至此，你已经走完"数据 → 手写训练 → 导出权重 → 纯 C 推理 → 数值验证"的完整闭环。
