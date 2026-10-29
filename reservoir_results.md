# Reservoir (ESN) 基线对比

ESN 为免训练模型：随机固定循环权重 + 岭回归读出。
超参：{'n_states': 1500, 'spectral_radius': 1.2, 'leak': 1.0, 'sparsity': 0.05, 'ridge': 1e-06, 'input_scaling': 0.5}。

**评估协议（与神经网络不完全相同，必须注意）**：

| 项 | 神经网络 | ESN |
|---|---|---|
| 训练样本 | 31,500 | 31,500（已对齐） |
| rollout 长度 | 250 步 | 150 步 |
| 测试轨迹数 | 40 × 2 种子 = 80 | 20 × 1 种子 = 20 |
| 训练种子 | 3（无噪声）/ 2（含噪） | 3 |

ESN 每次 rollout 都要重跑 50 步 1500×1500 矩阵乘，按原协议约 6 分钟/组。
缩减协议不影响「谁更强」的方向性结论（差距是倍数级），
但 ESN 的 VPT 绝对值被 150 步上限截断。


## c1 / noise=0.00 / ratio=1.00

| model | VPT (mean ± std) | n |
|---|---|---|
| ESN | 36.2 ± 6.8 | 3 |
| mlp | 99.9 ± 2.9 | 3 |
| lstm | 67.5 ± 20.1 | 3 |
| transformer | 49.2 ± 1.3 | 3 |

## c1 / noise=0.05 / ratio=1.00

| model | VPT (mean ± std) | n |
|---|---|---|
| ESN | 58.7 ± 4.2 | 3 |
| mlp | 66.5 ± 1.3 | 2 |
| lstm | 64.9 ± 2.0 | 2 |
| transformer | 44.2 ± 7.3 | 2 |

## c1 / noise=0.10 / ratio=1.00

| model | VPT (mean ± std) | n |
|---|---|---|
| ESN | 45.6 ± 3.5 | 3 |
| mlp | 62.1 ± 1.0 | 2 |
| lstm | 56.9 ± 0.1 | 2 |
| transformer | 54.1 ± 1.3 | 2 |

## c2 / noise=0.00 / ratio=0.50

| model | VPT (mean ± std) | n |
|---|---|---|
| ESN | 51.9 ± 2.1 | 3 |
| mlp | 83.0 ± 0.9 | 3 |
| lstm | 61.5 ± 11.8 | 3 |
| transformer | 32.0 ± 4.7 | 3 |

## c2 / noise=0.05 / ratio=0.50

| model | VPT (mean ± std) | n |
|---|---|---|
| ESN | 45.3 ± 2.5 | 3 |
| mlp | 44.2 ± 0.9 | 2 |
| lstm | 43.8 ± 1.1 | 2 |
| transformer | 32.8 ± 3.6 | 2 |

## c2 / noise=0.10 / ratio=0.50

| model | VPT (mean ± std) | n |
|---|---|---|
| ESN | 35.5 ± 0.8 | 3 |
| mlp | 39.7 ± 0.0 | 2 |
| lstm | 39.9 ± 0.0 | 2 |
| transformer | 34.5 ± 0.7 | 2 |

## c3 / noise=0.00 / ratio=0.33

| model | VPT (mean ± std) | n |
|---|---|---|
| ESN | 30.1 ± 2.8 | 3 |
| mlp | 78.5 ± 4.5 | 3 |
| lstm | 74.6 ± 10.5 | 3 |
| transformer | 32.8 ± 5.3 | 3 |

## c3 / noise=0.05 / ratio=0.33

| model | VPT (mean ± std) | n |
|---|---|---|
| ESN | 52.8 ± 5.9 | 3 |
| mlp | 51.1 ± 1.0 | 2 |
| lstm | 56.3 ± 1.4 | 2 |
| transformer | 32.1 ± 2.7 | 2 |

## c3 / noise=0.10 / ratio=0.33

| model | VPT (mean ± std) | n |
|---|---|---|
| ESN | 38.6 ± 3.4 | 3 |
| mlp | 43.1 ± 1.0 | 2 |
| lstm | 44.5 ± 3.0 | 2 |
| transformer | 38.2 ± 0.1 | 2 |

## c4 / noise=0.00 / ratio=0.17

| model | VPT (mean ± std) | n |
|---|---|---|
| ESN | 40.8 ± 2.1 | 3 |
| mlp | 67.9 ± 2.5 | 3 |
| lstm | 52.7 ± 3.3 | 3 |
| transformer | 29.0 ± 5.7 | 3 |

## c4 / noise=0.05 / ratio=0.17

| model | VPT (mean ± std) | n |
|---|---|---|
| ESN | 42.8 ± 1.3 | 3 |
| mlp | 27.6 ± 0.5 | 2 |
| lstm | 37.5 ± 0.4 | 2 |
| transformer | 20.9 ± 3.5 | 2 |

## c4 / noise=0.10 / ratio=0.17

| model | VPT (mean ± std) | n |
|---|---|---|
| ESN | 32.8 ± 2.0 | 3 |
| mlp | 27.6 ± 0.0 | 2 |
| lstm | 35.2 ± 0.1 | 2 |
| transformer | 24.6 ± 0.5 | 2 |
