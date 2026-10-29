# 含噪观测实验结果

只改变观测噪声水平，其余设置与多种子实验完全一致。
噪声只加在被观测到的位置；缺失位置仍填 0。

**评估协议已统一**：基线与含噪组均为 2 个评估种子 × 40 条轨迹 = 80 条，
因此两者的 std 可以直接比较。
基线为 3 个训练种子（noise=0，来自 `evals_seeds_re/`），
含噪组为 2 个训练种子 × 2 个噪声水平。


## c1: lorenz63 / none / ratio=1.00

| noise | model | VPT (mean ± std) | n |
|---|---|---|---|
| 0.00 | mlp | 99.9 ± 2.9 | 3 |
| 0.00 | lstm | 67.5 ± 20.1 | 3 |
| 0.00 | transformer | 49.2 ± 1.3 | 3 |
| 0.05 | mlp | 66.5 ± 1.3 | 2 |
| 0.05 | lstm | 64.9 ± 2.0 | 2 |
| 0.05 | transformer | 44.2 ± 7.3 | 2 |
| 0.10 | mlp | 62.1 ± 1.0 | 2 |
| 0.10 | lstm | 56.9 ± 0.1 | 2 |
| 0.10 | transformer | 54.1 ± 1.3 | 2 |

## c2: lorenz63 / temporal / ratio=0.50

| noise | model | VPT (mean ± std) | n |
|---|---|---|---|
| 0.00 | mlp | 83.0 ± 0.9 | 3 |
| 0.00 | lstm | 61.5 ± 11.8 | 3 |
| 0.00 | transformer | 32.0 ± 4.7 | 3 |
| 0.05 | mlp | 44.2 ± 0.9 | 2 |
| 0.05 | lstm | 43.8 ± 1.1 | 2 |
| 0.05 | transformer | 32.8 ± 3.6 | 2 |
| 0.10 | mlp | 39.7 ± 0.0 | 2 |
| 0.10 | lstm | 39.9 ± 0.0 | 2 |
| 0.10 | transformer | 34.5 ± 0.7 | 2 |

## c3: lorenz63 / temporal / ratio=0.33

| noise | model | VPT (mean ± std) | n |
|---|---|---|---|
| 0.00 | mlp | 78.5 ± 4.5 | 3 |
| 0.00 | lstm | 74.6 ± 10.5 | 3 |
| 0.00 | transformer | 32.8 ± 5.3 | 3 |
| 0.05 | mlp | 51.1 ± 1.0 | 2 |
| 0.05 | lstm | 56.3 ± 1.4 | 2 |
| 0.05 | transformer | 32.1 ± 2.7 | 2 |
| 0.10 | mlp | 43.1 ± 1.0 | 2 |
| 0.10 | lstm | 44.5 ± 3.0 | 2 |
| 0.10 | transformer | 38.2 ± 0.1 | 2 |

## c4: lorenz63 / temporal / ratio=0.17

| noise | model | VPT (mean ± std) | n |
|---|---|---|---|
| 0.00 | mlp | 67.9 ± 2.5 | 3 |
| 0.00 | lstm | 52.7 ± 3.3 | 3 |
| 0.00 | transformer | 29.0 ± 5.7 | 3 |
| 0.05 | mlp | 27.6 ± 0.5 | 2 |
| 0.05 | lstm | 37.5 ± 0.4 | 2 |
| 0.05 | transformer | 20.9 ± 3.5 | 2 |
| 0.10 | mlp | 27.6 ± 0.0 | 2 |
| 0.10 | lstm | 35.2 ± 0.1 | 2 |
| 0.10 | transformer | 24.6 ± 0.5 | 2 |

## 总结

- 8 个（配置 × 噪声）判断中，**7 个 MLP 显著领先**，0 个反转
- c1 @ noise=0.05: MLP−Transf = +22.2 (noise 7.3, ratio 3.1) -> 稳健（MLP 领先）
- c1 @ noise=0.10: MLP−Transf = +8.0 (noise 1.3, ratio 6.1) -> 稳健（MLP 领先）
- c2 @ noise=0.05: MLP−Transf = +11.4 (noise 3.6, ratio 3.2) -> 稳健（MLP 领先）
- c2 @ noise=0.10: MLP−Transf = +5.2 (noise 0.7, ratio 7.2) -> 稳健（MLP 领先）
- c3 @ noise=0.05: MLP−Transf = +19.0 (noise 2.7, ratio 7.1) -> 稳健（MLP 领先）
- c3 @ noise=0.10: MLP−Transf = +5.0 (noise 1.0, ratio 5.0) -> 稳健（MLP 领先）
- c4 @ noise=0.05: MLP−Transf = +6.7 (noise 3.5, ratio 1.9) -> 优势不显著
- c4 @ noise=0.10: MLP−Transf = +3.0 (noise 0.5, ratio 6.2) -> 稳健（MLP 领先）

## 噪声对各模型绝对水平的影响

| config | model | noise=0 | noise=0.05 | 保留比例 |
|---|---|---|---|---|
| c1 | mlp | 99.9 | 66.5 | 67% |
| c1 | lstm | 67.5 | 64.9 | 96% |
| c1 | transformer | 49.2 | 44.2 | 90% |
| c2 | mlp | 83.0 | 44.2 | 53% |
| c2 | lstm | 61.5 | 43.8 | 71% |
| c2 | transformer | 32.0 | 32.8 | 102% |
| c3 | mlp | 78.5 | 51.1 | 65% |
| c3 | lstm | 74.6 | 56.3 | 76% |
| c3 | transformer | 32.8 | 32.1 | 98% |
| c4 | mlp | 67.9 | 27.6 | 41% |
| c4 | lstm | 52.7 | 37.5 | 71% |
| c4 | transformer | 29.0 | 20.9 | 72% |

## 跨模型反转检测（同一配置、加噪后谁最好）

- c1 @ noise=0.05: MLP 仍最优 (VPT=66.5)
- c1 @ noise=0.10: MLP 仍最优 (VPT=62.1)
- c2 @ noise=0.05: MLP 仍最优 (VPT=44.2)
- **c2 @ noise=0.10: lstm 反超 MLP** (VPT=39.9)
- **c3 @ noise=0.05: lstm 反超 MLP** (VPT=56.3)
- **c3 @ noise=0.10: lstm 反超 MLP** (VPT=44.5)
- **c4 @ noise=0.05: lstm 反超 MLP** (VPT=37.5)
- **c4 @ noise=0.10: lstm 反超 MLP** (VPT=35.2)
