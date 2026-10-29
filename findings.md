# 网格结果速览（自动生成，需人工改写）

> 本文件由 `summarize.py` 自动生成，只做数字罗列。
> 写结论时必须自己判断，不要直接粘贴这些读数。


## lorenz63

- 单步 MSE 最优：**lstm** (mode=none, ratio=1.00) = 0.000018
- 长期 VPT 最优：**mlp** (mode=none, ratio=1.00) = 95.9 步

### 观测稀疏化下 VPT 的变化（temporal）

| 观测比例 | MLP | LSTM | Transf |
|---|---|---|---|
| 0.50 | 80.2 | 69.4 | 30.9 |
| 0.33 | 77.4 | 59.4 | 33.8 |
| 0.17 | 68.9 | 49.8 | 33.8 |

### 分裂检测（短时 vs 长期）



## lorenz96

- 单步 MSE 最优：**transformer** (mode=none, ratio=1.00) = 0.002603
- 长期 VPT 最优：**transformer** (mode=temporal, ratio=0.50) = 13.6 步

### 观测稀疏化下 VPT 的变化（uniform）

| 观测比例 | MLP | LSTM | Transf |
|---|---|---|---|
| 0.50 | 0.8 | 2.3 | 3.1 |
| 0.33 | 0.8 | 1.6 | 2.3 |
| 0.17 | 0.0 | 0.0 | 0.1 |

### 分裂检测（短时 vs 长期）

- ratio=0.17: **lstm** 短时最准 (RMSE@10=0.8466)，但长期最稳的是 **transformer** (VPT=0.1)
