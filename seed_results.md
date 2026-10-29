# 多种子重复结果

同一配置训练 3 个随机种子（掩码固定 seed=42，保证观测集合相同）。

评估时每个模型用 3 个评估种子各抽 40 条测试轨迹（共 120 条），
因此下表的 std 同时包含训练随机性与评估抽样随机性。


## c1: lorenz63 / none / ratio=1.00

| model | VPT (mean ± std) | RMSE@10 | test_mse | n |
|---|---|---|---|---|
| mlp | 99.9 ± 2.9 | 0.0051 | 0.00002 | 3 |
| lstm | 67.5 ± 20.1 | 0.0115 | 0.00002 | 3 |
| transformer | 49.2 ± 1.3 | 0.0191 | 0.00004 | 3 |

## c2: lorenz63 / temporal / ratio=0.50

| model | VPT (mean ± std) | RMSE@10 | test_mse | n |
|---|---|---|---|---|
| mlp | 83.0 ± 0.9 | 0.0060 | 0.00003 | 3 |
| lstm | 61.5 ± 11.8 | 0.0130 | 0.00006 | 3 |
| transformer | 32.0 ± 4.7 | 0.0491 | 0.00052 | 3 |

## c3: lorenz63 / temporal / ratio=0.33

| model | VPT (mean ± std) | RMSE@10 | test_mse | n |
|---|---|---|---|---|
| mlp | 78.5 ± 4.5 | 0.0076 | 0.00003 | 3 |
| lstm | 74.6 ± 10.5 | 0.0093 | 0.00003 | 3 |
| transformer | 32.8 ± 5.3 | 0.0593 | 0.00071 | 3 |

## c4: lorenz63 / temporal / ratio=0.17

| model | VPT (mean ± std) | RMSE@10 | test_mse | n |
|---|---|---|---|---|
| mlp | 67.9 ± 2.5 | 0.0091 | 0.00004 | 3 |
| lstm | 52.7 ± 3.3 | 0.0135 | 0.00004 | 3 |
| transformer | 29.0 ± 5.7 | 0.0690 | 0.00094 | 3 |

## 稳健性判断

- c1: MLP − Transformer = +50.7 (noise 2.9, ratio 17.3) → **稳健**
- c2: MLP − Transformer = +50.9 (noise 4.7, ratio 10.8) → **稳健**
- c3: MLP − Transformer = +45.7 (noise 5.3, ratio 8.5) → **稳健**
- c4: MLP − Transformer = +38.8 (noise 5.7, ratio 6.8) → **稳健**
