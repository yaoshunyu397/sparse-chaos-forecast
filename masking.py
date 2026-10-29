"""
masking.py -- 观测稀疏化（observation masking）

核心理念（这个文件是全项目最容易做错的地方，请读完再改）：
    任务始终是"预测完整状态"，
    只是"输入被遮挡"（部分可观测）。

    所以：target 永远是完整的 x_{t+1..t+H}，
          input 才是被 mask 过的 x_{t-L+1..t}。

四种遮挡方式（对应论文里"传感器有限 / 区域缺失 / 位置固定"等不同物理情境）：
    uniform   : 等间隔保留，保留约 k/dim 个维度（传感器数量有限）
    block     : 连续一段维度被遮掉（局部区域无观测）
    random    : 随机选固定的一组维度不观测（传感器位置固定）
    none      : 不遮挡（对照组）

两条铁律：
    1. 掩码在 train 和 test 上必须完全一致（否则模型会学到"测试时缺失维度的模式"，
       结论不可信）。本模块通过 --seed 与"固定掩码"来保证这一点。
    2. 被遮挡的位置填 0.0（标准化之后 0 = 训练集均值），
       不要用 NaN —— 大多数时序模型对 NaN 不友好。
       如需把"这里没有观测"作为显式信号，把 zero_fill 改成 -1.0 或其他哨兵值。

用法：
    # 生成并保存掩码（可复现）
    python -m masking --system lorenz63 --mode uniform --ratio 0.5 --out masks/l63_uniform_05.npy

    # 自检：打印各 mode 的实际保留比例
    python -m masking --selfcheck

    # 在代码里调用
    from masking import build_mask, apply_mask, load_data
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np

MASK_MODES = ("none", "uniform", "block", "random", "temporal")

# 说明（为什么需要 temporal）：
#   --- 维度掩码的致命限制 ---
#   Lorenz-63 只有 D=3。想得到 ratio=1/6 需要保留 0.5 个维度，不可能；
#   ratio=1/3 与 1/6 都会塌缩成"保留 1 维"，产生完全相同的掩码，
#   实验于是变成自我重复（pilot 阶段就是这样发现问题的）。
#   --- 解决 ---
#   temporal 模式不遮维度，而是遮"时间点"：
#     每 k 个时间步只保留 1 个观测（k = round(1/ratio)）。
#   这样任意比例都能精确实现，且物理含义同样自然（传感器采样频率降低）。
#   推荐：D 小（如 3）的系统用 temporal，D 大（如 40）的系统用空间掩码。


# --------------------------------------------------------------------------
# 掩码构造
# --------------------------------------------------------------------------
def uniform_mask(dim: int, ratio: float) -> np.ndarray:
    """
    等间隔保留。ratio = 保留的维度比例（0.5 -> 约一半维度被观测）。

    例：dim=40, ratio=0.5 -> 每隔 2 个取 1 个。
    返回 (dim,) 的 float32，1=观测，0=不观测。
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio 必须在 (0, 1] 之间，收到 {ratio}")
    n_keep = max(1, int(round(dim * ratio)))
    stride = max(1, int(round(dim / n_keep)))
    mask = np.zeros(dim, dtype=np.float32)
    mask[::stride] = 1.0
    return mask


def block_mask(dim: int, ratio: float) -> np.ndarray:
    """
    连续块遮挡：保留下来的维度集中在一段连续区间里，
    其余整段缺失（模拟"某片区域没有传感器"）。

    ratio 仍表示保留比例，但从开头连续保留。
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio 必须在 (0, 1] 之间，收到 {ratio}")
    n_keep = max(1, int(round(dim * ratio)))
    mask = np.zeros(dim, dtype=np.float32)
    mask[:n_keep] = 1.0
    return mask


def random_mask(dim: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """
    随机选固定的一组维度观测。seed 固定 -> 掩码固定，
    保证 train / test 用同一组传感器。
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio 必须在 (0, 1] 之间，收到 {ratio}")
    n_keep = max(1, int(round(dim * ratio)))
    idx = rng.choice(dim, size=n_keep, replace=False)
    mask = np.zeros(dim, dtype=np.float32)
    mask[idx] = 1.0
    return mask


def temporal_mask(input_len: int, ratio: float) -> np.ndarray:
    """
    时间维掩码：不遮挡任何维度，而是只保留部分时间点的观测。

    ratio = 保留的时间点比例。每 k = round(1/ratio) 个时间步保留 1 个。
    返回 (input_len,)，1 = 该时刻有观测，0 = 该时刻无观测。

    为什么需要它：D 小的系统（Lorenz-63 只有 3 维）用维度掩码时，
    不同 ratio 会塌缩成同一个掩码，实验变成自我重复。时间掩码可以在
    任意 D 上实现任意稀疏度。

    物理含义：传感器采样频率降低 / 通信丢包 / 观测间隔变大。
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio 必须在 (0, 1] 之间，收到 {ratio}")
    if ratio >= 1.0:
        return np.ones(input_len, dtype=np.float32)
    k = max(2, int(round(1.0 / ratio)))
    m = np.zeros(input_len, dtype=np.float32)
    m[::k] = 1.0
    return m


def build_mask(
    dim: int,
    mode: str = "uniform",
    ratio: float = 1.0,
    seed: int = 0,
    input_len: int | None = None,
) -> np.ndarray:
    """
    构造观测掩码。1 = 观测到，0 = 无观测。

    mode:
        "none"     全部观测                      -> (dim,)
        "uniform"  等间隔保留维度                -> (dim,)
        "block"    连续一段维度保留              -> (dim,)
        "random"   随机固定一组维度保留          -> (dim,)
        "temporal" 全部维度保留，但只保留部分时刻 -> (input_len,)

    temporal 模式必须提供 input_len。
    """
    if mode not in MASK_MODES:
        raise ValueError(f"mode 必须是 {MASK_MODES} 之一，收到 {mode!r}")

    if mode == "none" or ratio >= 1.0:
        return np.ones(dim, dtype=np.float32)

    if mode == "temporal":
        if input_len is None:
            raise ValueError("temporal 模式需要 input_len 参数")
        return temporal_mask(input_len, ratio)
    if mode == "uniform":
        return uniform_mask(dim, ratio)
    if mode == "block":
        return block_mask(dim, ratio)
    if mode == "random":
        return random_mask(dim, ratio, np.random.default_rng(seed))
    raise ValueError(mode)


# --------------------------------------------------------------------------
# 应用掩码
# --------------------------------------------------------------------------
def apply_mask(
    x: np.ndarray,
    mask: np.ndarray,
    zero_fill: float = 0.0,
    noise_std: float = 0.0,
    noise_seed: int | None = None,
) -> np.ndarray:
    """
    把观测掩码作用到输入张量上，并可选地加观测噪声。

    参数
    ----
    x        : (..., dim) 的输入（已标准化）
    mask     : 两种形状都支持
                 (dim,)     -> 维度掩码，广播到所有时间步（传感器位置固定）
                 (L, dim)   -> 时间-维度联合掩码（temporal 模式或逐时刻变化）
    zero_fill: 被遮挡位置填充值（标准化后 0.0 = 训练集均值）
    noise_std: 观测噪声的标准差（相对标准化后的尺度）。0 表示不加噪声。
    noise_seed: 加噪随机种子；None 表示每次不同（训练），固定值用于可复现评估。

    返回
    ----
    与 x 同形状的数组，被遮挡位置被 zero_fill 替换。

    注意
    ----
    * 噪声只加在"被观测的位置"上，这样"噪声"和"缺失"两种不确定性
      才不会被混淆。
    * temporal 掩码的语义是"这一时刻没有观测"，因此整行所有维度都置零，
      而不是保留某些维度 —— 这与"维度掩码"的物理含义不同，不能混用。
    """
    x = np.asarray(x, dtype=np.float32)
    m = mask.astype(np.float32)

    if m.ndim == 1:
        # 维度掩码：要求最后一维匹配，其余前置维度广播
        if x.shape[-1] != m.shape[0]:
            raise ValueError(
                f"输入最后一维 {x.shape[-1]} 与掩码维度 {m.shape[0]} 不一致"
            )
        m_b = m.reshape((1,) * (x.ndim - 1) + (-1,))
    elif m.ndim == 2:
        # 时间掩码：要求形状 (L, dim)，与输入末尾两维匹配
        if x.shape[-2:] != m.shape:
            raise ValueError(
                f"输入末尾两维 {x.shape[-2:]} 与时间掩码 {m.shape} 不一致"
            )
        m_b = m
        # temporal 语义：该时刻无观测则整行置零。若调用方传入的已经是
        # (L, dim) 的联合掩码，则按元素相乘即可，无需再做整行处理。
        m_b = m_b.reshape((1,) * (x.ndim - 2) + m.shape)
    else:
        raise ValueError(f"mask 必须是 1 维或 2 维，收到 {m.ndim} 维")

    out = x * m_b + (1.0 - m_b) * np.float32(zero_fill)

    if noise_std and noise_std > 0.0:
        rng = np.random.default_rng(noise_seed)
        noise = rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        out = out + noise * m_b  # 只加在观测到的位置

    return out


# --------------------------------------------------------------------------
# 数据装载 + 标准化（统一入口，训练和评估都走这里）
# --------------------------------------------------------------------------
def load_data(system: str, data_dir: str | None = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """返回 (train, test, meta)，形状均为 (N, T, dim)。"""
    data_dir = data_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train = np.load(os.path.join(data_dir, f"{system}_train.npy"))
    test = np.load(os.path.join(data_dir, f"{system}_test.npy"))
    with open(os.path.join(data_dir, f"{system}_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return train, test, meta


def standardize(
    train: np.ndarray, test: np.ndarray, meta: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    用训练集统计量做逐维标准化。返回 (train_z, test_z, mean, std)。
    只允许用训练集统计量 —— 这是防止信息泄露的基本纪律。
    """
    mean = np.asarray(meta["train_mean"], dtype=np.float32)
    std = np.asarray(meta["train_std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)  # 常数维度保护
    return (train - mean) / std, (test - mean) / std, mean, std


def observations_to_windows(
    x: np.ndarray,
    mask: np.ndarray,
    input_len: int,
    horizon: int = 1,
    stride: int = 1,
    zero_fill: float = 0.0,
    noise_std: float = 0.0,
    noise_seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    把连续轨迹切成 (inputs, targets) 窗口。

    inputs  : (N, input_len, dim)  —— 已经 apply_mask
    targets : (N, horizon,   dim)  —— 永远是完整状态，不做 mask

    x 是单条轨迹 (T, dim)。多轨迹请在外面循环或先 reshape。

    mask 支持两种：
        (dim,)          -> 维度掩码，所有窗口共用（传感器位置固定）
        (input_len,)    -> 时间掩码，转换为 (input_len, dim) 后共用
                           （每 k 步只观测一次；所有窗口的观测时刻一致，
                            这保证 train/test 与各窗口之间的一致性）

    noise_std:
        观测噪声强度（标准化尺度）。只加在被观测到的位置上，
        缺失位置仍是 zero_fill —— 这样"噪声"与"缺失"两种不确定性不会混淆。
        noise_seed=None 时每个窗口噪声不同（训练用）；
        给固定值则所有窗口噪声一致（评估可复现）。
    """
    T, dim = x.shape
    if T - input_len - horizon + 1 <= 0:
        raise ValueError("轨迹太短，无法切出窗口")

    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 2:
        raise ValueError("mask 只接受 (dim,) 或 (input_len,)")
    if m.ndim == 1 and m.shape[0] == input_len and dim != input_len:
        # 时间掩码 -> 转成 (input_len, dim) 的联合掩码
        m = np.repeat(m[:, None], dim, axis=1)
    elif m.ndim == 1 and m.shape[0] != dim:
        raise ValueError(
            f"掩码长度 {m.shape[0]} 既不等于维度 {dim}，也不等于输入长度 {input_len}"
        )

    starts = range(0, T - input_len - horizon + 1, stride)
    n = len(starts)
    inputs = np.empty((n, input_len, dim), dtype=np.float32)
    targets = np.empty((n, horizon, dim), dtype=np.float32)
    for i, s in enumerate(starts):
        inputs[i] = apply_mask(
            x[s : s + input_len],
            m,
            zero_fill=zero_fill,
            noise_std=noise_std,
            noise_seed=None if noise_seed is None else noise_seed + s,
        )
        targets[i] = x[s + input_len : s + input_len + horizon]
    return inputs, targets


# --------------------------------------------------------------------------
# 自检 / CLI
# --------------------------------------------------------------------------
def _selfcheck() -> None:
    print("mask 自检 —— 各 mode 的实际保留比例")
    print(f"{'mode':<10}{'ratio':<8}{'dim':<6}{'kept':<6}{'mean':<8}")
    for dim in (3, 40):
        for mode in ("none", "uniform", "block", "random"):
            for ratio in (1.0, 0.5, 1 / 3, 1 / 6):
                m = build_mask(dim, mode, ratio, seed=42)
                print(f"{mode:<10}{ratio:<8.3f}{dim:<6}{int(m.sum()):<6}{m.mean():<8.2f}")
    print(f"\n{'temporal':<10}{'ratio':<8}{'L':<6}{'kept':<6}{'mean':<8}")
    for L in (50,):
        for ratio in (1.0, 0.5, 1 / 3, 1 / 6):
            m = build_mask(3, "temporal", ratio, input_len=L)
            print(f"{'temporal':<10}{ratio:<8.3f}{L:<6}{int(m.sum()):<6}{m.mean():<8.2f}")

    print("\n维度掩码 vs 时间掩码 —— 关键区别")
    print("  uniform r=0.33 ->", build_mask(3, "uniform", 1 / 3, seed=42),
          "（只看见 1 个维度）")
    print("  uniform r=0.17 ->", build_mask(3, "uniform", 1 / 6, seed=42),
          "（与上面完全相同 —— 这就是必须用 temporal 的原因）")
    print("  temporal r=0.33 ->", build_mask(3, "temporal", 1 / 3, input_len=9),
          "（维度全看见，只保留部分时刻）")

    print("\napply_mask 自检 —— 被遮挡处是否确实填 0")
    x = np.ones((2, 50, 5), dtype=np.float32)
    m = build_mask(5, "uniform", 0.4, seed=1)
    y = apply_mask(x, m)
    assert np.allclose(y[..., m == 0], 0.0), "遮挡维度未被填 0"
    assert np.allclose(y[..., m == 1], 1.0), "观测维度被误改"
    print(" 通过：维度掩码 —— 观测维保持原值，遮挡维填 0")

    # 时间掩码：无观测的整行都应为 0
    x2 = np.ones((2, 50, 3), dtype=np.float32)
    mt = build_mask(3, "temporal", 0.5, input_len=50)
    y2 = apply_mask(x2, mt)
    assert np.allclose(y2[:, mt == 0, :], 0.0), "时间掩码未生效"
    assert np.allclose(y2[:, mt == 1, :], 1.0), "观测时刻被误改"
    print(" 通过：时间掩码 —— 无观测时刻整行置 0，有观测时刻保持原值")

    print("\n掩码一致性自检 —— 同一 seed 两次构造是否完全相同")
    a = build_mask(40, "random", 1 / 3, seed=7)
    b = build_mask(40, "random", 1 / 3, seed=7)
    assert np.array_equal(a, b), "random 掩码不可复现"
    print(" 通过：同 seed 掩码可复现（保证 train/test 一致）")


def main() -> None:
    ap = argparse.ArgumentParser(description="观测稀疏化掩码")
    ap.add_argument("--system", default="lorenz63")
    ap.add_argument("--mode", default="uniform", choices=list(MASK_MODES))
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="保存 .npy 的路径")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck()
        return

    train, test, meta = load_data(args.system)
    dim = meta["dim"]
    mask = build_mask(dim, args.mode, args.ratio, seed=args.seed)

    print(f"system={args.system} dim={dim} mode={args.mode} ratio={args.ratio}")
    print(f"保留维度数 = {int(mask.sum())} / {dim}")
    print(f"被遮挡维度索引 = {np.where(mask == 0)[0].tolist()}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.save(args.out, mask)
        print(f"已保存 -> {args.out}")

    # 顺手验证窗口切分不报错
    inputs, targets = observations_to_windows(
        train[0], mask, input_len=50, horizon=1, stride=10
    )
    print(f"窗口示例: inputs={inputs.shape}, targets={targets.shape}")


if __name__ == "__main__":
    main()
