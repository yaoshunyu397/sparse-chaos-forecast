"""
generate.py -- 生成混沌系统真值轨迹（RK4 积分，向量化实现）

为什么是这两个系统：
  - lorenz63 : 3 维，最经典的混沌吸引子。可预测性低、对初值极敏感。
  - lorenz96 : 40 维（F=8），有空间结构。Transformer 的作用只有在高维、
               有空间结构的系统上才会显现；只做 lorenz63 几乎必然得到
               "两者没差别"的结论，demo 就废了。

性能说明（重要）：
  早期版本对每条轨迹、每个采样点做 Python 循环，在 3 维系统上极慢
  （3 维数组的 numpy 开销远大于计算本身，10 分钟跑不完）。
  本版本把整个 batch 的所有轨迹并行积分：状态矩阵形状 (batch, dim)，
  RK4 的四个 k 都是整批矩阵运算。200 条轨迹约 1 秒。

用法：
    python data/generate.py --system all --n-traj 300        # 快速自测
    python data/generate.py --system all                     # 正式数据集
    python data/generate.py --system lorenz63 --n-traj 1000 --t-len 400

输出（data/ 下）：
    <system>_train.npy    (N_train, T_train, dim)
    <system>_test.npy     (N_test,  T_test,  dim)
    <system>_meta.json    维度 / 步长 / 参数 / 统计量

设计要点：
  1. 所有轨迹先丢弃 N_TRANSIENT 步瞬态，确保落在吸引子上；
  2. 采样间隔 ds = dt * sub_step（默认 0.05），避免数据在时间上过密；
  3. 固定随机种子，可复现；
  4. 一旦发散立即报警，而不是静默写入 inf/nan。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np

# --------------------------------------------------------------------------
# 默认超参：记住这三条，它们决定了实验能否算出差异
# --------------------------------------------------------------------------
DT = 0.01          # 积分步长
SUB_STEP = 5       # 每 5 个积分步存一个点 -> ds = 0.05
DS = DT * SUB_STEP
N_TRANSIENT = 1000  # 丢弃的瞬态步数（单位：采样点）

LORENZ63_PARAMS = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}
LORENZ96_PARAMS = {"F": 8.0}   # F=8 是经典混沌区

# 正式数据集的默认规模
DEFAULT_TRAIN = {"lorenz63": (1000, 400), "lorenz96": (300, 500)}
DEFAULT_TEST = {"lorenz63": (200, 800), "lorenz96": (100, 1000)}


# --------------------------------------------------------------------------
# 向量场（支持 (batch, dim) 批量输入）
# --------------------------------------------------------------------------
def rhs(x: np.ndarray, system: str, params: Dict[str, float]) -> np.ndarray:
    """x: (..., dim) -> (..., dim)"""
    if system == "lorenz63":
        sigma = params["sigma"]
        rho = params["rho"]
        beta = params["beta"]
        dx = sigma * (x[..., 1] - x[..., 0])
        dy = x[..., 0] * (rho - x[..., 2]) - x[..., 1]
        dz = x[..., 0] * x[..., 1] - beta * x[..., 2]
        return np.stack([dx, dy, dz], axis=-1)
    if system == "lorenz96":
        F = params["F"]
        return (
            (np.roll(x, -1, axis=-1) - np.roll(x, 2, axis=-1)) * np.roll(x, 1, axis=-1)
            - x
            + F
        )
    raise ValueError(f"unknown system: {system!r}")


def rk4_step(x: np.ndarray, dt: float, system: str, params: Dict[str, float]) -> np.ndarray:
    """整批 RK4 单步。x: (batch, dim)"""
    k1 = rhs(x, system, params)
    k2 = rhs(x + 0.5 * dt * k1, system, params)
    k3 = rhs(x + 0.5 * dt * k2, system, params)
    k4 = rhs(x + dt * k3, system, params)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# --------------------------------------------------------------------------
# 批量轨迹生成
# --------------------------------------------------------------------------
def _check_finite(x: np.ndarray, system: str, step: int) -> None:
    if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e6:
        raise FloatingPointError(
            f"轨迹发散 (system={system}, step={step})。"
            f"请检查参数是否落在混沌区，或初值范围是否过大。"
        )


def init_batch(rng: np.random.Generator, system: str, dim: int, n: int) -> np.ndarray:
    """在吸引子附近取初值。范围取自文献常用区间，避免一开始就发散。"""
    if system == "lorenz63":
        low = np.array([-20.0, -30.0, 0.0])
        high = np.array([20.0, 30.0, 50.0])
        return rng.uniform(low=low, high=high, size=(n, dim))
    if system == "lorenz96":
        # 混沌解大致在 O(F) 量级附近扰动
        return 8.0 * rng.standard_normal((n, dim)) + 8.0
    raise ValueError(system)


def generate_batch(
    n_traj: int,
    t_len: int,
    system: str,
    params: Dict[str, float],
    dim: int,
    rng: np.random.Generator,
    dt: float = DT,
    sub_step: int = SUB_STEP,
    n_transient: int = N_TRANSIENT,
    verbose: bool = True,
) -> np.ndarray:
    """
    整批生成 n_traj 条长度为 t_len 的轨迹。返回 (n_traj, t_len, dim)。

    实现：状态矩阵 (n_traj, dim) 同时积分，
    先丢瞬态，再每 sub_step 个积分步存一帧。
    """
    x = init_batch(rng, system, dim, n_traj)

    # ---- 丢弃瞬态（整批同时推进）----
    for _ in range(n_transient * sub_step):
        x = rk4_step(x, dt, system, params)
    _check_finite(x, system, -1)

    # ---- 正式采样 ----
    out = np.empty((n_traj, t_len, dim), dtype=np.float32)
    out[:, 0] = x
    for i in range(1, t_len):
        for _ in range(sub_step):
            x = rk4_step(x, dt, system, params)
        if i % 200 == 0 or i == t_len - 1:
            _check_finite(x, system, i)
        out[:, i] = x

    if verbose:
        print(
            f"    完成: {out.shape}  值域 [{out.min():.2f}, {out.max():.2f}]  "
            f"std={out.reshape(-1, dim).std(axis=0).mean():.3f}"
        )
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def process_system(
    system: str,
    params: Dict[str, float],
    dim: int,
    n_train: int,
    t_train: int,
    n_test: int,
    t_test: int,
    outdir: str,
    seed: int,
) -> Dict:
    t0 = time.time()
    rng_train = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 1)

    print(f"[{system}] 训练集 {n_train} x {t_train} x {dim}")
    train = generate_batch(n_train, t_train, system, params, dim, rng_train)

    print(f"[{system}] 测试集 {n_test} x {t_test} x {dim}")
    test = generate_batch(n_test, t_test, system, params, dim, rng_test)

    np.save(os.path.join(outdir, f"{system}_train.npy"), train)
    np.save(os.path.join(outdir, f"{system}_test.npy"), test)

    # ---- 标准化统计量：只允许来自训练集 ----
    warm = train[:, min(100, t_train // 4):, :].reshape(-1, dim)
    meta = {
        "system": system,
        "dim": dim,
        "dt": DT,
        "sub_step": SUB_STEP,
        "ds": DS,
        "params": params,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "seed": seed,
        "elapsed_sec": round(time.time() - t0, 2),
        "train_mean": [float(v) for v in warm.mean(axis=0)],
        "train_std": [float(v) for v in warm.std(axis=0)],
        "global_mean": float(warm.mean()),
        "global_std": float(warm.std()),
    }
    with open(os.path.join(outdir, f"{system}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[{system}] 保存完成，耗时 {meta['elapsed_sec']}s")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="生成混沌系统真值轨迹（向量化 RK4）")
    ap.add_argument("--system", default="all", choices=["lorenz63", "lorenz96", "all"])
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--n-traj", type=int, default=None,
                    help="覆盖训练轨迹数（自测用，如 50）")
    ap.add_argument("--t-len", type=int, default=None,
                    help="覆盖训练轨迹长度（自测用，如 120）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    systems = ["lorenz63", "lorenz96"] if args.system == "all" else [args.system]

    for sys_name in systems:
        params, dim = (
            (LORENZ63_PARAMS, 3) if sys_name == "lorenz63" else (LORENZ96_PARAMS, 40)
        )
        n_tr, t_tr = DEFAULT_TRAIN[sys_name]
        n_te, t_te = DEFAULT_TEST[sys_name]
        if args.n_traj is not None:
            n_tr = args.n_traj
            n_te = max(10, args.n_traj // 5)
        if args.t_len is not None:
            t_tr = args.t_len
            t_te = args.t_len

        process_system(
            sys_name, params, dim, n_tr, t_tr, n_te, t_te, args.outdir, args.seed
        )

    print("\n全部完成。下一步：")
    print("  python -m masking --selfcheck")
    print("  python -m masking --system lorenz63 --mode uniform --ratio 0.5")


if __name__ == "__main__":
    main()
