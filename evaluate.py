"""
evaluate.py -- 自回归 rollout 评估 + VPT + 结果图

为什么需要 rollout：
    单步 test_mse 只说明"一步预测准不准"，不回答真正的问题——
    **系统动力学与模型有效性的关系**。要回答它，必须看误差随时间如何增长。

做法：
    训练好的模型每次预测 1 步，把预测值反馈回输入窗口（同时继续施加同一个
    观测掩码），迭代 roll_T 步，与真值比较，得到误差曲线。

    这模拟的正是"只有稀疏观测，要靠模型自己往前推"的实际情形。

两个关键指标：
    VPT (Valid Prediction Time) : RMSE 首次超过阈值的时刻。回答"能预测多久"。
    Error growth rate           : 前期误差对数增长的斜率。回答"长期稳不稳"。

⚠️ 重要解读原则
    "短时 RMSE 低 + VPT 短" 是一个有价值的结果，不是失败：
    它说明模型学到的是相关性而非动力学。这正是我们要找的那种发现。

用法：
    # 评估单个模型
    python evaluate.py --run runs/lorenz63_transformer_uniform_r0.50.pt

    # 评估全部并出图
    python evaluate.py --all --roll-t 300

    # 生成最终对比图（邮件/README 用）
    python evaluate.py --report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import torch

from masking import load_data, standardize
from train import DEVICE, build_model

# 各系统的 VPT 阈值（标准化尺度下的 RMSE）。
# 取经验值：Lorenz-63 的吸引子尺度约 8，阈值 0.5 ≈ 6% 误差。
VPT_THRESHOLD = {"lorenz63": 0.5, "lorenz96": 0.5}
DEFAULT_ROLL_T = 300


# --------------------------------------------------------------------------
# 核心：自回归 rollout
# --------------------------------------------------------------------------
@torch.no_grad()
def prepare_input_mask(
    history: np.ndarray,
    mask: np.ndarray,
    noise_std: float = 0.0,
    noise_seed: int | None = None,
) -> np.ndarray:
    """
    把掩码作用到当前输入窗口上。支持两种掩码：

        (D,)   维度掩码 —— 传感器位置固定，所有时刻看同样的维度
        (L,)   时间掩码 —— 维度全可见，但只保留部分时刻（其余整行置 0）

    noise_std > 0 时给"被观测到的位置"加高斯噪声，缺失位置仍是 0。
    这样评估条件与训练条件一致，且噪声不会被误当成观测。

    返回与 history 同形状的已遮挡（可能带噪）输入。
    """
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 1 and m.shape[0] == history.shape[0]:
        # 时间掩码 -> 整行置零（"这一时刻没有观测"）
        m_b = m[:, None]
    else:
        # 维度掩码
        m_b = m.reshape(1, -1)

    out = history * m_b

    if noise_std and noise_std > 0.0:
        rng = np.random.default_rng(noise_seed)
        out = out + rng.normal(0.0, noise_std, size=history.shape).astype(np.float32) * m_b

    return out.astype(np.float32)


@torch.no_grad()
def rollout(
    model: torch.nn.Module,
    traj: np.ndarray,
    mask: np.ndarray,
    input_len: int,
    horizon: int,
    roll_t: int,
    n_traj: int = 50,
    seed: int = 0,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 traj (N, T, D) 上做自回归 rollout。

    返回:
        errors : (n_traj, roll_t)   每个时刻的 RMSE（标准化尺度）
        preds  : (n_traj, roll_t, D) 预测轨迹（用于定性画图）

    注意：模型被训练成一次预测 horizon 步；这里统一取输出的最后一步作为
    下一个状态，并把历史窗口截断保持 input_len。
    """
    rng = np.random.default_rng(seed)
    N, T, D = traj.shape
    n_traj = min(n_traj, N)
    idxs = rng.choice(N, size=n_traj, replace=False)

    model.eval()
    errors = np.full((n_traj, roll_t), np.nan, dtype=np.float32)
    preds = np.full((n_traj, roll_t, D), np.nan, dtype=np.float32)

    for k, n in enumerate(idxs):
        start = rng.integers(0, T - input_len - roll_t - 1)
        history = traj[n, start : start + input_len].copy()   # (L, D) 已标准化
        truth = traj[n, start + input_len : start + input_len + roll_t]  # (roll_t, D)

        for t in range(roll_t):
            x_in = prepare_input_mask(history, mask, noise_std=noise_std,
                                      noise_seed=None if noise_std == 0 else seed * 100003 + k * 1009 + t)
            xt = torch.from_numpy(x_in[None].astype(np.float32)).to(DEVICE)
            out = model(xt)                       # (1, H, D)
            next_state = out[0, -1].cpu().numpy() # 取最后一步
            preds[k, t] = next_state
            errors[k, t] = float(np.sqrt(np.mean((next_state - truth[t]) ** 2)))
            # 滚动：丢最早一步，接上预测
            history = np.vstack([history[1:], next_state[None]])

    return errors, preds


def valid_prediction_time(errors: np.ndarray, threshold: float) -> np.ndarray:
    """
    errors: (n_traj, roll_t)。返回 (n_traj,) 的 VPT（第一个超阈值的时刻）。
    从未超阈值则返回 roll_t（表示整个窗口内都有效）。
    """
    n, T = errors.shape
    vpt = np.full(n, T, dtype=np.float32)
    for i in range(n):
        bad = np.where(errors[i] > threshold)[0]
        if bad.size:
            vpt[i] = bad[0]
    return vpt


def error_growth_rate(errors: np.ndarray, upto: int = 50) -> float:
    """
    误差对数增长斜率：对 log(RMSE) 在前 upto 步做线性拟合。
    越大 = 发散越快 = 长期越不稳。
    """
    m = np.nanmean(errors[:, :upto], axis=0)
    m = np.clip(m, 1e-8, None)
    t = np.arange(m.size, dtype=np.float64)
    slope = np.polyfit(t, np.log(m), 1)[0]
    return float(slope)


# --------------------------------------------------------------------------
# 评估一个已训练的 run
# --------------------------------------------------------------------------
def evaluate_run(
    ckpt_path: str,
    roll_t: int = DEFAULT_ROLL_T,
    n_traj: int = 50,
    save_plot: bool = True,
    outdir: str = "figs",
    eval_dir: str = "evals",
    eval_seeds: List[int] | None = None,
) -> Dict:
    """
    评估一个检查点。

    eval_seeds: 用于多种子评估。若给定多个种子，则用每个种子各自抽一批
        测试轨迹做 rollout，结果取平均——这样种子间方差同时包含
        "训练随机性" 和 "评估抽样随机性"，比只换训练种子更诚实。
        None 表示只用 seed=0 抽一次。
    """
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    mask = np.asarray(ckpt["mask"], dtype=np.float32)
    dim = cfg["dim"]
    horizon = cfg["horizon"]
    input_len = cfg["input_len"]
    model_name = cfg["model"]

    # 从文件名解析 system / model / mode / ratio / seed / noise
    # 形如: lorenz63_transformer_uniform_r0.50            （无种子、无噪声）
    #       lorenz63_transformer_temporal_r0.33_s1        （多种子）
    #       lorenz63_lstm_temporal_r0.33_s0_n0.05         （加噪）
    run_id = os.path.splitext(os.path.basename(ckpt_path))[0]
    m = re.match(
        r"^(?P<system>[a-z0-9]+)_(?P<model>[a-z]+)_(?P<mode>[a-z]+)"
        r"_r(?P<ratio>[0-9.]+)(?:_s(?P<seed>\d+))?(?:_n(?P<noise>[0-9.]+))?$",
        run_id,
    )
    if not m:
        raise ValueError(
            f"无法解析 run_id: {run_id!r}。"
            f"期望形如 lorenz63_lstm_uniform_r0.50 / ..._r0.33_s1 / ..._s0_n0.05"
        )
    system = m.group("system")
    model_label = m.group("model")
    mode = m.group("mode")
    ratio = float(m.group("ratio"))
    seed = int(m.group("seed")) if m.group("seed") is not None else None
    # 噪声强度优先取检查点里记录的值，回退到文件名
    noise_std = float(cfg.get("noise_std", 0.0) or 0.0)
    if noise_std == 0.0 and m.group("noise") is not None:
        noise_std = float(m.group("noise"))

    train, test, meta = load_data(system)
    _, te_z, _, _ = standardize(train, test, meta)

    model = build_model(model_name, dim, horizon, input_len).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    seeds = eval_seeds if eval_seeds else [0]
    all_errors = []
    all_vpt = []
    thr = VPT_THRESHOLD.get(system, 0.5)
    for es in seeds:
        errors, preds = rollout(
            model, te_z, mask, input_len, horizon, roll_t,
            n_traj=n_traj, seed=es, noise_std=noise_std,
        )
        all_errors.append(errors)
        all_vpt.append(valid_prediction_time(errors, thr))
    errors = np.concatenate(all_errors, axis=0)   # 所有种子抽到的轨迹合并
    vpt = np.concatenate(all_vpt, axis=0)

    thr = VPT_THRESHOLD.get(system, 0.5)
    result = {
        "run_id": run_id,
        "system": system,
        "model": model_label,
        "mode": mode,
        "ratio": ratio,
        "seed": seed,
        "noise_std": noise_std,
        "threshold": thr,
        "vpt_mean": float(np.mean(vpt)),
        "vpt_median": float(np.median(vpt)),
        "short_rmse_10": float(np.mean(errors[:, :10])),
        "long_rmse": float(np.nanmean(errors)),
        "growth_rate": error_growth_rate(errors),
        "roll_t": roll_t,
        "n_traj": n_traj,
        "eval_seeds": list(seeds),
        "n_traj_total": int(errors.shape[0]),
    }

    if save_plot:
        os.makedirs(outdir, exist_ok=True)
        try:
            _plot_single(errors, result, os.path.join(outdir, f"{run_id}.png"))
        except Exception as e:
            print(f"  (画图失败，跳过: {type(e).__name__}: {e})")

    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, f"{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"  {run_id:<42} VPT={result['vpt_mean']:6.1f}  "
        f"RMSE@10={result['short_rmse_10']:.4f}  "
        f"growth={result['growth_rate']:+.4f}"
    )
    return result


# --------------------------------------------------------------------------
# 画图
# --------------------------------------------------------------------------
def _plot_single(errors: np.ndarray, result: Dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=140)
    mean = np.nanmean(errors, axis=0)
    lo = np.nanpercentile(errors, 25, axis=0)
    hi = np.nanpercentile(errors, 75, axis=0)
    t = np.arange(mean.size)
    ax.plot(t, mean, lw=1.6, label="mean RMSE")
    ax.fill_between(t, lo, hi, alpha=0.25, lw=0, label="IQR")
    ax.axhline(result["threshold"], ls="--", lw=1, color="crimson", label="VPT threshold")
    ax.axvline(result["vpt_mean"], ls=":", lw=1.2, color="black",
               label=f"VPT={result['vpt_mean']:.0f}")
    ax.set_yscale("log")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("RMSE (standardised)")
    ax.set_title(f"{result['run_id']}", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_report(outdir: str = "figs", evaldir: str = "evals") -> None:
    """
    生成核心对比图：VPT 与短时精度随观测密度的变化。

    每个系统选一个"主掩码模式"：
      - D 小的系统（lorenz63）只有 temporal
      - D 大的系统（lorenz96）优先用 uniform（最能类比"传感器数量"）
    这样同一张图里不会混入不可比的模式。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = glob.glob(os.path.join(evaldir, "*.json"))
    if not files:
        print("没有找到评估结果，先跑 evaluate.py --all")
        return
    rows = [json.load(open(f, encoding="utf-8")) for f in files]

    # 每个系统首选 uniform，其次 temporal；都没有就用任意一个含多比例的
    def pick_mode(system: str) -> str | None:
        modes = {r["mode"] for r in rows if r["system"] == system and r["mode"] != "none"}
        for pref in ("uniform", "temporal", "random", "block"):
            if pref in modes:
                return pref
        return None

    os.makedirs(outdir, exist_ok=True)
    for system in sorted({r["system"] for r in rows}):
        mode = pick_mode(system)
        if mode is None:
            print(f"{system}: 没有稀疏化结果，跳过报告图")
            continue
        sub = [r for r in rows if r["system"] == system and r["mode"] == mode]
        if not sub:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

        # 左：VPT vs 观测比例
        ax = axes[0]
        plotted = False
        for model in sorted({r["model"] for r in sub}):
            pts = sorted(
                [(r["ratio"], r["vpt_mean"]) for r in sub if r["model"] == model],
                key=lambda p: p[0],
            )
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=model)
                plotted = True
        ax.set_xlabel("fraction of observations retained")
        ax.set_ylabel("VPT (steps)")
        ax.set_title(f"{system} / {mode}: VPT vs observation density", fontsize=10)
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=8, frameon=False)

        # 右：短时 RMSE vs 观测比例
        ax = axes[1]
        for model in sorted({r["model"] for r in sub}):
            pts = sorted(
                [(r["ratio"], r["short_rmse_10"]) for r in sub if r["model"] == model],
                key=lambda p: p[0],
            )
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "s-", label=model)
        ax.set_xlabel("fraction of observations retained")
        ax.set_ylabel("RMSE over first 10 steps")
        ax.set_title(f"{system} / {mode}: short-term accuracy", fontsize=10)
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=8, frameon=False)

        fig.tight_layout()
        path = os.path.join(outdir, f"report_{system}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"已生成 {path} (mode={mode})")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="自回归 rollout 评估")
    ap.add_argument("--run", default=None, help="单个 .pt 检查点路径")
    ap.add_argument("--all", action="store_true", help="评估 runs/ 下全部")
    ap.add_argument("--report", action="store_true", help="生成对比报告图")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--roll-t", type=int, default=DEFAULT_ROLL_T)
    ap.add_argument("--n-traj", type=int, default=50)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if args.report:
        make_report()
        return

    if args.run:
        evaluate_run(args.run, args.roll_t, args.n_traj, save_plot=not args.no_plot)
        return

    if args.all:
        ckpts = sorted(glob.glob(os.path.join(args.runs_dir, "*.pt")))
        if not ckpts:
            print(f"{args.runs_dir}/ 下没有 .pt，先跑 train.py")
            return
        print(f"共 {len(ckpts)} 个检查点")
        for c in ckpts:
            evaluate_run(c, args.roll_t, args.n_traj, save_plot=not args.no_plot)
        make_report()
        return

    ap.print_help()


if __name__ == "__main__":
    main()
