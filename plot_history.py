"""
plot_history.py -- 训练曲线图

两个用途：
  1. 自检：val loss 是否随 epoch 下降、有没有过拟合/未收敛；
  2. 放进 README：证明模型确实训练到位，"单步误差很低"不是意外。

用法：
    python plot_history.py                      # 单组曲线
    python plot_history.py --system lorenz63 --mode temporal   # 按掩码模式对比
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_LABEL = {"mlp": "MLP", "lstm": "LSTM", "transformer": "Transformer"}
COLORS = {"mlp": "#d17b0f", "lstm": "#2b6cb0", "transformer": "#2f855a"}


def load_runs(runs_dir: str = "runs", systems: List[str] | None = None) -> List[Dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if systems and r["system"] not in systems:
            continue
        if "history" not in r or not r["history"]:
            continue
        out.append(r)
    return out


def plot_one(r: Dict, outdir: str = "figs") -> None:
    os.makedirs(outdir, exist_ok=True)
    h = r["history"]
    ep = [x["epoch"] for x in h]
    tr = [x["train"] for x in h]
    va = [x["val"] for x in h]

    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=140)
    ax.plot(ep, tr, lw=1.6, label="train", color="#2b6cb0")
    ax.plot(ep, va, lw=1.6, label="val", color="#c53030")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (log)")
    ax.set_title(r["run_id"], fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"hist_{r['run_id']}.png"))
    plt.close(fig)


def plot_compare(runs: List[Dict], system: str, mode: str, outdir: str = "figs") -> None:
    """
    同一系统、同一掩码模式下，三个模型的验证曲线对比。
    一张图里放多个观测比例：每个比例一条实线（模型色）。
    """
    sub = [r for r in runs if r["system"] == system and r["mode"] == mode]
    if not sub:
        return
    ratios = sorted({r["ratio"] for r in sub}, reverse=True)
    ncol = min(len(ratios), 3)
    nrow = (len(ratios) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), dpi=130,
                             squeeze=False)

    for k, ratio in enumerate(ratios):
        ax = axes[k // ncol][k % ncol]
        for m in ("mlp", "lstm", "transformer"):
            hit = [r for r in sub if r["model"] == m and abs(r["ratio"] - ratio) < 1e-9]
            if not hit:
                continue
            h = hit[0]["history"]
            ax.plot([x["epoch"] for x in h], [x["val"] for x in h],
                    lw=1.4, label=MODEL_LABEL[m], color=COLORS[m])
            ax.axhline(hit[0]["test_mse"], ls=":", lw=0.9, color=COLORS[m], alpha=0.7)
        ax.set_yscale("log")
        ax.set_title(f"ratio={ratio:.2f}", fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val MSE")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7, frameon=False)

    for k in range(len(ratios), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle(f"{system} / {mode} — validation loss (dotted = test MSE)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(outdir, f"hist_compare_{system}_{mode}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"已生成 {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="训练曲线")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--system", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--all", action="store_true", help="为每组画单独曲线")
    args = ap.parse_args()

    systems = [args.system] if args.system else None
    runs = load_runs(args.runs_dir, systems)
    if not runs:
        print("没有可画的训练历史。先跑 run_grid.py")
        return
    print(f"共 {len(runs)} 组训练历史")

    if args.all:
        for r in runs:
            plot_one(r, args.outdir)
        print(f"已生成 {len(runs)} 张单曲线图到 {args.outdir}/")

    targets = []
    for sys_name in sorted({r["system"] for r in runs}):
        for mode in sorted({r["mode"] for r in runs if r["system"] == sys_name}):
            targets.append((sys_name, mode))
    if args.system:
        targets = [t for t in targets if t[0] == args.system]
    if args.mode:
        targets = [t for t in targets if t[1] == args.mode]

    for sys_name, mode in targets:
        plot_compare(runs, sys_name, mode, args.outdir)


if __name__ == "__main__":
    main()
