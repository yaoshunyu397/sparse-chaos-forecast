"""
plot_phase.py -- 画"最优模型相图"与优势曲线

读 sweep_table.json（由 run_noise_sweep.py 生成），产出两张图：

  phase_diagram.png
      左：相图 —— x=观测比例，y=噪声强度，颜色=最优模型
      右：LSTM − MLP 的 VPT 差值热图（正值 = LSTM 更好）

  noise_curves.png
      四个配置各一张子图：三条 VPT-噪声曲线，标出交叉点

这两张图是这块工作最适合放进邮件/README 的东西：
一眼能看出"模型排名依赖观测质量"。

用法：
    python plot_phase.py
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

MODELS = ["mlp", "lstm", "transformer"]
LABEL = {"mlp": "MLP", "lstm": "LSTM", "transformer": "Transformer"}
COLOR = {"mlp": "#d17b0f", "lstm": "#2b6cb0", "transformer": "#2f855a"}

CONFIG_ORDER = ["c1", "c2", "c3", "c4"]
CONFIG_RATIO = {"c1": 1.0, "c2": 0.5, "c3": 1 / 3, "c4": 1 / 6}


def load_table(path: str = "sweep_table.json") -> Dict[str, Dict[str, float]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_matrix(
    table: Dict[str, Dict[str, float]],
    rows: List[str],
    cols: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (best_idx, delta)：
        best_idx[y,x] = 最优模型的下标（0=mlp,1=lstm,2=transformer）
        delta[y,x]    = LSTM - MLP 的 VPT 差（正值 = LSTM 更好）
    """
    best = np.full((len(rows), len(cols)), np.nan)
    delta = np.full((len(rows), len(cols)), np.nan)
    for i, cname in enumerate(rows):
        for j, noise in enumerate(cols):
            key = f"{cname}|{noise:.2f}"
            if key not in table:
                continue
            models = table[key]
            if not models:
                continue
            best_model = max(models, key=lambda m: models[m])
            best[i, j] = MODELS.index(best_model)
            if "mlp" in models and "lstm" in models:
                delta[i, j] = models["lstm"] - models["mlp"]
    return best, delta


def plot_phase(tab: Dict[str, Dict[str, float]], outdir: str = "figs") -> None:
    os.makedirs(outdir, exist_ok=True)
    all_noise = sorted({float(k.split("|")[1]) for k in tab})
    ratios = [CONFIG_RATIO[c] for c in CONFIG_ORDER]

    best, delta = build_matrix(tab, CONFIG_ORDER, all_noise)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), dpi=150)

    # ---- 左：相图 ----
    ax = axes[0]
    cmap = ListedColormap(["#d17b0f", "#2b6cb0", "#2f855a"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    # 用 ratio 作为 x（反序，越右观测越少），noise 作为 y
    im = ax.imshow(
        best, aspect="auto", origin="lower", cmap=cmap, norm=norm,
        extent=[-0.5, len(ratios) - 0.5, -0.5, len(all_noise) - 0.5],
    )
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f"{r:.2f}" for r in ratios])
    ax.set_yticks(range(len(all_noise)))
    ax.set_yticklabels([f"{n:.2f}" for n in all_noise])
    ax.set_xlabel("fraction of observations retained")
    ax.set_ylabel("observation noise (σ, standardised)")
    ax.set_title("Which model wins?", fontsize=10)
    # 手写图例
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=COLOR[m], label=LABEL[m]) for m in MODELS],
        loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
        frameon=False, fontsize=8,
    )

    # ---- 右：LSTM − MLP ----
    ax = axes[1]
    vmax = np.nanmax(np.abs(delta)) if np.isfinite(delta).any() else 1.0
    im2 = ax.imshow(
        delta, aspect="auto", origin="lower", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        extent=[-0.5, len(ratios) - 0.5, -0.5, len(all_noise) - 0.5],
    )
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f"{r:.2f}" for r in ratios])
    ax.set_yticks(range(len(all_noise)))
    ax.set_yticklabels([f"{n:.2f}" for n in all_noise])
    ax.set_xlabel("fraction of observations retained")
    ax.set_ylabel("observation noise (σ)")
    ax.set_title("LSTM − MLP (VPT steps; blue = LSTM better)", fontsize=10)
    # 在每个格子写数值
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            if np.isfinite(delta[i, j]):
                ax.text(
                    j, i, f"{delta[i, j]:+.0f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(delta[i, j]) > 0.5 * vmax else "black",
                )
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.03)

    fig.tight_layout()
    p = os.path.join(outdir, "phase_diagram.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"已生成 {p}")


def plot_curves(tab: Dict[str, Dict[str, float]], outdir: str = "figs") -> None:
    os.makedirs(outdir, exist_ok=True)
    all_noise = sorted({float(k.split("|")[1]) for k in tab})

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150, squeeze=False)
    for idx, cname in enumerate(CONFIG_ORDER):
        ax = axes[idx // 2][idx % 2]
        for model in MODELS:
            ys, xs = [], []
            for noise in all_noise:
                key = f"{cname}|{noise:.2f}"
                if key in tab and model in tab[key]:
                    xs.append(noise)
                    ys.append(tab[key][model])
            if xs:
                ax.plot(xs, ys, "o-", ms=3.5, lw=1.5,
                        label=LABEL[model], color=COLOR[model])
        ax.set_xlabel("observation noise σ")
        ax.set_ylabel("VPT (steps)")
        ax.set_title(f"{cname}  (ratio = {CONFIG_RATIO[cname]:.2f})", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, frameon=False)

    fig.suptitle("VPT vs observation noise — ranking is not stable", fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, "noise_curves.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"已生成 {p}")


def main() -> None:
    tab = load_table("sweep_table.json")
    plot_phase(tab)
    plot_curves(tab)


if __name__ == "__main__":
    main()
