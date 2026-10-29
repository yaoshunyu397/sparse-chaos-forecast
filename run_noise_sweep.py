"""
run_noise_sweep.py -- 噪声扫描，画"最优模型相图"

目的：
    前一轮只测了 5% 和 10% 两个点，结果显示 LSTM 在稀疏+有噪时反超 MLP。
    但 5%→10% 时优势是**非单调**的（c3 从 +5.2 缩到 +1.4），
    说明临界点在某个位置，必须扫出来，否则"噪声下 LSTM 更好"这句话
    只对特定噪声水平成立。

    扫完之后回答的是一个更强的问题：
        **在 (观测稀疏度, 噪声强度) 这张平面上，哪个模型最优？**

设计：
    噪声: 7 个水平（0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20）
    配置: c1..c4（同前）
    模型: mlp / lstm / transformer
    种子: 1 个（0）—— 用于定位趋势，不用于逐点置信区间

    基线 noise=0 复用 evals_seeds_re/（3 训练种子 × 80 条轨迹），不重跑。

用法：
    python run_noise_sweep.py
    python run_noise_sweep.py --report-only
    python plot_phase.py            # 画相图
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np

CONFIGS: Dict[str, Tuple[str, str, float]] = {
    "c1": ("lorenz63", "none", 1.0),
    "c2": ("lorenz63", "temporal", 0.5),
    "c3": ("lorenz63", "temporal", 1 / 3),
    "c4": ("lorenz63", "temporal", 1 / 6),
}
MODELS = ["mlp", "lstm", "transformer"]
NOISE_LEVELS = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
SEED = 0
EPOCHS = 30
OUTDIR = "runs_sweep"
EVALDIR = "evals_sweep"


def same_ratio(a: float, b: float, tol: float = 0.006) -> bool:
    return abs(a - b) < tol


def rid_of(system: str, model: str, mode: str, ratio: float,
           seed: int, noise: float) -> str:
    return f"{system}_{model}_{mode}_r{ratio:.2f}_s{seed}_n{noise:.2f}"


def train_all() -> None:
    from train import train_one

    todos = [
        (system, model, mode, ratio, noise)
        for (system, mode, ratio) in CONFIGS.values()
        for model in MODELS
        for noise in NOISE_LEVELS
    ]
    pending = []
    for system, model, mode, ratio, noise in todos:
        rid = rid_of(system, model, mode, ratio, SEED, noise)
        if os.path.exists(os.path.join(OUTDIR, f"{rid}.json")) and os.path.exists(
            os.path.join(OUTDIR, f"{rid}.pt")
        ):
            continue
        pending.append((system, model, mode, ratio, noise))

    print("=" * 74)
    print(f"噪声扫描: 共 {len(todos)} 组 | 已完成 {len(todos)-len(pending)} | 待跑 {len(pending)}")
    print(f"噪声水平: {NOISE_LEVELS}")
    print(f"配置: {list(CONFIGS)} | 模型: {MODELS} | 种子: {SEED}")
    print("=" * 74, flush=True)

    if not pending:
        print("训练部分已全部完成。")
        return

    t0 = time.time()
    ok, failed = 0, []
    for i, (system, model, mode, ratio, noise) in enumerate(pending, 1):
        rid = rid_of(system, model, mode, ratio, SEED, noise)
        print(f"\n[{i}/{len(pending)}] {rid}  (已用时 {time.time()-t0:.0f}s)", flush=True)
        try:
            res = train_one(
                system, model, mode, ratio, epochs=EPOCHS, outdir=OUTDIR,
                verbose=False, seed=SEED, noise_std=noise,
            )
            ok += 1
            print(f"    test_mse={res['test_mse']:.6f}", flush=True)
        except Exception as e:
            failed.append((rid, f"{type(e).__name__}: {e}"))
            print(f"    !! 失败: {type(e).__name__}: {e}", flush=True)

    print("\n" + "=" * 74)
    print(f"训练结束: 成功 {ok}/{len(pending)}，失败 {len(failed)}，"
          f"耗时 {(time.time()-t0)/60:.1f} 分钟")
    for rid, err in failed:
        print(f"  失败: {rid} -> {err}")
    print("=" * 74)


def evaluate_all(roll_t: int = 250, n_traj: int = 40) -> None:
    import evaluate

    os.makedirs(EVALDIR, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(OUTDIR, "*.pt")))
    if not ckpts:
        print(f"{OUTDIR}/ 下没有检查点")
        return

    print(f"\n评估 {len(ckpts)} 个检查点 ...", flush=True)
    t0 = time.time()
    for c in ckpts:
        rid = os.path.splitext(os.path.basename(c))[0]
        if os.path.exists(os.path.join(EVALDIR, f"{rid}.json")):
            continue
        try:
            evaluate.evaluate_run(
                c, roll_t=roll_t, n_traj=n_traj, save_plot=False,
                eval_dir=EVALDIR, eval_seeds=[0],
            )
        except Exception as e:
            print(f"  !! {rid}: {type(e).__name__}: {e}", flush=True)
    print(f"评估完成，耗时 {(time.time()-t0)/60:.1f} 分钟")


def collect() -> List[Dict]:
    """汇总基线 + 扫描结果。"""
    rows: List[Dict] = []
    # 基线 noise=0
    for p in glob.glob("evals_seeds_re/*.json"):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        r["noise_std"] = 0.0
        rows.append(r)
    for p in glob.glob(os.path.join(EVALDIR, "*.json")):
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def report() -> None:
    rows = collect()
    if not rows:
        print("没有结果")
        return

    # 汇总成 (config, noise) -> {model: (mean, std, n)}
    table: Dict[Tuple[str, float], Dict[str, Tuple[float, float, int]]] = {}
    for cname, (system, mode, ratio) in CONFIGS.items():
        for noise in [0.0] + NOISE_LEVELS:
            for model in MODELS:
                sub = [
                    r for r in rows
                    if r["system"] == system and r["mode"] == mode
                    and same_ratio(r["ratio"], ratio)
                    and r["model"] == model
                    and abs(r.get("noise_std", 0.0) - noise) < 1e-9
                ]
                if not sub:
                    continue
                v = np.array([r["vpt_mean"] for r in sub])
                table.setdefault((cname, noise), {})[model] = (
                    float(v.mean()), float(v.std()), len(sub)
                )

    print("\n" + "=" * 84)
    print("噪声扫描：最优模型相图")
    print("=" * 84)
    noise_axis = [0.0] + NOISE_LEVELS
    print(f"{'config':<7}" + "".join(f"{n:<9.2f}" for n in noise_axis))
    print("-" * 84)
    for cname in CONFIGS:
        cells = []
        for noise in noise_axis:
            models = table.get((cname, noise), {})
            if not models:
                cells.append("-".ljust(9))
                continue
            best = max(models, key=lambda m: models[m][0])
            tag = {"mlp": "MLP", "lstm": "LSTM", "transformer": "TF"}[best]
            cells.append(f"{tag}".ljust(9))
        print(f"{cname:<7}" + "".join(cells))
    print("-" * 84)
    print("（每格为 VPT 最高的模型）")

    # 数值表：各模型的 VPT 随噪声变化
    print("\n" + "=" * 84)
    print("各模型 VPT 随噪声的变化")
    print("=" * 84)
    for cname in CONFIGS:
        print(f"\n-- {cname} --")
        print(f"{'noise':<8}{'MLP':>10}{'LSTM':>10}{'Transf':>10}   最优")
        for noise in noise_axis:
            models = table.get((cname, noise), {})
            if not models:
                continue
            vals = [models.get(m, (float('nan'), 0, 0))[0] for m in MODELS]
            best = max(models, key=lambda m: models[m][0])
            print(f"{noise:<8.2f}" + "".join(f"{v:>10.1f}" for v in vals) + f"   {best}")

    # LSTM 相对 MLP 的优势随噪声变化（关键趋势）
    print("\n" + "=" * 84)
    print("LSTM − MLP（正值 = LSTM 更好）")
    print("=" * 84)
    out = ["# 噪声扫描结果\n",
           "全部为 Lorenz-63，单训练种子（seed=0），每模型 40 条测试轨迹。",
           "基线 noise=0 复用 3 种子 × 80 条轨迹。\n",
           "\n## 最优模型相图\n",
           "| config | " + " | ".join(f"{n:.2f}" for n in noise_axis) + " |",
           "|---|" + "---|" * len(noise_axis)]
    print(f"{'config':<7}" + "".join(f"{n:<9.2f}" for n in noise_axis))

    for cname in CONFIGS:
        cells = []
        deltas = []
        for noise in noise_axis:
            models = table.get((cname, noise), {})
            if not models or "mlp" not in models or "lstm" not in models:
                cells.append("-")
                deltas.append("-")
                continue
            best = max(models, key=lambda m: models[m][0])
            cells.append({"mlp": "MLP", "lstm": "LSTM", "transformer": "TF"}[best])
            deltas.append(f"{models['lstm'][0] - models['mlp'][0]:+.1f}")
        out.append(f"| {cname} | " + " | ".join(cells) + " |")
        print(f"{cname:<7}" + "".join(f"{d:<9}" for d in deltas))

    out.append("\n## LSTM − MLP（正值 = LSTM 更好）\n")
    out.append("| config | " + " | ".join(f"{n:.2f}" for n in noise_axis) + " |")
    out.append("|---|" + "---|" * len(noise_axis))
    for cname in CONFIGS:
        deltas = []
        for noise in noise_axis:
            models = table.get((cname, noise), {})
            if not models or "mlp" not in models or "lstm" not in models:
                deltas.append("-")
            else:
                deltas.append(f"{models['lstm'][0] - models['mlp'][0]:+.1f}")
        out.append(f"| {cname} | " + " | ".join(deltas) + " |")

    # 机器可读，供画图用
    serializable = {
        f"{cname}|{noise:.2f}": {
            m: table[(cname, noise)][m][0]
            for m in table.get((cname, noise), {})
        }
        for cname in CONFIGS for noise in noise_axis if (cname, noise) in table
    }
    with open("sweep_table.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    with open("noise_sweep.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n已写入 noise_sweep.md 与 sweep_table.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="噪声扫描")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--roll-t", type=int, default=250)
    ap.add_argument("--n-traj", type=int, default=40)
    args = ap.parse_args()

    if args.report_only:
        report()
        return
    if not args.eval_only:
        train_all()
    evaluate_all(args.roll_t, args.n_traj)
    report()


if __name__ == "__main__":
    main()
