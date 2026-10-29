"""
run_sweep_seeds.py -- 噪声扫描的多种子补充（修掉"84 组全是 seed=0"的缺陷）

问题背景：
    `run_noise_sweep.py` 产出的 84 组全部用 seed=0 训练。
    相图里 σ ≤ 0.02 区域的翻转因此无法归因：
    是训练随机性，还是评估抽样随机性？两者混在一起。
    单种子时得出的 "crossover near σ ≈ 0.03" 没有置信区间支撑。

本脚本：
    只补 seed=1 和 seed=2。seed=0 的结果复用 `runs_sweep/` 与 `evals_sweep/`，
    合并后得到 3 个训练种子的完整结果，可算 mean ± std。
    早期只有单种子时给出的"crossover near σ ≈ 0.03"结论，正是在这里被检验并修正的。

噪声水平（聚焦交叉区域）：
    0.01, 0.02, 0.05, 0.08, 0.10
    不含 0.15/0.20 —— 那两个点在单种子下结论已稳定（LSTM 领先），
    也不含 0.00 —— 无噪声基线已由 run_seeds.py 提供 3 个种子。

规模：4 配置 × 3 模型 × 5 噪声 × 2 种子 = 120 组

用法：
    python run_sweep_seeds.py
    python run_sweep_seeds.py --report-only
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
NOISE_LEVELS = [0.01, 0.02, 0.05, 0.08, 0.10]
SEEDS = [1, 2]
EPOCHS = 30
OUTDIR = "runs_sweep_seeds"
EVALDIR = "evals_sweep_seeds"
BASELINE_RUNS = "runs_sweep"      # seed=0
BASELINE_EVALS = "evals_sweep"    # seed=0


def same_ratio(a: float, b: float, tol: float = 0.006) -> bool:
    return abs(a - b) < tol


def rid_of(system: str, model: str, mode: str, ratio: float,
           seed: int, noise: float) -> str:
    return f"{system}_{model}_{mode}_r{ratio:.2f}_s{seed}_n{noise:.2f}"


def train_all() -> None:
    from train import train_one

    todos = [
        (system, model, mode, ratio, seed, noise)
        for (system, mode, ratio) in CONFIGS.values()
        for model in MODELS
        for seed in SEEDS
        for noise in NOISE_LEVELS
    ]
    pending = [
        t for t in todos
        if not (os.path.exists(os.path.join(OUTDIR, f"{rid_of(*t)}.json"))
                and os.path.exists(os.path.join(OUTDIR, f"{rid_of(*t)}.pt")))
    ]
    print("=" * 72)
    print(f"噪声扫描种子补充: 共 {len(todos)} 组 | 已完成 {len(todos)-len(pending)} | 待跑 {len(pending)}")
    print(f"种子: {SEEDS} | 噪声: {NOISE_LEVELS}")
    print("=" * 72, flush=True)

    if not pending:
        print("训练部分已完成。")
        return

    t0 = time.time()
    ok, failed = 0, []
    for i, (system, model, mode, ratio, seed, noise) in enumerate(pending, 1):
        rid = rid_of(system, model, mode, ratio, seed, noise)
        print(f"\n[{i}/{len(pending)}] {rid}  (已用时 {time.time()-t0:.0f}s)", flush=True)
        try:
            r = train_one(system, model, mode, ratio, epochs=EPOCHS,
                          outdir=OUTDIR, verbose=False, seed=seed, noise_std=noise)
            ok += 1
            print(f"    test_mse={r['test_mse']:.6f}", flush=True)
        except Exception as e:
            failed.append((rid, f"{type(e).__name__}: {e}"))
            print(f"    !! 失败: {type(e).__name__}: {e}", flush=True)

    print("\n" + "=" * 72)
    print(f"训练结束: 成功 {ok}/{len(pending)}，失败 {len(failed)}，"
          f"耗时 {(time.time()-t0)/60:.1f} 分钟")
    for rid, err in failed:
        print(f"  失败: {rid} -> {err}")
    print("=" * 72)


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
            evaluate.evaluate_run(c, roll_t=roll_t, n_traj=n_traj, save_plot=False,
                                  eval_dir=EVALDIR, eval_seeds=[0])
        except Exception as e:
            print(f"  !! {rid}: {type(e).__name__}: {e}", flush=True)
    print(f"评估完成，耗时 {(time.time()-t0)/60:.1f} 分钟")


def collect() -> List[Dict]:
    """合并 seed=0（旧目录）与 seed=1,2（新目录）。"""
    rows = []
    for d in (BASELINE_EVALS, EVALDIR):
        for p in glob.glob(os.path.join(d, "*.json")):
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            if "seed" not in r or r["seed"] is None:
                r["seed"] = 0
            rows.append(r)
    return rows


def report() -> None:
    rows = collect()
    if not rows:
        print("没有结果")
        return

    print("\n" + "=" * 86)
    print("噪声扫描 · 3 种子（mean ± std over 3 training seeds）")
    print("=" * 86)

    out = ["# 噪声扫描 · 多种子结果\n",
           "3 个训练种子（seed=0/1/2），掩码固定 seed=42，每模型 40 条测试轨迹。",
           "seed=0 来自 `evals_sweep/`，seed=1/2 来自 `evals_sweep_seeds/`。\n",
           "> ⚠️ **σ=0.15 / 0.20 只有 1 个训练种子**（来自早期单种子扫描），",
           "> std 为 0，**不能用于显著性判断**。表中以 `n=1` 标出。\n"]

    noise_axis = sorted({round(r.get("noise_std", 0.0), 2) for r in rows})
    # cname -> (mode, ratio)，两者一起用于筛选，避免 ratio 重复时混淆
    cfg_key = {c: (CONFIGS[c][1], CONFIGS[c][2]) for c in CONFIGS}

    # ---- 表 1：各模型 VPT 随噪声的变化 ----
    print(f"\n{'config':<7}{'noise':<8}{'n':<4}" + "".join(f"{m:>18}" for m in MODELS) + "   最优")
    out.append("## 各模型 VPT（mean ± std）\n")
    out.append("| config | noise | n | " + " | ".join(MODELS) + " | 最优 |")
    out.append("|---|---|---|" + "---|" * (len(MODELS) + 1))

    table: Dict[Tuple[str, float], Dict[str, Tuple[float, float, int]]] = {}
    for cname, (mode, ratio) in cfg_key.items():
        for noise in noise_axis:
            cells = {}
            for model in MODELS:
                sub = [r for r in rows if r["model"] == model
                       and r["mode"] == mode and same_ratio(r["ratio"], ratio)
                       and abs(r.get("noise_std", 0.0) - noise) < 1e-9]
                if sub:
                    v = np.array([r["vpt_mean"] for r in sub])
                    cells[model] = (float(v.mean()), float(v.std()), len(v))
            if not cells:
                continue
            table[(cname, noise)] = cells
            best = max(cells, key=lambda m: cells[m][0])
            n_seeds = max(c[2] for c in cells.values())
            flag = " (n=1)" if n_seeds < 2 else ""
            line = f"{cname:<7}{noise:<8.2f}{n_seeds:<4}"
            for m in MODELS:
                if m in cells:
                    mu, sd, n = cells[m]
                    line += f"{mu:>10.1f} ± {sd:<6.1f}"
                else:
                    line += f"{'-':>18}"
            line += f"   {best}{flag}"
            print(line)
            out.append(
                f"| {cname} | {noise:.2f} | {n_seeds} | "
                + " | ".join(f"{cells[m][0]:.1f} ± {cells[m][1]:.1f}" if m in cells else "-"
                             for m in MODELS)
                + f" | {best}{flag} |"
            )

    # ---- 表 2：LSTM − MLP 差值 + 稳健性判定 ----
    print("\n" + "=" * 86)
    print("LSTM − MLP（正值 = LSTM 更好）；「稳健」= 差值 / 噪声 > 3")
    print("=" * 86)
    out.append("\n## LSTM − MLP 与稳健性\n")
    out.append("| config | noise | LSTM−MLP | 噪声(std 取大) | 比值 | 判定 |")
    out.append("|---|---|---|---|---|---|")

    print(f"{'config':<7}{'noise':<8}{'LSTM-MLP':>10}{'noise':>10}{'ratio':>8}   判定")
    for cname in ("c1", "c2", "c3", "c4"):
        for noise in noise_axis:
            cells = table.get((cname, noise), {})
            if "mlp" not in cells or "lstm" not in cells:
                continue
            diff = cells["lstm"][0] - cells["mlp"][0]
            nz = max(cells["mlp"][1], cells["lstm"][1])
            ratio_v = abs(diff) / nz if nz > 0 else float("inf")
            if ratio_v > 3:
                verdict = "LSTM 领先" if diff > 0 else "MLP 领先"
            else:
                verdict = "无显著差异"
            print(f"{cname:<7}{noise:<8.2f}{diff:>+10.1f}{nz:>10.1f}{ratio_v:>8.1f}   {verdict}")
            out.append(f"| {cname} | {noise:.2f} | {diff:+.1f} | {nz:.1f} | "
                       f"{ratio_v:.1f} | {verdict} |")

    # ---- 表 3：交叉点定位 ----
    print("\n" + "=" * 86)
    print("交叉点：LSTM 首次在多数配置中显著领先的噪声水平")
    print("=" * 86)
    out.append("\n## 交叉点定位\n")

    for noise in noise_axis:
        wins = 0
        total = 0
        for cname in ("c1", "c2", "c3", "c4"):
            cells = table.get((cname, noise), {})
            if "mlp" in cells and "lstm" in cells:
                total += 1
                if cells["lstm"][0] > cells["mlp"][0]:
                    wins += 1
        if total:
            msg = f"noise={noise:.2f}: LSTM 在 {wins}/{total} 个配置中优于 MLP"
            print("  " + msg)
            out.append(f"- {msg}")

    # ---- 机器可读 ----
    serializable = {
        f"{c}|{n:.2f}": {m: {"mean": v[0], "std": v[1], "n": v[2]}
                         for m, v in cells.items()}
        for (c, n), cells in table.items()
    }
    with open("sweep_seeds_table.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    with open("noise_sweep_seeds.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n已写入 noise_sweep_seeds.md 与 sweep_seeds_table.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="噪声扫描多种子补充")
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
