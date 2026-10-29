"""
run_noise.py -- 含噪观测实验（结论稳健性检验）

为什么必须做：
    主网格与多种子实验都建立在"观测无噪声"这个不现实假设上。
    真实传感器有噪声。如果加噪后 MLP 的优势消失，那"简单模型更好"
    这个结论就得改写。

设计：
    只改一个变量——观测噪声水平。模型、数据、掩码、评估协议全部不变。
    无噪声基线直接复用 seed_results.md 里的结果，不重跑。

        噪声水平: 0.05（5%，约等于每维标准差的 5%）、0.10（10%）
        配置    : c1..c4（同 run_seeds.py）
        模型    : mlp / lstm / transformer
        种子    : 2 个（0, 1）

    噪声只加在"被观测到的位置"上；缺失位置仍是 0。
    训练与评估使用同一噪声水平。

用法：
    python run_noise.py                  # 训练 + 评估 + 报告
    python run_noise.py --eval-only
    python run_noise.py --report-only
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
SEEDS = [0, 1]
NOISE_LEVELS = [0.05, 0.10]
EPOCHS = 30
OUTDIR = "runs_noise"
EVALDIR = "evals_noise"


def same_ratio(a: float, b: float, tol: float = 0.006) -> bool:
    """容忍 run_id 的两位小数（r0.33 vs 1/3）。"""
    return abs(a - b) < tol


def run_id_of(system: str, model: str, mode: str, ratio: float,
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
        if not (
            os.path.exists(os.path.join(OUTDIR, f"{run_id_of(*t)}.json"))
            and os.path.exists(os.path.join(OUTDIR, f"{run_id_of(*t)}.pt"))
        )
    ]

    print("=" * 72)
    print(f"含噪实验: 共 {len(todos)} 组 | 已完成 {len(todos)-len(pending)} | 待跑 {len(pending)}")
    print(f"噪声水平: {NOISE_LEVELS} | 种子: {SEEDS} | epochs={EPOCHS}")
    print("=" * 72, flush=True)

    if not pending:
        print("训练部分已全部完成。")
        return

    t0 = time.time()
    ok, failed = 0, []
    for i, (system, model, mode, ratio, seed, noise) in enumerate(pending, 1):
        rid = run_id_of(system, model, mode, ratio, seed, noise)
        print(f"\n[{i}/{len(pending)}] {rid}  (已用时 {time.time()-t0:.0f}s)", flush=True)
        try:
            res = train_one(
                system, model, mode, ratio, epochs=EPOCHS,
                outdir=OUTDIR, verbose=False, seed=seed, noise_std=noise,
            )
            ok += 1
            print(f"    test_mse={res['test_mse']:.6f}", flush=True)
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
        print(f"{OUTDIR}/ 下没有检查点，先跑训练")
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
                eval_dir=EVALDIR, eval_seeds=[0, 1],
            )
        except Exception as e:
            print(f"  !! {rid}: {type(e).__name__}: {e}", flush=True)
    print(f"评估完成，耗时 {(time.time()-t0)/60:.1f} 分钟")


def load_seed_baseline() -> List[Dict]:
    """
    读取无噪声基线。

    优先用 evals_seeds_re/（与含噪组相同的评估协议：eval_seeds=[0,1]、
    n_traj=40 -> 80 条轨迹），保证 std 可比。
    回退到 evals_seeds/（旧协议，仅 40 条），此时会打印警告。
    """
    for d in ("evals_seeds_re", "evals_seeds"):
        paths = glob.glob(os.path.join(d, "*.json"))
        if not paths:
            continue
        if d == "evals_seeds":
            print("!! 警告：未找到 evals_seeds_re/，回退到 evals_seeds/。")
            print("   两组的评估样本量不同（40 vs 80），std 不可直接比较。")
        out = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            r["noise_std"] = 0.0
            out.append(r)
        return out
    return []


def report() -> None:
    rows: List[Dict] = load_seed_baseline()
    n_noise_rows = 0
    for p in glob.glob(os.path.join(EVALDIR, "*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        rows.append(r)
        n_noise_rows += 1

    if not n_noise_rows:
        print("没有含噪结果")
        return

    print("\n" + "=" * 80)
    print("含噪观测下的稳健性检验")
    print("=" * 80)
    print(f"基线(无噪声) {len(rows)-n_noise_rows} 组 | 含噪 {n_noise_rows} 组")

    out_lines = [
        "# 含噪观测实验结果\n",
        "只改变观测噪声水平，其余设置与多种子实验完全一致。",
        "噪声只加在被观测到的位置；缺失位置仍填 0。\n",
        "**评估协议已统一**：基线与含噪组均为 2 个评估种子 × 40 条轨迹 = 80 条，",
        "因此两者的 std 可以直接比较。",
        "基线为 3 个训练种子（noise=0，来自 `evals_seeds_re/`），",
        "含噪组为 2 个训练种子 × 2 个噪声水平。\n",
    ]

    noise_levels = [0.0] + NOISE_LEVELS
    verdict_lines = []

    for cname, (system, mode, ratio) in CONFIGS.items():
        print(f"\n-- {cname}: {system} / {mode} / ratio={ratio:.2f} --")
        print(f"{'noise':<8}{'model':<14}{'VPT mean±std':>20}{'n':>4}")
        out_lines.append(f"\n## {cname}: {system} / {mode} / ratio={ratio:.2f}\n")
        out_lines.append("| noise | model | VPT (mean ± std) | n |")
        out_lines.append("|---|---|---|---|")

        table = {}
        for noise in noise_levels:
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
                table[(noise, model)] = (float(v.mean()), float(v.std()), len(sub))
                print(f"{noise:<8.2f}{model:<14}{v.mean():>9.1f} ± {v.std():<7.1f}{len(sub):>4}")
                out_lines.append(
                    f"| {noise:.2f} | {model} | {v.mean():.1f} ± {v.std():.1f} | {len(sub)} |"
                )

        # 关键判断：加噪后 MLP 是否仍领先 Transformer
        for noise in NOISE_LEVELS:
            if (noise, "mlp") in table and (noise, "transformer") in table:
                mlp_m = table[(noise, "mlp")][0]
                tr_m = table[(noise, "transformer")][0]
                diff = mlp_m - tr_m
                noise_max = max(table[(noise, "mlp")][1], table[(noise, "transformer")][1])
                r = abs(diff) / noise_max if noise_max > 0 else float("inf")
                if diff > 0 and r > 3:
                    verdict = "稳健（MLP 领先）"
                elif diff > 0:
                    verdict = "优势不显著"
                elif r > 3:
                    verdict = "反转（Transformer 领先）"
                else:
                    verdict = "无差异"
                msg = (f"{cname} @ noise={noise:.2f}: MLP−Transf = {diff:+.1f} "
                       f"(noise {noise_max:.1f}, ratio {r:.1f}) -> {verdict}")
                print(f"   {msg}")
                verdict_lines.append((verdict, msg))

    print("\n" + "-" * 80)
    print("总结：加噪后结论是否保持")
    print("-" * 80)
    # 注意：这里必须用 startswith，用 endswith("稳健") 会把"不稳健"也数进去
    n_robust = sum(1 for verdict, _ in verdict_lines if verdict.startswith("稳健"))
    n_reversed = sum(1 for verdict, _ in verdict_lines if verdict.startswith("反转"))
    print(f"  共 {len(verdict_lines)} 个 (配置 × 噪声) 判断："
          f"{n_robust} 个 MLP 显著领先，{n_reversed} 个反转")
    out_lines.append("\n## 总结\n")
    out_lines.append(
        f"- {len(verdict_lines)} 个（配置 × 噪声）判断中，"
        f"**{n_robust} 个 MLP 显著领先**，{n_reversed} 个反转"
    )
    for verdict, v in verdict_lines:
        out_lines.append(f"- {v}")

    # ---- 附录：噪声对各模型绝对水平的打击，以及是否出现跨模型反转 ----
    out_lines.append("\n## 噪声对各模型绝对水平的影响\n")
    out_lines.append("| config | model | noise=0 | noise=0.05 | 保留比例 |")
    out_lines.append("|---|---|---|---|---|")
    print("\n" + "-" * 80)
    print("噪声对绝对水平的影响（VPT 保留比例）")
    print("-" * 80)
    for cname, (system, mode, ratio) in CONFIGS.items():
        for model in MODELS:
            sub0 = [r for r in rows if r["system"] == system and r["mode"] == mode
                    and same_ratio(r["ratio"], ratio) and r["model"] == model
                    and abs(r.get("noise_std", 0.0)) < 1e-9]
            sub5 = [r for r in rows if r["system"] == system and r["mode"] == mode
                    and same_ratio(r["ratio"], ratio) and r["model"] == model
                    and abs(r.get("noise_std", 0.0) - 0.05) < 1e-9]
            if sub0 and sub5:
                v0 = float(np.mean([r["vpt_mean"] for r in sub0]))
                v5 = float(np.mean([r["vpt_mean"] for r in sub5]))
                pct = v5 / v0 * 100 if v0 else float("nan")
                print(f"  {cname} {model:<12} {v0:6.1f} -> {v5:6.1f}  ({pct:5.1f}%)")
                out_lines.append(f"| {cname} | {model} | {v0:.1f} | {v5:.1f} | {pct:.0f}% |")

    # 跨模型反转检测：噪声下是否有非 MLP 模型在某个配置中反超
    out_lines.append("\n## 跨模型反转检测（同一配置、加噪后谁最好）\n")
    print("\n" + "-" * 80)
    print("跨模型反转检测（加噪后最优模型）")
    print("-" * 80)
    for cname, (system, mode, ratio) in CONFIGS.items():
        for noise in NOISE_LEVELS:
            best, best_v = None, -1.0
            for model in MODELS:
                sub = [r for r in rows if r["system"] == system and r["mode"] == mode
                       and same_ratio(r["ratio"], ratio) and r["model"] == model
                       and abs(r.get("noise_std", 0.0) - noise) < 1e-9]
                if not sub:
                    continue
                v = float(np.mean([r["vpt_mean"] for r in sub]))
                if v > best_v:
                    best, best_v = model, v
            if best and best != "mlp":
                print(f"  {cname} @ noise={noise:.2f}: 最优是 {best} (VPT={best_v:.1f})，MLP 不再是第一")
                out_lines.append(f"- **{cname} @ noise={noise:.2f}: {best} 反超 MLP** (VPT={best_v:.1f})")
            elif best:
                out_lines.append(f"- {cname} @ noise={noise:.2f}: MLP 仍最优 (VPT={best_v:.1f})")

    with open("noise_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print("\n已写入 noise_results.md")


def main() -> None:
    ap = argparse.ArgumentParser(description="含噪观测实验")
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
