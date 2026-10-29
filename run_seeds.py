"""
run_seeds.py -- 多种子重复实验（稳健性检验）

为什么必须做：
    主网格的每个配置只训了一次。如果 VPT 的种子间波动很大，
    "MLP 80 步 vs Transformer 31 步"这种差距可能只是噪声，
    任何结论都站不住。任何审稿人都会先问"这个差多少是噪声"。

设计：
    选 4 个关键配置（覆盖"全观测"锚点 + 三档稀疏度），
    三个模型各用 3 个种子重训 -> 4 x 3 x 3 = 36 组。

    ⚠️ 种子只影响训练（初始化 + 数据顺序），掩码始终固定 seed=42，
       否则不同种子之间连观测集合都不同，就不是可比的重复实验了。

用法：
    python run_seeds.py                 # 训练全部 36 组
    python run_seeds.py --eval-only     # 只评估已训练的
    python run_seeds.py --configs c1 c2 # 只跑部分配置
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np

# 关键配置：全观测锚点 + 三档稀疏度（Lorenz-63 的 temporal 主线）
CONFIGS: Dict[str, Tuple[str, str, float]] = {
    "c1": ("lorenz63", "none", 1.0),
    "c2": ("lorenz63", "temporal", 0.5),
    "c3": ("lorenz63", "temporal", 1 / 3),
    "c4": ("lorenz63", "temporal", 1 / 6),
}

MODELS = ["mlp", "lstm", "transformer"]
SEEDS = [0, 1, 2]
EPOCHS = 30
OUTDIR = "runs_seeds"
EVALDIR = "evals_seeds"
# 统一评估协议（eval_seeds=[0,1] -> 80 条轨迹），与含噪实验可比
EVALDIR_RE = "evals_seeds_re"


def run_id_of(system: str, model: str, mode: str, ratio: float, seed: int) -> str:
    return f"{system}_{model}_{mode}_r{ratio:.2f}_s{seed}"


def same_ratio(a: float, b: float, tol: float = 0.006) -> bool:
    """
    比较两个观测比例是否同一档。

    为什么不用 1e-9：run_id 里写的是两位小数（r0.33），
    而 CONFIGS 里写的是 1/3 = 0.3333...，二者相差 0.0033。
    严格的浮点相等会把它们判成不同配置 —— 这正是"36 组只统计出 2 组"的原因。
    容差取 0.006（小于相邻档位间距的一半：0.5→0.33 间距 0.167）。
    """
    return abs(a - b) < tol


def train_all(configs: List[str]) -> None:
    from train import train_one

    todos: List[Tuple[str, str, str, float, int]] = []
    for cname in configs:
        system, mode, ratio = CONFIGS[cname]
        for model in MODELS:
            for seed in SEEDS:
                todos.append((system, model, mode, ratio, seed))

    # 断点续跑
    pending = []
    for system, model, mode, ratio, seed in todos:
        rid = run_id_of(system, model, mode, ratio, seed)
        if os.path.exists(os.path.join(OUTDIR, f"{rid}.json")) and os.path.exists(
            os.path.join(OUTDIR, f"{rid}.pt")
        ):
            continue
        pending.append((system, model, mode, ratio, seed))

    print("=" * 70)
    print(f"多种子实验: 共 {len(todos)} 组 | 已完成 {len(todos)-len(pending)} | 待跑 {len(pending)}")
    print(f"配置: {configs} | 种子: {SEEDS} | epochs={EPOCHS}")
    print("=" * 70, flush=True)

    if not pending:
        print("训练部分已全部完成。")
        return

    t0 = time.time()
    ok, failed = 0, []
    for i, (system, model, mode, ratio, seed) in enumerate(pending, 1):
        rid = run_id_of(system, model, mode, ratio, seed)
        print(f"\n[{i}/{len(pending)}] {rid}  (已用时 {time.time()-t0:.0f}s)", flush=True)
        try:
            res = train_one(
                system, model, mode, ratio,
                epochs=EPOCHS, outdir=OUTDIR, verbose=False, seed=seed,
            )
            ok += 1
            print(f"    test_mse={res['test_mse']:.6f}", flush=True)
        except Exception as e:
            failed.append((rid, f"{type(e).__name__}: {e}"))
            print(f"    !! 失败: {type(e).__name__}: {e}", flush=True)

    print("\n" + "=" * 70)
    print(f"训练结束: 成功 {ok}/{len(pending)}，失败 {len(failed)}，"
          f"耗时 {(time.time()-t0)/60:.1f} 分钟")
    for rid, err in failed:
        print(f"  失败: {rid} -> {err}")
    print("=" * 70)


def evaluate_all(roll_t: int = 250, n_traj: int = 40) -> None:
    import evaluate

    os.makedirs(EVALDIR, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(OUTDIR, "*.pt")))
    if not ckpts:
        print(f"{OUTDIR}/ 下没有检查点，先跑训练")
        return

    print(f"\n评估 {len(ckpts)} 个检查点 ...", flush=True)
    t0 = time.time()
    for i, c in enumerate(ckpts, 1):
        rid = os.path.splitext(os.path.basename(c))[0]
        out_json = os.path.join(EVALDIR, f"{rid}.json")
        if os.path.exists(out_json):
            continue
        try:
            # 多个评估种子 -> 方差同时包含训练噪声与评估抽样噪声
            evaluate.evaluate_run(
                c,
                roll_t=roll_t,
                n_traj=n_traj,
                save_plot=False,
                eval_dir=EVALDIR,
                eval_seeds=[0, 1, 2],
            )
        except Exception as e:
            print(f"  !! {rid}: {type(e).__name__}: {e}", flush=True)
    print(f"评估完成，耗时 {(time.time()-t0)/60:.1f} 分钟")


def report() -> None:
    """
    按配置汇总 mean ± std，输出可直接引用的表格。

    优先读取 EVALDIR_RE（统一协议：eval_seeds=[0,1] -> 80 条轨迹），
    回退到 EVALDIR（旧协议：40 条）。两者数字会不同，必须明确用哪个。
    """
    rows = []
    src_dir = None
    for d in (EVALDIR_RE, EVALDIR):
        paths = sorted(glob.glob(os.path.join(d, "*.json")))
        if paths:
            src_dir = d
            for p in paths:
                with open(p, encoding="utf-8") as f:
                    rows.append(json.load(f))
            break
    if not rows:
        print("没有评估结果")
        return
    print(f"（结果来源: {src_dir}/ ）")

    # 读取训练 json 补充 test_mse
    for r in rows:
        tj = os.path.join(OUTDIR, f"{r['run_id']}.json")
        if os.path.exists(tj):
            with open(tj, encoding="utf-8") as f:
                r["test_mse"] = json.load(f)["test_mse"]

    print("\n" + "=" * 78)
    print("多种子重复结果（mean ± std over seeds）")
    print("=" * 78)

    out_lines = ["# 多种子重复结果\n",
                 "同一配置训练 3 个随机种子（掩码固定 seed=42，保证观测集合相同）。\n",
                 "评估时每个模型用 3 个评估种子各抽 40 条测试轨迹（共 120 条），",
                 "因此下表的 std 同时包含训练随机性与评估抽样随机性。\n"]

    # 完整性校验：每个配置都必须有结果，否则明确报错而不是静默跳过
    missing = []
    for cname, (system, mode, ratio) in CONFIGS.items():
        sub = [r for r in rows if r["system"] == system and r["mode"] == mode
               and same_ratio(r["ratio"], ratio)]
        if len(sub) < 6:
            missing.append((cname, len(sub)))
    if missing:
        print("\n!! 警告：以下配置的结果不足（预期每种 9 组 = 3 模型 x 3 种子）：")
        for cname, n in missing:
            print(f"   {cname}: 仅找到 {n} 组")
        print("   可能是评估未跑完，或 ratio 精度不匹配。\n")

    for cname, (system, mode, ratio) in CONFIGS.items():
        sub = [r for r in rows if r["system"] == system and r["mode"] == mode
               and same_ratio(r["ratio"], ratio)]
        if not sub:
            print(f"\n-- {cname}: 无结果，跳过 --")
            continue
        print(f"\n-- {cname}: {system} / {mode} / ratio={ratio:.2f} --")
        print(f"{'model':<14}{'VPT mean±std':>20}{'RMSE@10':>16}{'test_mse':>14}{'n':>4}")
        out_lines.append(f"\n## {cname}: {system} / {mode} / ratio={ratio:.2f}\n")
        out_lines.append("| model | VPT (mean ± std) | RMSE@10 | test_mse | n |")
        out_lines.append("|---|---|---|---|---|")

        for model in MODELS:
            ms = [r for r in sub if r["model"] == model]
            if not ms:
                continue
            vpt = np.array([r["vpt_mean"] for r in ms])
            rmse = np.array([r["short_rmse_10"] for r in ms])
            mse = np.array([r.get("test_mse", np.nan) for r in ms])
            # 变异系数：波动相对均值的比例，用来判断"能不能下结论"
            cv = vpt.std() / max(vpt.mean(), 1e-9)
            flag = "  <-- 波动大" if cv > 0.2 else ""
            print(
                f"{model:<14}{vpt.mean():>8.1f} ± {vpt.std():<7.1f}"
                f"{rmse.mean():>16.4f}{mse.mean():>14.5f}{len(ms):>4}{flag}"
            )
            out_lines.append(
                f"| {model} | {vpt.mean():.1f} ± {vpt.std():.1f} | "
                f"{rmse.mean():.4f} | {mse.mean():.5f} | {len(ms)} |"
            )

    # 关键判断：差距是否远大于噪声
    print("\n" + "-" * 78)
    print("稳健性判断（差距 / 噪声 比值 > 3 才算稳健）")
    print("-" * 78)
    out_lines.append("\n## 稳健性判断\n")

    for cname, (system, mode, ratio) in CONFIGS.items():
        sub = [r for r in rows if r["system"] == system and r["mode"] == mode
               and same_ratio(r["ratio"], ratio)]
        if len(sub) < 6:
            continue
        means = {}
        stds = {}
        for model in MODELS:
            ms = [r for r in sub if r["model"] == model]
            if ms:
                v = np.array([r["vpt_mean"] for r in ms])
                means[model], stds[model] = float(v.mean()), float(v.std())
        if "mlp" in means and "transformer" in means:
            diff = means["mlp"] - means["transformer"]
            noise = max(stds["mlp"], stds["transformer"])
            ratio_v = diff / noise if noise > 0 else float("inf")
            verdict = "稳健" if ratio_v > 3 else "不稳健，需更多种子"
            print(f"{cname}: MLP - Transformer = {diff:+.1f} 步，"
                  f"噪声(取大者) = {noise:.1f}，比值 = {ratio_v:.1f} -> {verdict}")
            out_lines.append(
                f"- {cname}: MLP − Transformer = {diff:+.1f} (noise {noise:.1f}, "
                f"ratio {ratio_v:.1f}) → **{verdict}**"
            )

    with open("seed_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print("\n已写入 seed_results.md")


def main() -> None:
    ap = argparse.ArgumentParser(description="多种子重复实验")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                    choices=list(CONFIGS.keys()))
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--roll-t", type=int, default=250)
    ap.add_argument("--n-traj", type=int, default=40)
    args = ap.parse_args()

    if args.report_only:
        report()
        return
    if not args.eval_only:
        train_all(args.configs)
    evaluate_all(args.roll_t, args.n_traj)
    report()


if __name__ == "__main__":
    main()
