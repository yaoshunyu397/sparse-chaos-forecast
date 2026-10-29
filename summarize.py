"""
summarize.py -- 汇总网格结果，输出表格与核心结论

把 runs/*.json（训练指标）与 evals/*.json（rollout 指标）合并成一张表，
并按"研究问题"的视角排序输出：

    研究问题：观测越稀疏，Transformer 相对 LSTM 的优势是扩大还是缩小？

输出三样东西：
    1. 终端表格：每个系统 x 掩码模式下，三模型在各观测比例的表现
    2. summary.csv：可直接粘进 Excel / README
    3. findings.md：自动生成的观察笔记（供你改写，不要直接照抄进邮件）

用法：
    python summarize.py
    python summarize.py --system lorenz63
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

MODELS = ["mlp", "lstm", "transformer"]
MODEL_LABEL = {"mlp": "MLP", "lstm": "LSTM", "transformer": "Transf"}


def load_all(runs_dir: str = "runs", eval_dir: str = "evals") -> List[Dict]:
    """合并训练结果与评估结果，按 run_id 关联。"""
    rows: Dict[str, Dict] = {}
    for p in glob.glob(os.path.join(runs_dir, "*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        rows[r["run_id"]] = {
            "run_id": r["run_id"],
            "system": r["system"],
            "model": r["model"],
            "mode": r["mode"],
            "ratio": r["ratio"],
            "n_params": r["n_params"],
            "test_mse": r["test_mse"],
            "best_val_mse": r["best_val_mse"],
            "elapsed_sec": r["elapsed_sec"],
        }
    for p in glob.glob(os.path.join(eval_dir, "*.json")):
        with open(p, encoding="utf-8") as f:
            e = json.load(f)
        rid = e["run_id"]
        if rid not in rows:
            continue
        rows[rid].update(
            {
                "vpt_mean": e["vpt_mean"],
                "vpt_median": e["vpt_median"],
                "short_rmse_10": e["short_rmse_10"],
                "long_rmse": e["long_rmse"],
                "growth_rate": e["growth_rate"],
            }
        )
    return sorted(rows.values(), key=lambda r: (r["system"], r["mode"], r["ratio"], r["model"]))


def fmt(v, nd=4, width=8) -> str:
    if v is None:
        return "-".rjust(width)
    return f"{v:.{nd}f}".rjust(width)


def print_tables(rows: List[Dict], system: str) -> None:
    sub = [r for r in rows if r["system"] == system]
    if not sub:
        return
    has_vpt = any("vpt_mean" in r for r in sub)

    print(f"\n{'='*78}\n{system}\n{'='*78}")

    for mode in ("none", "temporal", "uniform", "block", "random"):
        block = [r for r in sub if r["mode"] == mode]
        if not block:
            continue
        ratios = sorted({r["ratio"] for r in block})

        print(f"\n-- mode = {mode} --")
        header = f"{'ratio':<7}" + "".join(f"{MODEL_LABEL[m]:>12}" for m in MODELS)
        if has_vpt:
            print("  [短时精度 RMSE@10，越小越好]")
            print(header)
            for ratio in ratios:
                line = f"{ratio:<7.2f}"
                for m in MODELS:
                    hit = [r for r in block if r["model"] == m and abs(r["ratio"] - ratio) < 1e-9]
                    line += f"{fmt(hit[0].get('short_rmse_10') if hit else None):>12}"
                print(line)

            print("  [长期稳定性 VPT，越大越好]")
            print(header)
            for ratio in ratios:
                line = f"{ratio:<7.2f}"
                for m in MODELS:
                    hit = [r for r in block if r["model"] == m and abs(r["ratio"] - ratio) < 1e-9]
                    line += f"{fmt(hit[0].get('vpt_mean'), nd=1) if hit else '-':>12}"
                print(line)
        else:
            print("  [单步 test_mse，越小越好；尚未跑 evaluate.py]")
            print(header)
            for ratio in ratios:
                line = f"{ratio:<7.2f}"
                for m in MODELS:
                    hit = [r for r in block if r["model"] == m and abs(r["ratio"] - ratio) < 1e-9]
                    line += f"{fmt(hit[0]['test_mse'] if hit else None):>12}"
                print(line)


def write_csv(rows: List[Dict], path: str = "summary.csv") -> None:
    cols = [
        "system", "mode", "ratio", "model", "n_params",
        "test_mse", "short_rmse_10", "vpt_mean", "growth_rate",
    ]
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\n已写入 {path}")


def write_findings(rows: List[Dict], path: str = "findings.md") -> None:
    """
    自动生成观察笔记。注意：这些只是"读数"，结论必须你自己判断后改写。
    """
    lines = ["# 网格结果速览（自动生成，需人工改写）\n"]
    lines.append("> 本文件由 `summarize.py` 自动生成，只做数字罗列。")
    lines.append("> 写结论时必须自己判断，不要直接粘贴这些读数。\n")

    for system in sorted({r["system"] for r in rows}):
        sub = [r for r in rows if r["system"] == system]
        lines.append(f"\n## {system}\n")

        # 单步 MSE 最优
        best_mse = min(sub, key=lambda r: r["test_mse"])
        lines.append(
            f"- 单步 MSE 最优：**{best_mse['model']}** "
            f"(mode={best_mse['mode']}, ratio={best_mse['ratio']:.2f}) "
            f"= {best_mse['test_mse']:.6f}"
        )

        if not any("vpt_mean" in r for r in sub):
            lines.append("- VPT：尚未评估\n")
            continue

        best_vpt = max(sub, key=lambda r: r["vpt_mean"])
        lines.append(
            f"- 长期 VPT 最优：**{best_vpt['model']}** "
            f"(mode={best_vpt['mode']}, ratio={best_vpt['ratio']:.2f}) "
            f"= {best_vpt['vpt_mean']:.1f} 步"
        )

        # 关键对比：稀疏化时各模型 VPT 的变化
        # 每个系统各选一个主掩码模式（D 小的用 temporal，D 大的优先 uniform）
        modes_present = {r["mode"] for r in sub if r["mode"] != "none"}
        primary = next(
            (m for m in ("uniform", "temporal", "random", "block") if m in modes_present),
            None,
        )
        if primary is None:
            lines.append("- VPT：无稀疏化结果\n")
            continue
        lines.append(f"\n### 观测稀疏化下 VPT 的变化（{primary}）\n")
        lines.append("| 观测比例 | " + " | ".join(MODEL_LABEL[m] for m in MODELS) + " |")
        lines.append("|---|" + "---|" * len(MODELS))
        uni = [r for r in sub if r["mode"] == primary and "vpt_mean" in r]
        for ratio in sorted({r["ratio"] for r in uni}, reverse=True):
            cells = []
            for m in MODELS:
                hit = [r for r in uni if r["model"] == m and abs(r["ratio"] - ratio) < 1e-9]
                cells.append(f"{hit[0]['vpt_mean']:.1f}" if hit else "-")
            lines.append(f"| {ratio:.2f} | " + " | ".join(cells) + " |")

        # 分裂检测：短时最好但长期最差的模型
        lines.append("\n### 分裂检测（短时 vs 长期）\n")
        for ratio in sorted({r["ratio"] for r in uni}, reverse=True):
            hits = {m: next((r for r in uni if r["model"] == m and abs(r["ratio"] - ratio) < 1e-9), None)
                    for m in MODELS}
            hits = {k: v for k, v in hits.items() if v}
            if len(hits) < 2:
                continue
            best_short = min(hits, key=lambda k: hits[k]["short_rmse_10"])
            best_long = max(hits, key=lambda k: hits[k]["vpt_mean"])
            if best_short != best_long:
                lines.append(
                    f"- ratio={ratio:.2f}: **{best_short}** 短时最准 "
                    f"(RMSE@10={hits[best_short]['short_rmse_10']:.4f})，"
                    f"但长期最稳的是 **{best_long}** (VPT={hits[best_long]['vpt_mean']:.1f})"
                )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已写入 {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总网格结果")
    ap.add_argument("--system", default=None, help="只看某个系统")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--eval-dir", default="evals")
    args = ap.parse_args()

    rows = load_all(args.runs_dir, args.eval_dir)
    if not rows:
        print("没有结果。先跑 run_grid.py")
        return

    print(f"共 {len(rows)} 组结果（含评估的 {sum('vpt_mean' in r for r in rows)} 组）")

    systems = [args.system] if args.system else sorted({r["system"] for r in rows})
    for s in systems:
        print_tables(rows, s)

    write_csv(rows)
    write_findings(rows)


if __name__ == "__main__":
    main()
