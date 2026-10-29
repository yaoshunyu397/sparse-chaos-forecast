"""
run_grid.py -- 全量网格实验驱动（支持断点续跑）

为什么不直接 `train.py --grid`：
    1. 每完成一组立刻落盘，中途中断不丢进度；
    2. 已有结果默认跳过（--force 可强制重跑）；
    3. 输出带时间戳和进度，方便 tail 日志判断"卡住"还是"在跑"；
    4. 单组失败不终止整体。

⚠️ 重要设计修正（第一次网格暴露的问题）
    Lorenz-63 只有 D=3。维度掩码（uniform/block/random）在此系统上会塌缩：
    ratio=1/3 与 ratio=1/6 都只能保留 1 维，产生完全相同的掩码，
    实验变成自我重复 —— 表现为不同掩码/比例的 VPT 完全一致。
    因此按系统区分可用模式：

      D  小（lorenz63, D=3 ）：只用 temporal（时间维稀疏化）+ none
      D 大（lorenz96, D=40）：uniform/block/random（传感器空间布局）
                                + temporal + none

    temporal 的语义是"传感器采样频率降低"，在任意 D 上都能精确实现任意稀疏度，
    与空间掩码是两个独立、互补的稀疏维度。

实验矩阵：
    lorenz63: none + temporal(0.5, 1/3, 1/6)                          = 4
    lorenz96: none + uniform/block/random(0.5,1/3,1/6) + temporal(0.5,1/3,1/6) = 13
    两个系统 x 3 模型 -> (4 + 13) x 3 = 51 组

用法：
    python run_grid.py --epochs 30
    python run_grid.py --epochs 30 --systems lorenz63      # 只跑一个系统
    python run_grid.py --epochs 30 --force                 # 全量重跑
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import List, Tuple

from train import train_one

MODELS = ["mlp", "lstm", "transformer"]
SPATIAL_MODES = ["uniform", "block", "random"]
RATIOS = [0.5, 1 / 3, 1 / 6]

# 每个系统允许的稀疏化模式（原因见上方 docstring）
MODES_BY_SYSTEM = {
    "lorenz63": ["temporal"],                 # D=3，空间掩码会塌缩
    "lorenz96": SPATIAL_MODES + ["temporal"],  # D=40，两者都可用
}


def build_combos(systems: List[str]) -> List[Tuple[str, str, str, float]]:
    """生成实验矩阵：各系统一组 none 对照 + 各自允许的稀疏模式。"""
    combos: List[Tuple[str, str, str, float]] = []
    for sys_name, model in itertools.product(systems, MODELS):
        combos.append((sys_name, model, "none", 1.0))
        for mode in MODES_BY_SYSTEM[sys_name]:
            for ratio in RATIOS:
                combos.append((sys_name, model, mode, ratio))
    return combos


def run_id_of(s: str, m: str, mo: str, r: float) -> str:
    return f"{s}_{m}_{mo}_r{r:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="全量网格实验（断点续跑）")
    ap.add_argument("--systems", nargs="+", default=["lorenz63", "lorenz96"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--force", action="store_true", help="忽略已有结果，全部重跑")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    combos = build_combos(args.systems)

    # 预扫描：哪些已完成
    todo = []
    for c in combos:
        rid = run_id_of(*c)
        done_json = os.path.join(args.outdir, f"{rid}.json")
        done_pt = os.path.join(args.outdir, f"{rid}.pt")
        if not args.force and os.path.exists(done_json) and os.path.exists(done_pt):
            continue
        todo.append(c)

    print("=" * 72)
    print(f"实验矩阵: {len(combos)} 组 | 已完成: {len(combos)-len(todo)} | 待跑: {len(todo)}")
    print(f"epochs={args.epochs}  输出: {args.outdir}/")
    print("=" * 72, flush=True)

    if not todo:
        print("全部已完成。直接跑 python evaluate.py --all")
        return

    t_start = time.time()
    ok, failed = 0, []
    for i, (sys_name, model, mode, ratio) in enumerate(todo, 1):
        rid = run_id_of(sys_name, model, mode, ratio)
        t0 = time.time()
        print(f"\n[{i}/{len(todo)}] {rid}  (已用时 {time.time()-t_start:.0f}s)", flush=True)
        try:
            res = train_one(
                sys_name, model, mode, ratio,
                epochs=args.epochs, outdir=args.outdir, verbose=False,
            )
            ok += 1
            print(
                f"    完成 {time.time()-t0:.0f}s | test_mse={res['test_mse']:.5f} "
                f"| best_val={res['best_val_mse']:.5f}",
                flush=True,
            )
        except Exception as e:
            failed.append((rid, f"{type(e).__name__}: {e}"))
            print(f"    !! 失败: {type(e).__name__}: {e}", flush=True)

    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"网格结束: 成功 {ok} / 待跑 {len(todo)}，失败 {len(failed)}，总耗时 {elapsed/60:.1f} 分钟")
    for rid, err in failed:
        print(f"  失败: {rid} -> {err}")
    print("=" * 72)
    print("\n下一步: python evaluate.py --all")


if __name__ == "__main__":
    main()
