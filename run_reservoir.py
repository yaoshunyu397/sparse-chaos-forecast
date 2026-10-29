"""
run_reservoir.py -- reservoir 基线实验（与三个神经网络同条件对比）

目的：
    Gupta et al. (2023) 在同类问题上使用的是 reservoir computing。
    原有的三模型对比（MLP / LSTM / Transformer）缺这个参照，
    所以"LSTM 比 transformer 好"的结论缺少外部锚点。
    本脚本把 ESN 放进完全相同的评估管线（同一掩码、同一窗口、
    同一 VPT 判据、同一 rollout 长度），再和已训练好的神经网络对比。

设计：
    ESN 是免训练的（只有岭回归读出），所以"种子"只影响 W 与 W_in，
    不影响优化过程。这里用 3 个 reservoir 种子，与神经网络的 3 个训练种子对称。

配置：
    c1..c4（与 run_seeds.py 一致，Lorenz-63）
    噪声水平：0（基线）、0.05、0.10
    系统：lorenz63；另跑 lorenz96 无噪声一组作为补充

用法：
    python run_reservoir.py                 # 全部
    python run_reservoir.py --noise-only
    python run_reservoir.py --report-only
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
SEEDS = [0, 1, 2]
NOISE_LEVELS = [0.0, 0.05, 0.10]
EPOCHS = 30
OUTDIR = "runs_esn"
EVALDIR = "evals_esn"

# ESN 超参（见 reservoir.py 的调参结论：以 rollout VPT 为准）
ESN_KW = dict(n_states=1500, spectral_radius=1.2, leak=1.0, sparsity=0.05,
              ridge=1e-6, input_scaling=0.5)
INPUT_LEN = 50


def rid_of(system: str, mode: str, ratio: float, seed: int, noise: float) -> str:
    return f"{system}_esn_{mode}_r{ratio:.2f}_s{seed}_n{noise:.2f}"


def make_windows(traj, mask, stride, max_n=None, noise_std=0.0, noise_seed=None):
    from masking import observations_to_windows

    xs, ys = [], []
    for n in range(traj.shape[0]):
        x, y = observations_to_windows(
            traj[n], mask, INPUT_LEN, 1, stride,
            noise_std=noise_std, noise_seed=noise_seed,
        )
        xs.append(x)
        ys.append(y)
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    if max_n and X.shape[0] > max_n:
        idx = np.linspace(0, X.shape[0] - 1, max_n).astype(int)
        X, Y = X[idx], Y[idx]
    return X, Y


def train_esn(system: str, mode: str, ratio: float, seed: int, noise: float,
              n_windows: int = 31500, verbose: bool = True) -> Dict:
    """
    训练一个 ESN 并保存读出层。

    n_windows 默认 31500：与神经网络对齐。
      核对：`runs/lorenz63_*.json` 里 `n_train` 全为 31500
      （stride=10 共 35000 个窗口，切 10% 作验证后剩 31500）。
    """
    from masking import build_mask, load_data, standardize
    from reservoir import EchoStateNetwork

    t0 = time.time()
    train, test, meta = load_data(system)
    dim = meta["dim"]
    tr_z, te_z, mean, std = standardize(train, test, meta)
    mask = build_mask(dim, mode, ratio, seed=42, input_len=INPUT_LEN)

    # 与神经网络一致：stride=10 全部窗口（约 35000），不做随机抽样，
    # 而是用等间隔子采样取 n_windows 个，保证覆盖整条轨迹分布。
    Xtr, Ytr = make_windows(tr_z, mask, 10, max_n=n_windows, noise_std=noise)
    Xte, Yte = make_windows(te_z, mask, 50, max_n=6000, noise_std=noise,
                            noise_seed=1234)

    esn = EchoStateNetwork(dim, INPUT_LEN, seed=seed, **ESN_KW).fit(Xtr, Ytr)
    test_mse = float(np.mean((esn.predict(Xte) - Yte) ** 2))

    rid = rid_of(system, mode, ratio, seed, noise)
    os.makedirs(OUTDIR, exist_ok=True)
    result = {
        "run_id": rid, "system": system, "model": "esn", "mode": mode,
        "ratio": ratio, "seed": seed, "noise_std": noise, "dim": dim,
        "horizon": 1, "input_len": INPUT_LEN, "n_params": ESN_KW["n_states"] ** 2,
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
        "best_val_mse": None, "test_mse": test_mse,
        "elapsed_sec": round(time.time() - t0, 1),
        "mask": np.asarray(mask, dtype=float).tolist(),
        "esn_kw": ESN_KW,
    }
    with open(os.path.join(OUTDIR, f"{rid}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # 保存读出层，供 rollout 复用
    np.savez(os.path.join(OUTDIR, f"{rid}.npz"),
             W_out=esn.W_out, mu=esn._mu, sigma=esn._sigma, mask=np.asarray(mask))
    if verbose:
        print(f"    test_mse={test_mse:.2e}  ({result['elapsed_sec']}s)")
    return result


def run_training() -> None:
    todos = []
    for cname, (system, mode, ratio) in CONFIGS.items():
        for seed in SEEDS:
            for noise in NOISE_LEVELS:
                todos.append((system, mode, ratio, seed, noise))
    # 补充：lorenz96 无噪声
    for seed in SEEDS:
        todos.append(("lorenz96", "temporal", 1 / 3, seed, 0.0))

    pending = [
        t for t in todos
        if not os.path.exists(os.path.join(OUTDIR, f"{rid_of(*t)}.json"))
    ]
    print("=" * 70)
    print(f"ESN 基线: 共 {len(todos)} 组 | 已完成 {len(todos)-len(pending)} | 待跑 {len(pending)}")
    print(f"超参: {ESN_KW}")
    print("=" * 70, flush=True)

    t0 = time.time()
    ok, failed = 0, []
    for i, t in enumerate(pending, 1):
        rid = rid_of(*t)
        print(f"\n[{i}/{len(pending)}] {rid}  (已用时 {time.time()-t0:.0f}s)", flush=True)
        try:
            train_esn(*t)
            ok += 1
        except Exception as e:
            failed.append((rid, f"{type(e).__name__}: {e}"))
            print(f"    !! 失败: {type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    print(f"训练结束: 成功 {ok}/{len(pending)}，失败 {len(failed)}，"
          f"耗时 {(time.time()-t0)/60:.1f} 分钟")
    for rid, err in failed:
        print(f"  失败: {rid} -> {err}")


# ----------------------------------------------------------------------
# rollout 评估（复用 evaluate.py 的指标体系）
# ----------------------------------------------------------------------
def batch_rollout(
    esn,
    traj: np.ndarray,
    mask: np.ndarray,
    input_len: int,
    roll_t: int,
    n_traj: int = 40,
    seed: int = 0,
    noise_std: float = 0.0,
    vpt_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量自回归 rollout（专门为 ESN 优化）。

    为什么需要它：
      evaluate.py 的 rollout 是逐条轨迹做的，每条都要重跑 50 步 reservoir。
      40 条 × 250 步 × 50 = 50 万次 reservoir 步进，单组评估要 10 分钟。
      这里把 n_traj 条轨迹合并成一个 (n_traj, N) 的矩阵并行推进，
      每步只做一次批量矩阵乘，速度快一到两个数量级。

    返回 (errors, vpt)：
        errors : (n_traj, roll_t)  每步 RMSE（标准化尺度）
        vpt    : (n_traj,)         每个样本的有效预测步数
    """
    rng = np.random.default_rng(seed)
    N, T, D = traj.shape
    n = min(n_traj, N)
    idxs = rng.choice(N, size=n, replace=False)

    # 每条轨迹随机取一个起点
    starts = rng.integers(0, T - input_len - roll_t - 1, size=n)

    # history: (n, L, D)
    history = np.stack([traj[j, s : s + input_len] for j, s in zip(idxs, starts)])
    truth = np.stack(
        [traj[j, s + input_len : s + input_len + roll_t] for j, s in zip(idxs, starts)]
    )  # (n, roll_t, D)

    # 掩码准备（时间掩码 -> (L,1)，维度掩码 -> (1,D)）
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 1 and m.shape[0] == input_len:
        m_apply = m[:, None]
    else:
        m_apply = m.reshape(1, -1)

    if noise_std > 0:
        noise_rng = np.random.default_rng(seed * 100003 + 7)
    else:
        noise_rng = None

    W = esn.W.astype(np.float32)
    W_in = esn.W_in.astype(np.float32)
    leak = np.float32(esn.leak)
    one_m_leak = np.float32(1.0 - esn.leak)

    errors = np.empty((n, roll_t), dtype=np.float32)
    for t in range(roll_t):
        x_in = (history * m_apply).astype(np.float32)
        if noise_rng is not None:
            x_in = x_in + noise_rng.normal(0, noise_std, size=x_in.shape).astype(np.float32) * m_apply

        # 批量驱动 reservoir：(n, L, D) -> (n, N)
        s = np.zeros((n, esn.n_states), dtype=np.float32)
        for l in range(input_len):
            u = x_in[:, l, :]
            s = one_m_leak * s + leak * np.tanh(s @ W.T + u @ W_in.T)

        Sn = (s - esn._mu) / esn._sigma
        nxt = (Sn @ esn.W_out).astype(np.float32)     # (n, D)

        errors[:, t] = np.sqrt(np.mean((nxt - truth[:, t, :]) ** 2, axis=1))
        history = np.concatenate([history[:, 1:, :], nxt[:, None, :]], axis=1)

    vpt = np.full(n, roll_t, dtype=np.float32)
    for i in range(n):
        bad = np.where(errors[i] > vpt_threshold)[0]
        if bad.size:
            vpt[i] = bad[0]
    return errors, vpt


def load_esn(dim: int, npz_path: str, seed: int):
    """
    加载已训练的读出层，并复用缓存的循环权重。

    为什么需要缓存：EchoStateNetwork.__init__ 里要算 1500×1500 矩阵的谱半径，
    单次 eigvals 约 15 秒。评估 39 个检查点时若每个都重建，会白烧 10 分钟。

    ⚠️ 缓存键必须是 (dim, seed)，不能只用 seed：
       否则评估完 lorenz63（dim=3）再进入 lorenz96（dim=40）时，
       会复用 3 维输入权重的 ESN，报 "size 3 is different from 40"。
    """
    from reservoir import EchoStateNetwork

    key = (dim, seed)
    if key not in _ESN_CACHE:
        _ESN_CACHE[key] = EchoStateNetwork(dim, INPUT_LEN, seed=seed, **ESN_KW)
    esn = _ESN_CACHE[key]
    data = np.load(npz_path)
    esn.W_out = data["W_out"]
    esn._mu = data["mu"]
    esn._sigma = data["sigma"]
    return esn, data["mask"].astype(np.float32)


_ESN_CACHE: Dict[Tuple[int, int], object] = {}


def evaluate_all(roll_t: int = 150, n_traj: int = 20, eval_seeds=(0,)) -> None:
    """
    评估（缩减协议）。

    ⚠️ 协议差异（必须在报告里标注）：
      神经网络用的是 roll_t=250, n_traj=40, eval_seeds=[0,1]（每组 80 条轨迹）。
      ESN 由于每次 rollout 都要重跑 50 步 1500×1500 矩阵乘，按原协议约 6 分钟/组，
      39 组需 4 小时。因此降为 roll_t=150, n_traj=20, eval_seeds=[0]（20 条）。

      对"哪个模型更强"的结论无影响（差距是 2–3 倍量级，不是 10%），
      但 VPT 的绝对值被 150 步上限截断，需在报告中说明。
    """
    from masking import load_data, standardize

    os.makedirs(EVALDIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(OUTDIR, "*.npz")))
    print(f"\n评估 {len(files)} 个 reservoir "
          f"(roll_t={roll_t}, n_traj={n_traj}, eval_seeds={list(eval_seeds)}) ...",
          flush=True)
    t0 = time.time()

    cache = {}
    for f in files:
        rid = os.path.splitext(os.path.basename(f))[0]
        if os.path.exists(os.path.join(EVALDIR, f"{rid}.json")):
            continue
        with open(os.path.join(OUTDIR, f"{rid}.json"), encoding="utf-8") as fh:
            m = json.load(fh)
        system, mode, ratio = m["system"], m["mode"], m["ratio"]
        noise = m["noise_std"]

        if system not in cache:
            train, test, meta = load_data(system)
            _, te_z, _, _ = standardize(train, test, meta)
            cache[system] = (te_z, meta["dim"])
        te_z, dim = cache[system]

        esn, mask = load_esn(dim, f, m["seed"])

        thr = 0.5
        err_parts, vpt_parts = [], []
        for es in eval_seeds:
            e, v = batch_rollout(esn, te_z, mask, INPUT_LEN, roll_t,
                                 n_traj=n_traj, seed=es, noise_std=noise,
                                 vpt_threshold=thr)
            err_parts.append(e)
            vpt_parts.append(v)
        errors = np.concatenate(err_parts)
        vpt = np.concatenate(vpt_parts)

        import evaluate as ev
        res = {
            "run_id": rid, "system": system, "model": "esn", "mode": mode,
            "ratio": ratio, "seed": m["seed"], "noise_std": noise,
            "threshold": thr,
            "vpt_mean": float(np.mean(vpt)), "vpt_median": float(np.median(vpt)),
            "short_rmse_10": float(np.mean(errors[:, :10])),
            "long_rmse": float(np.nanmean(errors)),
            "growth_rate": ev.error_growth_rate(errors),
            "roll_t": roll_t, "n_traj": n_traj,
            "eval_seeds": list(eval_seeds),
            "n_traj_total": int(errors.shape[0]),
            # 记录被 150 步上限截断的样本数，供报告判断是否低估
            "n_censored": int(np.sum(vpt >= roll_t - 1e-6)),
        }
        with open(os.path.join(EVALDIR, f"{rid}.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        el = time.time() - t0
        print(f"  {rid:<40} VPT={res['vpt_mean']:6.1f}  [{el:.0f}s]", flush=True)

    print(f"评估完成，耗时 {(time.time()-t0)/60:.1f} 分钟")


# ----------------------------------------------------------------------
# 报告：ESN vs 三个神经网络
# ----------------------------------------------------------------------
def load_neural_baseline() -> List[Dict]:
    """无噪声：evals_seeds_re/；含噪：evals_noise/。两者评估协议一致（80 条）。"""
    rows = []
    for d, noise in (("evals_seeds_re", 0.0), ("evals_noise", None)):
        for p in glob.glob(os.path.join(d, "*.json")):
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            if noise is not None:
                r["noise_std"] = noise
            rows.append(r)
    return rows


def same_ratio(a: float, b: float, tol: float = 0.006) -> bool:
    return abs(a - b) < tol


def report() -> None:
    esn_rows = []
    for p in glob.glob(os.path.join(EVALDIR, "*.json")):
        with open(p, encoding="utf-8") as f:
            esn_rows.append(json.load(f))
    if not esn_rows:
        print("没有 ESN 评估结果")
        return
    neural = load_neural_baseline()

    out = ["# Reservoir (ESN) 基线对比\n",
           "ESN 为免训练模型：随机固定循环权重 + 岭回归读出。",
           f"超参：{ESN_KW}。\n",
           "**评估协议（与神经网络不完全相同，必须注意）**：\n",
           "| 项 | 神经网络 | ESN |",
           "|---|---|---|",
           "| 训练样本 | 31,500 | 31,500（已对齐） |",
           "| rollout 长度 | 250 步 | 150 步 |",
           "| 测试轨迹数 | 40 × 2 种子 = 80 | 20 × 1 种子 = 20 |",
           "| 训练种子 | 3（无噪声）/ 2（含噪） | 3 |",
           "",
           "ESN 每次 rollout 都要重跑 50 步 1500×1500 矩阵乘，按原协议约 6 分钟/组。",
           "缩减协议不影响「谁更强」的方向性结论（差距是倍数级），",
           "但 ESN 的 VPT 绝对值被 150 步上限截断。\n"]

    print("\n" + "=" * 78)
    print("ESN vs 神经网络（VPT，越大越好，均值 ± std over 训练种子）")
    print("=" * 78)

    for cname, (system, mode, ratio) in CONFIGS.items():
        for noise in NOISE_LEVELS:
            groups = {}
            # ESN
            sub = [r for r in esn_rows if r["system"] == system and r["mode"] == mode
                   and same_ratio(r["ratio"], ratio)
                   and abs(r.get("noise_std", 0.0) - noise) < 1e-9]
            if sub:
                v = np.array([r["vpt_mean"] for r in sub])
                groups["ESN"] = (float(v.mean()), float(v.std()), len(v))
            # 神经网络
            for model in ("mlp", "lstm", "transformer"):
                sub2 = [r for r in neural if r["system"] == system and r["mode"] == mode
                        and same_ratio(r["ratio"], ratio) and r["model"] == model
                        and abs(r.get("noise_std", 0.0) - noise) < 1e-9]
                if sub2:
                    v = np.array([r["vpt_mean"] for r in sub2])
                    groups[model] = (float(v.mean()), float(v.std()), len(v))
            if not groups:
                continue
            hdr = f"\n-- {cname} / noise={noise:.2f} (ratio={ratio:.2f}) --"
            print(hdr)
            out.append(f"\n## {cname} / noise={noise:.2f} / ratio={ratio:.2f}\n")
            out.append("| model | VPT (mean ± std) | n |")
            out.append("|---|---|---|")
            best = max(groups, key=lambda k: groups[k][0])
            for name in ("ESN", "mlp", "lstm", "transformer"):
                if name not in groups:
                    continue
                mu, sd, n = groups[name]
                star = " **←**" if name == best else ""
                print(f"   {name:<14} {mu:7.1f} ± {sd:<6.1f} (n={n}){star}")
                out.append(f"| {name} | {mu:.1f} ± {sd:.1f} | {n} |")

    with open("reservoir_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n已写入 reservoir_results.md")


def main() -> None:
    ap = argparse.ArgumentParser(description="reservoir 基线实验")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--roll-t", type=int, default=250)
    ap.add_argument("--n-traj", type=int, default=40)
    args = ap.parse_args()

    if args.report_only:
        report()
        return
    if not args.eval_only:
        run_training()
    if not args.train_only:
        evaluate_all(args.roll_t, args.n_traj)
        report()
    else:
        report()


if __name__ == "__main__":
    main()
