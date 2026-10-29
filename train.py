"""
train.py -- 训练 MLP / LSTM / Transformer 做混沌系统预测

任务定义（与 masking.py 保持一致）：
    input  = 被 mask 过的过去 L 步           (B, L, D)
    target = 未来 H 步的完整状态（从不 mask）  (B, H, D)

三个模型的意义：
    mlp         : 零基线。证明"深度学习不是随便就能用"。
    lstm        : 门控循环网络的代表，也是该任务最常见的基线之一。
    transformer : 注意力机制的代表，近年时序预测的默认选择之一。

超参在所有模型间保持统一 —— 不要给某个模型单独调参，
否则"谁更好"的结论不成立。

用法：
    # 冒烟测试（几十秒）
    python train.py --system lorenz63 --model lstm --epochs 3 --n-windows 2000

    # 正式训练（单次）
    python train.py --system lorenz63 --model transformer --mode uniform --ratio 0.5

    # 批量：全部 模型 x 系统 x 掩码 组合
    python train.py --grid --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from masking import (
    MASK_MODES,
    build_mask,
    load_data,
    observations_to_windows,
    standardize,
)

# --------------------------------------------------------------------------
# 全局设定
# --------------------------------------------------------------------------
INPUT_LEN = 50      # 输入窗口长度（约覆盖 Lorenz 的一个特征时间）
HORIZON = 1         # 默认单步；设为 10 则做直接多步预测
SEED = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# 数据集构建
# --------------------------------------------------------------------------
def build_windows(
    traj: np.ndarray,
    mask: np.ndarray,
    input_len: int,
    horizon: int,
    stride: int,
    max_windows: int | None = None,
    rng: np.random.Generator | None = None,
    noise_std: float = 0.0,
    noise_seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    traj: (N, T, D) 标准化后的轨迹。返回 (X, Y)，形状 (M, L, D) 和 (M, H, D)。

    max_windows 限定总数（超大时随机采样），用于控制训练时间。
    noise_std > 0 时给观测加高斯噪声（只加在被观测到的位置）。
    """
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for n in range(traj.shape[0]):
        x, y = observations_to_windows(
            traj[n], mask, input_len=input_len, horizon=horizon, stride=stride,
            noise_std=noise_std, noise_seed=noise_seed,
        )
        xs.append(x)
        ys.append(y)
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)

    if max_windows is not None and X.shape[0] > max_windows:
        rng = rng or np.random.default_rng(SEED)
        idx = rng.choice(X.shape[0], size=max_windows, replace=False)
        idx.sort()
        X, Y = X[idx], Y[idx]
    return X, Y


def make_tensors(X: np.ndarray, Y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32)),
    )


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
class MLP(nn.Module):
    """零基线：把 (B, L, D) 拉平后过 MLP，直接输出 (B, H*D)。"""

    def __init__(self, input_len: int, dim: int, horizon: int, hidden: int = 256):
        super().__init__()
        self.horizon = horizon
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(input_len * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon * dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        out = self.net(x.reshape(b, -1))
        return out.reshape(b, self.horizon, self.dim)


class LSTM(nn.Module):
    """2 层 LSTM，取最后时刻隐状态做预测头。"""

    def __init__(
        self, dim: int, horizon: int, hidden: int = 128, layers: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        self.horizon = horizon
        self.dim = dim
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, horizon * dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (B, L, hidden)
        h = out[:, -1]                  # 取最后一步
        b = x.shape[0]
        return self.head(h).reshape(b, self.horizon, self.dim)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class Transformer(nn.Module):
    """
    小 Transformer encoder：4 层、d=128、4 头。
    刻意保持小 —— 3 维系统用大模型只会过拟合。
    """

    def __init__(
        self,
        dim: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        ff: int = 256,
        dropout: float = 0.1,
        input_len: int = 50,
    ):
        super().__init__()
        self.horizon = horizon
        self.dim = dim
        self.inp = nn.Linear(dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=input_len + 8)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.inp(x))
        z = self.encoder(z)
        h = self.norm(z[:, -1])   # 用最后一步的表示做预测
        b = x.shape[0]
        return self.head(h).reshape(b, self.horizon, self.dim)


def build_model(name: str, dim: int, horizon: int, input_len: int) -> nn.Module:
    if name == "mlp":
        return MLP(input_len, dim, horizon)
    if name == "lstm":
        return LSTM(dim, horizon)
    if name == "transformer":
        return Transformer(dim, horizon, input_len=input_len)
    raise ValueError(f"unknown model: {name!r}")


# --------------------------------------------------------------------------
# 训练 / 评估
# --------------------------------------------------------------------------
def evaluate_loss(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, bs: int = 512) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            xb = X[i : i + bs].to(DEVICE)
            yb = Y[i : i + bs].to(DEVICE)
            loss = nn.functional.mse_loss(model(xb), yb, reduction="sum")
            total += float(loss)
            n += yb.numel()
    return total / max(n, 1)


def train_one(
    system: str,
    model_name: str,
    mode: str,
    ratio: float,
    horizon: int = HORIZON,
    input_len: int = INPUT_LEN,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 8,
    n_windows: int = 120_000,
    outdir: str = "runs",
    verbose: bool = True,
    seed: int = SEED,
    noise_std: float = 0.0,
) -> Dict:
    set_seed(seed)
    train, test, meta = load_data(system)
    dim = meta["dim"]
    tr_z, te_z, mean, std = standardize(train, test, meta)

    mask = build_mask(dim, mode, ratio, seed=42, input_len=input_len)

    rng = np.random.default_rng(seed)
    # 训练：跨轨迹 stride=10 降采样；测试：stride=25 进一步降
    # 噪声在训练与测试上同样施加（noise_seed=None -> 每次不同，训练用）
    Xtr, Ytr = build_windows(
        tr_z, mask, input_len, horizon, stride=10, max_windows=n_windows, rng=rng,
        noise_std=noise_std,
    )
    Xte, Yte = build_windows(
        te_z, mask, input_len, horizon, stride=25,
        max_windows=max(2000, n_windows // 10), rng=rng,
        noise_std=noise_std, noise_seed=1234,
    )

    # 切验证集
    n_val = max(500, int(0.1 * Xtr.shape[0]))
    Xval, Yval = make_tensors(Xtr[:n_val], Ytr[:n_val])
    Xtr, Ytr = make_tensors(Xtr[n_val:], Ytr[n_val:])
    Xte_t, Yte_t = make_tensors(Xte, Yte)

    model = build_model(model_name, dim, horizon, input_len).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 余弦退火：留 10% 的 epoch 作为最低学习率平台，避免训练末段 lr 直接砸到 0
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs), eta_min=lr * 0.1
    )

    best_val, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    hist: List[Dict] = []

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(Xtr.shape[0])
        tot, cnt = 0.0, 0
        for i in range(0, Xtr.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = Xtr[idx].to(DEVICE), Ytr[idx].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.detach().item() * xb.shape[0]
            cnt += xb.shape[0]
        sched.step()

        val = evaluate_loss(model, Xval, Yval)
        hist.append({"epoch": ep, "train": tot / max(cnt, 1), "val": val})

        if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
            print(
                f"    ep{ep:>3} train={tot/max(cnt,1):.5f} val={val:.5f} "
                f"lr={sched.get_last_lr()[0]:.2e}"
            )

        if val < best_val - 1e-9:
            best_val, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop @ ep{ep} (best val={best_val:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss = evaluate_loss(model, Xte_t, Yte_t)

    run_id = f"{system}_{model_name}_{mode}_r{ratio:.2f}_s{seed}"
    if noise_std > 0:
        run_id += f"_n{noise_std:.2f}"
    os.makedirs(outdir, exist_ok=True)
    result = {
        "run_id": run_id,
        "system": system,
        "model": model_name,
        "mode": mode,
        "ratio": ratio,
        "seed": seed,
        "noise_std": noise_std,
        "dim": dim,
        "horizon": horizon,
        "input_len": input_len,
        "n_params": int(n_params),
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "best_val_mse": float(best_val),
        "test_mse": float(test_loss),
        "elapsed_sec": round(time.time() - t0, 1),
        "history": hist,
        "mask": np.asarray(mask, dtype=float).tolist(),
        "train_mean": [float(v) for v in mean],
        "train_std": [float(v) for v in std],
    }
    with open(os.path.join(outdir, f"{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "dim": dim, "horizon": horizon, "input_len": input_len,
                "model": model_name, "noise_std": noise_std,
            },
            "mask": mask,
            "mean": mean,
            "std": std,
        },
        os.path.join(outdir, f"{run_id}.pt"),
    )

    if verbose:
        print(
            f"  完成 {run_id}: params={n_params/1e3:.1f}k "
            f"test_mse={test_loss:.5f} ({result['elapsed_sec']}s)"
        )
    return result


# --------------------------------------------------------------------------
# 批量网格
# --------------------------------------------------------------------------
def run_grid(
    systems: List[str],
    models: List[str],
    modes: List[str],
    ratios: List[float],
    epochs: int,
    outdir: str,
) -> None:
    combos = [
        (s, m, mo, r)
        for s in systems
        for m in models
        for mo in modes
        for r in ratios
        if not (mo == "none" and r != 1.0)  # none 模式只跑一次
    ]
    print(f"共 {len(combos)} 组实验，每组 ~epochs={epochs}")

    summary = []
    for i, (s, m, mo, r) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] {s} | {m} | {mo} | ratio={r:.2f}")
        try:
            res = train_one(s, m, mo, r, epochs=epochs, outdir=outdir, verbose=True)
            summary.append(res)
        except Exception as e:  # 单组失败不中断整个网格
            print(f"  !! 失败: {type(e).__name__}: {e}")

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "grid_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {k: v for k, v in r.items() if k != "history"}
                for r in summary
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n网格完成，{len(summary)}/{len(combos)} 组成功。摘要 -> {outdir}/grid_summary.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="训练混沌系统预测模型")
    ap.add_argument("--system", default="lorenz63", choices=["lorenz63", "lorenz96"])
    ap.add_argument("--model", default="lstm", choices=["mlp", "lstm", "transformer"])
    ap.add_argument("--mode", default="uniform", choices=list(MASK_MODES))
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-windows", type=int, default=120_000)
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="随机种子；不同种子会写入不同的 run_id（..._s<N>）")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="观测噪声标准差（标准化尺度），0 表示无噪声")
    ap.add_argument("--grid", action="store_true", help="跑全部组合")
    args = ap.parse_args()

    print(f"device={DEVICE}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    if args.grid:
        run_grid(
            systems=["lorenz63", "lorenz96"],
            models=["mlp", "lstm", "transformer"],
            modes=["none", "uniform", "block", "random"],
            ratios=[1.0, 0.5, 1 / 3, 1 / 6],
            epochs=args.epochs,
            outdir=args.outdir,
        )
    else:
        train_one(
            args.system,
            args.model,
            args.mode,
            args.ratio,
            horizon=args.horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            n_windows=args.n_windows,
            outdir=args.outdir,
            seed=args.seed,
            noise_std=args.noise,
        )


if __name__ == "__main__":
    main()
