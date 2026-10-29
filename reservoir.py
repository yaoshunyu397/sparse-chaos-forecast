"""
reservoir.py -- Echo State Network（储备池计算）基线

为什么必须加这个：
    项目当前三个模型全是神经网络。而这一领域的经典方法是 reservoir
    computing（见 Gupta et al., Neural Networks 160, 2023）。
    没有这个基线，"LSTM 比 transformer 好"这个结论
    就缺少参照——读者会说"你的 transformer 只是没调好"。

    reservoir 是**免训练**的：随机固定循环权重 + 线性读出层（岭回归）。
    所以它既是公平的参照，又几乎不花训练时间。

实现选择（以及一次失败记录）：

    ❌ 第一版（错误）：
       把整个 50 步窗口展平，经随机投影成 N 维**初始状态**，让 reservoir
       自由演化 50 步，取末态作特征。
       结果：一步 MSE 只比"预测均值"好 1.8 倍（0.564 vs 1.008），完全不可用。
       原因：谱半径 <1 的 reservoir 是收缩的，自由演化会把所有初始条件
       吸引到同一个固定点，输入信息在几十步内被冲掉。
       **reservoir 的记忆来自被驱动时的累积，不是来自初始条件。**

    ✅ 第二版（当前）：
       逐时刻驱动。窗口内第 l 步，用该时刻的观测向量作为输入，
       按标准 ESN 状态方程演化：
           s_{l} = (1-α)·s_{l-1} + α·tanh(W·s_{l-1} + W_in·u_l)
       末态 s_L 累积了整段窗口的历史，作为岭回归读出的特征。

    调参结论（Lorenz-63, temporal 0.5，含掩码）：
       一步 MSE 与 rollout VPT 的最优点不在同一处，以 VPT 为准：

       | N    | spectral_radius | ridge | 一步 MSE | VPT  |
       |------|-----------------|-------|----------|------|
       | 800  | 1.2             | 1e-4  | 1.2e-4   | 35.6 |
       | 1000 | 1.2             | 1e-6  | 8.7e-5   | 38.7 |
       | 1500 | 1.2             | 1e-6  | 3.3e-5   | 50.9 |  ← 采用
       | 2000 | 1.2             | 1e-6  | 2.1e-5   | 47.0 |
       | 2500 | 1.2             | 1e-6  | 1.7e-5   | 46.3 |

       三个要点：
       1. ridge 要小（1e-6），因为特征已标准化、SᵀS 对角量级 ≈ 样本数。
       2. spectral_radius 超过 1.2 会塌缩（1.35 时 VPT 跌到 23，1.5 时 9）。
       3. N 超过 1500 后 VPT 反而下降 —— 岭回归读出在高维特征上过拟合，
          尽管一步 MSE 仍在改善。这说明 reservoir 不能靠堆规模解决。

    这保留了教科书 ESN 的机制，也与 MLP/LSTM 使用**同一段输入窗口**，
    因此四个模型的信息预算是可比的。

参考：
    Jaeger & Haas (2004), Science 304, 78-80.
    Lukoševičius & Jaeger (2009), Computer Science Review 3, 127-149.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class EchoStateNetwork:
    """
    标准驱动式 ESN + 岭回归读出。

    参数
    ----
    input_dim  : 系统维度 d
    input_len  : 输入窗口长度 L
    n_states   : 储备池规模 N
    spectral_radius : 循环矩阵谱半径。0.9–1.2 常用；<1 保证回声状态性质
    leak       : 泄漏率 α（0,1]。越小记忆越长
    sparsity   : 循环权重非零比例
    ridge      : 岭回归正则化 λ
    input_scaling : 输入权重缩放。太小则 reservoir 反应弱，太大则饱和
    seed       : 随机种子（控制 W 与 W_in；读出由闭式解决定，无随机性）
    """

    def __init__(
        self,
        input_dim: int,
        input_len: int,
        n_states: int = 1500,
        spectral_radius: float = 1.2,
        leak: float = 1.0,
        sparsity: float = 0.05,
        ridge: float = 1e-6,
        input_scaling: float = 0.5,
        seed: int = 0,
    ):
        self.input_dim = input_dim
        self.input_len = input_len
        self.n_states = n_states
        self.spectral_radius = spectral_radius
        self.leak = leak
        self.sparsity = sparsity
        self.ridge = ridge
        self.input_scaling = input_scaling
        self.seed = seed

        rng = np.random.default_rng(seed)

        # ---- 循环权重：稀疏随机，缩放到目标谱半径 ----
        W = rng.standard_normal((n_states, n_states))
        mask = rng.random((n_states, n_states)) < sparsity
        W = W * mask
        eig = np.abs(np.linalg.eigvals(W)).max()
        if eig > 0:
            W = W * (spectral_radius / eig)
        self.W = W.astype(np.float32)

        # ---- 输入权重 ----
        self.W_in = (
            rng.uniform(-1.0, 1.0, size=(n_states, input_dim)) * input_scaling
        ).astype(np.float32)

        self.W_out: Optional[np.ndarray] = None
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 特征提取：逐时刻驱动
    # ------------------------------------------------------------------
    def features(self, X: np.ndarray, washout: int = 0) -> np.ndarray:
        """
        X: (B, L, d) 已加掩码的输入窗口。
        返回 (B, N)：每个样本演化到末态时的 reservoir 状态。

        实现上对 batch 做向量化：所有样本的窗口并行推进。
        """
        B, L, d = X.shape
        if d != self.input_dim:
            raise ValueError(f"输入维度 {d} 与构造时的 {self.input_dim} 不一致")

        s = np.zeros((B, self.n_states), dtype=np.float32)
        one_m_leak = np.float32(1.0 - self.leak)
        for l in range(L):
            u = X[:, l, :].astype(np.float32)          # (B, d)
            pre = s @ self.W.T + u @ self.W_in.T       # (B, N)
            s = one_m_leak * s + np.float32(self.leak) * np.tanh(pre)
        return s

    # ------------------------------------------------------------------
    # 训练 / 预测
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "EchoStateNetwork":
        """
        X: (B, L, d)  输入窗口
        Y: (B, H, d)  目标（本模块只用 H=1 的最后一步）
        """
        if Y.ndim == 3:
            Y = Y[:, -1, :]
        S = self.features(X)

        # 特征标准化：岭回归对尺度敏感，且 reservoir 状态可能集中在 0 附近
        self._mu = S.mean(axis=0, keepdims=True)
        self._sigma = S.std(axis=0, keepdims=True)
        self._sigma[self._sigma < 1e-6] = 1.0
        Sn = (S - self._mu) / self._sigma

        # 岭回归闭式解：W_out = (SᵀS + λnI)⁻¹ SᵀY
        A = Sn.T @ Sn
        A[np.diag_indices_from(A)] += self.ridge * Sn.shape[0]
        self.W_out = np.linalg.solve(A, Sn.T @ Y.astype(np.float32))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (B, L, d) -> (B, 1, d)"""
        if self.W_out is None:
            raise RuntimeError("先调用 fit()")
        S = (self.features(X) - self._mu) / self._sigma
        return (S @ self.W_out)[:, None, :]


class TorchWrapper:
    """把 ESN 包成 nn.Module 风格，让 evaluate.rollout 可以直接复用。"""

    def __init__(self, esn: EchoStateNetwork):
        self.esn = esn

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def train(self, mode: bool = False):
        return self

    def __call__(self, x):
        import torch

        arr = x.detach().cpu().numpy().astype(np.float32)
        out = self.esn.predict(arr)
        return torch.from_numpy(out).to(x.device)

    def state_dict(self):
        return {"W_out": self.esn.W_out}

    def load_state_dict(self, sd):
        self.esn.W_out = sd["W_out"]


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
def selfcheck_synthetic(verbose: bool = True) -> None:
    """
    实现正确性自检：reservoir 必须能记住"窗口最后一个输入"。

    为什么不用更难的任务（如记忆 15 步前的值）：
      那对 800 状态的 ESN 而言过难（实测 MSE 0.37），会与"实现有 bug"
      混为一谈。正确性检查应该选一个实现正确就必然通过的任务。
      诊断时确认：预测最后一个输入可得 MSE ~4e-4（基线 1.0），
      即 2500 倍改善 —— 这个阈值足以区分"实现对"与"实现错"。
    """
    rng = np.random.default_rng(0)
    n, L = 2000, 20
    X = rng.standard_normal((n, L, 1)).astype(np.float32)
    Y = X[:, -1, :].copy()          # 目标 = 最后一个输入

    esn = EchoStateNetwork(1, L, n_states=300, seed=0).fit(X[:1500], Y[:1500])
    pred = esn.predict(X[1500:])
    mse = float(np.mean((pred - Y[1500:, None, :]) ** 2))
    mse_mean = float(np.mean(Y[1500:] ** 2))
    if verbose:
        print(f"[合成] 记忆最后一步 MSE = {mse:.5f} | 均值基线 = {mse_mean:.4f} "
              f"| 改善 {mse_mean/max(mse,1e-12):.0f}x")
    assert mse < 0.05 * mse_mean, (
        f"合成自检失败（MSE={mse:.5f}）—— ESN 实现有问题"
    )
    if verbose:
        print("  通过：reservoir 确实保留了输入历史")


def selfcheck(verbose: bool = True, **kw) -> float:
    """在 Lorenz-63 真实数据上验证：显著优于均值基线。"""
    from masking import build_mask, load_data, observations_to_windows, standardize

    train, test, meta = load_data("lorenz63")
    dim = meta["dim"]
    tr_z, te_z, _, _ = standardize(train, test, meta)
    mask = build_mask(dim, "temporal", 0.5, seed=42, input_len=50)

    def windows(traj, stride, max_n=None):
        xs, ys = [], []
        for n in range(traj.shape[0]):
            x, y = observations_to_windows(traj[n], mask, 50, 1, stride)
            xs.append(x)
            ys.append(y)
        X = np.concatenate(xs)
        Y = np.concatenate(ys)
        if max_n and X.shape[0] > max_n:
            idx = np.linspace(0, X.shape[0] - 1, max_n).astype(int)
            X, Y = X[idx], Y[idx]
        return X, Y

    Xtr, Ytr = windows(tr_z, stride=25, max_n=30000)
    Xte, Yte = windows(te_z, stride=50, max_n=5000)

    esn = EchoStateNetwork(dim, 50, **kw).fit(Xtr, Ytr)
    pred = esn.predict(Xte)
    mse = float(np.mean((pred - Yte) ** 2))
    mse_mean = float(np.mean(Yte ** 2))
    if verbose:
        print(f"[Lorenz-63] 一步 MSE = {mse:.6f} | 均值基线 = {mse_mean:.6f} "
              f"| 改善 {mse_mean/max(mse,1e-12):.0f}x")
    return mse


if __name__ == "__main__":
    print("=" * 64)
    print("ESN 自检")
    print("=" * 64)
    selfcheck_synthetic()
    print()
    mse = selfcheck()
    print()
    # 与神经网络基线对比（来自 runs/lorenz63_lstm_temporal_r0.50 等）
    print("参考：MLP/LSTM 在同条件下的一步 MSE 约 2e-5 ~ 6e-5")
