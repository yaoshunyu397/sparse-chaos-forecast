"""
heat_equation.py -- 一维热方程的有限差分求解器与收敛阶验证

为什么写这个：
    这是通往动力学预测的数值基础。混沌系统的预测依赖时间积分，
    而时间积分的可信度取决于离散格式的收敛阶与稳定性——这一点只能
    通过对照解析解来验证，不能靠"图看起来对"。

    本文件实现两个格式并**分开验证空间阶与时间阶**：

        FTCS（显式）        : 时间 1 阶、空间 2 阶，条件稳定 (r ≤ 1/2)
        Crank-Nicolson（隐式）: 时间 2 阶、空间 2 阶，无条件稳定

    这个对比顺便说明一个工程取舍：隐式格式每步要解三对角线性系统，
    但允许大时间步；显式格式每步便宜，但步长被稳定性卡死。

问题设定：
    ∂u/∂t = α ∂²u/∂x²,   x ∈ (0, 1),  t > 0
    u(0, t) = u(1, t) = 0
    u(x, 0) = sin(πx)

解析解（单模，可精确写出）：
    u(x, t) = exp(-α π² t) sin(πx)          （取 α = 1）

单模的好处：离散解与解析解可以直接逐点比较，不需要傅里叶级数求和，
误差的阶数不会被高阶模污染。

用法：
    python heat_equation.py              # 跑全部验证
    python heat_equation.py --plot       # 另存误差曲线与解的动画帧

参考：
    LeVeque, R. J. (2007). Finite Difference Methods for Ordinary and
    Partial Differential Equations. SIAM. 第 2–3 章。
"""

from __future__ import annotations

import argparse
import math
from typing import Callable, Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# 解析解
# ----------------------------------------------------------------------
ALPHA = 1.0


def exact(x: np.ndarray, t: float, alpha: float = ALPHA) -> np.ndarray:
    """u(x,t) = exp(-α π² t) sin(πx)"""
    return np.exp(-alpha * math.pi ** 2 * t) * np.sin(math.pi * x)


# ----------------------------------------------------------------------
# 求解器
# ----------------------------------------------------------------------
def solve_ftcs(
    n_grid: int,
    t_end: float,
    n_steps: int,
    alpha: float = ALPHA,
    initial: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    显式 FTCS：u_i^{n+1} = u_i^n + r (u_{i+1}^n - 2u_i^n + u_{i-1}^n)

    参数
    ----
    n_grid  : 区间 (0,1) 内的**内部**网格点数。总点数 n_grid+2（含两端）。
    n_steps : 时间步数
    initial : 初值函数，默认 sin(πx)
    返回 (x, u) 在 t = t_end 的解，x 含两端点。

    r = α dt / dx² 必须 ≤ 1/2，否则格式不稳定（本函数不强制，便于演示失稳）。

    ⚠️ 演示失稳时初值必须含**高频成分**：
       稳定性上限来自最高频网格模态（k = n_grid），而默认初值 sin(πx)
       只有最低频模态。线性格式下各模态独立演化，光滑初值不会激发高频，
       即使 r > 1/2 也看不到发散 —— 见 study_stability()。
    """
    dx = 1.0 / (n_grid + 1)
    dt = t_end / n_steps
    r = alpha * dt / (dx * dx)

    x = np.linspace(0.0, 1.0, n_grid + 2)
    u = (initial(x) if initial is not None else np.sin(math.pi * x)).astype(np.float64)
    u[0] = u[-1] = 0.0

    for _ in range(n_steps):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        u_new[0] = u_new[-1] = 0.0
        u = u_new
    return x, u


def _thomas(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray,
            rhs: np.ndarray) -> np.ndarray:
    """
    三对角线性系统的 Thomas 算法（O(n)，不依赖 scipy）。

    A 的三条对角线：lower（长度 n-1）、diag（长度 n）、upper（长度 n-1）。
    """
    n = diag.size
    if n == 1:
        return rhs / diag

    cp = np.empty(n - 1, dtype=np.float64)   # 修改后的上对角线
    dp = np.empty(n, dtype=np.float64)       # 修改后的右端项

    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = upper[i] / m
        dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / m

    x = np.empty(n, dtype=np.float64)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def solve_crank_nicolson(
    n_grid: int,
    t_end: float,
    n_steps: int,
    alpha: float = ALPHA,
    initial: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crank-Nicolson（隐式，时间 2 阶、无条件稳定）

    离散：
        (u^{n+1} - u^n)/dt = α/2 (L u^{n+1} + L u^n)
    =>  -s u_{i-1}^{n+1} + (1+2s) u_i^{n+1} - s u_{i+1}^{n+1}
        = s u_{i-1}^n + (1-2s) u_i^n + s u_{i+1}^n,   s = α dt / (2 dx²)

    边界为零，所以 i=1 与 i=n_grid 处的邻居项直接消失。
    """
    dx = 1.0 / (n_grid + 1)
    dt = t_end / n_steps
    s = alpha * dt / (2.0 * dx * dx)

    x = np.linspace(0.0, 1.0, n_grid + 2)
    u = (initial(x) if initial is not None else np.sin(math.pi * x)).astype(np.float64)
    u[0] = u[-1] = 0.0

    n = n_grid                               # 内部未知量个数
    lower = np.full(max(n - 1, 0), -s)
    diag = np.full(n, 1.0 + 2.0 * s)
    upper = np.full(max(n - 1, 0), -s)

    for _ in range(n_steps):
        v = u[1:-1]                          # 内部点，长度 n
        rhs = (1.0 - 2.0 * s) * v
        if n > 1:
            rhs[:-1] += s * v[1:]
            rhs[1:] += s * v[:-1]
        v_new = _thomas(lower, diag, upper, rhs)
        u = np.concatenate([[0.0], v_new, [0.0]])
    return x, u


def exact_semidiscrete(
    n_grid: int, t: float, alpha: float = ALPHA
) -> Tuple[np.ndarray, np.ndarray]:
    """
    **半离散**问题的精确解（只离散空间，时间仍精确）。

    为什么需要它：
       要干净地测时间阶，必须把空间误差排除掉。
       直接与完全连续的解析解比较时，粗网格的空间误差会掩盖时间误差，
       而且 FTCS 在粗时间步下还会失稳（r > 1/2），误差完全不可用。

    推导：
        离散拉普拉斯（Dirichlet）的特征向量与特征值：
            φ_k,j = sin(kπ x_j),   λ_k = -(4α/dx²) sin²(kπ dx/2)
        把初值展开 u0 = Σ c_k φ_k，则半离散解为
            u_j(t) = Σ_k c_k exp(λ_k t) sin(kπ x_j)
        其中 c_k = 2/(N+1) Σ_j u0_j sin(kπ x_j)。

    返回 (x_inner, u_inner)，x 只含内部点。
    """
    N = n_grid
    dx = 1.0 / (N + 1)
    j = np.arange(1, N + 1)
    xj = j * dx
    u0 = np.sin(math.pi * xj)

    k = np.arange(1, N + 1)
    Phi = np.sin(np.outer(k, xj) * math.pi)          # (N, N)
    c = (2.0 / (N + 1)) * (Phi @ u0)
    lam = -4.0 * alpha * np.sin(k * math.pi * dx / 2.0) ** 2 / (dx * dx)
    u = Phi.T @ (c * np.exp(lam * t))
    return xj, u


# ----------------------------------------------------------------------
# 误差与收敛阶
# ----------------------------------------------------------------------
def linf_error(solver: Callable, n_grid: int, t_end: float,
               n_steps: int) -> float:
    """与**完全连续**的解析解比较——用于空间阶。"""
    x, u = solver(n_grid, t_end, n_steps)
    return float(np.max(np.abs(u - exact(x, t_end))))


def temporal_error(solver: Callable, n_grid: int, t_end: float,
                   n_steps: int) -> float:
    """与**半离散**精确解比较——用于时间阶，排除了空间误差。"""
    x, u = solver(n_grid, t_end, n_steps)
    _, u_ref = exact_semidiscrete(n_grid, t_end)
    return float(np.max(np.abs(u[1:-1] - u_ref)))


def observed_order(errors: List[float], params: List[float]) -> List[float]:
    """用相邻两级误差估计阶：p = log(e_{k}/e_{k+1}) / log(h_k/h_{k+1})"""
    orders = []
    for i in range(len(errors) - 1):
        if errors[i + 1] <= 0 or errors[i] <= 0:
            orders.append(float("nan"))
            continue
        orders.append(
            math.log(errors[i] / errors[i + 1]) / math.log(params[i] / params[i + 1])
        )
    return orders


# ----------------------------------------------------------------------
# 验证套件
# ----------------------------------------------------------------------
def study_spatial_order(t_end: float = 0.02) -> Dict[str, List[float]]:
    """
    空间阶：固定 dt 足够小，只加密 dx。

    期望 FTCS 与 CN 都是 2 阶。
    """
    # dt 固定为 t_end / n_steps，取 n_steps 大到时间误差可忽略
    n_steps = 20000
    n_grids = [8, 16, 32, 64, 128]
    out = {}
    for name, solver in (("FTCS", solve_ftcs), ("Crank-Nicolson", solve_crank_nicolson)):
        errs = [linf_error(solver, n, t_end, n_steps) for n in n_grids]
        h = [1.0 / (n + 1) for n in n_grids]
        out[name] = (errs, observed_order(errs, h))
    return out


def study_temporal_order(t_end: float = 0.02) -> Dict[str, List[float]]:
    """
    时间阶：固定 dx，只加密 dt，并与**半离散精确解**比较。

    期望 FTCS 1 阶、CN 2 阶。

    为什么必须用半离散精确解作参考：
      - 与完全连续解析解比较时，空间误差（N=40 时约 1e-3）会掩盖时间误差；
      - 更要紧的是 FTCS 的稳定性：N=40 时 dx² = 5.95e-4，
        要求 r = dt/dx² ≤ 0.5 即 dt ≤ 2.97e-4。
        若取 n_steps=20（dt=1e-3），r ≈ 1.68，格式本身已失稳，
        测出来的"误差"是发散量，阶数完全无意义（实测会出现 6.87、-0.50 这类值）。

    所以这里让所有 n_steps 都满足稳定条件：t_end=0.02、N=40 时 n_steps ≥ 68。
    """
    n_grid = 40                       # dx = 1/41, dx² ≈ 5.95e-4
    n_steps_list = [80, 160, 320, 640, 1280]   # 最小的 dt=2.5e-4 -> r≈0.42，稳定
    out = {}
    for name, solver in (("FTCS", solve_ftcs), ("Crank-Nicolson", solve_crank_nicolson)):
        errs = [temporal_error(solver, n_grid, t_end, ns) for ns in n_steps_list]
        dt = [t_end / ns for ns in n_steps_list]
        out[name] = (errs, observed_order(errs, dt))
    return out


def study_stability() -> Dict[str, float]:
    """
    稳定性边界：FTCS 在 r > 1/2 时应发散，CN 在任何 dt 下都稳定。

    ⚠️ 初值必须含高频成分。
       稳定性上限由最高频网格模态（k = n_grid）决定：
           g_k = 1 - 4r sin²(kπ dx / 2)
       对 k = n_grid、dx = 1/(n_grid+1)，sin² ≈ 1，故 |g| = |1-4r|。
       r = 0.6 时 |g| = 1.4 > 1 —— 但这个模态只有在初值里存在时才看得见。
       默认初值 sin(πx) 是最低频模态，其 g = 1 - 4r sin²(π dx/2) ≈ 0.987 < 1，
       即使 r = 0.6 也照样衰减。所以这里加一个 1e-3 的最高频扰动。
    """
    n_grid = 20
    t_end = 0.1                      # 步数多一点，让失稳模态有指数增长的空间
    dx = 1.0 / (n_grid + 1)

    def ic_with_high_mode(x: np.ndarray) -> np.ndarray:
        return np.sin(math.pi * x) + 1e-3 * np.sin(n_grid * math.pi * x)

    results = {}
    for label, r_target in (("FTCS r=0.4（稳定）", 0.4), ("FTCS r=0.6（失稳）", 0.6)):
        dt = r_target * dx * dx / ALPHA
        n_steps = max(1, int(round(t_end / dt)))
        _, u = solve_ftcs(n_grid, n_steps * dt, n_steps,
                          initial=ic_with_high_mode)
        results[label] = float(np.max(np.abs(u)))
    # CN 用同样的大 dt
    dt_big = 0.6 * dx * dx / ALPHA
    n_steps = max(1, int(round(t_end / dt_big)))
    _, u_cn = solve_crank_nicolson(n_grid, n_steps * dt_big, n_steps,
                                   initial=ic_with_high_mode)
    results["CN r=0.6（应当稳定）"] = float(np.max(np.abs(u_cn)))
    return results


def check_analytical_decay(verbose: bool = True) -> float:
    """
    确认解按 exp(-π² t) 衰减，而不是仅仅"看起来像 sin"。

    取 x=0.5（sin(πx)=1），解在 t 时刻应等于 exp(-π² t)。
    """
    t = 0.02
    x, u = solve_crank_nicolson(200, t, 400)
    idx = int(np.argmin(np.abs(x - 0.5)))
    got = float(u[idx])
    want = float(math.exp(-math.pi ** 2 * t))
    rel = abs(got - want) / want
    if verbose:
        print(f"  中点 x=0.5 处 u = {got:.6f}，解析值 {want:.6f}，相对误差 {rel:.2e}")
    assert rel < 1e-3, f"解析衰减验证失败：相对误差 {rel:.2e}"
    return rel


# ----------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------
def make_plot(outdir: str = "figs") -> None:
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    t_end = 0.02

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150)

    # 左：解与解析解对比
    ax = axes[0]
    x, u = solve_crank_nicolson(64, t_end, 400)
    ax.plot(x, exact(x, t_end), "-", lw=2.2, label="analytical", color="#2b6cb0")
    ax.plot(x, u, "o", ms=3.5, label="Crank–Nicolson", color="#c53030")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"u(x, t={t_end})", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    # 中：空间阶
    ax = axes[1]
    sp = study_spatial_order(t_end)
    for name, color in (("FTCS", "#d17b0f"), ("Crank-Nicolson", "#2f855a")):
        errs, _ = sp[name]
        n_grids = [8, 16, 32, 64, 128]
        h = [1.0 / (n + 1) for n in n_grids]
        ax.loglog(h, errs, "o-", ms=3.5, label=name, color=color)
    h = np.array([1.0 / (n + 1) for n in [8, 16, 32, 64, 128]])
    ax.loglog(h, 2e-3 * (h / h[0]) ** 2, "k--", lw=1, label=r"slope 2")
    ax.set_xlabel("grid spacing dx")
    ax.set_ylabel(r"$\|u - u_{\mathrm{exact}}\|_\infty$")
    ax.set_title("spatial convergence", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3, which="both")

    # 右：时间阶
    ax = axes[2]
    tp = study_temporal_order(t_end)
    n_steps_list = [20, 40, 80, 160, 320]
    dt = np.array([t_end / ns for ns in n_steps_list])
    for name, color in (("FTCS", "#d17b0f"), ("Crank-Nicolson", "#2f855a")):
        errs, _ = tp[name]
        ax.loglog(dt, errs, "s-", ms=3.5, label=name, color=color)
    ax.loglog(dt, 1e-1 * (dt / dt[0]), "k--", lw=1, label="slope 1")
    ax.loglog(dt, 1e-1 * (dt / dt[0]) ** 2, "k:", lw=1, label="slope 2")
    ax.set_xlabel("time step dt")
    ax.set_ylabel(r"$\|u - u_{\mathrm{exact}}\|_\infty$")
    ax.set_title("temporal convergence", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(outdir, "heat_equation.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"已生成 {path}")


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="一维热方程有限差分 + 收敛阶验证")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--t-end", type=float, default=0.02)
    args = ap.parse_args()

    print("=" * 72)
    print("一维热方程：FTCS vs Crank-Nicolson")
    print("=" * 72)

    print("\n[1] 解析衰减验证")
    check_analytical_decay()

    print("\n[2] 稳定性（FTCS 应在 r > 1/2 时发散）")
    st = study_stability()
    for k, v in st.items():
        print(f"  {k:<24} max|u| = {v:.4e}")
    assert st["FTCS r=0.6（失稳）"] > 1e3, "FTCS 在 r=0.6 时未发散，稳定性演示有问题"
    assert st["CN r=0.6（应当稳定）"] < 1.0, "CN 在 r=0.6 时不应发散"
    print("  ✔ FTCS 失稳、CN 稳定，与理论一致")

    print("\n[3] 空间阶（固定 dt，只加密 dx）—— 期望 2 阶")
    sp = study_spatial_order(args.t_end)
    for name, (errs, orders) in sp.items():
        print(f"  {name}")
        for e, o in zip(errs, orders):
            print(f"    err={e:.4e}  阶≈{o:.2f}" if not math.isnan(o)
                  else f"    err={e:.4e}")
        valid = [o for o in orders if not math.isnan(o)]
        mean_order = sum(valid) / len(valid)
        print(f"    平均观测阶 = {mean_order:.2f}")
        assert 1.7 < mean_order < 2.4, f"{name} 空间阶偏离 2：{mean_order:.2f}"

    print("\n[4] 时间阶（固定 dx，只加密 dt）—— FTCS 期望 1 阶，CN 期望 2 阶")
    tp = study_temporal_order(args.t_end)
    for name, (errs, orders) in tp.items():
        print(f"  {name}")
        for e, o in zip(errs, orders):
            print(f"    err={e:.4e}  阶≈{o:.2f}" if not math.isnan(o)
                  else f"    err={e:.4e}")
        valid = [o for o in orders if not math.isnan(o)]
        mean_order = sum(valid) / len(valid)
        print(f"    平均观测阶 = {mean_order:.2f}")
        if name == "FTCS":
            assert 0.7 < mean_order < 1.4, f"FTCS 时间阶应≈1，实测 {mean_order:.2f}"
        else:
            assert 1.6 < mean_order < 2.4, f"CN 时间阶应≈2，实测 {mean_order:.2f}"

    print("\n" + "=" * 72)
    print("全部验证通过")
    print("=" * 72)
    print("""
两点结论：
  1. 两个格式空间上都是 2 阶，因为它们都用中心差分近似 u_xx。
  2. 时间上 FTCS 只有 1 阶、CN 是 2 阶；代价是 CN 每步要解三对角系统。
  3. FTCS 的 dt 被稳定性(r ≤ 1/2)卡死：dx 减半则 dt 必须减到 1/4，
     所以细化到同样精度，显式格式的步数按 dx⁻² 增长。
     CN 无条件稳定，可以用大得多的时间步——这就是它值得多解一个线性系统的原因。""")

    if args.plot:
        make_plot()


if __name__ == "__main__":
    main()
