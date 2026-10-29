"""
test_pipeline.py -- 地基验收测试（不依赖 pytest，直接运行）

检查四件事：
  1. 数据文件存在、形状、无 nan/inf
  2. 标准化后训练集 ≈ 均值 0 / 方差 1
  3. 掩码在 train 与 test 上完全一致（这是全项目最容易被忽略的纪律）
  4. 窗口切分后：
       - target 未被 mask（始终是完整状态）
       - 被遮挡维度在 input 里确实填了 0
       - 不遮挡（ratio=1）时 input 与原始窗口逐元素相等

运行：
    python test_pipeline.py
"""

from __future__ import annotations

import numpy as np

from masking import (
    apply_mask,
    build_mask,
    load_data,
    observations_to_windows,
    standardize,
)

PASS = "\u2713"
FAIL = "\u2717"


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  {PASS} {msg}")


def test_system(system: str) -> None:
    print(f"\n[{system}]")
    train, test, meta = load_data(system)
    dim = meta["dim"]

    # ---- 1. 基本形状与有限性 ----
    check(train.shape == tuple(meta["train_shape"]), f"train 形状 {train.shape}")
    check(test.shape == tuple(meta["test_shape"]), f"test  形状 {test.shape}")
    check(train.shape[-1] == dim, f"维度 {dim}")
    check(np.all(np.isfinite(train)) and np.all(np.isfinite(test)), "无 nan/inf")

    # ---- 2. 标准化 ----
    tr_z, te_z, mean, std = standardize(train, test, meta)
    check(abs(tr_z.mean()) < 0.05, f"标准化后 train 均值 {tr_z.mean():+.4f} ≈ 0")
    # 注意：用 warmly 段算的统计量，全量标准化 std 允许略偏离 1
    check(0.5 < tr_z.std() < 1.5, f"标准化后 train 标准差 {tr_z.std():.3f} 在合理范围")

    # ---- 3. 掩码一致性（同 seed -> 同掩码）----
    for mode in ("uniform", "block", "random"):
        for ratio in (0.5, 1 / 3, 1 / 6):
            m_a = build_mask(dim, mode, ratio, seed=42)
            m_b = build_mask(dim, mode, ratio, seed=42)
            check(np.array_equal(m_a, m_b), f"{mode:<8} ratio={ratio:.3f} 掩码可复现")

    # ---- 3b. 时间掩码：必须能区分不同稀疏度（维度掩码在 D=3 上做不到）----
    L = 50
    temporal_obs = {}
    for ratio in (0.5, 1 / 3, 1 / 6):
        mt = build_mask(dim, "temporal", ratio, seed=42, input_len=L)
        n_obs = int(mt.sum())
        temporal_obs[ratio] = n_obs
        check(len(mt) == L, f"temporal ratio={ratio:.3f} 形状 {mt.shape} == ({L},)")
        check(0 < n_obs < L, f"temporal ratio={ratio:.3f} 保留 {n_obs}/{L} 个时刻")
    # 观测越稀疏 -> 保留的时刻越少（只看 ratio<1 的三档，它们的掩码长度才是 L）
    ordered = [temporal_obs[r] for r in (0.5, 1 / 3, 1 / 6)]
    check(
        ordered == sorted(ordered, reverse=True),
        f"temporal 观测数随稀疏度单调下降: {ordered}",
    )

    # 关键回归测试：D=3 时维度掩码 1/3 与 1/6 会塌缩，temporal 不会
    if dim == 3:
        m13 = build_mask(dim, "uniform", 1 / 3, seed=42)
        m16 = build_mask(dim, "uniform", 1 / 6, seed=42)
        check(np.array_equal(m13, m16), "已知限制：D=3 时 uniform 1/3 与 1/6 相同（故用 temporal）")
        check(
            temporal_obs[1 / 3] > temporal_obs[1 / 6],
            f"temporal 1/3 ({temporal_obs[1/3]}) 与 1/6 ({temporal_obs[1/6]}) 确有区分",
        )

    # ---- 4. 窗口切分 ----
    input_len, horizon = 50, 1
    modes = ("none", "uniform", "block", "random", "temporal")
    for mode in modes:
        ratios = (1.0,) if mode == "none" else (1.0, 0.5, 1 / 3, 1 / 6)
        for ratio in ratios:
            m = build_mask(dim, mode, ratio, seed=42, input_len=input_len)
            X, Y = observations_to_windows(
                tr_z[0], m, input_len=input_len, horizon=horizon, stride=50
            )
            is_temporal = mode == "temporal" and ratio < 1.0
            kept = int(m.sum()) if m.ndim == 1 and m.shape[0] == dim else f"{int(m.sum())}t"
            tag = f"{mode:<8} ratio={ratio:.3f} kept={kept}"

            if X.shape[1] != input_len or X.shape[2] != dim:
                raise AssertionError(f"{tag}: input 形状异常 {X.shape}")
            if Y.shape[1] != horizon or Y.shape[2] != dim:
                raise AssertionError(f"{tag}: target 形状异常 {Y.shape}")

            # 4a. target 必须是完整状态，绝不能被 mask 过
            starts = list(range(0, tr_z[0].shape[0] - input_len - horizon + 1, 50))
            expect_y = np.stack(
                [tr_z[0][s + input_len : s + input_len + horizon] for s in starts]
            )
            if not np.allclose(Y, expect_y):
                raise AssertionError(f"{tag}: target 被意外修改（应为完整状态）")

            # 4b. 被遮挡位置在 input 里必须是 0
            if is_temporal:
                # 时间掩码：无观测时刻的整行都应为 0
                if not np.allclose(X[:, m == 0, :], 0.0):
                    raise AssertionError(f"{tag}: 时间掩码未生效（无观测时刻未置零）")
            else:
                blocked = np.where(m == 0)[0]
                if len(blocked) and not np.allclose(X[..., blocked], 0.0):
                    raise AssertionError(f"{tag}: 遮挡维度未置零")

            # 4c. 不遮挡时 input 应与原始窗口逐元素相等
            if ratio >= 1.0:
                expect_x = np.stack([tr_z[0][s : s + input_len] for s in starts])
                if not np.allclose(X, expect_x):
                    raise AssertionError(f"{tag}: 无遮挡时 input 被误改")

            print(f"  {PASS} {tag}  X={X.shape} Y={Y.shape}")

    # ---- 5. apply_mask 的噪声只加在观测维 ----
    m = build_mask(dim, "uniform", 0.5, seed=1)
    base = np.ones((3, 20, dim), dtype=np.float32)
    noisy = apply_mask(base, m, noise_std=0.1, noise_seed=0)
    blocked = np.where(m == 0)[0]
    observed = np.where(m == 1)[0]
    if len(blocked):
        check(np.allclose(noisy[..., blocked], 0.0), "加噪后遮挡维仍为 0")
    check(not np.allclose(noisy[..., observed], 1.0), "观测维确实被加了噪声")
    # 同 seed 噪声可复现
    noisy2 = apply_mask(base, m, noise_std=0.1, noise_seed=0)
    check(np.array_equal(noisy, noisy2), "同 seed 噪声可复现")


def test_standardize_no_leak() -> None:
    """标准化只允许使用训练集统计量，这里验证 test 不会污染 mean/std。"""
    print("\n[信息泄露检查]")
    train, test, meta = load_data("lorenz63")
    tr_z, te_z, mean, std = standardize(train, test, meta)
    check(np.allclose(mean, meta["train_mean"], atol=1e-5), "mean 来自训练集 meta")
    check(np.allclose(std, meta["train_std"], atol=1e-5), "std 来自训练集 meta")
    # 用训练集自身统计量标准化后，训练集均值应为 0
    check(abs(tr_z.mean()) < 0.05, "训练集自标准化后均值 ≈ 0")


def main() -> None:
    print("=" * 60)
    print("地基验收：数据 / 标准化 / 掩码 / 窗口")
    print("=" * 60)
    for system in ("lorenz63", "lorenz96"):
        test_system(system)
    test_standardize_no_leak()
    print("\n" + "=" * 60)
    print("全部通过。地基可用，可以开始训模型。")
    print("=" * 60)


if __name__ == "__main__":
    main()
