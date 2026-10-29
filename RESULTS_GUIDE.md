# 结果解读指南（数据跑完后按此步骤操作）

> 本文件是**方法**，不是结论。数据出来前不要往里填任何数字。

---

## 一、先确认实验是否有效（看之前必查）

跑完 `summarize.py` 后，先回答这三个"体检"问题。**任何一条不过，实验作废，先修再解读。**

| 体检项 | 合格标准 | 不合格说明 |
|---|---|---|
| **无信息泄露** | `test_pipeline.py` 全绿 | 掩码/mask 在 train/test 不一致 |
| **模型确实学到了** | 所有 `test_mse` 显著 < 1（标准化尺度下方差为 1） | 训练失败或 mask 全零 |
| **稀疏化真的变难** | 同模型下，ratio↓ 时 RMSE@10 单调↑ | 掩码逻辑有 bug，或掩码未生效 |

**第三条最关键。** 如果观测比例从 0.5 降到 1/6，误差却没变，说明 mask 根本没起作用——这时候任何"发现"都是假的。

---

## 二、四种可能的结果，以及各自意味着什么

### 情形 A：Transformer 优势随稀疏化扩大

```
VPT:      Transf ────────────  (缓降)
          LSTM   ──────╲
                       ╲──────  (陡降)
              0.5   0.33   0.17
```

**解读**：注意力机制更善于从稀疏信息中重建状态。
**这句话能写**："The advantage of attention over recurrence widens as observations become sparser."
**注意**：这时要去检查 Block 模式（连续缺失）是否呈现**相反**趋势——如果 block 下 LSTM 反超，那真正的机制是"空间结构 vs 随机缺失"，而不是"注意力更强"。

### 情形 B：Transformer 优势随稀疏化缩小甚至反转

**解读**：Transformer 依赖较长历史上下文，稀疏观测下它与 LSTM 的输入信息量差距被放大，但它的长程依赖假设反而成了负担。
**这句话能写**："Under severe sparsity, the transformer's advantage does not survive; recurrence appears more robust to information loss."
**这是很有价值的结果**——因为它直接挑战"Transformer 什么都更好"的流行叙事。

### 情形 C：短时准、长期飘（分裂）

```
              RMSE@10        VPT
Transf         0.10          23
LSTM           0.16          15
MLP            0.13          30    ← 注意这个
```

**解读**：模型学到的是**观测间的相关性**，而不是**系统动力学**。单步误差低，是因为它对训练分布内的转变模式记得牢；一旦进入 rollout，误差累积速度快于真正的动力学模型。

**这句话能写**：
> "The transformer achieves the lowest single-step error but the shortest valid prediction horizon — consistent with it fitting observational correlations rather than the underlying dynamics."

**⚠️ 这是最容易得出的情形，也最容易说过头。** 想坐实"学到相关性"，需要额外证据（至少一个）：
- MLP/线性模型 VPT 反超 Transformer
- 误差增长率（growth_rate）与短期误差**不相关**
- 视觉上：Transformer 的预测轨迹"形状对但相位漂移"

只靠一张 VPT 表就宣称"它没学到动力学"，会被审稿人一问就倒。

### 情形 D：三者没差别

**解读**：在 3 维 / 40 维尺度上，模型容量不是瓶颈，**稀疏观测才是**。
**这句话能写**："At these system sizes, architecture choice is not the limiting factor; observation sparsity is."
**这不是失败。** 它把问题从"哪个模型好"重新定位成"多少观测才够"——一个更好的问题。接下来该做的是**扫更多观测比例**，找出临界点。

---

## 三、必做的三个交叉检验（防止结论站不住）

### 检验 1：随机种子稳定性

**只训一次就下结论是最大的坑。**

选 2–3 个关键组合，换 `SEED` 重训三次：
```bash
# 修改 train.py 的 SEED 或加 --seed 参数，跑三次
```
若三次的 VPT 波动超过 20%，**任何"谁更好"的结论都不能写**，只能写趋势。

### 检验 2：Block vs Uniform 的对照

这两种掩码模拟**不同物理情境**（区域缺失 vs 传感器稀少）。若结论只在 uniform 成立、block 下反转，那你的发现是"关于掩码方式的"，不是"关于动力学的"——**必须改措辞**。

### 检验 3：含噪观测

真实传感器有噪声。用 `apply_mask(..., noise_std=0.05)` 重跑一组关键实验：
- 若结论不变 → 稳健
- 若结论消失 → 你的发现依赖"无噪声"这个不现实假设，必须写明

---

## 四、把结果写成邮件里的那句话

**一张表 + 一句结论，比十张图有效。**

### 结构（三句话，不超过 80 词）

1. **设定**：我在什么系统、什么条件下测的
2. **发现**：一个具体的、可证伪的对比
3. **推断**：这暗示什么机制

### 模板（按实际情况挑选，**不要照抄**）

> **A（优势扩大）**
> "Across both Lorenz-63 and Lorenz-96, the transformer's valid prediction horizon degraded more slowly than the LSTM's as I removed observations — at 1/6 density it retained roughly X% of its fully-observed horizon, against Y% for the LSTM. This suggests the benefit of attention in this setting comes from reconstructing state from sparse cues, not from model capacity."

> **B（优势消失）**
> "The transformer was the most accurate model when the full state was observed, but its advantage vanished below 1/3 observation density, where the LSTM matched or exceeded it. The ordering of architectures was not stable across observation regimes, which I did not expect."

> **C（短时/长期分裂）**
> "The transformer had the lowest single-step error of the three models, yet the shortest valid prediction horizon — the opposite of what its training objective would suggest. This is consistent with it fitting correlations between observed components rather than the dynamics that generate them."

> **D（无差别）**
> "All three architectures degraded at similar rates, which suggests that at these system sizes the binding constraint is observation density rather than model capacity."

### ⚠️ 写作纪律

- **不要写 "proves"** → 写 "is consistent with" / "suggests"
- **不要写 "significantly"** 除非真做了统计检验
- **每个数字都要能从 CSV 里查到**——他会问"这个 X% 怎么算的"
- **pilot 规模的实验，明确说 "in my preliminary runs"**

---

## 五、README 该放什么

只放三样：

1. **一张核心图**：VPT vs 观测比例（两张子图：两个系统）
2. **一段发现**：上面的三句话
3. **复现命令**：一条能跑完的命令

其余全部折叠进附录。**他只有 3 分钟，前 10 行决定他会不会往下看。**

---

## 六、诚实清单（提交前自检）

- [ ] 每个 `test_mse` 都 < 1（模型真的学到了）
- [ ] 稀疏化确实让误差单调上升（掩码生效）
- [ ] 关键组合跑了 ≥ 3 个随机种子
- [ ] Block 与 Uniform 的结论方向一致，或已说明差异
- [ ] 加噪实验做过，或明确声明未做
- [ ] 邮件里的每个数字都能追溯到 CSV
- [ ] 没有用 "proves" / "significantly" 这类过强的词
- [ ] 明确标注了这是 pilot 还是完整实验
