# Quant 16 · 线性回归、核平滑与面试经典题：OLS、Gauss–Markov、Ridge/Lasso

在线性回归（Linear Regression）的面试中，Quant Research 候选人往往会觉得这部分内容过于基础而掉以轻心。但实际上，诸如 Two Sigma、DE Shaw 和 Citadel 等顶级机构极其喜欢在面试中用回归问题来考察你。他们考察的不是你是否听说过 OLS，而是你对概率论基础的掌握、推导代数的熟练度，以及——最重要的是——**你是否知道在哪些情况下标准统计模型会失效**。金融数据时刻伴随着异方差、自相关和多重共线性，如果你不知道如何应对这些问题，就无法通过 QR 轮面试。

```text
线性回归面试核心心智模型（Core Mental Models）：
1. 单变量 OLS 终极公式：熟记 \hat\beta = \rho (\sigma_y / \sigma_x) 以及 R^2 = \rho^2。这一个公式能秒杀几乎所有基础题。
2. 回归的不可逆性：y 对 x 的回归斜率与 x 对 y 的回归斜率乘积为 \rho^2 \le 1，永远不要想当然地取倒数。
3. 几何投影直觉：将 OLS 视作 y 在 X 列空间上的正交投影（Orthogonal Projection）。正交性是推导残差性质的钥匙。
4. BLUE 不依赖正态性：Gauss-Markov 定理证明 OLS 是最佳线性无偏估计量时不包含“正态性”假设。正态性仅用于精确的小样本 t/F 检验。
5. 惩罚项的几何效应：Lasso 的 \ell_1 菱形带来稀疏性（变量选择），Ridge 的 \ell_2 圆球带来缩减（应对共线性但保留所有变量）。
```

> 🧭 **核心知识全景导览**
> - **模块一：OLS 几何与代数**：矩阵推导 ｜ 正交投影 ｜ 单变量三大核心公式 ｜ 逆向回归陷阱
> - **模块二：Gauss–Markov / BLUE**：五大假设 ｜ 正态性的迷思 ｜ 异方差与自相关（White/Newey-West）
> - **模块三：变量选择与收缩（Shrinkage）**：子集选择 ｜ 岭回归（Ridge） ｜ Lasso ｜ 几何直觉与比较
> - **模块四：核平滑与局部回归**：Nadaraya-Watson 核回归 ｜ 边界偏差与局部线性回归 ｜ 维数灾难与 CV 选择
> - **模块五：面试经典题库**：相关系数极值（Green Book） ｜ CAPM与均值回归 ｜ 遗漏变量偏差 ｜ 测量误差 ｜ 多重共线性 ｜ R² 陷阱
> - **模块六：一分钟答题结构 + 避坑**
> - **模块七：快速自测选择题**

---

## 模块一：OLS 几何与代数（ESL 3.2）

### 1. 矩阵形式与正规方程（Normal Equations）
对于多元线性回归模型 $y = X\beta + \varepsilon$（其中 $X$ 为 $N \times (p+1)$ 的满秩矩阵），普通最小二乘法（Ordinary Least Squares, OLS）的目标是最小化残差平方和 $\operatorname{RSS}(\beta) = \|y - X\beta\|_2^2$。

对 $\beta$ 求导并令导数为零，我们得到**正规方程**：
$$
X^\top X\hat\beta = X^\top y
$$
当矩阵 $X$ 满秩（Full Rank）时，封闭解（Closed Form）为：
$$
\hat\beta = (X^\top X)^{-1}X^\top y
$$

### 2. OLS 的几何直觉（Orthogonal Projection）
在几何意义上，最小化残差平方和相当于将 $y$ **正交投影**到 $X$ 的列向量所张成的子空间 $\mathrm{Col}(X)$ 上。
- **拟合值** $\hat{y} = X\hat\beta = X(X^\top X)^{-1}X^\top y = H y$，其中 $H = X(X^\top X)^{-1}X^\top$ 称为**帽子矩阵（Hat Matrix）**或投影矩阵。
- **残差向量** $\hat\varepsilon = y - \hat{y} = (I - H)y$ 必定垂直于 $X$ 的列空间，即残差 $\perp$ $X$ 的列。
- 模型有效自由度为 $\mathrm{df} = \mathrm{tr}(H) = p+1$。

### 3. 面试必考单变量公式（Univariate Formulas）
对于简单的单变量回归 $y = \alpha + \beta x + \varepsilon$，面试官期望你不用笔就能默写以下关系：
$$
\hat\beta = \frac{\operatorname{Cov}(x, y)}{\operatorname{Var}(x)} = \rho \frac{\sigma_y}{\sigma_x}
$$
$$
\hat\alpha = \bar{y} - \hat\beta \bar{x}
$$
$$
R^2 = \rho^2
$$
> **陷阱（Trap）：回归的不可逆性**
> 面试中常问：“如果 $y$ 对 $x$ 回归的斜率是 2，那么 $x$ 对 $y$ 回归的斜率是多少？”
> **错误答案**：$1/2$。
> **正确解析**：根据公式，$\hat\beta_{y \sim x} = \rho \frac{\sigma_y}{\sigma_x}$，而 $\hat\beta_{x \sim y} = \rho \frac{\sigma_x}{\sigma_y}$。二者的乘积是：
> $$ \hat\beta_{y \sim x} \times \hat\beta_{x \sim y} = \rho^2 \le 1 $$
> 所以回归斜率并不是简单的倒数关系！这也正是**均值回归（Regression to the Mean）**的本质体现。

---

## 模块二：Gauss–Markov 定理与 BLUE（ESL 3.2.2）

Gauss-Markov 定理指出，在特定假设下，OLS 估计量是**最佳线性无偏估计量（Best Linear Unbiased Estimator, BLUE）**，即在所有线性的、无偏的估计量中，OLS 的方差最小。

### 1. Gauss-Markov 假设
1. **参数线性（Linearity in parameters）**：模型形式确实为 $y = X\beta + \varepsilon$。
2. **外生性（Exogeneity）**：误差项的条件均值为零，$E[\varepsilon \mid X] = 0$。
3. **同方差性（Homoskedasticity）**：所有残差具有相同的方差。
4. **无自相关（No serial correlation）**：残差之间相互独立。
5. **无完全多重共线性（No perfect multicollinearity）**：设计矩阵 $X$ 满秩。

### 2. 经典面试陷阱：正态性（Normality）的迷思
**“OLS 需要假设误差项服从正态分布吗？”**
**答案是：不需要！**
OLS 要成为 BLUE，完全不需要正态性假设。正态性仅仅在我们需要进行**精确的有限样本 $t$ 检验或 $F$ 检验**时才需要。这一点面试官会反复确认。

### 3. 违反假设的后果与补救
当金融数据（尤其是时间序列或横截面数据）违背假设时：
- **异方差（Heteroskedasticity） / 自相关（Autocorrelation）**：此时 OLS 估计量**仍然是无偏且一致的**（Unbiased & Consistent），但是**标准误（Standard Errors）算错了**（不再是最小方差），导致你的 $t$ 统计量可能偏大，出现虚假的显著性。
- **补救措施**：使用稳健标准误（Robust Standard Errors）。应对异方差使用 **White 标准误**；同时应对异方差和自相关使用 **Newey–West 标准误**。

---

## 模块三：变量选择与收缩（ESL 3.3–3.4）

面对大量可能存在共线性的特征因子，我们需要对模型进行限制（Regularization）。

### 1. 传统方法（Subset Selection）
- **最优子集（Best Subset）** / **逐步回归（Forward / Backward Stepwise）**：在高层面上，这些离散的选择过程能够选出较好的变量，但是由于选择的离散性，方差通常较大。

### 2. 岭回归（Ridge Regression）
引入 $\ell_2$ 范数惩罚项来控制系数大小：
$$
\hat\beta^{\mathrm{ridge}} = \arg\min_\beta \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2
$$
其封闭解为：
$$
\hat\beta^{\mathrm{ridge}} = (X^\top X + \lambda I)^{-1}X^\top y
$$
**特点**：极好地处理多重共线性问题（引入偏差，降低方差）；**不会将任何系数精确收缩到 0**。

### 3. Lasso 回归
引入 $\ell_1$ 范数惩罚项：
$$
\hat\beta^{\mathrm{lasso}} = \arg\min_\beta \|y - X\beta\|_2^2 + \lambda \|\beta\|_1
$$
**特点**：因为 $\ell_1$ 惩罚的几何形状是尖锐的“菱形（Diamond）”，在等高线相切时极易切在坐标轴或顶点上，从而能够将部分系数**精确收缩到 0**，起到**内建的特征选择（Sparsity）**作用。

### 4. 方法对比矩阵

| 方法 | 惩罚项 | 偏差与方差 | 产生稀疏性？ | 能处理多重共线性？ |
| :--- | :--- | :--- | :---: | :--- |
| **最优子集** | 限制变量数 | 离散过程，高方差 | 是 | 视保留的子集而定 |
| **Ridge** | $\lambda \|\beta\|_2^2$ (圆球) | 引入偏差，降低方差 | 否 | 能极好地处理，解唯一 |
| **Lasso** | $\lambda \|\beta\|_1$ (菱形) | 引入偏差，降低方差 | 是 | 能，但对高度相关的变量随机选一个 |

*(注：PCR / PLS 等降维方法本质上是生成少量“衍生方向”（Derived Directions）进行回归，与带惩罚的变量选择思路不同，无需长篇赘述。)*

---

## 模块四：核平滑与局部回归（ESL 6.1–6.3）

### 1. Nadaraya–Watson 核回归
通过加权平均的方法估计局部期望，权重由带带宽（Bandwidth）$h$ 的核函数决定。$h$ 的选择是一个经典的偏差-方差权衡（Bias-Variance Tradeoff）。

### 2. 局部线性回归（Local Linear Regression）
Nadaraya-Watson 相当于局部拟合常数（Local Constant），这在数据边界处会导致严重的**边界偏差（Boundary Bias）**。**局部线性回归**通过拟合一条直线来一阶抵消这一边界偏差，这被称为自动核修缮（Automatic Kernel Carpentry）。

### 3. 维数灾难与逃生指南（Curse of Dimensionality in $\mathbb{R}^p$）
核平滑在低维空间效果极佳，但面临维数灾难。在 $\mathbb{R}^p$ 中，所有的点在局部邻域内都很稀疏。
**实际解决方案**：使用**结构化核（Structured Kernels）**或者**可加结构（Additive Structure）**作为解脱困境的现实方法。通过交叉验证（CV，连接 ESL 第 7 章的 Effective df 等内容）来选择最优的 $h$ 或者正则化参数 $\lambda$。

---

## 模块五：面试经典题库

以下题目糅合了 Green Book、HOTS 及主流 QR 的常考题。

### 1. Green Book：相关系数极值推导
> **原题**：已知随机变量 $X$ 与 $Y$ 的相关系数 $\rho_{xy} = 0.8$，$X$ 与 $Z$ 的相关系数 $\rho_{xz} = 0.8$。求 $Y$ 与 $Z$ 的相关系数 $\rho_{yz}$ 的最大与最小值。

**思路 / 推导 / 要点**：
相关系数矩阵必须是**半正定（Positive Semi-Definite, PSD）**的。即 Gram Matrix 的行列式需 $\ge 0$：
$$
\det\begin{pmatrix}1 & 0.8 & 0.8 \\ 0.8 & 1 & \rho \\ 0.8 & \rho & 1\end{pmatrix} \ge 0
$$
展开得：$\rho \in [2 \times 0.8^2 - 1,\ 1] = [0.28,\ 1]$。最大值为 $1$（Y与Z同向），最小值为 $0.28$（由向量夹角的余弦加倍公式 $\cos(2\theta)$ 在 $\cos\theta=0.8$ 时得出）。

### 2. HOTS 风格：CAPM Beta 与均值回归
> **原题**：股票波动率为 $2\%$，市场波动率为 $1\%$，相关系数为 $0.5$。求 $\beta$ 和 $R^2$、残差波动率。如果今天该股票上涨了 $+4\%$，预测市场涨幅；如果收益率 IID，预测明天股票的涨幅。

**思路 / 推导 / 要点**：
- $\beta = \rho \frac{\sigma_{\text{stock}}}{\sigma_{\text{market}}} = 0.5 \times \frac{2\%}{1\%} = 1$。
- $R^2 = \rho^2 = 0.5^2 = 0.25$。
- 残差波动率 = $\sigma_{\text{stock}}\sqrt{1-\rho^2} = 2\% \times \sqrt{0.75} \approx 1.732\%$。
- **逆向回归陷阱**：用股票预测市场的斜率 $\beta_{\text{reverse}} = \rho \frac{\sigma_{\text{market}}}{\sigma_{\text{stock}}} = 0.25$。预测市场涨幅 = $0.25 \times 4\% = 1\%$。
- IID 情况下，今天的涨幅对明天无预测力，明日预期应为无条件均值（近似为 0）。

### 3. HOTS 仿射变换后的相关性
> **原题**：$\operatorname{Corr}(X+5,Y)$ 和 $\operatorname{Corr}(5X,Y)$ 是多少（若 $X,Y$ 原始相关为 $\rho$）？

**思路 / 推导 / 要点**：
$\operatorname{Corr}(X+5,Y)=\rho$（平移不变）。$\operatorname{Corr}(5X,Y)=\rho$（正数缩放不变）；若乘以负数，相关系数变号为 $-\rho$。

### 4. 遗漏变量偏差（Omitted Variable Bias）
> **原题**：如果少加了一个关键解释变量，剩余变量的回归系数偏差符号是什么？

**思路 / 推导 / 要点**：
若真模型为 $y=\beta x+\gamma z+\varepsilon$，却只回归 $x$，则
$$\operatorname{plim}\hat\beta=\beta+\gamma\frac{\operatorname{Cov}(x,z)}{\operatorname{Var}(x)}.$$
偏差符号由 $\gamma$ 与 $\operatorname{Cov}(x,z)$ 同号与否决定；面试常要求你定性判断偏高还是偏低。

### 5. 自变量测量误差（Measurement Error in x）
> **原题**：如果你的自变量 $x$ 带有噪音，回归系数会怎样？

**思路 / 推导 / 要点**：
会导致斜率向零衰减（Attenuation toward 0）：观测到的是 $x^*=x+u$，分母 $\operatorname{Var}(x^*)$ 被虚增，而分子协方差被稀释。

### 6. 多重共线性（Multicollinearity）陷阱
> **原题**：共线性的预测 vs 解释困境。

**思路 / 推导 / 要点**：
用 VIF（方差膨胀因子）直观解释：系数的标准误无限放大，失去个别解释力；但整体预测在样本内依然有效。

### 7. 对冲比例与金融期货（Hedging / Futures）
> **原题**：回归价格变化以获得对冲比率。

**思路 / 推导 / 要点**：
利用方差最小化，最优对冲比率严格等于现货收益对期货收益的单变量 OLS 斜率（$\hat\beta$）。

### 8. R² 陷阱（R² Traps）
> **原题**：日收益率 $R^2 \approx 1\%$ 是否意味着信号无用？

**思路 / 推导 / 要点**：
样本内增加变量永远不会降低 $R^2$（要看 Adjusted $R^2$ 或 OOS）。对于日频回报信号，$1\%$ 的 $R^2$ 可能是极好的强信号，这区分了真正做过回测的面试者与书本理论家。

---

## 模块六：一分钟答题结构 + 避坑指南

```text
现场面试速答清单：
1. 听到单变量回归求斜率：立刻脱口而出 "斜率 = \rho * (\sigma_y / \sigma_x)"。
2. 听到逆向回归求斜率：立刻警觉乘积为 \rho^2。不要回答倒数！预测极端值必须展示均值回归的特征。
3. 听到 OLS 的假设要求：大声说出 "BLUE不依赖正态性"，只有小样本检验才需要。
4. 听到异方差/自相关的影响：明确区分 "系数依旧无偏/一致" 和 "标准误算错（通常被低估，导致虚假显著）"，并能报出 White 或 Newey-West。
5. 看到 Lasso 和 Ridge：从几何角度切入，用“菱形”解释为什么 Lasso 会让系数变为零，用“圆球”解释 Ridge 的平滑缩减。
```

---

## 模块七：快速自测选择题

```quiz
title: 快速选择题 1
question: 在一元线性回归中，y 对 x 的拟合 R² 与相关系数 ρ 的关系是？
answer: C
A. R² = ρ
B. R² = 1 - ρ²
C. R² = ρ²
D. R² = ρ / (1 - ρ)
explanation: 在一元线性回归中，R² 在数学上严格等于相关系数的平方 ρ²。
```

```quiz
title: 快速选择题 2
question: 已知 y 对 x 的回归斜率为 0.5。如果我们将 x 对 y 进行回归，得到的斜率可能是多少？
answer: B
A. 一定是 2.0
B. 一定小于等于 2.0
C. 一定大于 2.0
D. 一定等于 0.5
explanation: \beta_{y\sim x} * \beta_{x\sim y} = \rho^2 \le 1。由于 \beta_{y\sim x} = 0.5，因此 \beta_{x\sim y} \le 1 / 0.5 = 2.0。直接回答倒数是面试中极易触发的陷阱。
```

```quiz
title: 快速选择题 3
question: 下列关于 Gauss-Markov 定理的说法，哪项是正确的？
answer: D
A. 只有在误差项服从正态分布时，OLS 才是 BLUE
B. OLS 在存在严重异方差时仍然是 BLUE
C. 当模型存在多重共线性时，OLS 估计量是有偏的
D. OLS 作为最佳线性无偏估计量（BLUE），其推导过程完全不需要正态性假设
explanation: OLS 成为 BLUE 需要的假设不包含正态性；正态性仅用于精确的有限样本统计推断（如 t 检验/F检验）。异方差会破坏 BLUE，但依然无偏。
```

```quiz
title: 快速选择题 4
question: 股票A和市场指数的相关系数是 0.6，A的波动率是市场的两倍。用 A 去跑市场的单变量回归，A的 Beta 是多少？
answer: C
A. 0.6
B. 0.3
C. 1.2
D. 0.833
explanation: \beta = \rho * (\sigma_A / \sigma_M) = 0.6 * 2 = 1.2。
```

```quiz
title: 快速选择题 5
question: 承接上题（Beta=1.2，相关系数=0.6），股票A今天突然大涨了 +4%（相当于 +2个标准差），你预测市场指数的涨幅是多少个标准差？
answer: A
A. +1.2 个标准差
B. +2.4 个标准差
C. +2.0 个标准差
D. +1.0 个标准差
explanation: 在标准化的 Z 分数单位下，预测公式就是均值回归 \hat{Z}_y = \rho * Z_x。0.6 * 2 = 1.2 个标准差。预测逆向变化时，系数是 0.6 * (1/2) = 0.3。
```

```quiz
title: 快速选择题 6
question: 关于 Ridge（岭回归）和 Lasso 的比较，以下哪句是错的？
answer: B
A. Ridge 回归对应的惩罚项是 L2 范数，而 Lasso 是 L1 范数
B. Ridge 回归可以自动将不重要的特征系数精确收缩为 0，从而实现特征选择
C. 当特征存在高度共线性时，Ridge 会有平滑缩减的作用
D. 两者都能通过增加正则化程度来降低模型方差，但同时会引入偏差
explanation: Lasso (L1) 具有尖锐的菱形几何边界，能将系数精确压缩到 0 实现特征选择；Ridge (L2) 只能将系数缩减到接近于 0，但不会绝对等于 0。
```

```quiz
title: 快速选择题 7
question: 局部线性回归（Local Linear Regression）相较于 Nadaraya-Watson 核平滑，主要解决了什么问题？
answer: A
A. 边界偏差（Boundary Bias）
B. 维数灾难
C. 异方差
D. 多重共线性
explanation: NW 回归相当于局部常数拟合，在数据边界处由于不对称性容易产生严重的边界偏差。局部线性回归能够一阶消除这种边界偏差。
```

```quiz
title: 快速选择题 8
question: 随机变量 X, Y, Z 的两两相关系数中，ρ_XY = 0.8，ρ_XZ = 0.8。请问 ρ_YZ 的最大与最小可能范围是？
answer: B
A. [-1, 1]
B. [0.28, 1]
C. [-0.28, 0.28]
D. [0, 1]
explanation: 根据 Gram 矩阵半正定性或夹角余弦公式计算，极值范围是 [0.28, 1]。
```

```quiz
title: 快速选择题 9
question: 在线性回归中，如果你错误地遗漏了一个对因变量有正向影响，且与自变量 x 正相关的关键变量 z，x 的回归系数会：
answer: A
A. 向上偏高（正向偏差，Positive Bias）
B. 向下偏低（负向偏差，Negative Bias）
C. 保持无偏，但标准误变大
D. 收缩到 0
explanation: 遗漏变量偏差公式中，偏差 = 遗漏变量系数 * Cov(x,z)/Var(x)。两个正数相乘为正，导致你的估计值被夸大。
```

```quiz
title: 快速选择题 10
question: 如果你的自变量 x 带有随机测量误差，这对 OLS 系数估计有什么直接影响？
answer: B
A. 导致系数产生向上的虚假显著偏差
B. 导致斜率向零衰减（Attenuation toward zero）
C. 导致 R² 急剧上升
D. 没有任何偏差影响，只影响标准误
explanation: 自变量的测量误差会虚增自变量的方差，导致 \beta = Cov(x, y) / Var(x) 中的分母变大，系数估计值向 0 衰减。
```
