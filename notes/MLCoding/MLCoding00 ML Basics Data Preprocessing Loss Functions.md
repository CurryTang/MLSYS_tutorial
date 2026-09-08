# ML Coding 00 · ML 基础：数据预处理、数据泄露与经典损失函数

在机器学习系统设计与算法工程实践中，扎实的统计学基础与严密的数据管道工程是构建高可用模型的基石。许多模型在离线评测中指标优异，上线后效果却断崖式下跌，其根源往往不在于复杂的模型架构，而在于数据泄露（Data Leakage）、不恰当的缺失值处理（Missing Data Imputation）、样本不平衡的评估陷阱或对损失函数（Loss Functions）统计假设的认知偏差。

本篇系统梳理工业界机器学习基础与工程落地的 6 大核心模块：
1. **数据泄露（Data Leakage）机理与全方位防御体系**
2. **缺失值机制（MCAR / MAR / MNAR）与处理策略权衡**
3. **样本不平衡处理体系、表征学习与 VAE 核心价值**
4. **分类、排序、校准与业务评估指标全景体系（含可折叠代码实现）**
5. **经典损失函数推导：线性回归 vs 逻辑回归，MSE vs MAE 及统计学收敛特性**
6. **核心机制辨析与高频问题清单**

---

## 模块一：数据泄露（Data Leakage）机理与防御体系

### 1. 数据泄露的本质与危害

数据泄露（Data Leakage）是指**在模型训练过程中，非预期地引入了训练集外部的信息（尤其是目标变量或未来测试数据）**。

```text
数据泄露生命周期与危害：
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ 训练/离线验证阶段       │ ───> │ 离线指标虚假繁荣        │ ───> │ 生产线上真实部署        │
│ 意外窥探未来/目标信息   │      │ 验证集 AUC 0.98+        │      │ 无法获取泄露特征        │
│ 产生虚假强相关性特征    │      │ (Overly Optimistic)     │      │ 线上效果断崖式崩塌 💥   │
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

数据泄露会导致严重的过拟合与“虚假繁荣”——模型在训练集和验证集上表现完美，但由于泄露的信息在真实的生产推理环境中根本不存在，模型在线上部署时性能会发生灾难性衰退。

---

### 2. 四大高频数据泄露场景与典型案例

#### 场景 1：目标泄露 / 代理特征（Target Leakage / Proxy Features）

**核心机制**：特征本身是在**目标事件发生之后**才被生成、更新或记录的，但在离线回溯构建样本时被误作为输入特征。

- **典型案例 1（贷款违约预测）**：用“账户注销日期（`account_closed_date`）”或“催收退款状态码（`refund_status_code`）”来预测用户是否会违约。在现实业务流中，只有用户发生违约并进入催收流程后，这些字段才会被写入数据库。
- **典型案例 2（疾病诊断）**：在预测患者是否患有某种罕见病时，把“是否开具了该病的专属处方特效药（`prescribed_treatment_drug`）”作为特征。医生是在确诊后才开药的，将其作为预测特征属于本末倒置。

#### 场景 2：预处理泄露 / 全局统计量污染（Preprocessing Leakage）

**核心机制**：在划分训练集/测试集之前，在**全量数据集（Global Dataset）**上统一计算了全局统计量并完成了数据转换。

- **典型案例 1（特征缩放与归一化）**：在 `train_test_split` 之前，直接对全量数据调用 `StandardScaler().fit_transform(X)`。测试集的均值和方差提前渗透进了训练集，导致测试集分布发生信息外泄。
- **典型案例 2（文本特征词表与 TF-IDF）**：在全量语料上拟合 `TfidfVectorizer`，使得词表（Vocabulary）和逆文档频率（IDF）包含了测试集的信息。
- **典型案例 3（高基数目标编码 Target Encoding）**：在没有按折（Out-of-Fold）隔离的情况下，直接用全量数据的目标均值替换类别特征，导致模型直接“背诵”了测试集的目标分布。

#### 场景 3：时间序列的时间泄露（Temporal / Look-Ahead Leakage）

**核心机制**：用“未来时间戳”的数据来预测“过去”发生的事件，破坏了时序数据的因果律（Causality）。

- **典型案例 1（金融量化 / 股票预测）**：使用未来 5 天的滚动移动平均线（Rolling SMA centered）作为今日交易信号的特征。
- **典型案例 2（错误的交叉验证切分）**：对时序/用户行为日志采用随机 K 折交叉验证（Random K-Fold）。第 1 天的测试样本可能被第 5 天的训练样本“剧透”，完全掩盖了概念漂移（Concept Drift）与时序因果性。

#### 场景 4：样本组 / 重复实体泄露（Group / Duplication Leakage）

**核心机制**：属于**同一个实体（Entity / Subject）**的多条强相关或重复样本，被随机拆分到了训练集与测试集两端。

- **典型案例 1（医学图像诊断）**：同一位患者拍摄了 10 张不同角度的胸透 CT 切片。如果随机划分，该患者的 8 张切片在训练集，2 张在测试集。卷积神经网络可能会记住该患者独特的骨骼阴影或设备伪影，而不是泛化的病理特征。
- **典型案例 2（多会话用户推荐）**：同一用户在同一天内的 20 次点击行为被随机分散到训练和测试集中。

---

### 3. 工业级数据泄露防御策略

| 防御策略 | 核心实施要点 | 关键工具 / 库支持 | 解决的泄露类型 |
|---|---|---|---|
| **先拆分，后拟合（Split First, Fit Later）** | 必须在数据集划分后，仅在训练集上调用 `fit()`，测试集仅调用 `transform()`。严禁在切分前做全局缩放或插补。 | `sklearn.pipeline.Pipeline`, `ColumnTransformer` | 预处理泄露 |
| **时序前向链式切分（Time-Based Splitting）** | 严格基于时间戳排序，仅使用历史时间窗口预测未来，使用滚动切分而非随机打乱。 | `TimeSeriesSplit`, `PurgedGroupTimeSeriesSplit` | 时间泄露 |
| **实体分组隔离（Group-Aware Splitting）** | 确保同一患者、同一设备或同一用户的所有数据严格锁定在单侧（同在训练集或同在测试集）。 | `GroupKFold`, `GroupShuffleSplit`, `StratifiedGroupKFold` | 实体分组泄露 |
| **推理时间线可用性审计（Inference Timeline Audit）** | 针对每个特征严格提问：“在生产环境发起预测请求的毫秒瞬间，该字段在数据库中是否已经生成并可用？” | 特征元数据注册表（Feature Store 如 Feast）、数据血缘系统 | 目标与代理特征泄露 |

---

### 4. Quick Coding：防泄露 Pipeline 与 GroupKFold 实战

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. 模拟生成带有缺失值、分组实体的数据
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
groups = np.repeat(np.arange(100), 10)  # 100 个独立用户，每个用户 10 条记录
X[np.random.rand(*X.shape) < 0.1] = np.nan  # 注入 10% 缺失值

# 2. 构建严密的防泄露管道 (Pipeline 封装 Imputer + Scaler + Model)
# 管道确保所有转换步骤仅在每一折的训练集上 fit，绝不窥探测试集
model_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=42))
])

# 3. 使用 GroupKFold 确保同一用户数据不跨折泄露
gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # 核心：fit 仅接触当前折的训练数据
    model_pipeline.fit(X_train, y_train)
    oof_preds[val_idx] = model_pipeline.predict_proba(X_val)[:, 1]

cv_auc = roc_auc_score(y, oof_preds)
print(f"严格防泄露 GroupKFold 5-Fold OOF AUC: {cv_auc:.4f}")
```

---

## 模块二：缺失值处理策略与统计权衡（Handling Missing Data）

### 1. 三大统计缺失机制（Missingness Mechanisms）

统计学家 Rubin 将数据缺失机制划分为以下三类：

```text
数据缺失机制分类：
┌──────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ 缺失机制类别                         │ 统计学数学定义与核心特征                                               │
├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. 完全随机缺失 (MCAR)               │ P(M | Y_obs, Y_mis) = P(M)                                             │
│    Missing Completely at Random      │ 缺失与任何已观测或未观测变量均无关（如传感器偶然丢包、问卷纸张偶发破损）│
├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. 随机缺失 (MAR)                    │ P(M | Y_obs, Y_mis) = P(M | Y_obs)                                     │
│    Missing at Random                 │ 缺失倾向依赖于其他已观测特征，但与缺失值本身无关（如老年人更少填写手机号│
│                                      │ 但在已知年龄的情况下，手机号缺失概率与手机号本身取值无关）             │
├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. 非随机缺失 (MNAR)                 │ P(M | Y_obs, Y_mis) 依赖于 Y_mis 本身                                  │
│    Missing Not at Random             │ 缺失本身携带强烈的未观测业务信号（如超高收入者或极低收入者更倾向于拒填│
│                                      │ 收入字段，缺失事实本身具有极高信息量）                                 │
└──────────────────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

### 2. 五大缺失值处理策略综合对比表

| 处理策略 | 适用场景 | 优势（Pros） | 劣势与权衡（Cons / Trade-offs） |
|---|---|---|---|
| **行/列直接删除（Listwise / Column Deletion）** | MCAR 机制且缺失率极低（<3%~5%）；或整列缺失率超过 80%~90%。 | 实现极简；若符合 MCAR 则不会引入人为合成的分布偏差。 | 严重损失样本量；若实际为 MAR 或 MNAR 会导致剧烈的**样本选择偏差（Selection Bias）**。 |
| **简单统计量填充（Mean / Median / Mode）** | 快速基线；数值型（中位数抗偏态）或类别型（众数/常量）；缺失率较低。 | 计算开销极低；在线实时推理部署成本低，易于持久化。 | **扭曲特征原有分布**，人为低估特征方差，完全破坏变量之间的协方差与相关性。 |
| **缺失指示变量（Missing Indicator: `is_missing`）** | MNAR 场景；“缺失这一事实本身”具有极强业务预测信号（如用户跳过可选收入填报）。 | 保留了“缺失行为”所蕴含的原生业务信号。 | 若盲目应用于所有特征会导致特征维度翻倍；可能在共线性与稀疏度上引入挑战。 |
| **基于模型的插补（Model-Based: KNN, MICE / IterativeImputer, MissForest）** | 特征间存在复杂的非线性交互；中等规模的高价值表格数据集。 | 充分保留特征间的多变量相关性、协方差与方差分布。 | 计算复杂度高；在线推理部署困难（需加载插补模型）；存在多级误差级联风险。 |
| **树模型原生默认路径路由（Native Tree Handling）** | LightGBM, XGBoost, CatBoost 等基于决策树的梯度提升模型。 | 无需手工插补；树分裂时通过评估将缺失值分配到左/右子树的最优增益自动选择默认路由。 | 仅限特定树模型使用；无法直接推广到线性模型、SVM 或深度神经网络。 |

---

### 3. 统计学深度权衡剖析

1. **方差收缩与分布扭曲（Variance Shrinkage）**：
   若使用均值填充 $x_{\text{imputed}} = \bar{x}$，填充后的样本方差计算为：

$$\text{Var}(X_{\text{imputed}}) = \frac{N_{\text{obs}}}{N_{\text{total}}} \text{Var}(X_{\text{obs}}) < \text{Var}(X_{\text{obs}})$$

   人为压低了特征方差，使后续基于方差的特征选择或线性模型权重估计产生统计偏差。
2. **多重插补（MICE: Multivariate Imputation by Chained Equations）的优势**：
   通过链式方程循环回归，针对每个缺失特征以其他特征作为自变量进行多轮迭代建模预测，并注入适度扰动残差，从而真实还原特征间的相关矩阵。

---

### 4. Quick Coding：带 Missing Indicator 的鲁棒插补 Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# 1. 构造包含 MNAR 信号的样本数据
df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 50, np.nan, 60],
    'income': [50000, np.nan, 120000, 80000, np.nan, 200000, 95000],  # 高收入倾向于缺失 (MNAR)
    'credit_score': [650, 700, 750, 680, 710, 790, 720]
})
y = np.array([0, 1, 0, 1, 0, 1, 0])

# 2. 构建组合插补器：同时获得 (中位数填充值 + 缺失二值指示标记)
numeric_features = ['age', 'income', 'credit_score']

numeric_transformer = FeatureUnion([
    ('imputed_features', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])),
    ('missing_indicators', MissingIndicator())  # 自动提取布尔标志列
])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features)]
)

full_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', Ridge())
])

full_pipeline.fit(df, y)
print("Pipeline 训练完成，转换后特征维度（含 is_missing 指示列）:", 
      full_pipeline.named_steps['preprocess'].transform(df).shape)
```

---

## 模块三：样本不平衡处理体系、表征学习与 VAE 核心价值

### 1. 不平衡数据处理的三个层级

在实际机器学习系统中，处理类别不平衡主要分布在三个层级：

* **数据层**：
  * **欠采样（Under-sampling）**：Random Under-sampling、Tomek Links（识别并移除异类最近邻对中的多数类以清晰类别分界）、ENN（Edited Nearest Neighbours，清理边界噪点）；
  * **过采样（Over-sampling）**：SMOTE（在特征空间通过 $k$ 近邻线性插值合成少数类样本）、ADASYN（根据少数类样本周围多数类的密集程度自适应分配插值权重）；
  * **针对性特征/样本增强**：长尾特征扰动与数据扩增。
* **算法与损失层**：
  * **代价敏感加权（Cost-sensitive / Class Weights）**：在损失函数中按类别频次反比赋予样本权重 $w_c \propto \frac{1}{N_c}$；
  * **聚焦损失（Focal Loss）**：引入调制因子 $(1 - p_t)^\gamma$，自适应衰减易分类样本对梯度的贡献；
  * **任务重构为单分类或异常检测**：One-Class SVM、Isolation Forest，避开极度不平衡的有监督分类直接学习正常样本分布支持集。
* **决策与后处理层**：
  * **动态阈值微调（Threshold Moving / Threshold Tuning）**：不直接采用 0.5 默认截断，依据验证集上的特定业务效用函数（如最大化 $F_\beta$ 或收益总和）搜索最优决策阈值；
  * **概率校准（Platt Scaling, Isotonic Regression）**：纠正因采样或加权导致的输出后验概率偏离真实经验发生率的问题；
  * **业务容量限制下的 Top-$k$ 截断**：按预测概率从高到低排序，仅截取前 $k$ 个最高风险/收益样本进入下游执行流。

---

### 2. 为什么严重不平衡有时对业务“没关系”？

1. **ROC-AUC 的排序不变性**：
   ROC-AUC 的统计学本质是 Wilcoxon-Mann-Whitney 统计量，等价于从正负样本中各随机抽取一个样本，正样本预测概率大于负样本预测概率的先验期望：
   $$ \text{AUC} = P(S^+ > S^-) $$
   排序关系仅取决于条件分布 $P(X \mid Y=1)$ 与 $P(X \mid Y=0)$ 在投影方向上的可分离度，在数学上完全独立于类别先验概率 $P(Y)$。即便负样本数量增加数倍，只要正负样本内部的分数分布未变，ROC-AUC 保持数学恒定。
2. **决策场景只依赖 Top 排序**：
   在推荐系统召回排序、量化多因子选股或风控初筛中，业务逻辑往往是选取固定容量的头部样本（如每日做多 Top 1% 股票，或人工审核前 500 笔可疑交易）。此时只要模型对头部的相对排序准确，绝对先验概率的偏移不影响最终决策集的构成。
3. **ROC-AUC 的“虚假繁荣”陷阱与 PR-AUC 边界**：
   虽然 ROC-AUC 对先验不敏感，但在极度不平衡下（如正例比例为 0.1%），假阳率公式为：
   $$ \text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}} $$
   庞大的 $\text{TN}$ 会极度稀释 $\text{FPR}$ 的分母。模型即便产生数千个误报（$\text{FP}$ 远超 $\text{TP}$），$\text{FPR}$ 依然极低，表现为 ROC-AUC 高达 0.98，但线上真实精确率（Precision）可能不足 5%。这也是为何实际反欺诈与故障诊断更推荐 **PR-AUC (Average Precision)**。

---

### 3. 对比学习 (SupCon)、SMOTE 与 Focal Loss 机制对比

| 方法 | 基本思想 | 适用范围 | 局限与边界 |
|---|---|---|---|
| **SMOTE** | 在特征空间中寻找少数类样本的 $k$ 近邻，通过线性插值合成新样本：$x_{\text{new}} = x + \lambda(x_{nn} - x)$。 | 中低维结构化表格数据；少数类分布连续且无大量边界重叠的场景。 | 高维稀疏特征下失效（维数灾难）；会盲目插值噪声与离群点，加剧类别混淆。 |
| **Focal Loss** | 在交叉熵损失上引入动态衰减因子 $(1 - p_t)^\gamma$，自适应降低易分类样本对梯度的贡献，强迫网络聚焦于难分样本。 | 密集预测（如目标检测）、高容量神经网络、样本极度不平衡且不希望修改采样率的端到端训练。 | 对标签噪声（Label Noise）极度敏感，因错误标注的样本天然会被视作“极难样本”而赋予极高权重。 |
| **对比学习 (SupCon)** | 利用样本间成对约束，拉近同类嵌入距离、推远异类嵌入距离，学习具有判别性的低维几何流形。 | 高维复杂表征学习（文本、时序、图表征）；长尾分布（Long-tailed recognition）；少样本冷启动。 | 训练开销大（依赖大 Batch Size 或 Memory Bank），需要精细构建正负样本对与表征投影头。 |

---

### 4. 变分自编码器（VAE）在噪声、不平衡与低信噪比数据中的价值

在量化金融高频时序与强噪声表格建模中，数据通常具有极低信噪比（$\text{SNR} < 0.05$）与非平稳性。变分自编码器（VAE，包括 CVAE、Bottleneck Autoencoder）展现出以下核心价值：

* **隐式高斯扰动与去噪正则化**：
  传统 Autoencoder 或 MLP 容易记忆微观结构中的高频随机噪声导致过拟合。VAE 通过将输入编码为均值 $\mu$ 与方差 $\sigma$，并引入重参数化技巧（Reparameterization Trick）采样：
  $$ z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$
  强制隐空间满足高斯先验 $\mathcal{N}(0, I)$。这种随机扰动相当于在隐特征上施加连续的数据增强，迫使解码器或下游预测头只关注宏观拓扑流形，过滤高频噪声。
* **非线性宏观状态因子（Market Regime）抽取**：
  复杂系统往往由少数不可观测的潜变量驱动（如系统状态切换、流动性枯竭）。线性主成分分析（PCA）无法捕获跨维度的非线性协同交互。VAE 的低维 Bottleneck $z$ 能提取正交、连续的非线性因子，作为下游树模型（LightGBM）或时序模型（Transformer）的稳健输入。
* **多任务联合预训练（End-to-End Joint Loss）**：
  将 VAE 特征重构损失与下游任务预测损失联合端到端优化：
  $$ \mathcal{L} = \mathcal{L}_{\text{prediction}}(y, \hat{y}) + \lambda_1 \mathcal{L}_{\text{recon}}(x, \hat{x}) + \lambda_2 D_{\text{KL}}(q(z \mid x) \parallel p(z)) $$
  无监督重构项充当强正则化约束，防止模型参数过早坍缩到局部假相关（Spurious Correlation）中。
* **不平衡与异常检测**：
  极端异常模式（系统崩盘、离群故障）在历史样本中极度稀缺。基于 VAE 的重构误差 $\|x - \hat{x}\|_2^2$ 或边缘似然估计，可以直接作为样本的“异常度评分”（Anomaly Score），用于头寸保护或风险兜底。

<details>
<summary><b>实现代码与解析：带下游联合预测头的 VAE 完整架构（PyTorch）</b></summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularVAEWithJointHead(nn.Module):
    """带下游联合预测头的变分自编码器架构"""
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        # 编码器 (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器 (Decoder: 特征重构去噪)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 联合预测头 (Prediction Head: 下游回归/分类任务)
        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧：z = mu + sigma * eps"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        pred_y = self.pred_head(z)
        return recon_x, pred_y, mu, logvar

def compute_joint_vae_loss(
    x: torch.Tensor, recon_x: torch.Tensor, 
    y_true: torch.Tensor, y_pred: torch.Tensor, 
    mu: torch.Tensor, logvar: torch.Tensor,
    lambda_recon: float = 1.0, lambda_kl: float = 0.01
):
    """端到端联合优化损失计算：预测 MSE + 重构 MSE + KL 散度约束"""
    pred_loss = F.mse_loss(y_pred.squeeze(-1), y_true)
    recon_loss = F.mse_loss(recon_x, x)
    # KL 散度：-0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    total_loss = pred_loss + lambda_recon * recon_loss + lambda_kl * kl_loss
    return total_loss, {
        "pred_loss": pred_loss.item(),
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item()
    }
```
</details>

---

## 模块四：分类、排序、校准与业务评估指标全景体系

```ml-metrics-demo
```

### 1. 十大常用评估指标全景比对矩阵

| 指标类型 | 指标名称 | 核心定义 / 计算公式 | 核心适用场景 | 盲区与潜在陷阱 |
|---|---|---|---|---|
| **排序类（阈值无关）** | **ROC-AUC** | TPR 对 FPR 的积分曲线下宽度，统计本质为 $P(S^+ > S^-)$。 | 评估模型全局分离能力；类别分布相对稳定或需要与先验解耦的模型横向对比。 | 负样本巨大时对假阳率钝化，在极度不平衡下易产生虚假繁荣。 |
|  | **PR-AUC (AP)** | Precision 对 Recall 积分曲线下宽度，加权阶梯面积：$\sum (R_k - R_{k-1})P_k$。 | **极度不平衡分类（反欺诈、违约预测、故障排查）核心指标**；聚焦于正样本检出率与查准率。 | 对负样本纯度不敏感；若无统一基准线（随机猜测基准为正类先验比例 $P$），跨数据集难以横向比较。 |
| **决策类（阈值相关）** | **F1-Score / $F_\beta$** | $F_\beta = (1 + \beta^2)\frac{P \cdot R}{\beta^2 P + R}$，调和平均数。 | 单一阈值上线决策；根据业务诉求微调检出偏好（如漏报代价高设 $\beta=2$）。 | 强依赖所选固定截断阈值；未考虑不同预测概率区间的风险分布。 |
|  | **Precision@k / Recall@k** | 预测置信度最高的前 $k$ 个样本中的查准率或查全率。 | 生产端具有严格吞吐上限（如人工复审团队每日限额、推荐系统前 $k$ 位展现）。 | 仅衡量头部排序质量，对 $k$ 之后的长尾分布完全盲区。 |
|  | **Balanced Accuracy** | $\frac{\text{TPR} + \text{TNR}}{2} = \frac{\text{Recall}_{\text{pos}} + \text{Recall}_{\text{neg}}}{2}$。 | 需要兼顾每一个类别的准确性，避免模型全部预测为多数类。 | 对极端分类器容易给出钝化评分，忽略了正负类在业务侧的不对称成本。 |
| **概率质量与校准** | **Log-Loss (Cross-Entropy)** | $-\frac{1}{N}\sum [y \ln p + (1-y)\ln(1-p)]$。 | 概率预测敏感任务（如点击率预估 CTR、期望收益定价）。 | 极易受高置信度错误分类的剧烈惩罚；受类别不平衡先验漂移影响极大。 |
|  | **Brier Score** | $\frac{1}{N}\sum (p_i - y_i)^2$，概率空间的均方误差。 | 衡量校准质量；可严格分解为可靠性（Reliability）、分辨率（Resolution）和不确定性。 | 无法直接替代分类决策阈值设计。 |
|  | **ECE (Expected Calibration Error)** | 概率分桶后，桶内置信度与真实标签比例的加权差绝对值。 | 风险定价系统、安全关键系统（医疗诊断、信贷授信）中的模型可信度验证。 | 结果受分桶策略（固定宽度 vs 等频分桶）影响较大，且不衡量区分能力。 |
| **量化 / 业务类** | **Rank IC (Information Coefficient)** | 预测打分排名与实际未来收益排名的 Spearman 秩相关系数。 | **量化截面 Alpha 因子有效性评价指标**；评估截面相对强弱。 | 无法衡量收益的非线性厚尾特征以及实际扣减交易滑点/换手率后的表现。 |
|  | **Expected Business Cost** | $\sum_{i,j} C_{ij} \cdot P(\hat{Y}=i, Y=j)$，业务代价矩阵。 | 终极线上决策：为误报（FP）和漏报（FN）赋予显式资金损失函数。 | 业务损失矩阵的量化成本往往难以动态精确建模。 |

---

### 2. 十大评估指标底层实现与解析（可折叠代码块）

<details>
<summary><b>实现 1：ROC-AUC（基于 Wilcoxon-Mann-Whitney 秩和检验算法）</b></summary>

```python
import numpy as np

def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """计算二分类 ROC-AUC
    
    数学原理：Wilcoxon-Mann-Whitney 统计量
    AUC = (sum(rank(S_pos)) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    支持平局分数（Tied Scores）的平均秩次处理。
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)
    
    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true 必须同时包含正例与负例样本")
        
    # 计算升序排列索引
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    
    # 平局分数的平均化（Fractional Ranking）
    sorted_scores = y_score[order]
    unique_scores, inverse_indices, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    if len(unique_scores) < len(y_score):
        tie_ranks = np.cumsum(counts) - (counts - 1) / 2.0
        ranks = tie_ranks[inverse_indices][np.argsort(order)]
    
    sum_pos_ranks = np.sum(ranks[pos_mask])
    u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))

# 测试验证
y_t = np.array([0, 0, 1, 1])
y_s = np.array([0.1, 0.4, 0.35, 0.8])
print("ROC-AUC:", compute_roc_auc(y_t, y_s))  # 0.75
```
</details>

<details>
<summary><b>实现 2：PR-AUC / Average Precision（梯步加权面积法）</b></summary>

```python
import numpy as np

def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """计算 PR-AUC / Average Precision (AP)
    
    数学定义：AP = sum_k (R_k - R_{k-1}) * P_k
    按预测分数降序排列，逐点计算 Precision 与 Recall 的梯步变化。
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    n_pos = tp[-1]
    
    if n_pos == 0:
        return 0.0
        
    precision = tp / (tp + fp)
    recall = tp / n_pos
    
    # 前驱召回率点 (R_0 = 0)
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    recall_diff = recall - recall_prev
    
    return float(np.sum(precision * recall_diff))

# 测试验证
print("Average Precision:", compute_average_precision(y_t, y_s))
```
</details>

<details>
<summary><b>实现 3：F1-Score 与 F-beta 评分（阈值决策指标）</b></summary>

```python
import numpy as np

def compute_f_beta(
    y_true: np.ndarray, 
    y_score: np.ndarray, 
    threshold: float = 0.5, 
    beta: float = 1.0, 
    eps: float = 1e-12
) -> float:
    """计算二分类在指定决策阈值下的 F-beta 评分
    
    数学公式：F_beta = (1 + beta^2) * (P * R) / (beta^2 * P + R)
    beta = 1.0: F1-Score (平衡精确率与召回率)
    beta = 2.0: 偏向召回率 (如反欺诈、重大疾病筛查，漏报成本极高)
    beta = 0.5: 偏向精确率 (如垃圾邮件拦截，误报成本极高)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = (np.asarray(y_score).ravel() >= threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    
    beta_sq = beta ** 2
    f_beta = (1.0 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + eps)
    return float(f_beta)

# 测试验证
print("F1-Score:", compute_f_beta(y_t, y_s, threshold=0.5, beta=1.0))
print("F2-Score:", compute_f_beta(y_t, y_s, threshold=0.5, beta=2.0))
```
</details>

<details>
<summary><b>实现 4：Precision@k 与 Recall@k（容量受限 Top-k 截断）</b></summary>

```python
import numpy as np

def compute_precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> tuple[float, float]:
    """计算置信度最高的 Top-k 样本中的查准率与查全率
    
    适用场景：人工复审名额受限、推荐系统前 k 个曝光位等。
    使用 argpartition 保证 O(N) 的选择复杂度。
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    n = len(y_true)
    k = min(max(1, k), n)
    
    # 快速获取 Top-k 索引
    top_k_indices = np.argpartition(-y_score, k - 1)[:k]
    
    hits = np.sum(y_true[top_k_indices] == 1)
    total_positives = np.sum(y_true == 1)
    
    p_at_k = float(hits / k)
    r_at_k = float(hits / total_positives) if total_positives > 0 else 0.0
    return p_at_k, r_at_k

# 测试验证
print("P@2, R@2:", compute_precision_recall_at_k(y_t, y_s, k=2))
```
</details>

<details>
<summary><b>实现 5：Balanced Accuracy（各类别召回率宏平均）</b></summary>

```python
import numpy as np

def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平衡准确率 Balanced Accuracy
    
    数学定义：各类别召回率（Sensitivity / Specificity）的未加权平均：
    Balanced_Acc = 0.5 * (TPR + TNR)
    完全克服多数类掩盖少数类预测失败的虚高准确率陷阱。
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.unique(y_true)
    
    recalls = []
    for c in classes:
        mask = (y_true == c)
        total_c = np.sum(mask)
        if total_c > 0:
            tp_c = np.sum((y_pred == c) & mask)
            recalls.append(tp_c / total_c)
            
    return float(np.mean(recalls)) if len(recalls) > 0 else 0.0

# 测试验证
print("Balanced Acc:", compute_balanced_accuracy(y_t, (y_s >= 0.5).astype(int)))
```
</details>

<details>
<summary><b>实现 6：Log-Loss / Binary Cross-Entropy（数值稳定对数损失）</b></summary>

```python
import numpy as np

def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """计算数值稳定的二元对数损失（Log-Loss / BCE）
    
    数学公式：-1/N * sum(y * ln(p) + (1-y) * ln(1-p))
    边界保护：将预测概率 clip 到 [eps, 1-eps] 防止 log(0) 产生 NaN。
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.clip(np.asarray(y_prob, dtype=float).ravel(), eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(loss)

# 测试验证
print("Log Loss:", compute_log_loss(y_t, y_s))
```
</details>

<details>
<summary><b>实现 7：Brier Score（概率空间均方误差与校准质量）</b></summary>

```python
import numpy as np

def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """计算二分类 Brier Score
    
    数学公式：BS = 1/N * sum (p_i - y_i)^2
    性质：
    1. 取值范围 [0, 1]，0 表示完美校准与完美分类；
    2. 可严格分解为：Brier = Reliability - Resolution + Uncertainty
       - Reliability (可靠性/校准误差)：概率预测是否匹配真实发生频率；
       - Resolution (分辨率)：模型区分类别的能力；
       - Uncertainty (固有不确定性)：事件先验方差 p*(1-p)。
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    return float(np.mean((y_prob - y_true) ** 2))

# 测试验证
print("Brier Score:", compute_brier_score(y_t, y_s))
```
</details>

<details>
<summary><b>实现 8：Expected Calibration Error / ECE（等宽分桶期望校准误差）</b></summary>

```python
import numpy as np

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """计算期望校准误差 ECE（Expected Calibration Error）
    
    数学公式：ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    衡量模型输出概率与真实观测经验频率之间的绝对校准差距。
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n = len(y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)
            
    return float(ece)

# 测试验证
print("ECE (10 bins):", compute_ece(y_t, y_s, n_bins=10))
```
</details>

<details>
<summary><b>实现 9：Rank IC（截面 Spearman 秩相关系数）</b></summary>

```python
import numpy as np

def compute_rank_ic(pred_scores: np.ndarray, true_returns: np.ndarray) -> float:
    """计算量化 Alpha 因子截面 Rank IC（Spearman 秩相关系数）
    
    数学定义：预测得分排序向量与实际未来收益排序向量的 Pearson 线性相关系数。
    评估因子对全市场资产相对表现的单调排序能力。
    """
    pred_scores = np.asarray(pred_scores).ravel()
    true_returns = np.asarray(true_returns).ravel()
    
    def rank_array(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        return ranks
        
    rank_p = rank_array(pred_scores)
    rank_y = rank_array(true_returns)
    
    # 协方差与方差计算
    cov = np.cov(rank_p, rank_y)[0, 1]
    std_p = np.std(rank_p, ddof=1)
    std_y = np.std(rank_y, ddof=1)
    
    if std_p == 0 or std_y == 0:
        return 0.0
    return float(cov / (std_p * std_y))

# 测试验证
print("Rank IC:", compute_rank_ic(y_s, y_t))
```
</details>

<details>
<summary><b>实现 10：Expected Business Cost（业务代价矩阵加权损失）</b></summary>

```python
import numpy as np

def compute_expected_business_cost(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    cost_matrix: np.ndarray
) -> float:
    """计算线上决策的期望业务代价
    
    参数定义：
    cost_matrix: 二维数组 C[pred_class, true_class]
    例如风控二分类代价矩阵：
    cost_matrix = [[C_00 (正常判为正常: 0元),   C_01 (盗刷漏报: 损失1000元)],
                   [C_10 (误封正常用户: 损失50元), C_11 (盗刷拦截: 挽损成本5元)]]
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    
    # 向量化直接索引对应成本
    costs = cost_matrix[y_pred, y_true]
    return float(np.mean(costs))

# 测试验证
cost_mat = np.array([
    [0.0, 1000.0],  # 预测为 0
    [50.0, 5.0]     # 预测为 1
])
y_p_hard = (y_s >= 0.5).astype(int)
print("平均业务单笔损失:", compute_expected_business_cost(y_t, y_p_hard, cost_mat))
```
</details>

---

## 模块五：经典损失函数剖析：线性回归 vs 逻辑回归，MSE vs MAE

### 1. 线性回归目标函数与高斯 MLE 概率推导

线性回归使用均方误差（Mean Squared Error, MSE）或普通最小二乘法（Ordinary Least Squares, OLS）作为目标函数：

$$\mathcal{L}_{\text{Linear}}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

#### 概率论推导（Gaussian MLE Derivation）

假设目标值 $y_i$ 与模型预测值 $\mathbf{w}^T \mathbf{x}_i$ 之间的残差 $\epsilon_i$ 独立同分布于均值为 0、方差为 $\sigma^2$ 的一维高斯分布：

$$y_i = \mathbf{w}^T \mathbf{x}_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2) \implies y_i \mid \mathbf{x}_i \sim \mathcal{N}(\mathbf{w}^T \mathbf{x}_i, \sigma^2)$$

其样本条件概率密度为：

$$p(y_i \mid \mathbf{x}_i; \mathbf{w}, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y_i - \mathbf{w}^T \mathbf{x}_i)^2}{2\sigma^2} \right)$$

构建全样本的对数似然函数 $\ell(\mathbf{w})$：

$$\ell(\mathbf{w}) = \sum_{i=1}^N \ln p(y_i \mid \mathbf{x}_i; \mathbf{w}, \sigma^2) = -\frac{N}{2} \ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

最大化对数似然 $\max_{\mathbf{w}} \ell(\mathbf{w})$ 等价于最小化负对数似然，常数项舍去后即精确等价于**最小化均方误差（MSE）**：

$$\arg\max_{\mathbf{w}} \ell(\mathbf{w}) \iff \arg\min_{\mathbf{w}} \frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

---

### 2. 逻辑回归目标函数与伯努利 MLE 推导

对于二分类问题 $y_i \in \{0, 1\}$，逻辑回归通过 Sigmoid 函数将线性输出映射为后验概率 $\hat{p}_i$：

$$\hat{p}_i = \sigma(\mathbf{w}^T \mathbf{x}_i) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}_i}}$$

假设 $y_i \mid \mathbf{x}_i$ 服从伯努利分布 $\text{Bernoulli}(\hat{p}_i)$，其概率质量函数为：

$$P(y_i \mid \mathbf{x}_i) = \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i}$$

全样本对数似然函数为：

$$\ell(\mathbf{w}) = \sum_{i=1}^N \left[ y_i \ln \hat{p}_i + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

取负均值得到二元交叉熵损失（Binary Cross-Entropy / Log Loss）：

$$\mathcal{L}_{\text{Logistic}}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

---

### 3. 为什么逻辑回归分类不能使用 MSE 损失？

许多初学者会问：“既然 MSE 能衡量误差，为什么不能直接在逻辑回归的 $\hat{p}_i = \sigma(\mathbf{w}^T \mathbf{x}_i)$ 上使用 MSE 损失？”

$$\mathcal{L}_{\text{MSE-Logistic}}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (y_i - \sigma(\mathbf{w}^T \mathbf{x}_i))^2$$

**不能使用 MSE 的三大根本原因**：

#### 原因 1：非凸性（Non-Convexity）与局部极小值陷阱

- **Log Loss** 与线性参数 $\mathbf{w}$ 结合是严格的**凸函数（Convex Function）**，其 Hessian 矩阵半正定，保证任意梯度下降算法都能收敛到全局全局最优解。
- **MSE** 与非线性的 Sigmoid 函数复合后，损失函数曲面变得高度**非凸（Non-Convex）**，存在大量平坦区域（Platoons）、鞍点（Saddle Points）和局部极小值（Local Minima），梯度下降极易卡死。

#### 原因 2：梯度消失与错误惩罚软弱（Vanishing Gradient on Severe Errors）

对比两者的参数梯度对残差的响应：

1. **MSE 损失关于参数 $\mathbf{w}$ 的梯度**：
   令 $z_i = \mathbf{w}^T \mathbf{x}_i$，根据链式法则：

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \mathbf{w}} = \frac{2}{N} \sum_{i=1}^N (\hat{p}_i - y_i) \cdot \sigma'(z_i) \cdot \mathbf{x}_i = \frac{2}{N} \sum_{i=1}^N (\hat{p}_i - y_i) \cdot \hat{p}_i(1 - \hat{p}_i) \cdot \mathbf{x}_i$$

   **致命缺陷**：当模型发生**严重错误预测**时（例如真实标签 $y_i = 1$，但模型输出 $\hat{p}_i = 0.0001$）：
   - 项 $(\hat{p}_i - y_i) \approx -1$（误差极大，理应受到剧烈惩罚）；
   - 但导数项 $\hat{p}_i(1 - \hat{p}_i) = 0.0001 \times 0.9999 \approx 0.0001 \to 0$！
   - 两者相乘导致**梯度几乎为 0**！模型在犯下大错时反而失去了学习动力，更新停滞。

2. **Log Loss 损失关于参数 $\mathbf{w}$ 的梯度**：

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \mathbf{w}} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i) \mathbf{x}_i$$

   **完美性质**：Sigmoid 的导数项 $\hat{p}_i(1-\hat{p}_i)$ 与 Log Loss 对 $\hat{p}$ 求导的分母**精准抵消**！梯度严格正比于预测误差 $(\hat{p}_i - y_i)$。当预测错得越离谱时，梯度越大，反向传播纠错越迅速。

#### 原因 3：概率校准（Well-Calibrated Probabilities）

Log Loss 源自伯努利最大似然估计，能够驱动模型输出真正收敛到真实的后验条件概率 $P(Y=1 \mid X)$；而 MSE 无法提供这种严格的概率校准保证。

---

### 4. MSE vs MAE 深度权衡与统计学收敛特性

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 \quad \text{vs.} \quad \text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

| 核心考量维度 | 均方误差（MSE / L2 Loss） | 平均绝对误差（MAE / L1 Loss） |
|---|---|---|
| **对离群异常值的敏感度** | **极度敏感**。残差被平方放大，单个极端异常值会产生巨大梯度，拉偏整个回归超平面。 | **高度鲁棒（Robust）**。误差按线性比例惩罚，受极端离群点的影响显著减小。 |
| **可导性与优化便利度** | **处处连续可导**。梯度 $\nabla_{\hat{y}} = -2(y - \hat{y})$ 平滑且随接近最优解自动缩小，易于梯度下降稳定收敛。 | **在 $e=0$ 处不可导**。梯度为固定符号阶跃函数（$\pm 1$），在极小值附近容易震荡，需使用次梯度或衰减学习率。 |
| **统计学收敛目标** | 最小化经验风险收敛到**条件均值（Conditional Mean）**：<br>$$\hat{y}^* = \mathbb{E}[y \mid \mathbf{x}]$$ | 最小化经验风险收敛到**条件中位数（Conditional Median）**：<br>$$\hat{y}^* = \text{Median}(y \mid \mathbf{x})$$ |

#### 数学证明：为什么 MSE 对应条件均值，而 MAE 对应条件中位数？

1. **MSE 的最优解是条件期望**：
   求期望风险极小值：$\min_c \mathbb{E}[(Y - c)^2]$
   对常数 $c$ 求导并令导数为 0：

$$\frac{d}{dc} \mathbb{E}[(Y - c)^2] = \mathbb{E}[-2(Y - c)] = -2\mathbb{E}[Y] + 2c = 0 \implies c^* = \mathbb{E}[Y]$$

2. **MAE 的最优解是中位数**：
   求期望风险极小值：$\min_c \mathbb{E}[|Y - c|]$
   对 $c$ 求导（利用 Leibniz 积分法则）：

$$\frac{d}{dc} \left( \int_{-\infty}^c (c - y) p(y)dy + \int_c^{\infty} (y - c) p(y)dy \right) = P(Y \le c) - P(Y > c) = 0$$

$$P(Y \le c) = P(Y > c) = 0.5 \implies c^* = \text{Median}(Y)$$

---

### 5. 折中方案：Huber Loss 与 Smooth L1 Loss

为了兼顾 MSE 的平滑可导性与 MAE 的抗离群鲁棒性，工业界常使用 **Huber Loss**：

$$\mathcal{L}_\delta(e) = \begin{cases} \frac{1}{2} e^2 & \text{for } |e| \le \delta \\ \delta \left( |e| - \frac{1}{2}\delta \right) & \text{for } |e| > \delta \end{cases}$$

- **小误差区间（$|e| \le \delta$）**：表现为 MSE，梯度为 $e$，连续平滑，便于微调收敛；
- **大误差区间（$|e| > \delta$）**：平滑过渡为 MAE，梯度被截断为固定的 $\pm \delta$，防止异常值梯度爆炸。

```text
损失函数梯度行为对比：
      误差 e 趋向无穷大时:
      • MSE 梯度: 2e ──> 趋向无穷 (梯度爆炸风险)
      • MAE 梯度: ±1 ──> 恒定常数 (零点不连续)
      • Huber 梯度: ±δ ──> 恒定有界且零点平滑！
```

---

### 6. Quick Coding：手写常用损失函数与导数验证

```python
import torch
import torch.nn as nn
import numpy as np

def custom_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """手写 MSE 损失"""
    return torch.mean((y_pred - y_true) ** 2)

def custom_mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """手写 MAE 损失"""
    return torch.mean(torch.abs(y_pred - y_true))

def custom_bce_loss(y_prob: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """手写数值稳定的二元交叉熵损失"""
    y_prob = torch.clamp(y_prob, min=eps, max=1.0 - eps)  # 防止 log(0) 溢出
    return -torch.mean(y_true * torch.log(y_prob) + (1.0 - y_true) * torch.log(1.0 - y_prob))

def custom_huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """手写 Huber Loss"""
    error = y_pred - y_true
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error, torch.tensor(delta))
    linear = abs_error - quadratic
    return torch.mean(0.5 * quadratic ** 2 + delta * linear)

# 单元测试与对齐验证
y_t = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32)
y_p = torch.tensor([0.9, 0.2, 0.8, 0.4], dtype=torch.float32)

# 验证与 PyTorch 官方原生实现严格数值等价
assert torch.allclose(custom_mse_loss(y_p, y_t), nn.MSELoss()(y_p, y_t))
assert torch.allclose(custom_mae_loss(y_p, y_t), nn.L1Loss()(y_p, y_t))
assert torch.allclose(custom_bce_loss(y_p, y_t), nn.BCELoss()(y_p, y_t))
assert torch.allclose(custom_huber_loss(y_p, y_t, delta=1.0), nn.HuberLoss(delta=1.0)(y_p, y_t))

print("✅ 所有损失函数数值测试均通过验证！")
```

---

## 模块六：核心机制辨析与系统问答清单

### Q1：如果训练集和测试集的分布不一致（Covariate Shift），如何设计交叉验证？
> **答**：
> 1. 先进行对抗验证（Adversarial Validation）：将训练集打标为 0，测试集打标为 1，训练一个二分类器（如 LightGBM）。若 AUC 远大于 0.5，说明存在明显的协变量偏移。
> 2. 利用对抗验证分类器的预测概率对训练样本计算重要性权重（Importance Weighting $w(x) = \frac{p_{\text{test}}(x)}{p_{\text{train}}(x)}$），或者选择与测试集概率分布最接近的训练样本构建验证集。

### Q2：为什么目标编码（Target Encoding）极易发生数据泄露？如何彻底防范？
> **答**：
> 1. 直接计算全量类别的目标均值会把样本自身的标签反哺给自己，产生严重的自相关泄露。
> 2. **防范标准**：采用 **K 折袋外目标编码（Out-of-Fold Target Encoding）**，计算当前样本所属类别的编码均值时，必须严格排除当前折（甚至排除当前样本自身），并施加经验贝叶斯平滑（Smoothing with prior mean）和高斯噪声扰动。

### Q3：为什么说最小化 MAE 比 MSE 更适合存在大量错误标记（Label Noise）的回归任务？
> **答**：
> 因为 MSE 会将离群点的巨大残差进行平方放大，导致模型被少数几个具有大标注错误的噪声样本“绑架”，过度扭曲模型拟合方向；而 MAE 的惩罚上限是线性的，对应的最优解是条件中位数，中位数对单侧尾部的极端噪声拥有天然的崩溃点（Breakdown Point）免疫力。

### Q4：在极度不平衡业务中，为什么即便 ROC-AUC 达到 0.98，模型上线后查准率依然可能崩溃？
> **答**：
> 核心根源在于假阳率公式 $\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}$。当负样本基数极大（如正负比 1:1000）时，巨大的 $\text{TN}$ 会稀释分母，使得即便模型产生了大量误报（例如 $\text{FP} = 1000$ 对比 $\text{TP} = 50$），$\text{FPR}$ 依然仅有千分之几，ROC 曲线显得极为优异。然而在实际业务中，查准率 $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{50}{1050} \approx 4.76\%$，导致人工审核资源被海量误报完全瘫痪。因此极端不平衡场景必须以 **PR-AUC（Average Precision）** 或 **Precision@k** 作为核心评估基准。

