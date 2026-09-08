# ML Coding 00 · ML Basics: Data Preprocessing, Data Leakage & Loss Functions

In machine learning system design and production engineering, a solid statistical foundation and rigorous data pipeline practices are essential prerequisites for building dependable models. Many machine learning models exhibit stellar offline evaluation metrics only to degrade catastrophically upon production rollout. The root causes rarely lie in model architectures, but rather in insidious data leakage, flawed missing data imputation, evaluation traps under extreme class imbalance, or a misalignment between loss function assumptions and problem characteristics.

This note systematically covers 6 foundational pillars of practical machine learning engineering:
1. **Data Leakage Mechanisms & End-to-End Prevention Strategies**
2. **Missing Data Mechanisms (MCAR / MAR / MNAR) & Imputation Trade-Offs**
3. **Imbalanced Data Handling, Representation Learning & Core Value of VAEs**
4. **Classification, Ranking, Calibration & Business Evaluation Metrics Framework (with Collapsible Implementations)**
5. **Loss Function Derivations: Linear vs. Logistic Regression, MSE vs. MAE & Statistical Convergence Targets**
6. **Core Concepts & System FAQ**

---

## Module 1: Data Leakage Mechanisms & Prevention

### 1. The Nature and Danger of Data Leakage

**Data leakage** occurs when **information from outside the training dataset (especially from the target variable or future test data) is inadvertently used to train a machine learning model**.

```text
Data Leakage Lifecycle & Impact:
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ Training / Offline Eval │ ───> │ Overly Optimistic Eval  │ ───> │ Production Deployment   │
│ Inadvertent look-ahead  │      │ Validation AUC 0.98+    │      │ Leaked signal missing   │
│ or target contamination │      │ (Spurious Correlations) │      │ Catastrophic drop 💥    │
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

Leakage creates an illusion of high predictive performance during offline experimentation. However, because the leaked information is unavailable at inference time in real-world systems, the model suffers a severe drop in production accuracy.

---

### 2. Four Canonical Data Leakage Scenarios

#### Scenario 1: Target Leakage / Proxy Features

**Mechanism**: Including a feature that is created, computed, or updated only **after** the target event occurs, but is mistakenly included in historical training records.

- **Case 1 (Loan Default Prediction)**: Using `account_closed_date` or `refund_status_code` as predictive features for whether a borrower will default. In real banking workflows, these columns are populated only after default proceedings and collections take place.
- **Case 2 (Medical Diagnosis)**: Using `prescribed_treatment_drug` to predict if a patient has a rare disease. Doctors prescribe the drug only after diagnosis; using it as an input inverts the causal graph.

#### Scenario 2: Preprocessing Leakage (Global Scaling & Imputation)

**Mechanism**: Computing preprocessing statistics (mean, variance, min-max bounds, TF-IDF vocabulary, or target encoding) across the **entire dataset** prior to splitting into train/test sets.

- **Case 1 (Feature Standardization)**: Calling `StandardScaler().fit_transform(X)` on the global dataset before `train_test_split`. Test set distribution parameters leak into training features.
- **Case 2 (NLP Vocabulary & TF-IDF)**: Fitting `TfidfVectorizer` globally, allowing inverse document frequency (IDF) and vocabulary tokens from the test split to contaminate training representations.
- **Case 3 (Target Encoding)**: Calculating category-level target averages on the full dataset without out-of-fold isolation, enabling the model to directly memorize test set target distributions.

#### Scenario 3: Temporal / Look-Ahead Leakage in Time Series

**Mechanism**: Using future timestamps to predict past events, violating the temporal arrow of causality.

- **Case 1 (Quantitative Finance / Market Prediction)**: Using a centered 5-day rolling moving average as a feature for today's trading signal.
- **Case 2 (Flawed Cross-Validation)**: Applying standard randomized K-Fold cross-validation to user activity logs or financial time series. Testing data from Day 1 is evaluated using training data from Day 5, obscuring non-stationarity and concept drift.

#### Scenario 4: Group / Duplication Leakage

**Mechanism**: Multiple highly correlated or repeated samples belonging to the **same entity (subject, patient, user session)** are randomly distributed across both train and test splits.

- **Case 1 (Medical Imaging)**: A patient undergoes 10 CT scans from varying angles. A randomized split assigns 8 scans to train and 2 to test. The CNN memorizes patient-specific artifacts (e.g., bone density, scanner noise) rather than generalized pathology.
- **Case 2 (Multi-Session Recommenders)**: User clicks from the same session are partitioned across train and test.

---

### 3. Industrial Data Leakage Prevention Blueprint

| Prevention Strategy | Implementation Principle | Recommended Tooling | Leakage Type Addressed |
|---|---|---|---|
| **Split First, Fit Later** | Enforce dataset partitioning before computing any statistical transformations. `fit()` exclusively on training data; `transform()` on test data. | `sklearn.pipeline.Pipeline`, `ColumnTransformer` | Preprocessing Leakage |
| **Time-Based Splitting** | Enforce strict chronological ordering. Train on historical windows; validate strictly on forward-looking out-of-time (OOT) sets. | `TimeSeriesSplit`, `PurgedGroupTimeSeriesSplit` | Temporal Leakage |
| **Group-Aware Partitioning** | Keep all records belonging to a given entity (user, patient, device) strictly inside either the train or test set. | `GroupKFold`, `GroupShuffleSplit`, `StratifiedGroupKFold` | Group / Subject Leakage |
| **Inference Timeline Audit** | Ask: "At the exact millisecond an inference request arrives in production, is this feature available in the feature store/DB?" | Feature Store (e.g., Feast), Data Lineage systems | Target & Proxy Leakage |

---

### 4. Quick Coding: Leak-Free Pipeline with GroupKFold

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. Generate synthetic grouped tabular data with missing values
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
groups = np.repeat(np.arange(100), 10)  # 100 distinct entities, 10 records each
X[np.random.rand(*X.shape) < 0.1] = np.nan  # Inject 10% missingness

# 2. Build leak-free pipeline encapsulating imputation, scaling, and estimator
model_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=42))
])

# 3. Perform GroupKFold cross-validation
gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # fit only touches the current training fold
    model_pipeline.fit(X_train, y_train)
    oof_preds[val_idx] = model_pipeline.predict_proba(X_val)[:, 1]

cv_auc = roc_auc_score(y, oof_preds)
print(f"Leak-Free GroupKFold 5-Fold OOF AUC: {cv_auc:.4f}")
```

---

## Module 2: Handling Missing Data: Strategies & Trade-Offs

### 1. Statistical Missingness Mechanisms

Rubin's classification framework categorizes missingness into three distinct regimes:

```text
Missing Data Regimes:
┌──────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Mechanism Category                   │ Mathematical Definition & Interpretation                               │
├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. MCAR (Missing Completely at Random)│ P(M | Y_obs, Y_mis) = P(M)                                             │
│                                      │ Missingness is entirely independent of observed and unobserved data.    │
│                                      │ (e.g., dropped sensor packets, randomly damaged paper forms).          │
├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. MAR (Missing at Random)           │ P(M | Y_obs, Y_mis) = P(M | Y_obs)                                     │
│                                      │ Missingness depends on observed features, but not on missing value.    │
│                                      │ (e.g., older patients less often report phone numbers, but conditional │
│                                      │ on age, missingness is independent of the phone number itself).        │
├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. MNAR (Missing Not at Random)      │ P(M | Y_obs, Y_mis) depends on Y_mis                                   │
│                                      │ Missingness directly depends on the unobserved value itself.           │
│                                      │ (e.g., very high/low income earners declining to disclose income).     │
└──────────────────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Comprehensive Missing Data Strategy Comparison

| Strategy | When to Use | Advantages (Pros) | Trade-Offs & Disadvantages (Cons) |
|---|---|---|---|
| **Listwise / Column Deletion** | MCAR with low missingness (<3%~5%); or when a column is >80% empty. | Simple; introduces no synthetic bias if truly MCAR. | Severe data loss; induces strong **selection bias** if data is MAR or MNAR. |
| **Simple Imputation (Mean / Median / Mode)** | Fast baseline; low missing rate on numeric/categorical features. | Extremely lightweight; easy online deployment. | **Distorts distribution**, artificially deflates feature variance, ignores cross-feature covariance. |
| **Missing Indicator (`is_missing`)** | MNAR scenarios where omission carries predictive signal (e.g., skipped optional credit check). | Explicitly preserves the informational signal of missingness. | Doubles the feature dimensionality if applied naively; potential collinearity. |
| **Model-Based Imputation (KNN, MICE / Iterative, MissForest)** | High-value tabular data with non-linear feature correlations. | Preserves multivariate distributions, variance, and feature interactions. | Computationally expensive; complex inference deployment; risk of cascading errors. |
| **Native Tree Routing (LightGBM, XGBoost, CatBoost)** | Gradient boosted decision trees. | Zero manual imputation; finds optimal split direction for missing values natively. | Restricted to specific tree libraries; unusable for neural nets or linear models. |

---

### 3. Deep Statistical Trade-Off Analysis

1. **Variance Deflation under Mean Imputation**:
   When replacing missing entries with the sample mean $x_{\text{imputed}} = \bar{x}$, the imputed variance drops:

$$\text{Var}(X_{\text{imputed}}) = \frac{N_{\text{obs}}}{N_{\text{total}}} \text{Var}(X_{\text{obs}}) < \text{Var}(X_{\text{obs}})$$

   This deflation produces biased standard errors, narrower confidence intervals, and skewed regression weights.
2. **MICE (Multivariate Imputation by Chained Equations)**:
   MICE iteratively specifies a univariate conditional model for each variable with missing data given all other variables, adding residual stochastic noise to preserve true uncertainty.

---

### 4. Quick Coding: Missing Indicator Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# 1. Synthetic dataset with MNAR income missingness
df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 50, np.nan, 60],
    'income': [50000, np.nan, 120000, 80000, np.nan, 200000, 95000],
    'credit_score': [650, 700, 750, 680, 710, 790, 720]
})
y = np.array([0, 1, 0, 1, 0, 1, 0])

# 2. Composite transformer: Imputed values + Boolean Missing Indicators
numeric_features = ['age', 'income', 'credit_score']

numeric_transformer = FeatureUnion([
    ('imputed_features', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])),
    ('missing_indicators', MissingIndicator())
])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features)]
)

full_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', Ridge())
])

full_pipeline.fit(df, y)
print("Pipeline fitted successfully. Output shape (with is_missing flags):", 
      full_pipeline.named_steps['preprocess'].transform(df).shape)
```

---

## Module 3: Imbalanced Data Handling, Representation Learning & Core Value of VAEs

### 1. Three Tiers of Imbalanced Data Processing

In production machine learning systems, handling severe class imbalance spans three distinct architectural tiers:

* **Data Tier**:
  * **Under-sampling**: Random Under-sampling, Tomek Links (identifying and removing majority class instances in minimally separated opposite-class pairs to clarify decision boundaries), ENN (Edited Nearest Neighbours to prune noisy border points);
  * **Over-sampling**: SMOTE (synthetic minority over-sampling via $k$-nearest neighbor linear interpolation in feature space), ADASYN (adaptive synthetic sampling adjusting interpolation weights based on local majority density);
  * **Targeted Feature/Sample Augmentation**: Long-tail jittering and targeted domain augmentation.
* **Algorithm & Loss Tier**:
  * **Cost-Sensitive / Class Weighting**: Re-weighting the loss inversely proportional to class frequencies $w_c \propto \frac{1}{N_c}$;
  * **Focal Loss**: Modulating cross-entropy by $(1 - p_t)^\gamma$ to down-weight well-classified easy examples and focus gradient updates on hard instances;
  * **Re-framing as One-Class Classification or Anomaly Detection**: One-Class SVM, Isolation Forest, modeling the support of normal instances without supervised label degradation.
* **Decision & Post-Processing Tier**:
  * **Dynamic Threshold Tuning / Moving**: Replacing default 0.5 cutoffs with validation-optimized operating thresholds chosen to maximize business utility (e.g., $F_\beta$ or net expected revenue);
  * **Probability Calibration**: Platt Scaling or Isotonic Regression to recover empirical posterior frequencies distorted by sampling or loss re-weighting;
  * **Top-$k$ Ranking Truncation**: Ranking predictions descending by score and dispatching only the top-$k$ instances to human auditing or downstream execution pipelines.

---

### 2. Why Severe Imbalance is Often "Harmless" in Practice

1. **Rank Invariance of ROC-AUC**:
   The statistical nature of ROC-AUC is the Wilcoxon-Mann-Whitney U statistic, which equals the prior expectation that a randomly chosen positive instance receives a higher model score than a randomly chosen negative instance:
   $$ \text{AUC} = P(S^+ > S^-) $$
   This ranking relationship depends exclusively on the separability of the conditional distributions $P(X \mid Y=1)$ and $P(X \mid Y=0)$ along the projection vector, remaining mathematically invariant to the class prior $P(Y)$. Scaling negative instances by $10\times$ or $1000\times$ leaves ROC-AUC unaltered as long as internal score shapes remain constant.
2. **Decision Pipelines Depend Exclusively on Top Rankings**:
   In recommender retrieval/ranking, quantitative cross-sectional equity selection, and fraud triage, production capacity dictates selecting a fixed top tranche (e.g., longing the top 1% scored equities daily, or routing the top 500 suspicious transactions to fraud investigators). As long as relative orderings among top candidates are preserved, shifts in baseline prior probability do not affect the composition of the actionable cohort.
3. **The "False Prosperity" Trap of ROC-AUC & The PR-AUC Frontier**:
   While ROC-AUC is prior-invariant, in extreme imbalance (e.g., 0.1% positive prevalence), the False Positive Rate formula is:
   $$ \text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}} $$
   An overwhelming $\text{TN}$ denominator heavily dilutes $\text{FPR}$. Even if the model generates thousands of false alarms ($\text{FP} \gg \text{TP}$), $\text{FPR}$ remains near zero, yielding an inflated ROC-AUC of 0.98+ while production Precision may fall below 5%. In safety-critical and fraud detection systems, **PR-AUC (Average Precision)** is the mandatory golden standard.

---

### 3. Comparison: Supervised Contrastive Learning (SupCon), SMOTE, and Focal Loss

| Method | Core Mechanism | Primary Target Domain | Limitations & Failure Modes |
|---|---|---|---|
| **SMOTE** | Finds $k$-nearest neighbors of minority samples in feature space and synthesizes new instances via linear interpolation: $x_{\text{new}} = x + \lambda(x_{nn} - x)$. | Low-to-moderate dimensional structured tabular data with continuous support. | Fails in high-dimensional sparse representations (curse of dimensionality); blindly interpolates noise and outliers, blurring class borders. |
| **Focal Loss** | Adds a modulating factor $(1 - p_t)^\gamma$ to cross-entropy, dynamically scaling down gradients from easy negatives. | Dense prediction (object detection), deep neural networks with extreme foreground/background skew without resampling. | Highly vulnerable to label noise, as incorrectly labeled samples are treated as maximally hard examples and given huge gradient weights. |
| **Contrastive Learning (SupCon)** | Employs pairwise/multi-sample pull-push geometric objectives to contract intra-class embeddings and repel inter-class representations. | High-dimensional rich representations (text, time series, graphs); long-tailed recognition; few-shot cold starts. | Substantial compute footprint (requires large batch sizes or memory banks); requires careful construction of augmentation views and projection heads. |

---

### 4. Value of Variational Autoencoders (VAEs) in Noisy, Imbalanced & Financial Datasets

In domains characterized by ultra-low signal-to-noise ratios ($\text{SNR} < 0.05$) and non-stationarity (e.g., high-frequency financial order book flow, telemetry feeds), VAEs and their variants (CVAE, Bottleneck AE) provide critical structural inductive biases:

* **Implicit Gaussian Perturbation & Denoising Regularization**:
  Standard Autoencoders and MLPs memorize high-frequency microstructure noise, causing severe overfitting. VAEs encode inputs into mean $\mu$ and log-variance $\sigma$, sampling via the reparameterization trick:
  $$ z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$
  Constraining the latent representation against a standard normal prior $\mathcal{N}(0, I)$ acts as continuous manifold data augmentation, forcing downstream prediction heads to rely solely on robust topological signals.
* **Nonlinear Regime / Macro Factor Extraction**:
  Complex financial and industrial systems are governed by unobserved latent regimes (e.g., liquidity contraction, volatility regimes). Linear PCA fails across nonlinear cross-asset dynamics. A low-dimensional VAE bottleneck $z$ extracts orthogonal, smooth, continuous representations serving as noise-resilient inputs to gradient-boosted decision trees (LightGBM) or Transformers.
* **End-to-End Multi-Task Joint Loss**:
  Optimizing prediction loss jointly with self-supervised reconstruction:
  $$ \mathcal{L} = \mathcal{L}_{\text{prediction}}(y, \hat{y}) + \lambda_1 \mathcal{L}_{\text{recon}}(x, \hat{x}) + \lambda_2 D_{\text{KL}}(q(z \mid x) \parallel p(z)) $$
  The reconstruction objective serves as an inductive anchor preventing model weights from collapsing into spurious correlations.
* **Imbalance & Anomaly Detection**:
  Extreme tail risks (market crashes, rare equipment failures) are exceedingly scarce. The VAE reconstruction error $\|x - \hat{x}\|_2^2$ or marginal log-likelihood acts directly as an **Anomaly Score**, providing a natural risk circuit breaker or position sizing metric.

<details>
<summary><b>Implementation & Explanation: Tabular VAE with Joint Prediction Head (PyTorch)</b></summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularVAEWithJointHead(nn.Module):
    """Variational Autoencoder with downstream joint prediction head"""
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder (feature denoising reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Downstream Joint Prediction Head (regression or classification)
        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * eps"""
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
    """Multi-task joint loss: Prediction MSE + Denoising Reconstruction MSE + KL Divergence"""
    pred_loss = F.mse_loss(y_pred.squeeze(-1), y_true)
    recon_loss = F.mse_loss(recon_x, x)
    # KL(q(z|x) || N(0, I))
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

## Module 4: Comprehensive Evaluation Metrics Framework & Vectorized Implementations

### 1. Landscape of Top 10 Evaluation Metrics

| Metric Family | Metric Name | Mathematical Definition / Formula | Ideal Use Cases | Pitfalls & Failure Modes |
|---|---|---|---|---|
| **Ranking (Threshold-Free)** | **ROC-AUC** | Area under the TPR vs. FPR curve, statistically $P(S^+ > S^-)$. | Global discriminative power; benchmark comparisons across stable class balances. | Insensitive to false alarm surges when negatives are massive; over-optimistic under heavy skew. |
|  | **PR-AUC (AP)** | Area under the Precision vs. Recall curve: $\sum (R_k - R_{k-1})P_k$. | **Gold standard for imbalanced classification (fraud, rare defaults, defect triage)**. | Baseline is positive prevalence $P$, making uncalibrated cross-dataset comparisons difficult. |
| **Decision (Threshold-Dependent)** | **F1-Score / $F_\beta$** | $F_\beta = (1 + \beta^2)\frac{P \cdot R}{\beta^2 P + R}$, harmonic mean. | Single-threshold deployment; asymmetric error costs ($\beta=2$ weights recall higher). | Rigidly tied to an arbitrary decision boundary; ignores probability confidence calibration. |
|  | **Precision@k / Recall@k** | Precision or Recall within the top-$k$ ranked predictions. | Capacity-constrained environments (auditing teams with daily quotas, top-10 recommendations). | Blind to model quality outside the top-$k$ cutoff. |
|  | **Balanced Accuracy** | $\frac{\text{TPR} + \text{TNR}}{2} = \frac{\text{Recall}_{\text{pos}} + \text{Recall}_{\text{neg}}}{2}$. | Equal weighting across classes to prevent majority-class trivial predictors from scoring high. | Can obscure asymmetric financial costs between positive and negative errors. |
| **Probability Quality & Calibration** | **Log-Loss (Cross-Entropy)** | $-\frac{1}{N}\sum [y \ln p + (1-y)\ln(1-p)]$. | Strict probability output tasks (Click-Through Rate estimation, expected value pricing). | Severely penalizes confident wrong classifications; vulnerable to base-rate distribution shift. |
|  | **Brier Score** | $\frac{1}{N}\sum (p_i - y_i)^2$, mean squared error in probability space. | Calibration evaluation; decomposes into Reliability, Resolution, and Uncertainty. | Cannot directly establish a operational binary classification cutoff. |
|  | **ECE (Expected Calibration Error)** | Bin-weighted absolute difference between empirical accuracy and mean confidence. | Safety-critical and risk-pricing systems (credit underwriting, medical diagnosis). | Sensitive to binning strategy (fixed-width vs. equal-frequency) and ignores discrimination capacity. |
| **Quantitative & Business** | **Rank IC (Information Coefficient)** | Spearman rank correlation between predicted scores and actual future returns. | **Benchmark metric for quantitative cross-sectional Alpha factor evaluation**. | Does not capture non-linear tail gains, market impact, slippage, or turnover drag. |
|  | **Expected Business Cost** | $\sum_{i,j} C_{ij} \cdot P(\hat{Y}=i, Y=j)$, business cost matrix. | Ground-truth operational evaluation: maps FP and FN directly to monetary loss. | Accurate real-world cost matrices can be difficult to specify dynamically. |

---

### 2. Low-Level Vectorized Implementations (Collapsible Code Blocks)

<details>
<summary><b>Implementation 1: ROC-AUC (Wilcoxon-Mann-Whitney U-Rank Algorithm)</b></summary>

```python
import numpy as np

def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes binary ROC-AUC using the Wilcoxon-Mann-Whitney U statistic.
    
    Formula: AUC = (sum(rank(S_pos)) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    Includes fractional tie-breaking.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)
    
    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true must contain both positive and negative samples")
        
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    
    # Handle tied scores via fractional ranking
    sorted_scores = y_score[order]
    unique_scores, inverse_indices, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    if len(unique_scores) < len(y_score):
        tie_ranks = np.cumsum(counts) - (counts - 1) / 2.0
        ranks = tie_ranks[inverse_indices][np.argsort(order)]
    
    sum_pos_ranks = np.sum(ranks[pos_mask])
    u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))

# Verification
y_t = np.array([0, 0, 1, 1])
y_s = np.array([0.1, 0.4, 0.35, 0.8])
print("ROC-AUC:", compute_roc_auc(y_t, y_s))  # 0.75
```
</details>

<details>
<summary><b>Implementation 2: PR-AUC / Average Precision (Step-Wise Trapezoidal Rule)</b></summary>

```python
import numpy as np

def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes PR-AUC / Average Precision (AP).
    
    Formula: AP = sum_k (R_k - R_{k-1}) * P_k
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
    
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    recall_diff = recall - recall_prev
    
    return float(np.sum(precision * recall_diff))

# Verification
print("Average Precision:", compute_average_precision(y_t, y_s))
```
</details>

<details>
<summary><b>Implementation 3: F1-Score & F-beta Score (Threshold-Based Metric)</b></summary>

```python
import numpy as np

def compute_f_beta(
    y_true: np.ndarray, 
    y_score: np.ndarray, 
    threshold: float = 0.5, 
    beta: float = 1.0, 
    eps: float = 1e-12
) -> float:
    """Computes F-beta score for binary classification at a chosen threshold.
    
    Formula: F_beta = (1 + beta^2) * (P * R) / (beta^2 * P + R)
    beta = 1.0: F1-Score (balanced precision and recall)
    beta = 2.0: Recall-weighted (e.g., fraud prevention, medical diagnostics)
    beta = 0.5: Precision-weighted (e.g., spam detection)
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

# Verification
print("F1-Score:", compute_f_beta(y_t, y_s, threshold=0.5, beta=1.0))
print("F2-Score:", compute_f_beta(y_t, y_s, threshold=0.5, beta=2.0))
```
</details>

<details>
<summary><b>Implementation 4: Precision@k & Recall@k (Capacity-Constrained Top-k)</b></summary>

```python
import numpy as np

def compute_precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> tuple[float, float]:
    """Computes Precision@k and Recall@k using O(N) selection via argpartition."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    n = len(y_true)
    k = min(max(1, k), n)
    
    top_k_indices = np.argpartition(-y_score, k - 1)[:k]
    
    hits = np.sum(y_true[top_k_indices] == 1)
    total_positives = np.sum(y_true == 1)
    
    p_at_k = float(hits / k)
    r_at_k = float(hits / total_positives) if total_positives > 0 else 0.0
    return p_at_k, r_at_k

# Verification
print("P@2, R@2:", compute_precision_recall_at_k(y_t, y_s, k=2))
```
</details>

<details>
<summary><b>Implementation 5: Balanced Accuracy (Macro-Averaged Recall)</b></summary>

```python
import numpy as np

def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Balanced Accuracy: 0.5 * (TPR + TNR).
    
    Overcomes majority-class dominance in unbalanced datasets.
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

# Verification
print("Balanced Acc:", compute_balanced_accuracy(y_t, (y_s >= 0.5).astype(int)))
```
</details>

<details>
<summary><b>Implementation 6: Log-Loss / Binary Cross-Entropy (Numerically Stable)</b></summary>

```python
import numpy as np

def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """Computes numerically stable Binary Cross-Entropy (Log-Loss).
    
    Formula: -1/N * sum(y * ln(p) + (1-y) * ln(1-p))
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.clip(np.asarray(y_prob, dtype=float).ravel(), eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(loss)

# Verification
print("Log Loss:", compute_log_loss(y_t, y_s))
```
</details>

<details>
<summary><b>Implementation 7: Brier Score (Probability Mean Squared Error & Calibration Decomposition)</b></summary>

```python
import numpy as np

def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes Brier Score: 1/N * sum((p_i - y_i)^2).
    
    Ranges from 0 (perfect probability calibration) to 1.
    Decomposes into Reliability - Resolution + Uncertainty.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    return float(np.mean((y_prob - y_true) ** 2))

# Verification
print("Brier Score:", compute_brier_score(y_t, y_s))
```
</details>

<details>
<summary><b>Implementation 8: Expected Calibration Error / ECE (Uniform Binning)</b></summary>

```python
import numpy as np

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) via uniform probability binning.
    
    Formula: ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
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

# Verification
print("ECE (10 bins):", compute_ece(y_t, y_s, n_bins=10))
```
</details>

<details>
<summary><b>Implementation 9: Rank IC (Cross-Sectional Spearman Rank Correlation)</b></summary>

```python
import numpy as np

def compute_rank_ic(pred_scores: np.ndarray, true_returns: np.ndarray) -> float:
    """Computes quantitative cross-sectional Rank Information Coefficient (Spearman Rank Correlation)."""
    pred_scores = np.asarray(pred_scores).ravel()
    true_returns = np.asarray(true_returns).ravel()
    
    def rank_array(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        return ranks
        
    rank_p = rank_array(pred_scores)
    rank_y = rank_array(true_returns)
    
    cov = np.cov(rank_p, rank_y)[0, 1]
    std_p = np.std(rank_p, ddof=1)
    std_y = np.std(rank_y, ddof=1)
    
    if std_p == 0 or std_y == 0:
        return 0.0
    return float(cov / (std_p * std_y))

# Verification
print("Rank IC:", compute_rank_ic(y_s, y_t))
```
</details>

<details>
<summary><b>Implementation 10: Expected Business Cost (Cost-Matrix Weighted Loss)</b></summary>

```python
import numpy as np

def compute_expected_business_cost(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    cost_matrix: np.ndarray
) -> float:
    """Computes Expected Business Cost using a cost matrix C[pred_class, true_class].
    
    Example fraud detection cost matrix:
    cost_matrix = [[C_00 (True Negative: $0),   C_01 (False Negative / missed fraud: $1000)],
                   [C_10 (False Positive: $50), C_11 (True Positive interception fee: $5)]]
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    
    costs = cost_matrix[y_pred, y_true]
    return float(np.mean(costs))

# Verification
cost_mat = np.array([
    [0.0, 1000.0],
    [50.0, 5.0]
])
y_p_hard = (y_s >= 0.5).astype(int)
print("Expected Unit Cost:", compute_expected_business_cost(y_t, y_p_hard, cost_mat))
```
</details>

---

## Module 5: Loss Functions: Linear vs. Logistic Regression, and MSE vs. MAE

### 1. Linear Regression Objective & Gaussian MLE Derivation

Linear regression optimizes the Ordinary Least Squares (OLS) / Mean Squared Error (MSE) loss:

$$\mathcal{L}_{\text{Linear}}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

#### Probabilistic Derivation via Gaussian MLE

Assume the true target $y_i$ is generated by a linear function with additive Gaussian residual noise $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$:

$$y_i = \mathbf{w}^T \mathbf{x}_i + \epsilon_i \implies y_i \mid \mathbf{x}_i \sim \mathcal{N}(\mathbf{w}^T \mathbf{x}_i, \sigma^2)$$

The likelihood of observing sample $y_i$ is:

$$p(y_i \mid \mathbf{x}_i; \mathbf{w}, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y_i - \mathbf{w}^T \mathbf{x}_i)^2}{2\sigma^2} \right)$$

The total log-likelihood $\ell(\mathbf{w})$ across $N$ i.i.d. observations is:

$$\ell(\mathbf{w}) = -\frac{N}{2} \ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

Maximizing log-likelihood $\max_{\mathbf{w}} \ell(\mathbf{w})$ is mathematically equivalent to **minimizing the Mean Squared Error (MSE)**:

$$\arg\max_{\mathbf{w}} \ell(\mathbf{w}) \iff \arg\min_{\mathbf{w}} \frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

---

### 2. Logistic Regression Objective & Bernoulli MLE Derivation

For binary classification $y_i \in \{0, 1\}$, logistic regression models posterior probability via the sigmoid function:

$$\hat{p}_i = \sigma(\mathbf{w}^T \mathbf{x}_i) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}_i}}$$

Assuming $y_i \mid \mathbf{x}_i \sim \text{Bernoulli}(\hat{p}_i)$, the probability mass function is:

$$P(y_i \mid \mathbf{x}_i) = \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i}$$

The dataset log-likelihood is:

$$\ell(\mathbf{w}) = \sum_{i=1}^N \left[ y_i \ln \hat{p}_i + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

Negating and scaling by $1/N$ gives the Binary Cross-Entropy / Log Loss objective:

$$\mathcal{L}_{\text{Logistic}}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

---

### 3. Why Classification with Logistic Regression Cannot Use MSE Loss

Why MSE is unsuitable for training classification models on sigmoid probabilities:

$$\mathcal{L}_{\text{MSE-Logistic}}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (y_i - \sigma(\mathbf{w}^T \mathbf{x}_i))^2$$

**Three Fundamental Reasons**:

#### Reason 1: Non-Convex Optimization Landscape

- **Log Loss** with linear parameters $\mathbf{w}$ is strictly **convex**, guaranteeing that gradient descent reaches the global minimum.
- **MSE** compounded with the non-linear sigmoid $\sigma(z)$ produces a **non-convex** error surface with flat plateaus, saddle points, and suboptimal local minima.

#### Reason 2: Vanishing Gradients on Severe Errors

Let $z_i = \mathbf{w}^T \mathbf{x}_i$. Compare parameter gradients under both objectives:

1. **MSE Gradient**:

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \mathbf{w}} = \frac{2}{N} \sum_{i=1}^N (\hat{p}_i - y_i) \cdot \hat{p}_i(1 - \hat{p}_i) \mathbf{x}_i$$

   **Critical Flaw**: If the model makes a severe error (e.g., true $y_i = 1$, but predicted $\hat{p}_i = 0.0001$):
   - The error term $(\hat{p}_i - y_i) \approx -1$ is large.
   - However, the derivative term $\hat{p}_i(1 - \hat{p}_i) \approx 0.0001 \to 0$.
   - Their product causes the gradient to **vanish to zero**! The model stops learning precisely when it is most wrong.

2. **Log Loss (BCE) Gradient**:

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \mathbf{w}} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i) \mathbf{x}_i$$

   **Optimal Behavior**: The sigmoid derivative in the chain rule cancels out the denominator of the log derivative. The gradient is directly proportional to the residual $(\hat{p}_i - y_i)$. Large errors yield large gradients.

#### Reason 3: Probability Calibration

Log loss corresponds to the proper scoring rule under Bernoulli MLE, producing outputs that converge to true posterior probabilities $P(Y=1 \mid X)$.

---

### 4. MSE vs. MAE: Statistical Convergence & Trade-Offs

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 \quad \text{vs.} \quad \text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

| Dimension | Mean Squared Error (MSE / L2) | Mean Absolute Error (MAE / L1) |
|---|---|---|
| **Outlier Sensitivity** | **Highly sensitive**. Squares residuals, disproportionately penalizing large deviations. | **Robust**. Scales linearly with error; outliers do not dominate the gradient. |
| **Optimization & Smoothness** | **Smooth & differentiable everywhere**. Gradients shrink gracefully near the minimum. | **Non-differentiable at $e=0$**. Gradient is a step function ($\pm 1$), causing oscillations around the optimum without learning rate decay. |
| **Statistical Target** | Converges to the **Conditional Mean**: <br>$$\hat{y}^* = \mathbb{E}[y \mid \mathbf{x}]$$ | Converges to the **Conditional Median**: <br>$$\hat{y}^* = \text{Median}(y \mid \mathbf{x})$$ |

#### Proof: MSE Minimizes to Mean, MAE Minimizes to Median

1. **MSE converges to Mean**:
   Minimize $\min_c \mathbb{E}[(Y - c)^2]$. Differentiating w.r.t. $c$:

$$\frac{d}{dc} \mathbb{E}[(Y - c)^2] = \mathbb{E}[-2(Y - c)] = -2\mathbb{E}[Y] + 2c = 0 \implies c^* = \mathbb{E}[Y]$$

2. **MAE converges to Median**:
   Minimize $\min_c \mathbb{E}[|Y - c|]$. Applying Leibniz's rule:

$$\frac{d}{dc} \left( \int_{-\infty}^c (c - y) p(y)dy + \int_c^{\infty} (y - c) p(y)dy \right) = P(Y \le c) - P(Y > c) = 0 \implies c^* = \text{Median}(Y)$$

---

### 5. The Hybrid Compromise: Huber Loss

**Huber Loss** combines the smoothness of MSE near zero with the outlier robustness of MAE for large errors:

$$\mathcal{L}_\delta(e) = \begin{cases} \frac{1}{2} e^2 & \text{for } |e| \le \delta \\ \delta \left( |e| - \frac{1}{2}\delta \right) & \text{for } |e| > \delta \end{cases}$$

```text
Gradient behavior comparison as error e -> infinity:
• MSE Gradient:   2e  ──> Grows unboundedly (risk of exploding gradients)
• MAE Gradient:   ±1  ──> Constant (discontinuous at 0)
• Huber Gradient: ±δ  ──> Bounded and smooth at 0!
```

---

### 6. Quick Coding: Custom Loss Functions & PyTorch Alignment

```python
import torch
import torch.nn as nn

def custom_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2)

def custom_mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_pred - y_true))

def custom_bce_loss(y_prob: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    y_prob = torch.clamp(y_prob, min=eps, max=1.0 - eps)
    return -torch.mean(y_true * torch.log(y_prob) + (1.0 - y_true) * torch.log(1.0 - y_prob))

def custom_huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    error = y_pred - y_true
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error, torch.tensor(delta))
    linear = abs_error - quadratic
    return torch.mean(0.5 * quadratic ** 2 + delta * linear)

# Verification against native PyTorch implementations
y_t = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32)
y_p = torch.tensor([0.9, 0.2, 0.8, 0.4], dtype=torch.float32)

assert torch.allclose(custom_mse_loss(y_p, y_t), nn.MSELoss()(y_p, y_t))
assert torch.allclose(custom_mae_loss(y_p, y_t), nn.L1Loss()(y_p, y_t))
assert torch.allclose(custom_bce_loss(y_p, y_t), nn.BCELoss()(y_p, y_t))
assert torch.allclose(custom_huber_loss(y_p, y_t, delta=1.0), nn.HuberLoss(delta=1.0)(y_p, y_t))

print("✅ All custom loss functions matched PyTorch native references!")
```

---

## Module 6: Core Concepts & System FAQ

### Q1: How do you design cross-validation when train and test distributions differ (Covariate Shift)?
> **Answer**:
> 1. Run **Adversarial Validation**: Train a binary classifier (e.g., LightGBM) to distinguish between train ($y=0$) and test ($y=1$).
> 2. If AUC $\gg 0.5$, use the predicted probabilities for **Importance Weighting** ($w(x) = \frac{p_{\text{test}}(x)}{p_{\text{train}}(x)}$) or select training instances with the highest test similarity to build the validation set.

### Q2: Why is Target Encoding prone to catastrophic leakage, and how is it prevented?
> **Answer**:
> Using the global target average for a category leaks a sample's own ground-truth label into its feature representation. Standard prevention requires **Out-of-Fold (OOF) Target Encoding** (calculating target statistics strictly on the remaining $K-1$ folds) combined with empirical Bayes smoothing and additive Gaussian jitter.

### Q3: Why is MAE preferred over MSE for training models on datasets with severe label noise?
> **Answer**:
> MSE squares errors, allowing mislabeled samples with huge residuals to dominate the loss and pull the regression curve off trajectory. MAE penalizes errors linearly and optimizes for the conditional median, which has a higher statistical breakdown point against extreme outliers.

### Q4: Why can a model with ROC-AUC = 0.98 suffer from near-zero Precision in production?
> **Answer**:
> The root cause is the False Positive Rate definition $\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}$. Under extreme class imbalance (e.g., 1 positive per 1,000 negatives), the vast negative pool $\text{TN}$ inflates the denominator. Even if the model produces 1,000 false alarms for every 50 true positives, $\text{FPR} \approx 0.001$, appearing stellar on the ROC curve. However, operational precision $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{50}{1050} \approx 4.76\%$, causing triage queues to collapse under false alarms. Consequently, extreme skew mandates **PR-AUC (Average Precision)** or **Precision@k** as primary benchmarks.

