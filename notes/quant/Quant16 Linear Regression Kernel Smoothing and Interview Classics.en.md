# Quant 16 · Linear Regression, Kernel Smoothing & Interview Classics: OLS, Gauss–Markov, Ridge/Lasso

In quantitative research interviews, candidates often underestimate linear regression as being too basic. However, top-tier firms like Two Sigma, DE Shaw, and Citadel frequently use regression questions to probe your understanding. They are not checking if you've heard of OLS; they are testing your grasp of probability fundamentals, your algebraic fluency, and—most importantly—**whether you know when standard statistical models fail**. Financial data is riddled with heteroskedasticity, autocorrelation, and multicollinearity. If you don't know how to handle these violations, you will not pass the QR loops.

```text
Core Mental Models for Regression Interviews:
1. The Ultimate Univariate OLS Formula: Memorize \hat\beta = \rho (\sigma_y / \sigma_x) and R^2 = \rho^2. This alone solves a massive fraction of basic questions.
2. Regression Asymmetry: The product of the slope of y on x and the slope of x on y is \rho^2 \le 1. Never assume they are simply reciprocals.
3. Geometric Projection: View OLS as the orthogonal projection of y onto the column space of X. Orthogonality is the key to deriving residual properties.
4. BLUE Does Not Require Normality: The Gauss-Markov theorem proves OLS is BLUE without assuming normal errors. Normality is only needed for exact finite-sample t and F tests.
5. Geometric Effect of Penalties: Lasso's \ell_1 diamond induces sparsity (variable selection), while Ridge's \ell_2 sphere induces shrinkage (handles collinearity but retains all variables).
```

> 🧭 **Core Knowledge Landscape**
> - **Module 1: OLS Geometry & Algebra**: Normal Equations | 5 Dimensions of Residual Orthogonality & ANOVA | Coefficients vs. Covariance | Reverse Regression Trap
> - **Module 2: Gauss–Markov, Statistical Inference & Core Lemma Sheet**: Estimator Properties | t/F Tests & Restricted Models | Prediction vs Confidence Intervals | LOOCV & Leverage | Measurement Errors & OVB
> - **Module 3: Variable Selection & Shrinkage**: Best Subset | Ridge Regression | Lasso | Geometric Intuition & Comparison
> - **Module 4: Kernel Smoothing & Local Regression**: Conditional Expectation & Essence of Kernels | Nadaraya-Watson | Boundary Bias & Local Linear | Curse of Dimensionality
> - **Module 5: Classic Interview Question Bank (Green Book + HOTS + Top QR Loops)**: Correlation Bounds | Equicorrelated Matrix Lower Bound | Cholesky Simulation | CAPM & Reverse Regression | Affine Invariance | Omitted Variable Bias | Measurement Error | Multicollinearity & VIF | Optimal Futures Hedge Ratio | FWL Theorem & Factor Neutralization | Regression Without Intercept Trap | R² vs. Real-World IC
> - **Module 6: One-Minute Answer Checklist**

---

## Module 1: OLS Geometry and Algebra (ESL 3.2)

### 1. Matrix Form, Normal Equations, and Closed-Form Solution
Consider the standard multivariate linear regression model:
$$
y = X\beta + \varepsilon
$$
where target vector $y \in \mathbb{R}^N$, design matrix $X \in \mathbb{R}^{N \times (p+1)}$ (with the leading column typically set to $\mathbf{1}$ for the intercept, and assuming full column rank $\operatorname{rank}(X) = p+1 < N$), and coefficient vector $\beta \in \mathbb{R}^{p+1}$. Ordinary Least Squares (OLS) minimizes the Residual Sum of Squares:
$$
\operatorname{RSS}(\beta) = \|y - X\beta\|_2^2 = (y - X\beta)^\top (y - X\beta) = y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X \beta
$$
Differentiating with respect to $\beta$ (applying matrix calculus rules $\nabla_\beta (\beta^\top A) = A$ and $\nabla_\beta (\beta^\top A \beta) = 2A\beta$):
$$
\nabla_\beta \operatorname{RSS}(\beta) = -2 X^\top y + 2 X^\top X \beta = \mathbf{0}
$$
This yields the foundational **Normal Equations**:
$$
X^\top X \hat\beta = X^\top y
$$
When $X$ has full column rank, the Gram matrix $X^\top X$ is symmetric positive definite and strictly invertible, providing the unique analytic closed-form solution:
$$
\hat\beta = (X^\top X)^{-1} X^\top y
$$

---

### 2. Residual Orthogonality: Algebraic Identity and Geometric Projection
Define the fitted vector $\hat{y} = X\hat\beta$ and the sample residual vector $e = y - \hat{y} = y - X\hat\beta$.
The orthogonality of residuals forms the geometric bedrock of linear modeling, exhibiting five fundamental properties:

#### (1) Residuals are Orthogonal to Every Regressor ($X^\top e = \mathbf{0}$)
Directly rewriting the first-order optimality condition:
$$
-2 X^\top (y - X\hat\beta) = \mathbf{0} \implies X^\top e = \mathbf{0}
$$
Expressed column by column: for any predictor column $X_j$ ($j = 0, 1, \dots, p$):
$$
X_j^\top e = \sum_{i=1}^N X_{ij} e_i = 0 \iff X_j \perp e
$$
**Statistical Intuition**: The sample dot product between the residuals and every included regressor is identically zero (and their sample covariance is zero after mean-centering). This guarantees that **all linear predictive signal present in the explanatory variables has been fully extracted into $\hat\beta$, leaving zero residual linear signal**.

#### (2) The Magic of the Intercept: Residual Sum Vanishes ($\mathbf{1}^\top e = 0 \implies \bar{e} = 0$)
If the regression includes an intercept, the first column of the design matrix is the vector of ones $X_0 = \mathbf{1} = (1, 1, \dots, 1)^\top$.
Evaluating the orthogonality condition for $X_0 = \mathbf{1}$:
$$
\mathbf{1}^\top e = \sum_{i=1}^N e_i = 0 \implies \bar{e} = \frac{1}{N} \sum_{i=1}^N e_i \equiv 0
$$
**Two Crucial Corollaries**:
1. **The sample mean of OLS residuals is strictly zero**;
2. **The regression hyperplane passes directly through the sample center of mass $(\bar{x}, \bar{y})$**: Since $\bar{e} = \bar{y} - \bar{x}^\top \hat\beta = 0$, it follows that $\bar{y} = \bar{x}^\top \hat\beta$.
> **Classic Interview Trap: Regression Without Intercept (Through the Origin)**
> Interviewers often ask: "Is the mean of OLS residuals always zero?"
> **Wrong Answer**: "Yes, always."
> **Correct Explanation**: **Only when an intercept is included!** If the model is forced through the origin ($y = X\beta$ with no constant column), $\mathbf{1} \notin \operatorname{Col}(X)$, so $\mathbf{1}^\top e = \sum e_i \ne 0$, and the mean residual does not vanish!

#### (3) Residuals are Orthogonal to Fitted Values ($\hat{y}^\top e = 0$)
Because fitted values $\hat{y} = X\hat\beta$ reside entirely within the column space $\operatorname{Col}(X)$:
$$
\hat{y}^\top e = (X\hat\beta)^\top e = \hat\beta^\top (X^\top e) = \hat\beta^\top \mathbf{0} = 0
$$
The fitted prediction vector $\hat{y}$ and residual vector $e$ are strictly perpendicular in $\mathbb{R}^N$ ($\hat{y} \perp e$).
- Hat matrix $H = X(X^\top X)^{-1}X^\top$ is the orthogonal projection operator onto $\operatorname{Col}(X)$ (symmetric and idempotent: $H^2 = H, H^\top = H$);
- Annihilator matrix $M = I - H$ is the orthogonal projection operator onto the orthogonal complement $\operatorname{Col}(X)^\perp$ ($M^2 = M, M^\top = M, HM = \mathbf{0}$);
- Model effective degrees of freedom is $\mathrm{df} = \operatorname{tr}(H) = p+1$.

#### (4) Pythagorean Theorem & Variance Decomposition (ANOVA / Geometric Origin of $R^2$)
The observation vector decomposes uniquely into two orthogonal vectors: $y = \hat{y} + e$. By the Pythagorean theorem in Euclidean space:
$$
\|y\|^2 = \|\hat{y} + e\|^2 = \|\hat{y}\|^2 + \|e\|^2 + 2 \underbrace{\hat{y}^\top e}_{= 0} = \|\hat{y}\|^2 + \|e\|^2
$$
When an intercept is present, centering all vectors by subtracting the sample mean vector $\bar{y}\mathbf{1}$:
$$
(y - \bar{y}\mathbf{1}) = (\hat{y} - \bar{y}\mathbf{1}) + e
$$
The inner product cross-term evaluates to:
$$
(\hat{y} - \bar{y}\mathbf{1})^\top e = \hat{y}^\top e - \bar{y} (\mathbf{1}^\top e) = 0 - \bar{y}(0) = 0
$$
Because the cross-term vanishes identically, the squared Euclidean norms decompose into the canonical Analysis of Variance (ANOVA) identity:
$$
\underbrace{\sum_{i=1}^N (y_i - \bar{y})^2}_{\text{Total Sum of Squares } \mathrm{TSS}} = \underbrace{\sum_{i=1}^N (\hat{y}_i - \bar{y})^2}_{\text{Explained Sum of Squares } \mathrm{ESS}} + \underbrace{\sum_{i=1}^N e_i^2}_{\text{Residual Sum of Squares } \mathrm{RSS}}
$$
This leads to the coefficient of determination:
$$
R^2 = \frac{\mathrm{ESS}}{\mathrm{TSS}} = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}} \in [0, 1]
$$
*(Note: In regression without intercept, $(\hat{y})^\top e \ne 0$ in deviation-from-mean space, breaking $\mathrm{TSS} = \mathrm{ESS} + \mathrm{RSS}$ and potentially producing negative $R^2$.)*

#### (5) Key Distinction: Sample Residual Algebraic Orthogonality vs. Population Error Exogeneity
Distinguishing the sample residual from the unobserved population disturbance is a classic litmus test:
- **Sample Residual Orthogonality ($X^\top e = 0$)**: An **algebraic/numerical identity**. It is a direct mathematical consequence of setting the gradient of RSS to zero. Regardless of whether the true data-generating process is linear, or whether heteroskedasticity or measurement errors exist, the calculated sample residuals $e$ are guaranteed to be orthogonal to $X$ by construction.
- **Population Error Exogeneity ($\mathbb{E}[\varepsilon \mid X] = 0 \implies \mathbb{E}[X^\top \varepsilon] = \mathbf{0}$)**: A **structural population assumption**. It asserts that unobserved latent shocks $\varepsilon$ are mean-independent of $X$. In practice, omitted variable bias, simultaneity, or selection bias violate this assumption (endogeneity).
> **Interview Follow-Up**: "In a misspecified model with omitted variables, are the OLS residuals still orthogonal to the regressors?"
> **Standard Answer**: The sample residuals $e$ remain **strictly orthogonal** to the included regressors (algebraic necessity); however, the true population errors $\varepsilon$ are **no longer orthogonal** to the regressors, causing $\hat\beta$ to be structurally biased.

---

### 3. Deep Connection Between Regression Coefficients and Covariance
Regression coefficients act as projection operators mapping variance and covariance structures.

#### (1) Univariate OLS: Ratio of Covariance to Regressor Variance
For simple univariate regression $y = \alpha + \beta x + \varepsilon$ with intercept:
$$
\hat\beta = \frac{\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^N (x_i - \bar{x})^2} = \frac{\widehat{\operatorname{Cov}}(x, y)}{\widehat{\operatorname{Var}}(x)} = \hat\rho_{xy} \frac{s_y}{s_x}
$$
$$
\hat\alpha = \bar{y} - \hat\beta \bar{x}
$$
$$
R^2 = \hat\rho_{xy}^2
$$
- **Correlation $\rho$ vs. Slope $\beta$**:
  - **Correlation $\rho = \frac{\operatorname{Cov}(x, y)}{\sigma_x \sigma_y} \in [-1, 1]$**: **Dimensionless** and **symmetric** ($\rho_{xy} = \rho_{yx}$). It measures the purity/signal-to-noise ratio of the linear alignment, geometrically representing $\cos \theta$ between unit vectors;
  - **Regression Slope $\beta = \rho \frac{\sigma_y}{\sigma_x}$**: **Dimensional** (units of $y$ per unit of $x$) and **asymmetric** ($\beta_{y \sim x} \ne \beta_{x \sim y}$). It quantifies the marginal physical expected rate of change in $y$ given a 1-unit increase in $x$;
  - If both variables are standardized ($Z$-scores, $\sigma_x = \sigma_y = 1$), the slope and correlation **coincide numerically**: $\hat\beta = \hat\rho$.

#### (2) Asymmetry of Regression & "Regression to the Mean"
Interview trap: "If regressing $y$ on $x$ yields a slope of 2, does regressing $x$ on $y$ yield a slope of $1/2$?"
- **Wrong Answer**: $1/2$.
- **Mathematical Reality**:
  $$
  \hat\beta_{y \sim x} = \rho \frac{\sigma_y}{\sigma_x}, \quad \hat\beta_{x \sim y} = \rho \frac{\sigma_x}{\sigma_y}
  $$
  Multiplying both slopes:
  $$
  \hat\beta_{y \sim x} \times \hat\beta_{x \sim y} = \rho^2 \le 1
  $$
  Whenever real data contains noise ($|\rho| < 1$):
  $$
  \hat\beta_{x \sim y} = \frac{\rho^2}{\hat\beta_{y \sim x}} < \frac{1}{\hat\beta_{y \sim x}}
  $$
- **Geometric & Galton Origins**:
  If $\hat\beta_{y \sim x} = 2$, then $\sigma_y / \sigma_x \ge 2$ and $\rho \le 1$. The forward and reverse regression lines intersect at the centroid $(\bar{x}, \bar{y})$ with a non-zero angular separation $\theta > 0$, collapsing onto each other only when $|\rho| = 1$. This encapsulates Francis Galton's 1886 insight on "regression toward mediocrity": an exceptionally tall father's son is predicted to be above average, but closer to the population mean.

#### (3) Multivariate OLS: Inverse Covariance Matrix & Cross-Covariance Vector
Centering all regressors and the target ($X \in \mathbb{R}^{N \times p}$, $y \in \mathbb{R}^N$):
- Regressor sample covariance matrix: $\hat{\boldsymbol{\Sigma}}_{XX} = \frac{1}{N} X^\top X \in \mathbb{R}^{p \times p}$;
- Cross-covariance vector: $\hat{\boldsymbol{\Sigma}}_{Xy} = \frac{1}{N} X^\top y \in \mathbb{R}^{p \times 1}$.
The multivariate OLS closed form is expressed purely through covariances:
$$
\hat\beta = (X^\top X)^{-1} X^\top y = \hat{\boldsymbol{\Sigma}}_{XX}^{-1} \hat{\boldsymbol{\Sigma}}_{Xy}
$$
- **Orthogonal Regressors Decouple**: If all predictors are pairwise uncorrelated ($\boldsymbol{\Sigma}_{XX} = \operatorname{diag}(\sigma_1^2, \dots, \sigma_p^2)$ is diagonal):
  $$
  \hat\beta_j = \frac{\operatorname{Cov}(X_j, y)}{\operatorname{Var}(X_j)}
  $$
  Every multivariate coefficient **collapses strictly to its separate univariate regression coefficient**!
- **Correlated Regressors & The Whitening/Decorrelation Operator**:
  When predictors correlate, the cross-covariance $\operatorname{Cov}(X_j, y)$ is contaminated by indirect confounding paths through other predictors.
  The inverse covariance matrix $\hat{\boldsymbol{\Sigma}}_{XX}^{-1}$ functions as a **linear decorrelation operator**: it strips out common co-movements and isolates the unique, marginal contribution of each variable.

#### (4) Partial Covariance & The Frisch–Waugh–Lovell (FWL) Theorem
How does a single coefficient $\hat\beta_j$ in multiple regression reconcile with covariance?
By the FWL Theorem:
$$
\hat\beta_j = \frac{\operatorname{Cov}(\tilde{X}_j, y)}{\operatorname{Var}(\tilde{X}_j)} = \frac{\operatorname{Cov}(\tilde{X}_j, \tilde{y})}{\operatorname{Var}(\tilde{X}_j)}
$$
where $\tilde{X}_j$ is the residual from regressing $X_j$ onto all remaining predictors $X_{-j}$ (capturing the orthogonal, non-redundant variation of $X_j$), and $\tilde{y}$ is the residual from regressing $y$ onto $X_{-j}$.
- **Direct Derivation of Variance Inflation Factor (VIF)**:
  Since $\operatorname{Var}(\tilde{X}_j) = \operatorname{Var}(X_j) (1 - R_{j \mid -j}^2)$, where $R_{j \mid -j}^2$ is the $R^2$ from regressing $X_j$ onto $X_{-j}$:
  $$
  \operatorname{Var}(\hat\beta_j) = \frac{\sigma^2}{\sum_{i=1}^N \tilde{x}_{ij}^2} = \frac{\sigma^2}{(N-1)\operatorname{Var}(X_j)} \cdot \underbrace{\frac{1}{1 - R_{j \mid -j}^2}}_{\mathrm{VIF}_j}
  $$
  As multicollinearity intensifies ($R_{j \mid -j}^2 \to 1$), the denominator variance $\operatorname{Var}(\tilde{X}_j) \to 0$, driving the parameter variance to infinity.

#### (5) Four Canonical Quantitative Finance Mappings
1. **CAPM Asset Beta**:
   $$ \beta_i = \frac{\operatorname{Cov}(R_i, R_m)}{\operatorname{Var}(R_m)} $$
   An asset's systematic risk exposure is the covariance between asset excess returns and market excess returns, normalized by market variance.
2. **Minimum-Variance Optimal Hedge Ratio**:
   Holding spot $\Delta S$ and shorting $h$ futures contracts $\Delta F$, the hedged portfolio variance is:
   $$ \min_h \operatorname{Var}(\Delta S - h \Delta F) = \operatorname{Var}(\Delta S) - 2h \operatorname{Cov}(\Delta S, \Delta F) + h^2 \operatorname{Var}(\Delta F) $$
   First-order condition yields:
   $$ h^* = \frac{\operatorname{Cov}(\Delta S, \Delta F)}{\operatorname{Var}(\Delta F)} \equiv \beta_{\Delta S \sim \Delta F} $$
   The optimal hedge ratio is algebraically identical to the univariate OLS slope of spot changes regressed on futures changes!
3. **Omitted Variable Bias (OVB Formula)**:
   If the true model is $y = \beta_1 x_1 + \beta_2 x_2 + \varepsilon$, omitting $x_2$ yields:
   $$ \hat\beta_1^{\text{short}} = \frac{\operatorname{Cov}(x_1, y)}{\operatorname{Var}(x_1)} = \beta_1 + \beta_2 \cdot \underbrace{\frac{\operatorname{Cov}(x_1, x_2)}{\operatorname{Var}(x_1)}}_{\beta_{x_2 \sim x_1}} $$
   The bias equals the true coefficient of the omitted variable multiplied by the auxiliary regression coefficient of the omitted on the included variable.
4. **Barra Multi-Factor Risk & Factor Neutralization**:
   Raw alpha factors $F_{\text{raw}}$ are often exposed to systematic risks like Size and Industry. Running OLS:
   $$ F_{\text{raw}} = X_{\text{risk}} \gamma + F_{\text{neutral}} $$
   By residual orthogonality, $F_{\text{neutral}} \perp X_{\text{risk}}$, guaranteeing that the neutralized alpha factor has zero linear covariance with the underlying risk factors.

---

## Module 2: Gauss–Markov Theorem, Statistical Inference & Core Problem-Solving Lemma Sheet

The Gauss–Markov theorem along with statistical inference in Classical Normal Linear Models (CNLM) forms the foundational theoretical toolkit across quant interviews, econometrics exams, and PhD qualifiers. This module curates the essential lemmas, proofs, and algebraic identities frequently utilized in technical assessments.

---

### 1. Gauss–Markov Assumptions & The Essence of BLUE
The Gauss–Markov theorem states that under specific conditions, the OLS estimator is the **Best Linear Unbiased Estimator (BLUE)**—namely, among all linear unbiased estimators, OLS achieves the minimum variance (its covariance matrix difference is positive semi-definite).

1. **Linearity in Parameters**: The true model satisfies $y = X\beta + \varepsilon$;
2. **Strict Exogeneity**: $\mathbb{E}[\varepsilon \mid X] = \mathbf{0}$;
3. **Spherical Disturbances**:
   - **Homoskedasticity**: $\operatorname{Var}(\varepsilon_i \mid X) = \sigma^2$;
   - **No Autocorrelation**: $\operatorname{Cov}(\varepsilon_i, \varepsilon_j \mid X) = 0 \quad (i \ne j)$;
   - In matrix form: $\operatorname{Var}(\varepsilon \mid X) = \sigma^2 I_N$;
4. **No Full Multicollinearity**: $\operatorname{rank}(X) = k = p+1 \le N$.

> **Classic Interview Trap: The Normality Myth**
> **"Does OLS require normally distributed errors to be BLUE?"**
> **Answer: NO!**
> Normality is completely unnecessary for OLS to be BLUE. The theorem requires only first-moment (exogeneity) and second-moment (spherical errors) conditions. Normality is strictly required only for **exact finite-sample $t$-tests and $F$-tests**, and for proving that OLS achieves the Cramér–Rao Lower Bound (making it the Uniformly Minimum-Variance Unbiased Estimator, UMVUE).

---

### 2. Core Problem-Solving Lemma Sheet

#### [Lemma 1] Fundamental Algebraic & Moment Properties of OLS
- **Linearity**: $\hat\beta = (X^\top X)^{-1} X^\top y = C y$, where weight matrix $C = (X^\top X)^{-1} X^\top$ satisfies $C X = I_k$.
- **Conditional Unbiasedness**:
  $$ \mathbb{E}[\hat\beta \mid X] = \mathbb{E}[C(X\beta + \varepsilon) \mid X] = \beta + C \underbrace{\mathbb{E}[\varepsilon \mid X]}_{= \mathbf{0}} = \beta $$
- **Conditional Covariance Matrix**:
  $$ \operatorname{Var}(\hat\beta \mid X) = \operatorname{Var}(Cy \mid X) = C \operatorname{Var}(\varepsilon \mid X) C^\top = C (\sigma^2 I_N) C^\top = \sigma^2 (X^\top X)^{-1} $$
  - Variance of the $j$-th coefficient: $\operatorname{Var}(\hat\beta_j \mid X) = \sigma^2 [(X^\top X)^{-1}]_{jj}$;
  - Covariance between two coefficients: $\operatorname{Cov}(\hat\beta_j, \hat\beta_m \mid X) = \sigma^2 [(X^\top X)^{-1}]_{jm}$.
- **Unbiased Residual Variance Estimator**:
  $$ \hat\sigma^2 = s^2 = \frac{e^\top e}{N - k} = \frac{\sum_{i=1}^N e_i^2}{N - k} $$
  where $N$ is sample size and $k = p+1$ is the number of estimated parameters (including intercept).
  - **Derivation Proof (Quadratic Form Expectation Lemma)**:
    Residuals express as $e = (I - H)y = (I - H)(X\beta + \varepsilon) = (I - H)\varepsilon$.
    Residual sum of squares is the quadratic form $e^\top e = \varepsilon^\top (I - H) \varepsilon$.
    Applying the expectation lemma $\mathbb{E}[\varepsilon^\top A \varepsilon] = \operatorname{tr}(A \operatorname{Var}(\varepsilon)) + \mathbb{E}[\varepsilon]^\top A \mathbb{E}[\varepsilon]$:
    $$ \mathbb{E}[e^\top e \mid X] = \operatorname{tr}\left( (I - H) \sigma^2 I_N \right) + \mathbf{0} = \sigma^2 \operatorname{tr}(I - H) = \sigma^2 (N - \operatorname{tr}(H)) = \sigma^2 (N - k) $$
    Dividing both sides by $N - k$ yields $\mathbb{E}[\hat\sigma^2 \mid X] = \sigma^2$.

#### [Lemma 2] Statistical Inference Distributional Lemmas under Normality
Assuming conditional normality $\varepsilon \mid X \sim \mathcal{N}(\mathbf{0}, \sigma^2 I_N)$:
- **Independence Lemma (Core Corollary of Cochran's Theorem)**:
  $$ \hat\beta \text{ and the sample residuals } e \text{ (and } \hat\sigma^2 \text{) are strictly statistically independent!} $$
  **Algebraic Proof**: $\hat\beta = C y$ and $e = (I - H)y$. Their cross-covariance evaluates to:
  $$ \operatorname{Cov}(\hat\beta, e \mid X) = C \operatorname{Var}(y \mid X) (I - H)^\top = \sigma^2 C (I - H) = \sigma^2 \left( (X^\top X)^{-1}X^\top - (X^\top X)^{-1}X^\top H \right) = \mathbf{0} $$
  Under joint Gaussianity, zero covariance implies strict statistical independence: $\hat\beta \perp e$.
- **Residual Sum of Squares Chi-Square Distribution**:
  $$ \frac{e^\top e}{\sigma^2} = \frac{(N - k)\hat\sigma^2}{\sigma^2} \sim \chi^2(N - k) $$
- **Single-Coefficient $t$-Test**:
  Testing $H_0: \beta_j = \beta_{j,0}$ (typically testing significance $\beta_{j,0} = 0$):
  $$ t = \frac{\hat\beta_j - \beta_{j,0}}{\operatorname{SE}(\hat\beta_j)} = \frac{\hat\beta_j - \beta_{j,0}}{\sqrt{\hat\sigma^2 [(X^\top X)^{-1}]_{jj}}} \sim t(N - k) $$
- **Multiple Linear Restrictions $F$-Test**:
  Testing joint hypothesis $H_0: R\beta = r$ ($q$ linear restrictions, $R$ is $q \times k$ with full row rank):
  $$ F = \frac{(R\hat\beta - r)^\top [R(X^\top X)^{-1} R^\top]^{-1} (R\hat\beta - r) / q}{\hat\sigma^2} \sim F(q, N - k) $$
  - **Problem-Solving Shortcut (Restricted $R$ vs. Unrestricted $UR$)**:
    $$ F = \frac{(\operatorname{RSS}_R - \operatorname{RSS}_{UR}) / q}{\operatorname{RSS}_{UR} / (N - k)} = \frac{(R_{UR}^2 - R_R^2) / q}{(1 - R_{UR}^2) / (N - k)} $$
  - **Overall Regression Significance Test** ($H_0: \beta_1 = \dots = \beta_p = 0$, with $q = p$):
    $$ F = \frac{\mathrm{ESS} / p}{\mathrm{RSS} / (N - p - 1)} = \frac{R^2 / p}{(1 - R^2) / (N - p - 1)} \sim F(p, N - p - 1) $$
  - **Equivalence of $t$ and $F$**: For a single restriction ($q=1$), $t^2 \equiv F$.

#### [Lemma 3] Prediction Intervals vs. Confidence Intervals
Given a new query point $x_0 \in \mathbb{R}^k$:
- **Confidence Interval for Conditional Mean Response ($\mathbb{E}[y_0 \mid x_0] = x_0^\top \beta$)**:
  Fitted point $\hat{y}_0 = x_0^\top \hat\beta$. Variance stems strictly from parameter estimation error:
  $$ \operatorname{Var}(\hat{y}_0 \mid X) = x_0^\top \operatorname{Var}(\hat\beta \mid X) x_0 = \sigma^2 x_0^\top (X^\top X)^{-1} x_0 $$
  $1-\alpha$ Confidence Interval: $\hat{y}_0 \pm t_{1-\alpha/2, N-k} \cdot \hat\sigma \sqrt{x_0^\top (X^\top X)^{-1} x_0}$.
- **Prediction Interval for an Individual New Observation ($y_0 = x_0^\top \beta + \varepsilon_0$)**:
  Prediction error $e_0 = y_0 - \hat{y}_0 = \varepsilon_0 - x_0^\top(\hat\beta - \beta)$. Because future disturbance $\varepsilon_0$ is independent of the training sample:
  $$ \operatorname{Var}(e_0 \mid X) = \operatorname{Var}(\varepsilon_0) + \operatorname{Var}(\hat{y}_0 \mid X) = \sigma^2 \left[ 1 + x_0^\top (X^\top X)^{-1} x_0 \right] $$
  $1-\alpha$ Prediction Interval: $\hat{y}_0 \pm t_{1-\alpha/2, N-k} \cdot \hat\sigma \sqrt{1 + x_0^\top (X^\top X)^{-1} x_0}$.
> **Key Takeaway**: Prediction variance strictly exceeds confidence variance by $\sigma^2$ (the irreducible error variance). Hence, **prediction intervals are always strictly wider than confidence intervals**; even as $N \to \infty$, prediction interval width does not collapse to zero, remaining bounded at $\pm z_{\alpha/2}\sigma$.

#### [Lemma 4] Leave-One-Out Cross-Validation & Leverage
- **Hat Matrix Diagonal (Leverage $H_{ii}$)**:
  $H_{ii} = x_i^\top (X^\top X)^{-1} x_i$ measures the outlier distance of point $i$ in predictor space.
  Properties: $0 \le H_{ii} \le 1$, $\sum_{i=1}^N H_{ii} = k$, with average leverage $\bar{H} = k/N$.
- **Leave-One-Out Residual Formula (via Sherman–Morrison Lemma)**:
  Without retraining $N$ separate models, the out-of-fold prediction error when omitting sample $i$ is:
  $$ e_{(-i)} = y_i - \hat{y}_{(-i)} = \frac{e_i}{1 - H_{ii}} $$
  Yielding an exact one-step computation for LOOCV:
  $$ \mathrm{LOOCV} = \frac{1}{N} \sum_{i=1}^N \left( \frac{e_i}{1 - H_{ii}} \right)^2 $$
- **Sample Deletion Effect on Coefficients (Foundation of Cook's Distance)**:
  $$ \hat\beta - \hat\beta_{(-i)} = \frac{(X^\top X)^{-1} x_i e_i}{1 - H_{ii}} $$

#### [Lemma 5] Omitted Variable Bias & Irrelevant Regressors
- **Omitted Variable Bias (OVB)**:
  If the true data-generating process is $y = X_1 \beta_1 + X_2 \beta_2 + \varepsilon$, but $X_2$ is omitted:
  $$ \mathbb{E}[\hat\beta_1^{\text{short}} \mid X] = \beta_1 + \underbrace{(X_1^\top X_1)^{-1} X_1^\top X_2}_{\hat\Gamma_{2 \sim 1}} \beta_2 $$
  **Unbiasedness Condition**: The short regression is unbiased if and only if $\beta_2 = \mathbf{0}$ (omitted variables have zero true impact) or $X_1^\top X_2 = \mathbf{0}$ (omitted variables are orthogonal to included variables).
- **Including Irrelevant Variables (Overfitting)**:
  If the true model does not contain $X_2$ ($\beta_2 = \mathbf{0}$), but $X_2$ is erroneously included:
  - $\hat\beta_1^{\text{long}}$ **remains unbiased** ($\mathbb{E}[\hat\beta_1^{\text{long}}] = \beta_1$);
  - But variance inflates: $\operatorname{Var}(\hat\beta_1^{\text{long}}) \ge \operatorname{Var}(\hat\beta_1^{\text{short}})$, with equality holding if and only if $X_1 \perp X_2$.

#### [Lemma 6] Measurement Error (Errors-in-Variables / Attenuation Bias)
- **Regressor Measurement Error (Attenuation Bias)**:
  True model $y = \beta x^* + \varepsilon$, with observed $x = x^* + u$ ($u \sim (0, \sigma_u^2)$ independent of $x^*, \varepsilon$):
  $$ \operatorname{plim}_{N \to \infty} \hat\beta = \beta \cdot \frac{\sigma_{x^*}^2}{\sigma_{x^*}^2 + \sigma_u^2} < \beta $$
  **Takeaway**: Noise in independent variables attenuates the coefficient estimate toward zero (systematic underestimation).
- **Dependent Variable Measurement Error**:
  If observed $y = y^* + v$ ($v$ independent of $x$), $\hat\beta$ **remains unbiased and consistent**, only inflating error variance to $\sigma^2 + \sigma_v^2$ and reducing statistical power.

#### [Lemma 7] Scale & Affine Invariance
- **Predictor Rescaling**: If $x_{\text{new}} = c \cdot x$, then $\hat\beta_{\text{new}} = \frac{1}{c} \hat\beta$;
- **Target Rescaling**: If $y_{\text{new}} = d \cdot y$, then $\hat\beta_{\text{new}} = d \cdot \hat\beta$;
- **Centering / Shifting**: Adding constants to $x$ or $y$ leaves the slope $\hat\beta$ **strictly invariant**, altering only the intercept $\hat\alpha$;
- **Invariance**: Non-zero affine scaling and shifts leave **$t$-statistics, $F$-statistics, $R^2$, and $p$-values completely unchanged**.

---

### 3. Violations of Assumptions & Remedies (White / Newey–West / GLS)
When empirical financial data violates Gauss–Markov conditions:
- **Heteroskedasticity / Autocorrelation**:
  OLS remains unbiased and consistent, but ceases to be BLUE. Standard errors computed via $\sigma^2(X^\top X)^{-1}$ are severely underestimated, generating spurious significance.
- **Remedies**:
  1. **White Heteroskedasticity-Consistent Standard Errors (HC0 / Sandwich Estimator)**:
     $$ \operatorname{Var}_{\text{White}}(\hat\beta) = (X^\top X)^{-1} \left( \sum_{i=1}^N e_i^2 x_i x_i^\top \right) (X^\top X)^{-1} $$
  2. **Newey–West Heteroskedasticity and Autocorrelation Consistent (HAC)**:
     Incorporates a Bartlett lag-decay kernel to handle serial autocorrelation in financial time series.
  3. **Generalized Least Squares (GLS / WLS, Aitken's Theorem)**:
     If error covariance $\operatorname{Var}(\varepsilon \mid X) = \sigma^2 \boldsymbol{\Omega}$ is known, pre-multiplying by $P = \boldsymbol{\Omega}^{-1/2}$ yields the BLUE estimator:
     $$ \hat\beta_{\text{GLS}} = (X^\top \boldsymbol{\Omega}^{-1} X)^{-1} X^\top \boldsymbol{\Omega}^{-1} y $$

---

## Module 3: Variable Selection and Shrinkage (ESL 3.3–3.4)

When faced with numerous potentially collinear predictors, we must restrict or regularize the model.

### 1. Traditional Methods (Subset Selection)
- **Best Subset / Forward & Backward Stepwise**: At a high level, these discrete selection procedures can find good subsets of variables, but the discrete nature of the selection process often leads to high variance.

### 2. Ridge Regression
Introduces an $\ell_2$ norm penalty to shrink coefficients:
$$
\hat\beta^{\mathrm{ridge}} = \arg\min_\beta \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2
$$
The closed-form solution is:
$$
\hat\beta^{\mathrm{ridge}} = (X^\top X + \lambda I)^{-1}X^\top y
$$
**Key Traits**: Excellent at handling multicollinearity (trades a little bias for a large reduction in variance). However, **it does not shrink any coefficient exactly to zero**.

### 3. Lasso Regression
Introduces an $\ell_1$ norm penalty:
$$
\hat\beta^{\mathrm{lasso}} = \arg\min_\beta \|y - X\beta\|_2^2 + \lambda \|\beta\|_1
$$
**Key Traits**: The geometric shape of the $\ell_1$ penalty is a sharp "diamond". The elliptical contours of the sum of squares are highly likely to hit the corners of the diamond on the axes, allowing the Lasso to shrink some coefficients **exactly to zero**, effectively performing **built-in variable selection (sparsity)**.

### 4. Method Comparison Matrix

| Method | Penalty | Bias vs. Variance | Produces Sparsity? | Handles Multicollinearity? |
| :--- | :--- | :--- | :---: | :--- |
| **Best Subset** | Restricts # of variables | Discrete, high variance | Yes | Depends on the subset kept |
| **Ridge** | $\lambda \|\beta\|_2^2$ (Sphere) | Increases bias, lowers var | No | Yes, beautifully; unique solution |
| **Lasso** | $\lambda \|\beta\|_1$ (Diamond) | Increases bias, lowers var | Yes | Yes, but picks randomly among highly correlated features |

*(Note: Dimensionality reduction techniques like PCR/PLS essentially create "derived directions" to regress on. They differ from penalized regression and are usually just mentioned at a high level during interviews.)*

---

## Module 4: Kernel Smoothing & Local Regression (ESL 6.1–6.3)

In the preceding three modules, we thoroughly examined linear regression (OLS, Ridge, Lasso). All these classical models rest upon a **Global Parametric Assumption**: namely, that the true underlying data-generating function satisfies $f(X) = X\beta$ globally across the entire input domain. However, across modern quantitative finance—such as fitting option implied volatility smiles/surfaces, capturing nonlinear price impact curves from high-frequency order flow imbalance, or mining localized alpha signals—true relationships are inherently curved, regime-dependent, or state-contingent.

When we seek to discard rigid global linear assumptions, we arrive at the intersection of classical statistics and machine learning: **Nonparametric Smoothing**. Following the theoretical architecture of ESL Chapter 6, this module begins from first principles with conditional expectation, demonstrates how "kernels" naturally emerge as the bridge connecting density estimation to regression, and rigorously derives the mechanics of local polynomial regression.

---

### 1. Theoretical Foundations: The Regression Objective, The Essence of "Kernel", and Bridging Two Paradigms

#### (1) The Statistical Essence of Regression: The Conditional Expectation Function
In probability and statistics, the ultimate objective of regression is finding a predictor function $f(X)$ that minimizes the expected mean squared prediction error $\mathbb{E}[(Y - f(X))^2]$. By the law of total expectation and the orthogonal projection theorem in $L^2$ probability space, the unique theoretical minimizer is the **conditional expectation function (regression function)**:
$$
f(x_0) = \mathbb{E}[Y \mid X = x_0] = \int y \, p(y \mid x_0) \, dy = \frac{\int y \, p(x_0, y) \, dy}{p(x_0)}
$$
- **Global Parametric School (Modules 1–3)**: Imposes a rigid global functional assumption $f(x) \approx x^\top \beta$, estimating a single fixed set of parameters $\hat\beta$ across all samples. Its strengths are low estimation variance and high computational efficiency, but it suffers from severe **model misspecification bias**.
- **Nonparametric Local School (This Module)**: Presumes no global parametric structure on $f(x)$. Instead, it adheres to **memory-based learning (lazy learning)**—"to predict at query point $x_0$, inspect only the observations in the local neighborhood of $x_0$".

#### (2) From Conditional Expectation to Nadaraya–Watson: Natural Plug-in via Kernel Density Estimation
Since conditional expectation is the ratio between the integrated joint density and the marginal density, statisticians Nadaraya (1964) and Watson (1964) proposed an elegant, foundational breakthrough: **can we directly estimate both the numerator and denominator using nonparametric Parzen window Kernel Density Estimation (KDE)?**

Let the kernel function with bandwidth $\lambda$ be $K_\lambda(x_0, x) = \frac{1}{\lambda} D\left(\frac{|x - x_0|}{\lambda}\right)$:
1. **Denominator (Marginal input density $\hat{p}(x_0)$)**:
   $$ \hat{p}(x_0) = \frac{1}{N} \sum_{i=1}^N K_\lambda(x_0, x_i) $$
2. **Numerator (Joint density integral $\int y \hat{p}(x_0, y) dy$)**:
   Using a 2D independent product kernel to estimate the joint density $\hat{p}(x_0, y) = \frac{1}{N} \sum_{i=1}^N K_\lambda(x_0, x_i) K_{h_y}(y, y_i)$, substitute this into the integral over $y$:
   $$ \int y \, \hat{p}(x_0, y) \, dy = \frac{1}{N} \sum_{i=1}^N K_\lambda(x_0, x_i) \underbrace{\int y K_{h_y}(y, y_i) \, dy}_{= y_i} = \frac{1}{N} \sum_{i=1}^N K_\lambda(x_0, x_i) y_i $$

Dividing the estimated numerator by the estimated denominator yields the **Nadaraya–Watson kernel regression estimator** directly and unconditionally:
$$
\hat{f}(x_0) = \frac{\int y \hat{p}(x_0, y) dy}{\hat{p}(x_0)} = \frac{\sum_{i=1}^N K_\lambda(x_0, x_i) y_i}{\sum_{i=1}^N K_\lambda(x_0, x_i)}
$$
**Key Takeaway**: Kernel regression is not an ad-hoc heuristic weighting rule; it is the mathematically exact **nonparametric plug-in estimator of the true conditional expectation $\mathbb{E}[Y \mid X = x]$**.

#### (3) Machine Learning Clarification: Localization Kernels vs. Mercer / RKHS Kernels
ESL Chapter 6 explicitly warns: **Do not confuse the localization kernels in this chapter with the "kernel trick" used in Support Vector Machines!**

| Dimension | Localization Kernel (ESL Chapter 6) | Mercer / RKHS Kernel (ESL Chapters 5.8 & 12) |
| :--- | :--- | :--- |
| **Mathematical Definition** | Local neighborhood decay window $K_\lambda(x_0, x_i) = D\left(\frac{\|x_i - x_0\|}{\lambda}\right)$ | Positive semi-definite continuous kernel $K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$ |
| **Core Mechanism** | **Neighborhood weighting in the original input space** (memory-based localization) | **Implicit mapping into a high/infinite-dimensional Reproducing Kernel Hilbert Space (RKHS)** |
| **Computation Paradigm** | **Lazy Learning**: Virtually zero offline training; all computation occurs at query time | **Eager Learning**: Solves a global dual quadratic program or kernel matrix inversion during training |
| **Canonical Applications** | Nadaraya-Watson, local linear regression (LOESS/Lowess), volatility surface smoothing | Support Vector Machines (SVM), Kernel Ridge Regression, Gaussian Processes (GP) |

#### (4) The Continuum Spectrum: Continuous Transition from Global OLS to Local Nearest Neighbors
The objective function of local weighted least squares is:
$$
\min_{\beta(x_0)} \sum_{i=1}^N K_\lambda(x_0, x_i) \left[ y_i - b(x_i)^\top \beta(x_0) \right]^2
$$
The bandwidth $\lambda$ acts as a continuous dial between global rigidity and local flexibility:
- When **$\lambda \to \infty$**: Kernel weights become uniform constants $K_\lambda \to \text{const}$, and local regression **strictly degenerates to Global Ordinary Least Squares (OLS)** (minimum variance, but high potential misspecification bias; $\mathrm{df} = 2$);
- When **$\lambda \to 0$**: Kernel weights vanish everywhere except at the single observation closest to $x_0$, and local regression **degenerates to 1-Nearest Neighbor (1-NN) interpolation** (zero bias, but unbounded variance; $\mathrm{df} = N$);
- **Finite Bandwidth $\lambda \in (0, \infty)$**: Forms a continuous spectrum balancing the bias-variance tradeoff between the global parametric extreme and the local nonparametric extreme.

---

### 2. From k-NN to Nadaraya–Watson Kernel Regression (ESL 6.1)
- **Defects of the k-NN Running Mean**:
  A simple $k$-nearest-neighbor running mean estimates $\hat{f}(x) = \frac{1}{k}\sum_{x_i \in N_k(x)} y_i$. As $x$ shifts smoothly, observations enter and leave the neighborhood $N_k(x)$ abruptly in discrete steps, yielding an unnaturally jagged and discontinuous curve $\hat{f}(x)$.
- **The Nadaraya–Watson Kernel Estimator (1964)**:
  Replace the 0-1 indicator weights with a smoothly decaying **kernel weighting function** $K_\lambda(x_0, x_i) = D\left(\frac{|x_i - x_0|}{\lambda}\right)$:
  $$
  \hat{f}(x_0) = \frac{\sum_{i=1}^N K_\lambda(x_0, x_i) y_i}{\sum_{i=1}^N K_\lambda(x_0, x_i)} = \sum_{i=1}^N l_i(x_0) y_i
  $$
  where the normalized equivalent weights $l_i(x_0) = \frac{K_\lambda(x_0, x_i)}{\sum_{j=1}^N K_\lambda(x_0, x_j)}$ satisfy non-negativity and $\sum_{i=1}^N l_i(x_0) = 1$.
  - **Local Constant Fit Equivalence**: The Nadaraya–Watson estimate is mathematically equivalent to solving a local weighted least squares problem for a constant:
    $$
    \hat{f}(x_0) = \arg\min_c \sum_{i=1}^N K_\lambda(x_0, x_i)(y_i - c)^2
    $$
- **Comparison of Three Common Kernels**:
  1. **Epanechnikov Kernel**: $D(t) = \frac{3}{4}(1 - t^2) \cdot \mathbb{I}(|t| \le 1)$. Compact support. Asymptotically optimal in the sense of minimizing mean squared error (AMSE) among nonnegative kernels, though its derivative is discontinuous at the support boundaries.
  2. **Tri-cube Kernel (Default in Cleveland's LOESS)**: $D(t) = (1 - |t|^3)^3 \cdot \mathbb{I}(|t| \le 1)$. Compact support, twice continuously differentiable at the support boundaries, with a flatter peak and smoother transitions.
  3. **Gaussian Kernel**: $D(t) = \frac{1}{\sqrt{2\pi}} e^{-t^2/2}$. Infinite support, infinitely differentiable everywhere, with bandwidth parameter $\lambda$ acting as the standard deviation.
- **Bandwidth $\lambda$ & The Bias-Variance Tradeoff**:
  - $\lambda \to 0$ (narrow window): Dominated by only one or very few points $\implies$ **low bias, high variance** (interpolates data, extreme overfitting).
  - $\lambda \to \infty$ (wide window): All points receive equal weight $\implies$ **high bias, low variance** (degenerates to the global sample mean $\bar{y}$, severe underfitting).
  - **Metric Bandwidth vs. k-NN Adaptive Bandwidth**:
    - Constant metric bandwidth $\lambda$ maintains a constant neighborhood radius, keeping bias roughly uniform across space, but causes variance to spike in sparse data regions.
    - $k$-NN adaptive bandwidth $h_k(x_0) = |x_0 - x_{[k]}|$ fixes the effective sample size $k$, ensuring uniform variance, but broadens the window in sparse regions, increasing bias.

### 3. The Fatal Flaw: Boundary Bias & Mathematical Analysis
Why is the Nadaraya–Watson estimator often rejected as an inadequate baseline in quantitative research?
- **Intuitive Flaw**:
  In the interior of the data cloud, neighbors are balanced symmetrically to the left and right of $x_0$, so overestimates and underestimates cancel out.
  At the boundary of the support (e.g., $x_0 = 0$ on domain $[0, 1]$), all available neighbors lie strictly on one side ($x_i > x_0$). If the true underlying function has a non-zero slope ($f'(x_0) > 0$), all neighboring points systematically evaluate above $f(x_0)$, causing the weighted average to **systematically overestimate the true value**.
- **Taylor Series Derivation of Bias Orders**:
  Expanding the true function $f(x_i)$ around $x_0$:
  $$
  f(x_i) = f(x_0) + f'(x_0)(x_i - x_0) + \frac{f''(x_0)}{2}(x_i - x_0)^2 + O((x_i - x_0)^3)
  $$
  Taking the conditional expectation $\mathbb{E}[\hat{f}(x_0) \mid X] = \sum_{i=1}^N l_i(x_0) f(x_i)$, and noting $\sum l_i(x_0) = 1$:
  $$
  \operatorname{Bias}(\hat{f}(x_0)) = \mathbb{E}[\hat{f}(x_0)] - f(x_0) = f'(x_0) \underbrace{\sum_{i=1}^N l_i(x_0)(x_i - x_0)}_{\text{First Moment}} + \frac{f''(x_0)}{2} \sum_{i=1}^N l_i(x_0)(x_i - x_0)^2 + O(h^3)
  $$
  - **Interior Region**: Symmetric support leads to $\sum l_i(x_0)(x_i - x_0) = 0$, so the first-order term cancels. The bias is dominated by curvature: **$O(h^2) f''(x_0)$**.
  - **Boundary Region**: One-sided support leaves $\sum l_i(x_0)(x_i - x_0) = O(h) \ne 0$. The boundary bias degrades to **$O(h) f'(x_0)$**—an entire order of magnitude worse in convergence speed!

### 4. Local Linear Regression & "Automatic Kernel Carpentry" (ESL 6.1.1)
To eliminate the $O(h)$ boundary bias, local linear regression upgrades the model from a local constant to a local tangent line at every query point $x_0$.

- **Weighted Least Squares (WLS) Formulation**:
  At query point $x_0$, solve:
  $$
  \min_{\alpha(x_0), \beta(x_0)} \sum_{i=1}^N K_\lambda(x_0, x_i) \left[ y_i - \alpha(x_0) - \beta(x_0)(x_i - x_0) \right]^2
  $$
  Because regressors are centered at $(x_i - x_0)$, the evaluated prediction at $x = x_0$ is simply the intercept: $\hat{f}(x_0) = \hat{\alpha}(x_0)$.

- **Matrix Solution & The Equivalent Kernel**:
  Let basis vector $b(x) = (1, x - x_0)^\top$, and let design matrix $\mathbf{B}_{N \times 2}$ have row $i$ as $(1, x_i - x_0)$. Let weight diagonal matrix $\mathbf{W}(x_0) = \operatorname{diag}(K_\lambda(x_0, x_1), \dots, K_\lambda(x_0, x_N))$.
  From standard weighted normal equations:
  $$
  \begin{pmatrix} \hat{\alpha}(x_0) \\ \hat{\beta}(x_0) \end{pmatrix} = \left( \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{B} \right)^{-1} \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{y}
  $$
  The point prediction is linear in $y$:
  $$
  \hat{f}(x_0) = e_1^\top \left( \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{B} \right)^{-1} \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{y} = \sum_{i=1}^N l_i(x_0) y_i
  $$
  where row vector $l(x_0)^\top = e_1^\top \left( \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{B} \right)^{-1} \mathbf{B}^\top \mathbf{W}(x_0)$ is the **equivalent kernel**.

- **Why is it called "Automatic Kernel Carpentry"?**
  By the matrix identity $\left( \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{B} \right) \cdot \left[ \left( \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{B} \right)^{-1} e_1 \right] = e_1$:
  $$
  \mathbf{B}^\top \mathbf{W}(x_0) l(x_0) = \begin{pmatrix} 1 \\ 0 \end{pmatrix}
  $$
  Writing out the two rows explicitly:
  1. Row 1 (Zeroth Moment): $\sum_{i=1}^N l_i(x_0) = 1$ (preserves level unbiasedness)
  2. Row 2 (First Moment): $\sum_{i=1}^N l_i(x_0)(x_i - x_0) = 0$ (**the first-order moment is strictly zero everywhere, even on asymmetric boundaries!**)
  
  Substituting this back into the Taylor bias formula, the term $f'(x_0) \sum l_i(x_0)(x_i - x_0) \equiv 0$ **vanishes identically**!
  At boundary points, the equivalent kernel $l_i(x_0)$ automatically adapts its shape (increasing weights on near points and dipping slightly negative on distant points) to cancel the first-order slope bias, reducing boundary bias from $O(h)$ to $O(h^2)$ with zero manual tuning.

- **Polynomial Degree Tradeoffs (ESL 6.1.2)**:
  - **Local Quadratic ($d=2$)**: In regions of high interior curvature ($|f''(x)| \gg 0$), local linear fits exhibit "trimming hills and filling valleys" bias. Local quadratic fits remove this curvature bias (bias becomes $O(h^4)$), but increase variance considerably at boundaries.
  - **Odd-Degree Dominance**: Asymptotic MSE is dominated by boundary behavior. Moving from degree 0 to degree 1 drastically reduces boundary bias with negligible variance penalty. Moving from degree 1 to degree 2 does not improve boundary bias order while inflating boundary variance.
  - $\implies$ **Industry Rule of Thumb: Default to local linear ($d=1$)**.

### 5. Bandwidth Selection & Effective Degrees of Freedom (ESL 6.2 / Ch.7)
- **Linear Smoother & Smoother Matrix**:
  Predictions across all training points form a linear mapping: $\hat{\mathbf{y}} = \mathbf{S}_\lambda \mathbf{y}$, where row $i$ of $\mathbf{S}_\lambda$ is $l(x_i)^\top$.
- **Effective Degrees of Freedom**:
  Paralleling the projection hat matrix in OLS where $\operatorname{df} = \operatorname{tr}(H) = p+1$, the effective degrees of freedom for a kernel smoother is:
  $$
  \operatorname{df}_\lambda = \operatorname{tr}(\mathbf{S}_\lambda)
  $$
  - As $\lambda \to 0$, $\mathbf{S}_\lambda \to \mathbf{I}_N \implies \operatorname{df}_\lambda = N$ (full interpolation, maximal overfitting).
  - As $\lambda \to \infty$, local linear regression converges to global OLS $\implies \operatorname{df}_\lambda = 2$ (intercept + slope).
- **Leave-One-Out Cross-Validation (LOOCV) Shortcut**:
  For any linear smoother, LOOCV requires no expensive retraining loops:
  $$
  \operatorname{CV}(\lambda) = \frac{1}{N} \sum_{i=1}^N \left( \frac{y_i - \hat{f}_\lambda(x_i)}{1 - S_{\lambda, ii}} \right)^2
  $$
  Or via Generalized Cross-Validation (GCV):
  $$
  \operatorname{GCV}(\lambda) = \frac{1}{N} \sum_{i=1}^N \left( \frac{y_i - \hat{f}_\lambda(x_i)}{1 - \operatorname{tr}(\mathbf{S}_\lambda)/N} \right)^2
  $$

### 6. Multidimensional Smoothing & Escaping the Curse of Dimensionality (ESL 6.3–6.4)
- **Multivariate Local Linear Regression in $\mathbb{R}^p$**:
  Basis vector expands to $b(x) = (1, (x - x_0)^\top)^\top \in \mathbb{R}^{p+1}$ with radial kernel $K_\lambda(x_0, x) = D\left(\frac{\|x - x_0\|_2}{\lambda}\right)$. Excellent for 2D/3D applications such as implied volatility surface calibration across strike and maturity.
- **The Curse of Dimensionality**:
  For dimensions $p \ge 4$, local smoothing collapses due to two geometric realities:
  1. **Empty Space Phenomenon**: To enclose a fraction $r$ of sample observations in a unit hypercube in $\mathbb{R}^p$, the required neighborhood radius is $e_p(r) = r^{1/p}$.
     - For $p=1$, capturing $1\%$ of the sample requires radius $e = 0.01$ (truly localized).
     - For $p=10$, capturing $1\%$ requires $e = (0.01)^{0.1} \approx 0.63$ (covers over $60\%$ of each feature axis; completely non-local!). Nonparametric MSE convergence slows to $O(N^{-4/(4+p)})$.
  2. **Boundary Proliferation**: In high-dimensional hyperspheres, almost all volume resides in a thin outer shell ($1 - (1-\epsilon)^p \to 1$). Every sample point is effectively on the boundary.
- **Escape Routes: Structured Nonparametric Models (ESL 6.4)**:
  Top quantitative desks avoid unstructured high-dimensional smoothing by imposing domain structure:
  1. **Structured Kernels**: Use a positive semi-definite metric matrix $\mathbf{A} \succeq 0$: $K_{\lambda, \mathbf{A}}(x_0, x) = D\left(\frac{(x - x_0)^\top \mathbf{A} (x - x_0)}{\lambda}\right)$ to eliminate noise dimensions.
  2. **Generalized Additive Models (GAM / ESL Ch.9)**:
     Decompose the target into additive univariate functions:
     $$f(X) = \alpha + \sum_{j=1}^p g_j(X_j)$$
     Solved via the **Backfitting Algorithm**: iteratively smooth partial residuals $y - \alpha - \sum_{k \ne j} g_k(x_k)$ against $X_j$ using 1D local linear regression. Preserves 1D nonparametric convergence rates $O(N^{-2/5})$ while modeling nonlinear factor dependencies.
  3. **Varying-Coefficient Models (The Quantitative Finance Standard)**:
     $$f(X, Z) = \sum_{j=1}^q \beta_j(Z) X_j$$
     Factors $X$ enter linearly, but their factor loadings $\beta(Z)$ vary smoothly according to low-dimensional macroeconomic regime indicators $Z$ (e.g., market volatility, interest rate levels, liquidity spreads). Fits separate local WLS regressions conditional on $Z=z_0$, creating **regime-switching dynamic factor models**.

---

## Module 5: Classic Interview Question Bank (Green Book + HOTS + Top QR Loops)

This module curates high-frequency regression and correlation problems from Xinfeng Zhou's *A Practical Guide to Quantitative Finance Interviews* (the "Green Book"), Timothy Crack's *Heard on the Street* (HOTS), and quantitative researcher loops at Citadel, Two Sigma, and DE Shaw. Each solution details algebraic derivations, Hilbert space geometry, and practitioner traps.

---

### 1. Green Book Classic: Correlation Coefficient Bounds (Gram Matrix PSD & Geometric Angles)

> **Problem Statement (Green Book 3.6 / Two Sigma Classic)**:
> Let $X, Y, Z$ be zero-mean, unit-variance random variables. The correlation between $X$ and $Y$ is $\rho_{xy} = 0.8$, and the correlation between $X$ and $Z$ is $\rho_{xz} = 0.8$.
> 1. Find the maximum and minimum possible values of the correlation between $Y$ and $Z$, $\rho_{yz}$;
> 2. Generalize to arbitrary correlations $\rho_{xy} = a$ and $\rho_{xz} = b$.

**Intuition & Mental Model**:
Correlation matrices must be **Positive Semi-Definite (PSD)**. Geometrically, centered unit-variance random variables form unit vectors in Hilbert space ($L^2$), where the correlation equals the cosine of the angle between vectors: $\rho = \cos\theta$.

**Step-by-Step Derivation**:

**Method 1: Positive Semi-Definite Gram Matrix**
The correlation matrix $\mathbf{R}$ must satisfy $\mathbf{R} \succeq 0$, which requires $\det(\mathbf{R}) \ge 0$:
$$
\mathbf{R} = \begin{pmatrix} 1 & 0.8 & 0.8 \\ 0.8 & 1 & \rho \\ 0.8 & \rho & 1 \end{pmatrix}
$$
Expanding along the first row:
$$
\begin{aligned}
\det(\mathbf{R}) &= 1 \cdot (1 - \rho^2) - 0.8 \cdot (0.8 - 0.8\rho) + 0.8 \cdot (0.8\rho - 0.8) \\
&= 1 - \rho^2 - 0.64 + 0.64\rho + 0.64\rho - 0.64 \\
&= -\rho^2 + 1.28\rho - 0.28 \ge 0
\end{aligned}
$$
Multiplying by $-1$:
$$
\rho^2 - 1.28\rho + 0.28 \le 0
$$
Solving the quadratic equation $\rho^2 - 1.28\rho + 0.28 = 0$:
$$
\rho = \frac{1.28 \pm \sqrt{1.28^2 - 4(0.28)}}{2} = \frac{1.28 \pm \sqrt{1.6384 - 1.12}}{2} = \frac{1.28 \pm \sqrt{0.5184}}{2} = \frac{1.28 \pm 0.72}{2}
$$
Hence:
- $\rho_{\max} = \frac{1.28 + 0.72}{2} = \boxed{1.0}$
- $\rho_{\min} = \frac{1.28 - 0.72}{2} = \boxed{0.28}$

**Method 2: Euclidean Vector Angle (Triangle Inequality)**
In $L^2$, $\langle U, V \rangle = \operatorname{Corr}(U, V) = \cos\theta$:
- $\cos\theta_{xy} = 0.8 \implies \theta_{xy} = \theta_0 = \arccos(0.8)$;
- $\cos\theta_{xz} = 0.8 \implies \theta_{xz} = \theta_0 = \arccos(0.8)$.
By spherical/Euclidean triangle inequalities, the angle $\theta_{yz}$ between $Y$ and $Z$ satisfies:
$$
|\theta_{xy} - \theta_{xz}| \le \theta_{yz} \le \theta_{xy} + \theta_{xz} \implies 0 \le \theta_{yz} \le 2\theta_0
$$
Since $\cos\theta$ is monotonically decreasing on $[0, \pi]$:
1. **Maximum correlation (minimal angle)**: When $\theta_{yz} = 0$, $Y$ and $Z$ are collinear:
   $$ \rho_{\max} = \cos(0) = \boxed{1.0} $$
2. **Minimum correlation (maximal angle)**: When $\theta_{yz} = 2\theta_0$, $Y$ and $Z$ lie on opposite sides of $X$ in the same plane:
   $$ \rho_{\min} = \cos(2\theta_0) = 2\cos^2\theta_0 - 1 = 2(0.8)^2 - 1 = 2(0.64) - 1 = \boxed{0.28} $$

**General Formula**:
For $\rho_{xy} = a, \rho_{xz} = b$, let $\theta_a = \arccos a, \theta_b = \arccos b$:
$$
\rho_{yz} \in \left[ ab - \sqrt{(1 - a^2)(1 - b^2)},\; ab + \sqrt{(1 - a^2)(1 - b^2)} \right]
$$

---

### 2. Green Book Advanced: Minimum Correlation Bound in an Equicorrelated Matrix

> **Problem Statement (Green Book 3.6 / Citadel Core Question)**:
> Suppose there are $n$ assets $X_1, X_2, \dots, X_n$, each with variance $\sigma^2 > 0$. The pairwise correlation between any two distinct assets is identical: $\operatorname{Corr}(X_i, X_j) = \rho, \forall i \ne j$.
> 1. Find the theoretical admissible range of $\rho$ such that the correlation matrix is valid (positive semi-definite);
> 2. What happens to the lower bound as $n \to \infty$? What is the fundamental takeaway for portfolio diversification?

**Step-by-Step Derivation**:

**Method 1: Eigenvalue Decomposition**
The equicorrelated matrix $\mathbf{R}_{n \times n}$ has the algebraic structure:
$$
\mathbf{R} = (1 - \rho)\mathbf{I}_n + \rho \mathbf{1}\mathbf{1}^\top
$$
where $\mathbf{1} = (1, 1, \dots, 1)^\top \in \mathbb{R}^n$.
1. For eigenvector $\mathbf{1}$:
   $$ \mathbf{R}\mathbf{1} = (1 - \rho)\mathbf{1} + \rho \mathbf{1}(\mathbf{1}^\top \mathbf{1}) = (1 - \rho)\mathbf{1} + n\rho \mathbf{1} = [1 + (n - 1)\rho]\mathbf{1} $$
   Thus, $\lambda_1 = 1 + (n - 1)\rho$ (multiplicity 1).
2. For any vector $v$ orthogonal to $\mathbf{1}$ ($v \perp \mathbf{1}$, spanning an $(n-1)$-dimensional subspace):
   $$ \mathbf{R}v = (1 - \rho)v + \rho \mathbf{1}(\mathbf{1}^\top v) = (1 - \rho)v $$
   Thus, $\lambda_2 = \dots = \lambda_n = 1 - \rho$ (multiplicity $n - 1$).

Positive semi-definiteness ($\mathbf{R} \succeq 0$) requires all eigenvalues to be non-negative:
$$
\begin{cases}
1 - \rho \ge 0 \implies \rho \le 1 \\
1 + (n - 1)\rho \ge 0 \implies \rho \ge -\frac{1}{n - 1}
\end{cases}
$$
Thus, the exact valid range is:
$$
\boxed{-\frac{1}{n - 1} \le \rho \le 1}
$$

**Method 2: Equal-Weighted Portfolio Variance (10-Second Interview Shortcut)**
Consider the sum portfolio $S = \sum_{i=1}^n X_i$. Total variance must be non-negative:
$$
\operatorname{Var}(S) = \sum_{i=1}^n \operatorname{Var}(X_i) + \sum_{i \ne j} \operatorname{Cov}(X_i, X_j) = n\sigma^2 + n(n - 1)\rho\sigma^2 = n\sigma^2[1 + (n - 1)\rho] \ge 0
$$
Since $n\sigma^2 > 0$, this immediately yields $1 + (n - 1)\rho \ge 0 \implies \rho \ge -\frac{1}{n - 1}$.

**Financial Takeaway**:
- For $n = 2$: $\rho \ge -1$ (two assets can be perfectly negatively correlated);
- For $n = 3$: $\rho \ge -1/2 = -0.5$;
- As $n \to \infty$: $\lim_{n \to \infty} \left(-\frac{1}{n - 1}\right) = 0$.
**Conclusion**: In an infinite universe of assets, they cannot all be mutually negatively correlated. Systematic market risk forces the average pairwise correlation to be bounded below by 0.

---

### 3. Green Book / Simulation: Validating Correlation Matrices & Cholesky Simulation

> **Problem Statement (Green Book 3.6 / Quant Research Loop)**:
> Given pairwise correlations among three assets: $\rho_{12} = 0.6, \rho_{23} = 0.8, \rho_{13} = 0$.
> 1. Is this correlation matrix mathematically valid?
> 2. If valid, describe how to generate correlated Monte Carlo asset paths using the Cholesky decomposition.

**Step-by-Step Solution**:

**Step 1: Verify Positive Semi-Definiteness**
Construct the matrix:
$$
\mathbf{R} = \begin{pmatrix} 1 & 0.6 & 0 \\ 0.6 & 1 & 0.8 \\ 0 & 0.8 & 1 \end{pmatrix}
$$
Check principal minors:
- $1 \times 1$ minor: $1 > 0$
- $2 \times 2$ minor: $1 - 0.6^2 = 0.64 > 0$
- Determinant:
  $$ \det(\mathbf{R}) = 1(1 - 0.8^2) - 0.6(0.6 - 0) + 0 = 0.36 - 0.36 = 0 $$
All principal minors are $\ge 0$ and $\det(\mathbf{R}) = 0$. Hence, $\mathbf{R}$ is a **valid positive semi-definite matrix** (residing on the boundary of degeneracy where the vectors are coplanar).

**Step 2: Cholesky Factorization & Simulation**
Compute lower-triangular $\mathbf{L}$ such that $\mathbf{R} = \mathbf{L}\mathbf{L}^\top$:
1. $l_{11} = \sqrt{1} = 1$;
2. $l_{21} = 0.6 / 1 = 0.6$, $l_{22} = \sqrt{1 - 0.6^2} = 0.8$;
3. $l_{31} = 0 / 1 = 0$, $l_{32} = (0.8 - 0 \times 0.6) / 0.8 = 1.0$, $l_{33} = \sqrt{1 - 0^2 - 1.0^2} = 0$.

$$
\mathbf{L} = \begin{pmatrix} 1 & 0 & 0 \\ 0.6 & 0.8 & 0 \\ 0 & 1 & 0 \end{pmatrix}
$$
**Simulation Algorithm**:
Sample independent standard normals $Z = (Z_1, Z_2, Z_3)^\top \sim \mathcal{N}(0, \mathbf{I})$. The correlated variables are generated via $X = \mathbf{L}Z$:
$$
\begin{pmatrix} X_1 \\ X_2 \\ X_3 \end{pmatrix} = \begin{pmatrix} Z_1 \\ 0.6 Z_1 + 0.8 Z_2 \\ Z_2 \end{pmatrix}
$$
Verifying covariances: $\mathbb{E}[X_1 X_2] = 0.6$, $\mathbb{E}[X_2 X_3] = 0.8$, $\mathbb{E}[X_1 X_3] = 0$. Exact match!

---

### 4. HOTS Classic: CAPM Beta, Variance Decomposition & The Reverse Regression Trap

> **Problem Statement (Heard on the Street / QuantVault Interview Classic)**:
> Stock A has daily volatility $\sigma_A = 2\%$, market index M has volatility $\sigma_M = 1\%$, and their correlation is $\rho = 0.5$.
> 1. Calculate stock A's CAPM $\beta$ against M, model $R^2$, and residual idiosyncratic volatility $\sigma_\varepsilon$;
> 2. If stock A surged $+4\%$ today, what is your best estimate of market M's return today?
> 3. Under the IID return assumption, what is your prediction for stock A's return tomorrow?

**Step-by-Step Derivation & Pitfalls**:

**Part 1: Forward Regression Parameters**
- **CAPM Beta**:
  $$ \beta_{A \sim M} = \rho \frac{\sigma_A}{\sigma_M} = 0.5 \times \frac{2\%}{1\%} = \boxed{1.0} $$
- **Coefficient of Determination $R^2$**:
  $$ R^2 = \rho^2 = 0.5^2 = \boxed{0.25 = 25\%} $$
- **Residual Volatility**:
  $$ \sigma_\varepsilon = \sigma_A \sqrt{1 - R^2} = 2\% \times \sqrt{1 - 0.25} = 2\% \times \frac{\sqrt{3}}{2} = \boxed{\sqrt{3}\% \approx 1.732\%} $$

**Part 2: The Reverse Regression Trap**
> **Interviewer Trap**: "Since $\beta = 1.0$, if stock A moves $+4\%$, does the market also move $+4\% / 1.0 = +4\%$?"
> **Fatal Flaw**: Regressions do NOT invert! You cannot simply algebraically rearrange $y = \beta x$.

**Correct Derivation**:
To predict $R_M$ given $R_A = +4\%$, we must construct the reverse regression conditioning on $R_A$:
$$
\beta_{M \sim A} = \rho \frac{\sigma_M}{\sigma_A} = 0.5 \times \frac{1\%}{2\%} = \boxed{0.25}
$$
Thus, the expected market return is:
$$
\mathbb{E}[R_M \mid R_A = 4\%] = \beta_{M \sim A} \times 4\% = 0.25 \times 4\% = \boxed{+1\%}
$$
**Standardized Variable Intuition (Regression to the Mean)**:
In $Z$-scores, $z_A = \frac{+4\%}{\sigma_A} = \frac{4\%}{2\%} = +2$ ($2\sigma$ shock).
Conditioning yields: $\hat{z}_M = \rho z_A = 0.5 \times 2 = +1$ ($1\sigma$ shock).
Converting back: $1 \times \sigma_M = +1\%$. Since $|\rho| < 1$, extreme observations always predict less extreme partners!

**Part 3: IID Tomorrow Forecast**
> **Interviewer Trap**: "Since it rose $4\%$ today, will it drop tomorrow to mean-revert?"
> **Correct Answer**: Expected return tomorrow is the unconditional mean (**approximately 0%**)!
Returns were specified as **IID**. Cross-sectional regression to the mean is purely a property of bivariate conditioning at a single snapshot, NOT negative time-series autocorrelation.

---

### 5. HOTS 4.5: Correlation Under Affine Transformations

> **Problem Statement (Heard on the Street Question 4.5)**:
> Given $\operatorname{Corr}(X, Y) = \rho$:
> 1. Find $\operatorname{Corr}(X + 5, Y)$;
> 2. Find $\operatorname{Corr}(5X, Y)$;
> 3. Find $\operatorname{Corr}(-5X + 3, 2Y - 7)$.

**Step-by-Step Derivation**:
Using the bilinear property of covariance $\operatorname{Cov}(aX + b, cY + d) = ac \operatorname{Cov}(X, Y)$ and scale property of standard deviations $\sigma_{aX+b} = |a|\sigma_X$:
$$
\operatorname{Corr}(aX + b, cY + d) = \frac{ac \operatorname{Cov}(X, Y)}{|a|\sigma_X |c|\sigma_Y} = \frac{ac}{|a||c|} \rho = \operatorname{sgn}(ac) \rho
$$
1. **Translation ($a=1, c=1$)**: $\operatorname{Corr}(X + 5, Y) = \boxed{\rho}$ (translation invariant);
2. **Positive Scaling ($a=5, c=1$)**: $\operatorname{Corr}(5X, Y) = \boxed{\rho}$ (scale invariant);
3. **Opposite Sign Scaling ($a=-5, c=2$)**: $ac = -10 < 0 \implies \operatorname{Corr}(-5X + 3, 2Y - 7) = \boxed{-\rho}$.

---

### 6. Top Quant Loop: Omitted Variable Bias (OVB) Formula & Signing the Bias

> **Problem Statement (Citadel / Two Sigma Core Multifactor Question)**:
> Suppose the true data generating process is $y = \beta_1 x_1 + \beta_2 x_2 + \varepsilon$ with $\mathbb{E}[\varepsilon \mid x_1, x_2] = 0$. A researcher mistakenly omits $x_2$ and estimates $y = \alpha x_1 + u$.
> 1. Derive the large-sample probability limit $\operatorname{plim}\hat\alpha$ and state the omitted variable bias formula;
> 2. **Quant Case Study**: If $x_1$ is a short-term momentum factor and $x_2$ is an industry momentum factor ($\beta_2 > 0$), and high-momentum stocks cluster in high-momentum industries ($\operatorname{Cov}(x_1, x_2) > 0$), is the univariate momentum slope overestimated or underestimated?

**Step-by-Step Derivation**:
The univariate OLS estimator is:
$$
\hat\alpha = \frac{\sum x_{1i} y_i}{\sum x_{1i}^2} = \beta_1 + \beta_2 \frac{\sum x_{1i} x_{2i}}{\sum x_{1i}^2} + \frac{\sum x_{1i} \varepsilon_i}{\sum x_{1i}^2}
$$
Taking the probability limit as $N \to \infty$:
$$
\operatorname{plim}\hat\alpha = \beta_1 + \beta_2 \frac{\operatorname{Cov}(x_1, x_2)}{\operatorname{Var}(x_1)}
$$
The **Omitted Variable Bias** is:
$$
\operatorname{Bias} = \operatorname{plim}\hat\alpha - \beta_1 = \boxed{\beta_2 \frac{\operatorname{Cov}(x_1, x_2)}{\operatorname{Var}(x_1)}}
$$
**Quant Takeaway**:
Since $\beta_2 > 0$ and $\operatorname{Cov}(x_1, x_2) > 0$, $\operatorname{Bias} > 0$. The univariate momentum exposure is **substantially overestimated**, confusing industry Beta risk with idiosyncratic stock Alpha.

---

### 7. Top Quant Loop: Measurement Error in Regressors & Attenuation Bias

> **Problem Statement (Two Sigma / DE Shaw Core Signal Question)**:
> The true economic model is $y = \beta x^* + \varepsilon$ with $\beta \ne 0$ and $\mathbb{E}[\varepsilon \mid x^*] = 0$. Due to market microstructure noise (bid-ask bounce, stale quotes), $x^*$ cannot be directly observed. Instead, the trader observes $x = x^* + u$, where $u \sim (0, \sigma_u^2)$ is white noise independent of $x^*$ and $\varepsilon$.
> 1. Derive the probability limit $\operatorname{plim}\hat\beta$ when regressing $y$ on the noisy proxy $x$;
> 2. Explain why this causes "attenuation bias" and why increasing sample size $N \to \infty$ does NOT fix it.

**Step-by-Step Derivation**:
$$
\hat\beta = \frac{\widehat{\operatorname{Cov}}(x, y)}{\widehat{\operatorname{Var}}(x)}
$$
1. **Numerator**: $\operatorname{Cov}(x^* + u, \beta x^* + \varepsilon) = \beta \operatorname{Var}(x^*) = \beta \sigma_{x^*}^2$;
2. **Denominator**: $\operatorname{Var}(x^* + u) = \sigma_{x^*}^2 + \sigma_u^2$.
$$
\operatorname{plim}\hat\beta = \beta \cdot \boxed{\frac{\sigma_{x^*}^2}{\sigma_{x^*}^2 + \sigma_u^2}} = \beta \cdot \frac{1}{1 + \frac{\sigma_u^2}{\sigma_{x^*}^2}}
$$
**Conclusion**:
The reliability ratio $\frac{\sigma_{x^*}^2}{\sigma_{x^*}^2 + \sigma_u^2} < 1$ shrinks the slope **toward zero**. OLS is inconsistent under measurement error in $x$. Remedy requires Instrumental Variables (IV) or Kalman filtering.

---

### 8. Classical Statistics: Multicollinearity, VIF & The Prediction vs. Interpretation Paradox

> **Problem Statement (QR Interview Standard)**:
> 1. State the analytic formula for the variance of the $j$-th regression coefficient $\operatorname{Var}(\hat\beta_j)$ and define the Variance Inflation Factor (VIF);
> 2. Explain why severe multicollinearity destroys factor interpretation but leaves in-sample predictions virtually unharmed.

**Step-by-Step Derivation**:
In multivariate OLS, $\operatorname{Var}(\hat\beta) = \sigma^2 (X^\top X)^{-1}$. Expanding the $j$-th diagonal entry:
$$
\operatorname{Var}(\hat\beta_j) = \frac{\sigma^2}{\sum_{i=1}^N (x_{ij} - \bar{x}_j)^2 (1 - R_j^2)} = \frac{\sigma^2}{\operatorname{TSS}_j} \cdot \operatorname{VIF}_j
$$
where $R_j^2$ is the $R^2$ from regressing $x_j$ on all remaining regressors, and $\operatorname{VIF}_j = \frac{1}{1 - R_j^2}$.

**The Geometric Paradox**:
- **Interpretation collapses**: As $R_j^2 \to 1$, $\operatorname{VIF}_j \to \infty$. Standard errors blow up, $t$-statistics drop to zero, and individual coefficient signs become erratic.
- **Prediction remains solid**: Geometrically, the subspace $\mathrm{Col}(X)$ is well-defined. The orthogonal projection $\hat{y} = Hy$ onto the subspace is unique and numerically stable, even if the individual basis vectors spanning that plane are nearly collinear.

---

### 9. Green Book 4.5 / HOTS: Optimal Futures Hedge Ratio Derivation

> **Problem Statement (Green Book 4.5 / HOTS Derivatives Question)**:
> An asset manager holds spot asset $S$ and hedges using futures contracts $F$. Over the hedging period, spot price change is $\Delta S$ and futures price change is $\Delta F$. The hedged portfolio change is $\Delta \Pi = \Delta S - h \Delta F$.
> 1. Find the hedge ratio $h^*$ that minimizes portfolio variance;
> 2. Show that $h^*$ is identical to the univariate OLS slope and derive the percentage variance reduction.

**Step-by-Step Derivation**:
1. **Minimize Portfolio Variance**:
   $$ \operatorname{Var}(\Delta \Pi) = \sigma_S^2 + h^2 \sigma_F^2 - 2h \operatorname{Cov}(\Delta S, \Delta F) $$
   Setting the derivative with respect to $h$ to zero:
   $$ \frac{d \operatorname{Var}(\Delta \Pi)}{dh} = 2h \sigma_F^2 - 2\operatorname{Cov}(\Delta S, \Delta F) = 0 \implies h^* = \frac{\operatorname{Cov}(\Delta S, \Delta F)}{\operatorname{Var}(\Delta F)} = \rho \frac{\sigma_S}{\sigma_F} $$
2. **OLS Equivalence & Variance Reduction**:
   This is mathematically identical to the OLS slope of regressing $\Delta S$ on $\Delta F$.
   Substituting $h^*$ back into the variance:
   $$ \operatorname{Var}^*(\Delta \Pi) = \sigma_S^2(1 - \rho^2) $$
   Percentage risk reduction is $\frac{\sigma_S^2 - \sigma_S^2(1 - \rho^2)}{\sigma_S^2} = \boxed{\rho^2 = R^2}$.

---

### 10. Geometric Orthogonalization: The Frisch–Waugh–Lovell (FWL) Theorem & Factor Neutralization

> **Problem Statement (Two Sigma / Citadel Core Quantitative Architecture)**:
> In the partitioned regression model $y = X_1 \beta_1 + X_2 \beta_2 + \varepsilon$:
> 1. Show how to obtain $\hat\beta_1$ without joint matrix inversion $(X^\top X)^{-1}$ using stepwise projections;
> 2. Explain the Frisch–Waugh–Lovell (FWL) theorem and its equivalence to "factor neutralization" in multi-factor alpha models.

**Step-by-Step Derivation**:
Define projection matrix $P_2 = X_2(X_2^\top X_2)^{-1}X_2^\top$ and residual-maker matrix $M_2 = \mathbf{I} - P_2$.
1. **Regress $y$ on $X_2$**: $\tilde{y} = M_2 y$ (residuals of $y$ net of $X_2$);
2. **Regress $X_1$ on $X_2$**: $\tilde{X}_1 = M_2 X_1$ (residuals of each column of $X_1$ net of $X_2$);
3. **Residual-on-Residual Regression**:
   $$ \hat\beta_1^* = (\tilde{X}_1^\top \tilde{X}_1)^{-1}\tilde{X}_1^\top \tilde{y} = (X_1^\top M_2 X_1)^{-1} X_1^\top M_2 y $$
By partitioned matrix algebra, $\hat\beta_1^*$ **is algebraically identical to the joint OLS estimator $\hat\beta_1$**.

**Quant Finance Equivalence**:
- Approach A: Neutralize the raw alpha factor against industry dummies and log-market-cap via cross-sectional regression, then regress future returns on neutralized alpha;
- Approach B: Run a joint multiple regression of future returns on raw alpha, industry dummies, and market cap simultaneously.
**FWL guarantees that Approach A and Approach B yield identical alpha returns and slopes!**

---

### 11. Classical Trap: Regression Without Intercept & Negative R²

> **Problem Statement (Quant Interview Pitfall)**:
> In empirical tests of no-arbitrage models, researchers sometimes force the intercept to zero ($y = X\beta + \varepsilon$).
> 1. Why does the sum of residuals $\sum_{i=1}^N \hat\varepsilon_i$ no longer equal zero?
> 2. Why can the standard coefficient of determination $R^2$ become negative?

**Step-by-Step Derivation**:
1. **Residual Sum Zero Condition**:
   Normal equations state $X^\top \hat\varepsilon = 0$. When an intercept is included, $X$ contains the constant vector $\mathbf{1}$, so $\mathbf{1}^\top \hat\varepsilon = \sum \hat\varepsilon_i = 0$. Without an intercept, $\mathbf{1} \notin \mathrm{Col}(X)$, so the residuals do NOT sum to zero.
2. **Breakdown of TSS Decomposition**:
   $$ \operatorname{TSS} = \sum (y_i - \bar{y})^2 = \operatorname{RSS} + \operatorname{ESS} - 2\bar{y}\sum_{i=1}^N \hat\varepsilon_i $$
   Since $\sum \hat\varepsilon_i \ne 0$, the cross-term does not vanish: $\operatorname{TSS} \ne \operatorname{ESS} + \operatorname{RSS}$.
   If the zero-intercept line fits worse than the horizontal line $y = \bar{y}$, $\operatorname{RSS} > \operatorname{TSS}$, producing **$R^2 = 1 - \frac{\operatorname{RSS}}{\operatorname{TSS}} < 0$**.

---

### 12. Quant Reality: The Enormous Commercial Value of Daily R² ≈ 1%

> **Problem Statement (Citadel / Millennium Final Round Question)**:
> A candidate states in an interview: "My equity alpha signal only had an $R^2$ of $1\%$ when predicting next-day returns, so I discarded it as pure noise."
> From the perspective of a quantitative Portfolio Manager, refute this using the **Fundamental Law of Active Management**.

**Step-by-Step Derivation**:
In a univariate regression, $R^2 = \rho^2 \implies |\rho| = \sqrt{R^2} = \sqrt{0.01} = \boxed{0.10}$.
The signal possesses an **Information Coefficient (IC) of 0.10**.

**Fundamental Law of Active Management (Grinold & Kahn)**:
$$ \operatorname{IR} \approx \operatorname{IC} \times \sqrt{\text{Breadth}} $$
For a universe of $N = 1000$ stocks over $T = 252$ trading days:
- Even conservatively assuming effective independent cross-sectional breadth of $N_{\text{eff}} = 100$:
  $$ \text{Breadth} = 252 \times 100 = 25,200 \implies \operatorname{IR} \approx 0.10 \times \sqrt{25,200} \approx 15.87 $$
- Even considering only time-series breadth ($T = 252$, single-stock portfolio):
  $$ \operatorname{IR} \approx 0.10 \times \sqrt{252} \approx 1.59 $$
In systematic equity market-neutral funds, an annualized Sharpe ratio of $1.5 \sim 2.0$ represents an exceptional, world-class alpha capacity! Claiming $R^2 = 1\%$ is useless immediately disqualifies a candidate for failing to understand financial signal-to-noise ratios.

---

## Module 6: One-Minute Answer Checklist

```text
Live Interview Quick Reflexes:
1. When asked for a univariate slope: Instantly output "Slope = \rho * (\sigma_y / \sigma_x)". Do not try to derive least squares on the spot.
2. When asked for a reverse regression slope: Remember the product is \rho^2. Never say the reciprocal! It's a test of mean reversion.
3. When asked about OLS assumptions: Explicitly state "BLUE does not require normality." Normality is for finite-sample hypothesis testing only.
4. When asked about heteroskedasticity/autocorrelation: Clarify that the coefficients are "still unbiased and consistent," but the standard errors are incorrect (usually understated, causing false significance).
5. When asked to contrast Lasso and Ridge: Invoke geometry. Use the "diamond" to explain Lasso's exact zeros and the "sphere" for Ridge's smooth shrinkage.
6. When asked about Kernel Smoothing vs. Local Regression: Highlight that "Nadaraya-Watson local constant has an O(h) boundary bias; local linear regression achieves automatic kernel carpentry (first moment strictly vanishes) to reduce boundary bias to O(h^2); in high dimensions, escape the curse of dimensionality using GAMs or varying-coefficient models".
```

---
