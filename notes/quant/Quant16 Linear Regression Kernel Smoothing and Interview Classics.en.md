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
> - **Module 1: OLS Geometry & Algebra**: Matrix Derivations | Orthogonal Projection | Core Univariate Formulas | Reverse Regression Trap
> - **Module 2: Gauss–Markov / BLUE**: The 5 Assumptions | The Normality Myth | Heteroskedasticity & Autocorrelation (White/Newey-West)
> - **Module 3: Variable Selection & Shrinkage**: Best Subset | Ridge Regression | Lasso | Geometric Intuition & Comparison
> - **Module 4: Kernel Smoothing & Local Regression**: Nadaraya-Watson | Boundary Bias & Local Linear Regression | Curse of Dimensionality
> - **Module 5: Interview Classics**: Correlation Bounds (Green Book) | CAPM & Mean Reversion | Omitted Variable Bias | Measurement Error | Multicollinearity | R² Traps
> - **Module 6: One-Minute Answer Checklist**

---

## Module 1: OLS Geometry and Algebra (ESL 3.2)

### 1. Matrix Form and Normal Equations
For the multivariate linear regression model $y = X\beta + \varepsilon$ (where $X$ is an $N \times (p+1)$ full-rank matrix), Ordinary Least Squares (OLS) minimizes the residual sum of squares $\operatorname{RSS}(\beta) = \|y - X\beta\|_2^2$.

Setting the derivative with respect to $\beta$ to zero yields the **normal equations**:
$$
X^\top X\hat\beta = X^\top y
$$
When $X$ has full column rank, the closed-form solution is:
$$
\hat\beta = (X^\top X)^{-1}X^\top y
$$

### 2. OLS Geometric Intuition (Orthogonal Projection)
Geometrically, minimizing the residual sum of squares is equivalent to orthogonally projecting $y$ onto the subspace spanned by the columns of $X$, denoted as $\mathrm{Col}(X)$.
- **Fitted values**: $\hat{y} = X\hat\beta = X(X^\top X)^{-1}X^\top y = H y$, where $H = X(X^\top X)^{-1}X^\top$ is the **hat matrix** (or projection matrix).
- **Residual vector**: $\hat\varepsilon = y - \hat{y} = (I - H)y$ must be orthogonal to the column space of $X$ (i.e., $\hat\varepsilon \perp$ columns of $X$).
- The effective degrees of freedom of the model is $\mathrm{df} = \mathrm{tr}(H) = p+1$.

### 3. Must-Know Univariate Formulas
For simple univariate regression $y = \alpha + \beta x + \varepsilon$, interviewers expect you to know these relationships instantly:
$$
\hat\beta = \frac{\operatorname{Cov}(x, y)}{\operatorname{Var}(x)} = \rho \frac{\sigma_y}{\sigma_x}
$$
$$
\hat\alpha = \bar{y} - \hat\beta \bar{x}
$$
$$
R^2 = \rho^2
$$
> **Trap: Regression Asymmetry**
> Interviewers frequently ask: "If you regress $y$ on $x$ and get a slope of 2, what is the slope of $x$ on $y$?"
> **Wrong Answer**: $1/2$.
> **Correct Explanation**: Based on the formula, $\hat\beta_{y \sim x} = \rho \frac{\sigma_y}{\sigma_x}$, and $\hat\beta_{x \sim y} = \rho \frac{\sigma_x}{\sigma_y}$. Their product is:
> $$ \hat\beta_{y \sim x} \times \hat\beta_{x \sim y} = \rho^2 \le 1 $$
> Thus, the reverse slope is not simply the reciprocal! This mathematical fact lies at the heart of **regression to the mean**.

---

## Module 2: Gauss–Markov Theorem & BLUE (ESL 3.2.2)

The Gauss-Markov theorem states that under specific assumptions, the OLS estimator is the **Best Linear Unbiased Estimator (BLUE)**—meaning it has the minimum variance among all linear, unbiased estimators.

### 1. Gauss-Markov Assumptions
1. **Linearity in parameters**: The true model is $y = X\beta + \varepsilon$.
2. **Exogeneity**: The conditional mean of the errors is zero, $E[\varepsilon \mid X] = 0$.
3. **Homoskedasticity**: The errors have constant variance, $\operatorname{Var}(\varepsilon_i \mid X) = \sigma^2$.
4. **No serial correlation**: The errors are independent of each other.
5. **No perfect multicollinearity**: The design matrix $X$ has full rank.

### 2. The Classic Interview Trap: The Normality Myth
**"Does OLS require the error terms to be normally distributed?"**
**Answer: No!**
OLS is BLUE regardless of whether the errors are normal. The normality assumption is only required when conducting **exact finite-sample $t$-tests and $F$-tests**, or if you want the OLS estimator to exactly match Maximum Likelihood Estimation (MLE). Interviewers will frequently test you on this distinction.

### 3. Consequences of Violations and Remedies
When financial data (especially time series or cross-sectional) violates these assumptions:
- **Heteroskedasticity / Autocorrelation**: The OLS coefficients remain **unbiased and consistent**, but the **standard errors are wrong** (they are no longer minimal). This often results in overly large $t$-statistics and spurious significance.
- **Remedies**: Use robust standard errors. Use **White standard errors** for heteroskedasticity, and **Newey-West standard errors** when dealing with both heteroskedasticity and autocorrelation.

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

### 1. Nadaraya–Watson Kernel Regression
Estimates the local conditional expectation via a weighted average, with weights determined by a kernel and a bandwidth parameter $h$. Choosing $h$ is a classic bias-variance tradeoff.

### 2. Local Linear Regression
Nadaraya-Watson is equivalent to a **local constant** fit. Near the boundaries of the data, the asymmetric neighborhood leads to severe **boundary bias**. **Local linear regression** explicitly fits a local line, which cancels out this boundary bias to the first order—a phenomenon known as automatic kernel carpentry.

### 3. The Curse of Dimensionality in $\mathbb{R}^p$
Kernel smoothing works flawlessly in low dimensions but is crippled by the curse of dimensionality. In high dimensions ($\mathbb{R}^p$), all points are essentially on the boundary, and data within any local neighborhood becomes incredibly sparse.
**Practical Solutions**: In practice, one must rely on **structured kernels** or **additive models** to bypass the curse. Bandwidths ($h$) or regularization parameters ($\lambda$) are typically chosen using Cross-Validation (CV).

---

## Module 5: Interview Classics

The following questions combine standard QR scenarios, "Green Book" staples, and HOTS puzzles.

### 1. Green Book: Correlation Coefficient Bounds
> **Question**: Suppose the correlation between random variables $X$ and $Y$ is $\rho_{xy} = 0.8$, and the correlation between $X$ and $Z$ is $\rho_{xz} = 0.8$. Find the minimum and maximum possible values for $\rho_{yz}$.

**Solution & Intuition**:
A correlation matrix must be **Positive Semi-Definite (PSD)**. This means the determinant of the Gram matrix must be $\ge 0$:
$$
\det\begin{pmatrix}1 & 0.8 & 0.8 \\ 0.8 & 1 & \rho \\ 0.8 & \rho & 1\end{pmatrix} \ge 0
$$
Expanding this yields: $\rho \in [2(0.8)^2 - 1,\ 1] = [0.28,\ 1]$. The maximum is $1$ (if Y and Z align), and the minimum $0.28$ can be verified via the cosine double-angle formula $\cos(2\theta)$ where $\cos\theta=0.8$.

### 2. HOTS Style: CAPM Beta and Mean Reversion
> **Question**: A stock has an annual volatility of $2\%$, the market has a volatility of $1\%$, and their correlation is $0.5$. Find $\beta$, $R^2$, and the residual volatility. If the stock jumped $+4\%$ today, what is your best guess for the market's move? If returns are IID, what is your prediction for the stock tomorrow?

**Solution & Intuition**:
- $\beta = \rho \frac{\sigma_{\text{stock}}}{\sigma_{\text{market}}} = 0.5 \times \frac{2\%}{1\%} = 1$.
- $R^2 = \rho^2 = 0.5^2 = 0.25$.
- Residual volatility = $\sigma_{\text{stock}}\sqrt{1-\rho^2} = 2\% \times \sqrt{0.75} \approx 1.732\%$.
- **The reverse regression trap**: To predict the market given the stock, the slope is $\beta_{\text{reverse}} = \rho \frac{\sigma_{\text{market}}}{\sigma_{\text{stock}}} = 0.25$. Predicted market move = $0.25 \times 4\% = 1\%$. (Since the stock moved $+2\sigma$, the market is predicted to move $+1\sigma$ via $\rho \times 2\sigma$).
- Under the IID assumption, today's move has no bearing on tomorrow. The expected move tomorrow is the unconditional mean (approximately zero).

### 3. HOTS: Correlation Under Affine Transformations
> **Question**: If $\operatorname{Corr}(X,Y) = \rho$, what is $\operatorname{Corr}(X+5, Y)$? What is $\operatorname{Corr}(5X, Y)$?

**Solution & Intuition**:
Correlation is invariant under translation and positive scaling: $\operatorname{Corr}(X+5,Y)=\rho$ and $\operatorname{Corr}(5X,Y)=\rho$. Multiplying by a negative constant flips the sign: $\operatorname{Corr}(-5X,Y)=-\rho$.

### 4. Omitted Variable Bias
> **Question**: If you run a simple regression of $y$ on $x$ but mistakenly omit a key variable $z$, what is the sign of the bias on the coefficient of $x$?

**Solution & Intuition**:
The bias is equal to: (coefficient of the omitted variable) $\times \frac{\operatorname{Cov}(x,z)}{\operatorname{Var}(x)}$. You must evaluate whether the omitted variable positively/negatively impacts $y$ and whether it is positively/negatively correlated with $x$ to sign the bias.

### 5. Measurement Error in x
> **Question**: What happens to your regression coefficient if your independent variable $x$ is measured with random noise?

**Solution & Intuition**:
The coefficient suffers from **attenuation bias** (biased toward zero). The noise inflates $\operatorname{Var}(x)$ in the denominator of $\hat\beta = \frac{\operatorname{Cov}(x,y)}{\operatorname{Var}(x)}$ while leaving the covariance unchanged.

### 6. Multicollinearity Trap
> **Question**: Does severe multicollinearity render your model useless for prediction?

**Solution & Intuition**:
No. While it inflates the standard errors of individual coefficients (making interpretation impossible—measured by VIF), **the model's overall in-sample predictive power remains intact**. It will still predict well out-of-sample as long as the collinear structure holds.

### 7. Hedging Ratios and Futures
> **Question**: You want to hedge a spot position $S$ using futures $F$. What is the variance-minimizing hedge ratio $h$?

**Solution & Intuition**:
By minimizing $\operatorname{Var}(\Delta S - h \Delta F)$, the optimal ratio $h = \frac{\operatorname{Cov}(\Delta S, \Delta F)}{\operatorname{Var}(\Delta F)}$. This is mathematically identical to the OLS slope of regressing $\Delta S$ on $\Delta F$.

### 8. The R² Worship Trap
> **Question**: If a daily stock return model only achieves an $R^2$ of 1%, is it a terrible model?

**Solution & Intuition**:
Not at all. In high-frequency or daily finance, the signal-to-noise ratio is minuscule. A true, stable out-of-sample $R^2$ of 1% on daily returns can yield an outstanding Sharpe ratio when scaled with leverage and breadth. Never judge financial models by textbook macroeconomic $R^2$ standards.

---

## Module 6: One-Minute Answer Checklist

```text
Live Interview Quick Reflexes:
1. When asked for a univariate slope: Instantly output "Slope = \rho * (\sigma_y / \sigma_x)". Do not try to derive least squares on the spot.
2. When asked for a reverse regression slope: Remember the product is \rho^2. Never say the reciprocal! It's a test of mean reversion.
3. When asked about OLS assumptions: Explicitly state "BLUE does not require normality." Normality is for finite-sample hypothesis testing only.
4. When asked about heteroskedasticity/autocorrelation: Clarify that the coefficients are "still unbiased and consistent," but the standard errors are incorrect (usually understated, causing false significance).
5. When asked to contrast Lasso and Ridge: Invoke geometry. Use the "diamond" to explain Lasso's exact zeros and the "sphere" for Ridge's smooth shrinkage.
```

---
