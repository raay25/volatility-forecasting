# Volatility Forecasting with GARCH

This project is an exploratory investigation into forecasting the volatility of daily stock returns. It began with a simple question: **can a GARCH model capture the way volatility clusters over time?**

The analysis started with Amazon and was later expanded to a collection of large publicly traded companies. Along the way, the project moved from a single train/test split to time-series cross-validation and from an intentionally naive baseline to much stronger rolling-variance benchmarks.

The complete investigation, including the explanations, realizations, issues, plots, and outputs, is in [`discovery.ipynb`](discovery.ipynb).

## Background

Daily returns are calculated from adjusted closing prices as

$$
r_t = \frac{P_t-P_{t-1}}{P_{t-1}} \times 100.
$$

A standard regression model often assumes that its error variance is constant. However, financial returns rarely behave that way: world events can cause shocks that lead to periods of larger volatility, for example.

An ARCH($p$) model allows the current conditional variance to depend on past squared shocks ($\epsilon$):

$$
\sigma_t^2 = \alpha_0 + \sum_{i=1}^{p}\alpha_i\epsilon_{t-i}^2.
$$

GARCH($p,q$) extends this idea by also including previous conditional variances:

$$
\sigma_t^2 = \omega + \sum_{i=1}^{p}\alpha_i\epsilon_{t-i}^2 + \sum_{j=1}^{q}\beta_j\sigma_{t-j}^2.
$$

For a GARCH(1,1) model, this becomes

$$
\sigma_t^2 = \omega + \alpha_1\epsilon_{t-1}^2 + \beta_1\sigma_{t-1}^2.
$$

The model therefore combines a recent shock with its previous estimate of volatility. This is what lets it represent volatility persistence.

## The investigation

### 1. Exploring daily returns

Historical prices were downloaded with `yfinance`, initially for Amazon. Daily percentage returns were calculated with pandas and plotted to inspect their behavior.

An Augmented Dickey-Fuller test produced an extremely small p-value, providing strong evidence against a unit root in the return series. An autocorrelation plot of absolute returns then showed significant positive autocorrelation at several lags. Together, these results supported treating returns as stationary while recognizing that their magnitude, and therefore their volatility, was persistent.

### 2. Fitting ARCH and GARCH models

Initial ARCH and GARCH models were fit with a zero conditional mean. The analysis focused on one-day-ahead variance forecasts, with squared returns plotted as a noisy observable proxy for the latent conditional variance.

The connection comes from the shock equation

$$
r_t = \sigma_t z_t, \qquad E[z_t^2]=1.
$$

Given the information available before day $t$,

$$
E[r_t^2 \mid \mathcal{F}_{t-1}] = \sigma_t^2.
$$

An individual squared return is not the true variance (it also contains the random factor $z_t^2$) but its conditional expectation equals the variance. This makes squared returns useful, though noisy, for evaluating volatility forecasts.

### 3. Investigating the GARCH orders

Values of $p$ and $q$ from 1 through 5 were compared in several ways:

- AIC favored a more complex GARCH(1,3) specification.
- BIC, which penalizes additional parameters more strongly, favored GARCH(1,1).
- One out-of-sample split favored GARCH(3,2), although the leading scores were extremely close and tended to use $q=2$.
- A later train/validation/test split produced leading models with $q=1$.

The sensitivity to a single split motivated the use of expanding-window time-series cross-validation. Each fold trained on all observations available up to that point and evaluated forecasts on the following block. The training period grew with each fold while chronological order was preserved.

Cross-validation still found only marginal differences among several specifications. GARCH(1,1) was retained because it was competitive, converged reliably, and achieved this with fewer parameters. This was a deliberate choice in favor of **parsimony** rather than selecting a more complicated model for a tiny score improvement.

### 4. Creating a baseline

The first baseline used yesterday's squared return as today's variance forecast:

$$
\hat{\sigma}_t^2 = r_{t-1}^2.
$$

GARCH appeared to beat this baseline by an enormous margin. Further inspection showed that this was largely a weakness of the baseline: after an unusually calm day, $r_{t-1}^2$ can be nearly zero. Even a moderate return on the following day then receives a catastrophically poor Gaussian log-likelihood score.

Rather than treating this misleading victory as the conclusion, the baseline was improved by averaging recent squared returns:

$$
\hat{\sigma}_t^2 = \frac{1}{w}\sum_{i=1}^{w}r_{t-i}^2,
$$

where rolling windows of 10, 20, and 50 trading days were tested. These baselines preserve the simple idea that recent volatility predicts near-future volatility while greatly reducing the noise of a single squared return.

### 5. Expanding the comparison

The final experiment applied GARCH(1,1) and the rolling baselines to 19 large-company return series using data beginning in 2010.

For each company:

1. The first 80% of observations formed the development period.
2. Five expanding-window folds within that period evaluated the rolling windows without using future observations.
3. GARCH was fit without fitting its parameters on the final 20%.
4. Fixed-parameter, one-day-ahead GARCH forecasts and rolling-variance forecasts were scored on the same final test period.

Forecasts were compared using the average Gaussian log-likelihood of the observed returns:

$$
\ell_t = -\frac{1}{2}\left[\log(2\pi) + \log(\hat{\sigma}_t^2) + \frac{r_t^2}{\hat{\sigma}_t^2}\right].
$$

A larger likelihood is better. This score rewards forecasts that assign high density to the returns that actually occur and strongly penalizes forecasts that are too confident.

## Findings

In the saved multi-company run, GARCH produced the following results against the rolling baselines:

| Baseline | Companies where GARCH scored higher | Mean GARCH log-score improvement |
|---|---:|---:|
| 10-day rolling variance | 19 of 19 | 0.136 |
| 20-day rolling variance | 19 of 19 | 0.058 |
| 50-day rolling variance | 16 of 19 | 0.028 |

The main finding is therefore not that GARCH defeats every alternative by a huge margin. It is more nuanced:

> GARCH(1,1) generally produced better one-day-ahead probabilistic volatility forecasts than rolling historical variance, but its advantage became small against a well-smoothed 50-day baseline.

The 50-day baseline slightly outperformed GARCH for TSM, Visa, and Procter & Gamble in this run. That does not make GARCH ineffective. Instead, it shows that a simple rolling estimator captures a substantial amount of volatility persistence.

Some broader lessons from the project were:

- Volatility persistence is the central useful signal captured by both approaches.
- Model complexity does not guarantee meaningfully better forecasts.
- Results from a single time-series split can be unstable.
- A baseline should be challenged and improved when its failures make the main model look implausibly strong.
- Small forecast improvements may still be useful, but they should not be overstated.

## Running the notebook

Create a Python environment and install the required packages:

```bash
python -m pip install jupyter yfinance pandas numpy matplotlib statsmodels arch tabulate
```

Then start Jupyter and run the notebook from top to bottom:

```bash
jupyter notebook discovery.ipynb
```

Internet access is required when the notebook downloads historical market data.

## Possible extensions

Natural next steps would include Student's $t$ innovations, asymmetric models such as GJR-GARCH or EGARCH, formal tests of forecast-score differences, and evaluation against realized-volatility measures. These are extensions rather than requirements for the current investigation: the notebook already answers its central question and documents how that answer became more credible as the methodology improved.

## Sources

- Hansen, P. R., & Lunde, A. (2005). [*A Forecast Comparison of Volatility Models: Does Anything Beat a GARCH(1,1)?*](https://doi.org/10.1002/jae.800), *Journal of Applied Econometrics*, 20(7), 873–889.
- Gao, M. [*GARCH Estimation and Forecasting*](https://mingze-gao.com/posts/garch-estimation/). A practical overview of estimating and evaluating GARCH models.
- Hyndman, R. J., & Athanasopoulos, G. [*Forecasting: Principles and Practice* — Time Series Cross-Validation](https://otexts.com/fpptr/tscv.html). The expanding-window evaluation procedure used as the basis for the notebook's cross-validation approach.
