---
title: "What SPY Gamma Exposure Told Me About Next-Day Volatility"
description: "An offline study of SPY option gamma, next-day realized variance, and the limits hidden inside a clean quantitative pipeline."
date: 2026-07-13
image: images/gamma-surface-cover.png
categories: ["Quantitative Research", "Options"]
---

# What SPY Gamma Exposure Told Me About Next-Day Volatility

Options commentary often gives gamma a confident story. Concentrated dealer gamma should dampen price moves; short gamma should amplify them. The story is plausible, but a useful research pipeline has to separate the mechanism from what the available data can identify.

I tested a narrower question with committed SPY data: is an open-interest-weighted gamma measure observed on day $t$ associated with realized intraday variance on the next trading day, $t+1$? The January 2024 sample contains 21 exposure dates and 20 properly aligned factor-response pairs. It produces a clear answer for this small window: no persuasive association.

That null result is worth keeping. It also exposes a more consequential issue in the factor definition. The implementation aggregates positive option gamma without inferring dealer inventory direction, so its stored `net_gamma_exposure` is not signed dealer gamma.

## The measurement comes before the story

For option contract $i$ on date $t$, define $OI_{i,t}$ as open interest in contracts, $M=100$ as the standard US equity-option contract multiplier, $S_t$ as the SPY close in dollars, and $\Gamma_{i,t}$ as option gamma, the change in delta for a one-dollar move in SPY. The engine computes contract-level exposure $g_{i,t}$ as

$$
g_{i,t} = OI_{i,t} M S_t^2 \Gamma_{i,t}.
$$

It then sums across the set $\mathcal{O}_t$ of valid option rows observed on date $t$:

$$
G_t = \sum_{i \in \mathcal{O}_t} g_{i,t}.
$$

The $S_t^2$ scaling converts a local curvature measure into the implementation's dollar-scaled exposure convention. This is a descriptive option-market factor. It does not reveal who owns each contract, whether a market maker is long or short it, or how much has already been hedged.

The core expression in the cleaner is deliberately plain:

```python
gamma_exposure = (
    open_interest
    * CONTRACT_MULTIPLIER
    * spot_close
    * spot_close
    * gamma
)
```

That plainness makes an important fact easy to audit: no call/put or dealer-position sign enters the multiplication.

## Building a next-day response without lookahead

For minute $j$ on trading day $t$, let $P_{t,j}$ be the minute close and let $r_{t,j}$ be its log return:

$$
r_{t,j} = \log(P_{t,j}) - \log(P_{t,j-1}).
$$

If day $t$ contains $n_t$ valid minute returns, daily realized variance is

$$
RV_t = \sum_{j=1}^{n_t} r_{t,j}^2.
$$

The research row pairs $G_t$ with $RV_{t+1}$. Here, $t+1$ means the next observed trading date, not the next calendar day. The dataset builder sorts exposure dates, shifts that calendar by one row, and keeps a response only when its date matches the shifted exposure date. This avoids pairing Friday's factor with Saturday or leaking Friday's realized move into Friday's predictor.

```python
ordered_exposures = exposures.sort("trade_date")
exposure_calendar = ordered_exposures.with_columns(
    pl.col("trade_date").shift(-1).alias("_next_exposure_date")
)
aligned = exposure_calendar.join(
    response_payload,
    left_on="_next_exposure_date",
    right_on="_response_join_date",
    how="inner",
)
```

The offline boundary is just as useful as the date alignment. Both raw Parquet files are committed, the run does not call ClickHouse, and every table used below can be regenerated with one command. That limits accidental differences between an exploratory notebook and the reported result.

## What the 20 observations show

The daily factor ranges from 2.98 trillion to 4.74 trillion in the engine's exposure units. The following day's realized variance moves quite differently.

![Daily SPY gamma exposure and following-day realized variance](images/01-daily-alignment.png)

The two series have visible day-to-day variation, but their peaks and troughs do not line up consistently. A chart can suggest that absence; rank statistics give it a precise test.

Spearman's rank correlation measures monotonic association. Define $R(G_t)$ as the rank of $G_t$ and $R(RV_{t+1})$ as the rank of the following day's realized variance. With $\operatorname{Cov}$ denoting covariance and $\sigma$ denoting standard deviation, the statistic is

$$
\rho_s = \frac{\operatorname{Cov}\left(R(G_t), R(RV_{t+1})\right)}
{\sigma_{R(G)}\sigma_{R(RV)}}.
$$

For this sample, $\rho_s=0.030$ with a two-sided p-value of 0.900. A Kruskal-Wallis test across five gamma buckets gives $H=4.786$ and a p-value of 0.310. Neither test rejects the null of no systematic relationship at conventional significance levels.

| Check | Result | Interpretation |
|---|---:|---|
| Aligned observations | 20 | One month is a diagnostic sample, not a basis for a stable estimate |
| Spearman rank correlation | 0.030 | Almost no monotonic association |
| Spearman p-value | 0.900 | The observed rank relationship is compatible with noise |
| Kruskal-Wallis statistic | 4.786 | Bucket distributions differ too little for this sample |
| Kruskal-Wallis p-value | 0.310 | No rejection across the five buckets |

The quintile view tells the same story. Each bucket contains only four observations. Mean next-day variance rises through the middle buckets, then falls in the highest bucket. The 95% percentile-bootstrap intervals overlap widely.

![Next-day realized variance by gamma-exposure quintile](images/02-quantile-variance.png)

A monotonic gamma effect should leave a more orderly sequence than this. The fifth bucket's mean, $4.04 \times 10^{-5}$, sits below the third bucket's $5.91 \times 10^{-5}$. With four days per bucket, either value can move sharply when one observation changes.

![Scatter plot of gamma exposure against next-day realized variance](images/03-factor-scatter.png)

The scatter plot makes the sample-size problem tangible. The fitted line is descriptive, not a forecast, and the rank correlation printed on the chart is the statistic that matters here.

## The audit found a bigger limitation than the p-value

The raw snapshot contains 168,762 option rows. Cleaning retains 131,215 rows and excludes 37,547 rows with non-positive open interest, or 22.25% of the input. Another 24,331 retained rows have zero gamma. Those diagnostics are useful because they make the input filter visible rather than silently shrinking the sample.

The factor audit matters more. In every one of the 21 daily snapshots:

- `net_gamma_exposure` equals `absolute_gamma_exposure` exactly;
- the largest negative-gamma strike distance is undefined;
- all contract gamma contributions entering the aggregate are non-negative.

The result follows directly from the formula. Standard call gamma and put gamma are both positive for a long option. Without a position-side assumption, the aggregate measures open-interest-weighted gamma mass. Calling it signed dealer gamma would add information that the dataset does not contain.

This distinction changes the economic interpretation. A dealer inventory model might assign signs from customer flow, trade direction, or an explicit heuristic. Each choice adds assumptions and possible measurement error. The current factor avoids those assumptions, but it cannot test the usual long-dealer-gamma versus short-dealer-gamma story.

## Small samples also disable the sophisticated appendices

The configured volatility-regime classifier needs 20 prior observations before it labels a day. The walk-forward predictive comparison also needs 20 training rows before scoring its first out-of-sample forecast. Once day-$t$ exposures are aligned with day-$t+1$ responses, only 20 rows remain. Both output tables are therefore empty by design.

That is the correct behavior. Lowering the thresholds until a model produces a score would create a result with almost no evaluation history. The empty tables document insufficient evidence more honestly than a fitted model on the full month.

The first-half and second-half rank correlations also change sign: $-0.188$ for the first ten observations and $0.152$ for the last ten. Their p-values are 0.603 and 0.676. This split does not prove instability, since each half is tiny, but it gives no reason to treat the full-sample estimate as durable.

## What I would change before using this factor

The next research run needs more dates before it needs more model complexity. A multi-year sample would permit past-only regime labels, walk-forward forecasts, and leave-one-month-out checks that contain actual observations.

I would also rename the current factor to `open_interest_weighted_gamma` so the code states exactly what the data support. A separate dealer-gamma estimate could then encode and test a documented sign convention. Comparing the unsigned and signed versions would show whether an apparent effect comes from gamma concentration or from the inventory assumption.

Finally, daily open interest is stale within the session and does not identify intraday positioning changes. If the purpose is to explain next-day variance, dated snapshots, corporate-action handling, option expiry effects, and the exact observation timestamp all belong in the data contract.

The January run does not validate the market narrative. It validates the research discipline: preserve time order, expose the factor convention, and allow the output to be empty or statistically unremarkable when the data cannot support a stronger claim.
