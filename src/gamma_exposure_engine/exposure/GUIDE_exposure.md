# GUIDE_exposure

## Part 1: Conceptual Explanation

This folder turns one option snapshot row per contract into unsigned daily
gamma-structure factors. The raw data supplies open interest and vanilla-option
gamma, but no position owner or dealer sign. The output must therefore describe
gamma mass rather than dealer inventory.

For contract $i$ on date $t$, define $OI_{i,t}$ as open interest in contracts,
$M=100$ as shares per contract, $S_t$ as the underlying close in dollars, and
$\Gamma_{i,t}$ as gamma in inverse dollars. Contract gamma mass is

$$
m_{i,t}=OI_{i,t}M S_t^2\Gamma_{i,t}.
$$

The result has dollar units under the engine's unit-proportional-move scaling.
Multiplying it by $0.01$ expresses the same local curvature for a one-percent
spot move. It is not predicted hedge flow because actual hedge trading also
depends on ownership, inventory sign, rebalancing policy, and market movement.

Cleaning removes missing essentials, non-positive open interest, invalid
prices, expired rows, unsupported option types, and negative vanilla gamma.
Zero gamma and invalid quotes survive with diagnostic flags. Aggregation then
builds total mass, near-spot and front-expiry shares, the distance to the
largest mass node, call-versus-put composition, and strike-expiry concentration.

## Part 2: Code Reference

- `cleaning.py`: `clean_options_snapshot` validates rows and creates
  `open_interest_weighted_gamma`; `summarize_cleaning_diagnostics` reports each
  exclusion and surviving warning.
- `aggregation.py`: `build_strike_gamma_map` and `build_expiry_gamma_map`
  summarize mass by location; `build_daily_gamma_factors` creates the daily
  research columns.
- `__init__.py`: exports the public cleaning and aggregation entry points.

Start with `cleaning.py` to understand the data and unit contract, then read
`aggregation.py` to see how each daily factor is formed.

## Part 3: Short Journal

- 2026-07-13: Removed signed dealer-gamma labels because the raw input contains
  no position sign; the public factors now describe only observable unsigned
  gamma mass, and negative vanilla gamma is treated as invalid data.
