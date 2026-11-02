# LSTM-DDPG trading research

A dependency-free (plain JS/Bun) implementation of a recurrent encoder + DDPG
actor-critic for EURUSD trading, with manual backpropagation, truncated BPTT,
audited causal features, supervised behavior cloning, anchored fine-tuning,
walk-forward retraining, seed ensembles, and reproducible runs.

> **Status: research prototype, not production trading software.** Behavior
> cloning produces a modest, walk-forward-positive baseline on H4, but the
> result is regime-dependent and not validated for live trading. See
> [Current results](#current-results).

## Quick start

```bash
# 1. Put the data where the trainer expects it:
#      data/EURUSD_D1.json   (schema below)
# 2. Install and verify the manual gradients
bun install
bun run gradcheck

# 2b. Optional: download a 16-pair daily universe (Yahoo Finance)
#     Writes data/Y_<SYMBOL>_D1.json (5.3k-7.8k bars per pair)
bun run fetch-data

# 3. Audit the features before training (walk-forward IC + stability)
bun run audit

# 4. Train a reproducible model (writes saved_models/first_*)
bun run train --tf D1 --seed 1 --prefix first

# 5. Evaluate the behavior-cloned policy
#    (use first_best_model for the anchored-DDPG version)
bun run backtest --tf D1 --model first_bc_model --folds 3

# 6. Honest out-of-sample check: 5 folds, embargo, per-fold teacher
bun run walkforward --tf H4 --folds 5 --start 0.4 --seeds 1,2,3 --epochs 0 --select-teacher

# 7. Cross-sectional selection across the 16-pair universe
bun run fetch-data
bun run universe --folds 5 --start 0.4 --seeds 1,2 --data-points 5000 --concurrency 6

# 8. Optional: compare seeds on one split
bun run sweep --tf H4 --epochs 0 --seeds 1,2,3 --folds 3
```

The default DDPG run is 30 epochs x 50 steps, batch 16, with 5 behavior-cloning
epochs. A single seed takes roughly 1-2 minutes on a desktop. Backtests print
their metrics and write `backtest_results/<model>_trades.csv` and
`_summary.json`.

Each run writes four checkpoints:

- `<prefix>_pretrain_model`: supervised encoder before cloning
- `<prefix>_bc_model`: behavior-cloned actor (usually the best policy)
- `<prefix>_best_model`: best-validation checkpoint after DDPG fine-tuning
- `<prefix>_final_model`: last DDPG epoch

Prefer walk-forward evaluation over a single split, and never select features,
thresholds, or checkpoints on the test tail.

## Ways to use this repo

| Goal | Command | What to look at |
|---|---|---|
| Sanity-check the math | `bun run gradcheck` | `GRADIENT CHECK PASSED` |
| Smoke-test the pipeline | `bun run train --epochs 1 --steps 10 --pretrain-steps 5 --no-save` | runs without NaN/Inf |
| Fetch multi-pair data | `bun run fetch-data` | `data/Y_<SYMBOL>_D1.json` |
| Audit features | `bun run audit` or `bun run audit --data data/EURUSD_H4.json --horizons 30,60,120` | IC, stability, candidates |
| Train + evaluate one split | `bun run train --tf H4 --seed 1 --prefix h4` then `bun run backtest --tf H4 --model h4_bc_model --folds 3` | PF, expectancy, folds |
| Multi-pair walk-forward | `bun run walkforward --tf D1 --data data/Y_USDJPY_D1.json --label Y_USDJPY --folds 5 --start 0.4 --seeds 1,2,3 --epochs 0 --select-teacher` | per-symbol OOS folds |
| Regime-gated walk-forward | add `--regime-gate` to the command above | gate choice / skipped-regime lines |
| Training-only universe basket | `bun run universe --folds 5 --start 0.4 --seeds 1,2 --data-points 5000 --concurrency 6` | eligible basket vs all-symbol basket |
| Walk-forward + ensemble | `bun run walkforward --tf H4 --folds 5 --start 0.4 --seeds 1,2,3 --epochs 0 --select-teacher` | per-fold OOS returns, teacher choices |
| Ensemble existing models | `bun run backtest --tf H4 --ensemble h4_seed1_bc_model,h4_seed2_bc_model,h4_seed3_bc_model` | averaged policy metrics |
| Anchored DDPG fine-tune | `bun run train --tf H1 --seed 1 --prefix anchored` | `anchored_best_model` vs `anchored_bc_model` |
| Compare seeds | `bun run sweep --tf H4 --epochs 0 --seeds 1,2,3 --folds 3` | profitable seeds, positive folds |
| Random policy floor | `bun run backtest --random --trades 20` | reference |
| Skip supervised stages | `bun run train --no-pretrain --no-bc` | unrestrained DDPG |

### Reading a backtest

Start with **Expectancy (R)**, **Profit factor**, and **Max drawdown**, then check
whether the result holds across **folds**, **seeds**, and **walk-forward
retraining**. A single positive fold, PF above ~1.1 on <100 trades, or one
profitable seed is not evidence. A usable signal should stay positive across
most out-of-sample blocks and multiple seeds with a consistent direction mix.

### Tuning cheatsheet

| Symptom | First knob to change |
|---|---|
| Policy is always flat | reduce `--min-sigma`, raise `flatPenalty`, lower the size dead zone |
| Policy is always one direction | actor/RL layer: use BC + anchor, more seeds, or more BC epochs |
| Stops never hit / all `close` exits | reduce `--range-window`, cap SL/TP multiples |
| Validation improves then collapses | raise `--bc-anchor`, freeze the encoder, lower actor LR |
| Over-trades after gains | raise spread cost, add turnover/drawdown penalties |
| Results swing by seed | more seeds + ensemble, walk-forward, more data |
| No result survives walk-forward | regime dependence or overfitting: rework features/horizon |

## Pipeline

```
OHLCV bars -> 12 audited causal features -> standardized (train-region stats)
          -> 60-step LSTM (BPTT) -> hidden state (x fixed state gain)
          -> actor  -> [direction, size, SL fraction, TP fraction]
          -> critic <- (state, action)

Horizons: D1 20 bars, H4 120 bars, H1 480 bars (~20 trading days)
SL/TP distances = rangeWindow bars of price range * (0.5 + k * fraction)
Reward           = tanh((PnL_R - cost_R) / 2) * size   (risk-normalized)
```

- Features: `ret1z/5z/20z/60z`, `atrRel`, `volRatio`, `dist20`, `dist50`,
  `slope20`, `donchian20`, `rsi14`, `bollingerB`. See
  [Signal audit](#signal-audit-feature-lab).
- The LSTM is genuinely recurrent: 60 bars, one 12-feature vector per step,
  hidden/cell state carried, truncated BPTT from the final hidden state.
- Training stages: (1) encoder pretraining on forward-return direction,
  (2) behavior cloning of the actor from the audited contrarian teacher,
  (3) anchored DDPG fine-tuning with a frozen encoder by default.
- Gradients are exact (numerically checked with `bun run gradcheck`).
- Costs: spread from the data file (`spread * point`, ~1 pip for the bundled
  EURUSD file) is charged on every simulated trade and in the training reward.
- Split: first 80% of the selected file for training, last 20% for evaluation.
  Feature standardization uses training-region statistics only. Walk-forward
  runs move this boundary per fold and retrain from scratch.

## Requirements and data

- [Bun](https://bun.sh) >= 1.1
- Data files in `data/` (gitignored). Expected schema:

```json
{
  "spread": 10,
  "point": 0.00001,
  "period": 1440,
  "time": [ ... ],
  "open": [ ... ],
  "high": [ ... ],
  "low": [ ... ],
  "close": [ ... ],
  "volume": [ ... ]
}
```

The EURUSD intraday files come from `fractal-solutions/paed-framework` (`data/`):

| Profile | File | Bars |
|---|---|---|
| D1 | EURUSD_D1.json | 5,014 |
| H4 | EURUSD_H4.json | 25,848 |
| H1 | EURUSD_H1.json | 100,000 |
| M15 | EURUSD_M15.json | 100,000 |

`bun run fetch-data` downloads a multi-pair daily universe from Yahoo Finance
into `data/Y_<SYMBOL>_D1.json`:

```
EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD EURGBP
EURJPY GBPJPY EURCHF EURAUD EURCAD AUDJPY CADJPY CHFJPY
```

Coverage is 5.3k-7.8k daily bars per pair (USDJPY back to 1996; EURGBP to 1999;
most others 2003+), which is more history than the Dukascopy daily file for
EURUSD. Yahoo FX volume is unavailable, so those files store zero volume and the
`volumeZ` feature is inert on them.

Use `--data data/Y_USDJPY_D1.json` and a `--label` to keep checkpoints separate.
`bun run audit` automatically picks up every `data/EURUSD_*.json`; pass
`--data` explicitly for the Yahoo files.

## Install and verify

```bash
bun install
bun run gradcheck     # LSTM BPTT, Dense, critic, actor finite differences
```

## Training

```bash
bun run train --tf D1 --seed 1 --prefix d1_s1      # D1, 20-bar horizon (default)
bun run train --tf H4 --seed 1 --prefix h4_s1      # H4, 120-bar horizon
bun run train --tf H1 --seed 1 --prefix h1_s1      # H1, 480-bar horizon
bun run train --tf H4 --epochs 0 --bc-epochs 5 --seed 1 --prefix bc_only
bun run train --no-bc --no-pretrain --prefix plain_ddpg
```

| Flag | Meaning |
|---|---|
| `--tf D1\|H4\|H1` | Timeframe profile: data file, horizon, SL/TP range window, teacher |
| `--epochs`, `--steps` | DDPG epochs and gradient steps per epoch |
| `--batch` | Replay minibatch size |
| `--hidden` | LSTM / critic hidden size (default 32) |
| `--bars` | Override `barsPredicted` (trade horizon in bars) |
| `--range-window` | Override bars used for the volatility-based SL/TP distance |
| `--features` | Comma-separated override of the feature list |
| `--data-points` | Limit trailing bars used (default: full file) |
| `--pretrain-epochs`, `--pretrain-steps` | Encoder pretraining (0 = skip) |
| `--no-pretrain` | Disable encoder pretraining |
| `--bc-epochs`, `--bc-steps` | Actor behavior cloning (default 5 epochs x 50 steps) |
| `--no-bc` | Disable behavior cloning and anchoring |
| `--teacher` | Teacher feature for cloning (default per profile) |
| `--teacher-sign` | `-1` contrarian (default), `+1` momentum |
| `--bc-anchor` | Anchor strength toward the BC policy (default 1.0) |
| `--no-freeze-encoder` | Let the encoder update during DDPG (frozen after BC by default) |
| `--actor-lr`, `--critic-lr`, `--encoder-lr`, `--pretrain-lr` | Learning rates |
| `--sigma`, `--min-sigma`, `--noise-decay` | OU exploration schedule |
| `--gamma` | Discount factor |
| `--seed` | Seed `Math.random` for a reproducible run |
| `--prefix` | Checkpoint name prefix |
| `--no-save` | Do not write checkpoints |
| `--train-fraction` | Fraction used for training (default 0.8; walk-forward sets it) |
| `--test-end` | End fraction of the evaluation block (default 1.0) |
| `--embargo` | Gap in bars between train and test (default = `barsPredicted`) |
| `--gate-feature`, `--gate-min`, `--gate-max` | Regime gate: trade only inside a standardized feature band |

| Profile | File | Bars | Horizon | Range window | Teacher |
|---|---|---|---|---|---|
| `D1` | EURUSD_D1.json | 5,014 | 20 bars (~1 month) | 20 bars | `ret1z` |
| `H4` | EURUSD_H4.json | 25,848 | 120 bars (~1 month) | 60 bars | `bollingerB` |
| `H1` | EURUSD_H1.json | 100,000 | 480 bars (~1 month) | 240 bars | `ret20z` |

Checkpoints store weights plus metadata (timeframe, feature names, lookback,
hidden size, horizon, range window, state gain, seed, spread, train/test
fractions, anchor/freeze flags). Loading rejects mismatches instead of silently
simulating a different setup.

### Behavior cloning (supervised actor init)

The audit signals are contrarian, so the actor is initialized from a teacher:

| Profile | Teacher | Sign |
|---|---|---|
| D1 | `ret1z` | contrarian |
| H4 | `bollingerB` | contrarian |
| H1 | `ret20z` | contrarian |

The teacher label is the sign of the last state bar's teacher feature. The
actor's direction output is trained with MSE to that label while the pretrained
encoder is held fixed; size/SL/TP keep their initialized values. The resulting
`<prefix>_bc_model` checkpoint is saved before DDPG starts:

```bash
bun run train --tf H1 --epochs 0 --bc-epochs 5 --seed 1 --prefix bc_h1
bun run backtest --tf H1 --model bc_h1_bc_model --folds 3
```

Set `--epochs 0` to skip DDPG and evaluate the cloned policy alone.

**Anchored fine-tuning:** when BC is enabled, DDPG starts from the cloned actor
with the encoder frozen and adds an L2 anchor toward the cloned policy
(`--bc-anchor`, default 1.0). This prevents unrestricted DDPG from destroying
the BC policy. Use `--no-freeze-encoder` to let the encoder move; `--no-bc`
disables cloning and anchoring.

```bash
bun run train --tf H1 --seed 1 --prefix anchored --bc-anchor 1.0
bun run backtest --tf H1 --model anchored_bc_model --folds 3
bun run backtest --tf H1 --model anchored_best_model --folds 3
```

## Backtesting

```bash
bun run backtest --tf H4 --model h4_seed1_bc_model --folds 3
bun run backtest --tf H4 --ensemble h4_seed1_bc_model,h4_seed2_bc_model,h4_seed3_bc_model
bun run backtest --random --trades 20
```

`--tf` (and matching `--bars`/`--range-window`/`--features`/`--train-fraction`/
`--test-end`) must match the checkpoint.

| Flag | Meaning |
|---|---|
| `--tf D1\|H4\|H1` | Timeframe profile; must match the checkpoint |
| `--model <name>` | Checkpoint name without `.json` (default `final_model`) |
| `--ensemble <a,b,c>` | Average the action vectors of several checkpoints |
| `--folds <n>` | Split the evaluation period into `n` contiguous folds |
| `--trades <n>` | Cap the number of trades evaluated |
| `--random` | Use randomly initialized networks (no checkpoint) |
| `--seed` | Seed the random init for `--random` |
| `--train-fraction`, `--test-end` | Evaluation block boundaries (walk-forward) |
| `--gate-feature`, `--gate-min`, `--gate-max` | Apply the regime gate used/saved by a model |
| `--no-csv` | Skip writing CSV/JSON results |
| `--data` | Alternate data file path |
| `--bars`, `--range-window`, `--features` | Override profile values (must match checkpoint) |

Results are written to `backtest_results/` (gitignored) as
`<model>_trades.csv` and `<model>_summary.json`.

### Metric definitions

- **Trade**: entry at the close of bar `i-1`, state from bars `[i-60, i)`,
  simulated over the profile horizon (20 D1 / 120 H4 / 480 H1), non-overlapping.
- **R**: price move divided by the initial stop distance.
  `netR = pnlR - costR`; risk-normalization makes stop sizes comparable.
- **Expectancy**: mean `netR` per trade.
- **Profit factor**: gross winning PnL / gross losing PnL.
- **Total return / CAGR**: compounded equity from position-sized trade PnL minus
  spread. Years and CAGR use the file's `period` (intraday aware).
- **Max drawdown**: peak-to-trough on that equity curve.
- **Sharpe (per trade)**: mean/std of trade returns times `sqrt(trades)`.
- **Folds**: the evaluation period is split into contiguous blocks; each fold's
  metrics are computed independently.

## Walk-forward retraining and seed ensembles

A single 80/20 holdout can be dominated by one regime. The walk-forward runner
retrains from scratch on an expanding window for each block, evaluates each seed
on the next block, and then evaluates the seed ensemble on that block:

```bash
# Behavior-cloned policy, 5 expanding folds, 3-seed ensemble, embargoed
bun run walkforward --tf H4 --folds 5 --start 0.4 --seeds 1,2,3 --epochs 0

# Same, but re-select the teacher/feature per fold from training data only
bun run walkforward --tf H4 --folds 5 --start 0.4 --seeds 1,2,3 --epochs 0 --select-teacher

# Anchored DDPG fine-tune on the same protocol
bun run walkforward --tf H1 --folds 3 --seeds 1,2 --epochs 10 --steps 25

# Custom split: first training window ends at 60%, then 3 equal blocks
bun run walkforward --tf D1 --folds 3 --start 0.6 --seeds 1,2,3 --epochs 0
```

Defaults: `--start 0.5`, `--folds 3`, `--epochs 0` (BC only), `--bc-epochs 5`.
The embargo is `barsPredicted` by default (20 D1 / 120 H4 / 480 H1) and can be
overridden with `--embargo <bars>`. The test block starts after the embargo, so
training labels cannot overlap it.

`--select-teacher` re-runs the univariate audit on each fold's training window
and chooses the teacher feature and sign with the strongest stable IC (signed
features only), then trains every seed for that fold with that teacher. This is
fold-by-fold feature selection with no test-data leakage.

`--regime-gate` re-audits regime bands on the same training window. It scores
quintile bands of volatility/trend features for the selected teacher, requires
stability across training sub-folds, and passes the best band to training and
backtest (`--gate-feature/--gate-min/--gate-max`). Outside the band the policy is
forced flat. The gate is *selective*: if the ungated teacher baseline is already
positive in training, no gate is applied. This avoids weakening symbols that
already work, but it can still be fooled when the training baseline is
optimistic (see GBPUSD below).

Each fold writes `<tf>_wf<fold>_s<seed>_*` checkpoints; the aggregate summary is
written to
`backtest_results/walkforward_<label>_<model>[_sel][_gate]_s<n>_f<n>_st<start>.json`.
The table reports per-fold seed-average return, ensemble return, PF, trades, and
drawdown; the aggregate is the compounded ensemble return across OOS folds.

`backtest --ensemble a,b,c` averages the four action outputs (`direction`,
`size`, `SL fraction`, `TP fraction`) across models and simulates the averaged
action.

## Universe selection (training-only)

`universe.js` runs the walk-forward for every symbol, then decides eligibility
per symbol and per fold using **only the training window**:

1. select a teacher from the signed audited features (same rule as
   `--select-teacher`);
2. compute the teacher's net return over five training sub-folds;
3. require positive overall training net return and at least 4/5 positive
   sub-folds, with >=200 observations.

It then builds an equal-weight basket of eligible symbols on each out-of-sample
block. No test return is used for selection.

```bash
bun run fetch-data
bun run universe --folds 5 --start 0.4 --seeds 1,2 \
  --data-points 5000 --concurrency 6
```

Notes:

- `--data-points 5000` trims each pair to the last 5,000 bars so the fold
  fractions are approximately aligned across symbols (they end on the same date
  but have slightly different trading calendars).
- Checkpoints are written to the OS temp directory (`lstm-ddpg-universe-models`),
  not the repo. The summary is
  `backtest_results/universe_s<n>_f<n>_st<start>_d<bars>.json`.
- This is a screen, not proof: 16 symbols x 5 folds creates many opportunities
  for a false winner. Any result must be pre-registered and then validated on
  data not used for the rule, not re-tuned until it looks good.

## Multi-seed sweep

```bash
bun run sweep --tf H4 --epochs 0 --seeds 1,2,3 --folds 3
bun run sweep --seeds 1,2 --epochs 10 --steps 30 --pretrain-epochs 2 --pretrain-steps 20
```

Each seed trains with `<tf>_seed<SEED>_*` checkpoints and runs the backtest,
then prints a per-seed table and aggregate line. With `--epochs 0` the sweep
backtests the BC checkpoint instead of a DDPG best checkpoint.

## Signal audit (feature lab)

Audit candidate features with causal, purged walk-forward folds before feeding
them to the model:

```bash
bun run audit                                              # all EURUSD_*.json
bun run audit --data data/EURUSD_D1.json --horizons 5,10,20
bun run audit --data data/EURUSD_H4.json --horizons 30,60,120
bun run audit --horizons 5,10,20 --folds 5 --min-ic 0.02 --min-stability 0.7
```

For each feature and horizon it prints walk-forward Spearman IC, fold
stability, and directional hit rate, then lists candidates. Per-timeframe
CSV/JSON is written to `signal_audit/` (gitignored). Features are causal and
never normalized with future data:

```
ret1z ret5z ret20z ret60z            vol-normalized momentum
atrRel volRatio rangeRatio shock      volatility / regime
dist20 dist50 slope20                 trend
donchian20 rsi14 bollingerB           range / oscillator
gap volumeZ                           structure / volume
```

Reading it: `|IC| ~ 0.02-0.03` with stability `>= 0.7` across folds is worth
testing; one high-IC fold is not. A negative IC with hit rate below 50% means
the feature is **contrarian**. Always compare the edge with round-trip cost.

Initial audit (all data, univariate event study; `net edge` is a rough
`(2p-1)*mean|move| - cost` estimate, not a strategy backtest):

| Data | Bars | Best signal at ~20-day horizon | Stability | Hit (contrarian) | Net edge |
|---|---|---|---|---|---|
| D1 | 5,014 | `ret1z` | 0.80 | 52.2% | ~6.3 bps/trade |
| H4 | 25,848 | `bollingerB` / `rsi14` | 0.80 | 50.8% / 50.6% | ~2.1 / 1.3 bps |
| H1 | 100,000 | `ret20z` / `bollingerB` | 0.80 | 50.7% / 50.6% | ~1.8 / 1.3 bps |
| M15 | 100,000 | stable mean reversion, edge < spread | 1.00 | ~48.6% raw | < 1 bp, net negative |

Teacher choices were re-audited on the training region only: D1 `ret1z` IC
-0.045 at stability 1.0, H4 `bollingerB` -0.035 at 0.8, H1 `ret20z` -0.024 at
0.8. All stable candidates are contrarian/mean-reverting around a ~20-day
horizon; M15's edge is smaller than the spread.

## Current results

### Expanded walk-forward: 5 folds, start 0.4, embargo, 3-seed ensemble

**Fixed teacher per profile** (D1 `ret1z`, H4 `bollingerB`):

| Dataset | Fold returns | Total | Positive folds | Fold max DD |
|---|---|---|---|---|
| D1 h20 | +3.37 / +1.00 / -1.44 / -2.02 / -1.15% | -0.34% | 2/5 | -0.82 / -1.02 / -2.11 / -2.60 / -1.51% |
| H4 h120 | -0.93 / -1.97 / -1.79 / -1.79 / -0.24% | **-6.55%** | **0/5** | -3.19 / -3.69 / -2.40 / -3.93 / -3.45% |

H1 was not rerun with the fixed teacher at 5 folds; its 3-fold result was +0.32%
(1/3). The earlier 3-fold H4 result (+5.62%, 3/3) did not survive more folds and
the embargo.

**Fold-by-fold teacher selection** (chosen from each fold's training data only):

| Dataset | Fold returns | Total | Positive folds | Teachers chosen | Fold max DD |
|---|---|---|---|---|---|
| H4 h120 (3 seeds) | +5.81 / -1.53 / -2.04 / +0.54 / +1.56% | **+4.22%** | 3/5 | ret60z, ret60z, rsi14, rsi14, rsi14 | -1.76 / -2.88 / -2.55 / -2.09 / -2.48% |
| D1 h20 (3 seeds) | +8.52 / +0.46 / -3.72 / -3.19 / -1.33% | +0.26% | 2/5 | bollingerB, bollingerB, ret60z(+), bollingerB, bollingerB | -1.59 / -3.13 / -3.76 / -3.89 / -2.98% |
| H1 h480 (2 seeds) | -8.51 / -3.22 / +2.92 / -0.53 / -2.56% | **-11.67%** | 1/5 | slope20, ret60z, ret60z, ret60z, ret60z | -10.39 / -4.89 / -1.72 / -3.84 / -5.88% |

### Multi-pair Yahoo D1 (5 folds, start 0.4, embargo, 3-seed ensemble)

| Symbol | No-gate total | Positive folds | Selective-gate total | Positive folds | Gate behavior |
|---|---|---|---|---|---|
| USDJPY | **+14.35%** | 4/5 | +14.35% | 4/5 | skipped (training baseline positive) |
| AUDUSD | +10.16% | 4/5 | **+11.07%** | **5/5** | gated folds 1-2, skipped 3-5 |
| EURUSD | -5.84% | 2/5 | -5.84% | 2/5 | gated fold 1 only, no effect |
| GBPUSD | **-15.98%** | 1/5 | -15.98% | 1/5 | skipped (training baseline positive) |

A forced (non-selective) gate told a different story: it rescued GBPUSD to
+1.05% (4/5 folds) but only 1-10 trades per fold, while breaking USDJPY
(+14.35% -> -5.44%) and cutting AUDUSD to +1.29%. So a universal gate is not the
fix. The selective gate helps AUDUSD slightly and leaves the good symbols alone,
but it skipped GBPUSD because its *training* baseline was positive, even though
its OOS folds failed.

### Training-only universe basket (16 pairs, 5 folds, 2 seeds, last 5,000 bars)

| Basket | Total | Positive folds |
|---|---|---|
| Eligible (selected from training only) | **+1.91%** | 3/5 |
| All symbols, equal weight | -2.50% | 2/5 |
| Test-fitted oracle (not achievable) | +4.26% | - |

Per-symbol OOS totals ranged from -17.46% (EURAUD) to +11.94% (USDJPY,
5/5 positive folds). 7 of 16 symbols were positive overall. Selection added
value over holding everything, but it is noisy:

- USDJPY was the strongest symbol but was eligible in only 2/5 folds, so the
  basket captured only part of its return.
- EURAUD (-17.46%) and USDCHF (-10.36%) were eligible in 3/5 folds, so the
  criterion also admitted losers.
- The basket's +1.91% is not a tradable result yet: two seeds, approximate fold
  alignment, spread-only costs, and no out-of-sample confirmation of the
  selection rule.

**Anchored DDPG fine-tune** (H1, 3-fold protocol, 10 epochs x 25 steps, 2 seeds):

| Dataset | Fold returns | Total | Positive folds | Fold max DD |
|---|---|---|---|---|
| H1 h480 | -2.02 / +3.93 / -1.06% | +0.75% | 1/3 | -2.76 / -0.93 / -11.39% |

**Single 80/20 split** (historical, optimistic):

| Dataset | BC seeds | Avg return | Avg PF | Seeds profitable | Folds+ |
|---|---|---|---|---|---|
| D1 | 1,2,3 | +0.79% | 1.117 | 2/3 | 1/3, 1/3, 2/3 |
| H4 | 1,2,3 | +1.63% | 1.145 | 3/3 | 2/3 each |
| H1 | 1,2,3 | +3.97% | 1.414 | 3/3 | 2/3 each |

What this means:

- More folds plus an embargo destroyed the fixed-teacher H4 result: 0/5 folds
  and -6.55%. The earlier +5.62% was concentrated in one favourable block.
- Fold-by-fold teacher selection recovers H4 to +4.22% with 3/5 positive folds,
  but fold 1 (+5.81%) still carries most of it, and the selected teacher changes
  across folds (ret60z -> rsi14). That is evidence of regime dependence, not a
  stable signal.
- D1 is essentially flat (+0.26%, 2/5) and its total is dominated by fold 1
  (+8.52%). H1 fails outright (-11.67%, 1/5).
- Across the Yahoo majors, the edge is **cross-sectional**: USDJPY and AUDUSD
  survive the 5-fold/embargo protocol (+14.35% and +10.16%), while EURUSD and
  GBPUSD fail (-5.84% and -15.98%). More data did not create a universal edge;
  it revealed which pairs carry it.
- Regime gating is not a reliable universal fix. Forced gating rescued GBPUSD
  but broke USDJPY; selective gating helps AUDUSD slightly but still fails to
  detect GBPUSD because its training baseline is optimistic.
- Training-only universe selection improves on holding everything (+1.91% vs
  -2.50% basket) but is noisy: it missed most of USDJPY's profitable folds and
  admitted losing symbols. It is a useful screen, not yet a portfolio rule.
- Anchored DDPG is not additive: +0.75% over 3 folds with 1/3 positive folds and
  a -11.4% drawdown in one fold.
- Unrestricted DDPG earlier collapsed to constant direction; BC plus per-fold
  selection is the best current policy, but it only works on H4 and not across
  all regimes.

Reproduce:

```bash
# EURUSD H4
bun run walkforward --tf H4 --folds 5 --start 0.4 --seeds 1,2,3 --epochs 0 --select-teacher

# Multi-pair Yahoo D1 (repeat per symbol)
bun run fetch-data
bun run walkforward --tf D1 --data data/Y_USDJPY_D1.json --label Y_USDJPY --folds 5 --start 0.4 --seeds 1,2,3 --epochs 0 --select-teacher
bun run walkforward --tf D1 --data data/Y_AUDUSD_D1.json --label Y_AUDUSD --folds 5 --start 0.4 --seeds 1,2,3 --epochs 0 --select-teacher --regime-gate
```

## Known limitations / next steps

1. **The next step is a pre-registered go/no-go, not more tuning**: the
   training-only universe basket is implemented and beats the all-symbol basket
   (+1.91% vs -2.50%) but is noisy. Freeze the rule now, add realistic
   commission/slippage, and validate on a period not used for any selection.
   Acceptance example: positive in >=4/5 OOS folds across >=3 seeds and a
   drawdown/turnover budget. If it fails, stop instead of adding filters.
2. **Regime gating is not solved**: forced gates can rescue a weak pair but break
   a working one; the selective rule overfits the training baseline (GBPUSD).
   Treat gating as research only until it passes the same held-out gate.
3. **Policy learning**: BC is the best policy; anchored DDPG is not reliably
   additive. Keep `bc_model` as the reference and only change the RL layer after
   pair selection is stable.
4. **Signal**: the audited edge is modest and contrarian around ~20 days. Keep
   the feature/teacher set pre-registered; adding indicators without an audit
   will overfit.
5. **Data**: 16 daily pairs are fetched and screened, but folds are only
   approximately aligned (all end on the same date, different calendars) and the
   universe run used 2 seeds. More history and exact date alignment are needed
   before treating the basket as a portfolio.
6. **Reward**: risk-normalized R is implemented, but portfolio drawdown,
   turnover, and slippage beyond the spread are not. Add them if the policy
   over-trades.
7. **Compute**: pure JS BPTT is correct but slow. Vectorized/typed-array math
   would enable larger walk-forward sweeps.
8. **Validation**: model selection still uses a slice of the training region.
   Rolling walk-forward validation and seed ensembles are the preferred
   protocol.
9. **Determinism**: `--seed` covers weight init, replay sampling, and
   exploration, but not OS threading/number formatting.

## Troubleshooting

| Error / symptom | Cause | Fix |
|---|---|---|
| `Checkpoint ... expects N features per bar` | checkpoint from a different feature set | retrain with the current `--tf`, or use a matching tag |
| `ENOENT ... data/EURUSD_D1.json` | data missing or wrong working directory | put the file in `data/` or pass `--data` |
| `Index ... out of bounds` | data too short for the 80/20 split and 60-bar state | use more bars or reduce `lookback` |
| NaN loss or weights | numerical blow-up | run `bun run gradcheck`, lower LRs, keep `clipNorm` |
| All windows flat / 0 trades | actor collapsed or noise too small | use BC, raise `--sigma`, lower `--min-sigma` |
| Very different results per run | no seed set | pass `--seed <n>`; compare seeds/walk-forward |
| Walk-forward fold fails | empty evaluation block | use `--start`/`--folds` that leave enough bars |
| `bun run backtest` uses old `final_model` | legacy checkpoint | train first, or pass e.g. `--tf H4 --model h4_seed1_bc_model` |

## File layout

```
main.js                 # environment, features, pretrain, BC, anchored DDPG
backtest.js             # sequential/fold backtest, ensembles, metrics, CSV/JSON
walkforward.js          # expanding-window retraining + per-fold ensembles/gates
universe.js             # training-only per-fold symbol selection + basket
sweep.js                # multi-seed train + backtest comparison
fetchYahoo.js           # downloads multi-pair daily OHLC from Yahoo Finance
signalAudit.js          # walk-forward feature IC/stability/hit-rate audit
gradcheck.js            # finite-difference gradient verification
utils/nn.js             # Dense, activations, Adam, soft updates
utils/features.js       # causal feature factory + rank/IC helpers
utils/dataProcessor.js  # normalization, features, windows, price ranges
utils/rng.js            # seeded RNG + Gaussian sampler
models/lstm.js          # recurrent encoder, truncated BPTT
models/ddpg.js          # actor (direction/size/SL/TP head) + critic
saved_models/           # checkpoints (legacy files are incompatible)
data/                   # input data (gitignored)
backtest_results/       # generated backtest/walk-forward results (gitignored)
signal_audit/           # generated feature-audit results (gitignored)
```

## License

No license file is included. Add one before distributing or using commercially.
