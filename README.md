# LSTM-DDPG trading research

A dependency-free (plain JS/Bun) implementation of a recurrent encoder + DDPG
actor-critic for EURUSD daily-bar trading, with manual backpropagation,
truncated BPTT, risk-normalized rewards, walk-forward backtests, and seeded
reproducible runs.

> **Status: research prototype, not production trading software.** The current
> baseline has no demonstrated edge (see [Current results](#current-results)).
> The repository is structured so the learning math, the environment, and the
> evaluation can be iterated independently.

## Quick start

```bash
# 1. Put the data where the trainer expects it:
#      data/EURUSD_D1.json   (schema below)
# 2. Install and verify the manual gradients
bun install
bun run gradcheck

# 3. Audit the features before training (walk-forward IC + stability)
bun run audit

# 4. Train a reproducible model (writes saved_models/first_*)
bun run train --seed 1 --prefix first

# 5. Evaluate the cloned policy (use first_best_model for the DDPG version)
bun run backtest --model first_bc_model --folds 3

# 6. Optional: compare seeds under the same protocol
bun run sweep --seeds 1,2,3 --folds 3
```

The default run is 30 epochs x 50 steps, batch 16, and takes roughly 90 s per
seed on a desktop. A backtest prints its metrics, folds, and writes
`backtest_results/<model>_trades.csv` and `_summary.json`.

You get three checkpoints per run: `first_pretrain_model` (supervised encoder
before DDPG), `first_best_model` (best validation score inside the training
region), and `first_final_model` (last epoch). Always evaluate `best` (or a seed
ensemble), never the checkpoint you trained on the test tail with.

## Ways to use this repo

| Goal | Command | What to look at |
|---|---|---|
| Sanity-check the math | `bun run gradcheck` | `GRADIENT CHECK PASSED` |
| Smoke-test the pipeline | `bun run train --epochs 1 --steps 10 --pretrain-steps 5 --no-save` | runs without NaN/Inf |
| Baseline a model | `bun run train --seed 1 --prefix first` then `bun run backtest --model first_best_model --folds 3` | PF, expectancy, folds |
| Check robustness | `bun run sweep --seeds 1,2,3 --folds 3` | profitable seeds, positive folds |
| Test an idea quickly | add `--epochs 10 --steps 30 --pretrain-epochs 2 --pretrain-steps 20` | 20-30 s per seed |
| Evaluate random policy | `bun run backtest --random --trades 20` | reference floor |
| Skip supervised pretraining | `bun run train --no-pretrain --prefix ddpgenly` | compare with pretrain |
| Audit features/timeframes | `bun run audit` or `bun run audit --data data/EURUSD_H4.json --horizons 30,60,120` | IC, stability, candidates |
| Evaluate the supervised policy | `bun run train --tf H1 --epochs 0 --bc-epochs 5 --seed 1 --prefix bc_h1` then `bun run backtest --tf H1 --model bc_h1_bc_model --folds 3` | BC policy without DDPG |

### Reading a backtest

Start with **Expectancy (R)**, **Profit factor**, and **Max drawdown**, then check
whether the result holds across **folds** and **seeds**. A single positive fold,
a PF above ~1.1 on <100 trades, or one profitable seed is not evidence. A usable
signal should show positive expectancy across most folds and multiple seeds with
consistent direction mix, not one lucky regime.

### Tuning cheatsheet

| Symptom | First knob to change |
|---|---|
| Policy is always flat | reduce `--min-sigma`, raise `flatPenalty`, lower the size dead zone |
| Policy is always one direction | it is the actor/RL layer, not the features: try supervised actor init/behavior cloning, more seeds, and an ensemble |
| Stops never hit / all `close` exits | reduce `--range-window`, cap SL/TP multiples |
| Validation improves then collapses | lower actor LR, increase `--patience`, decay noise faster |
| Over-trades after gains | add/raise spread cost, add turnover/drawdown penalties |
| Results swing wildly by seed | more seeds + ensemble, more data, stronger regularization |
| No result survives the test tail | overfitting or no signal: rework features/horizon, use walk-forward retraining |

## Pipeline

```
OHLCV bars -> 12 audited causal features -> standardized (train-region stats)
          -> 60-step LSTM (BPTT) -> 32-d hidden state (x fixed state gain)
          -> actor  -> [direction, size, SL fraction, TP fraction]
          -> critic <- (state, action)

Horizons: D1 20 bars, H4 120 bars, H1 480 bars (~20 trading days)
SL/TP distances = rangeWindow bars of price range * (0.5 + k * fraction)
Reward           = tanh((PnL_R - cost_R) / 2) * size   (risk-normalized)
```

- Features are the audited set: `ret1z/5z/20z/60z`, `atrRel`,
  `volRatio`, `dist20`, `dist50`, `slope20`, `donchian20`, `rsi14`,
  `bollingerB`. See [Signal audit](#signal-audit-feature-lab).
- The LSTM is genuinely recurrent: 60 bars, one 12-feature vector per step,
  hidden/cell state carried, truncated BPTT from the final hidden state.
- Training has three stages: (1) encoder pretraining on forward-return
  direction, (2) **behavior cloning** of the actor from the audited contrarian
  teacher (checkpoint `<prefix>_bc_model`), then (3) optional DDPG fine-tuning.
- Gradients are exact (checked numerically with `bun run gradcheck`).
- The DDPG actor is trained with the deterministic policy gradient through the
  critic (`-dQ/da * da/dtheta`), and the shared encoder also receives
  `-dQ/ds`. Targets are soft-updated and use TD3-style target smoothing.
- Costs: spread from the data file (`spread * point`, ~1 pip for the bundled
  EURUSD file) is charged on every simulated trade and is also present in the
  training reward.
- Train/test split: first 80% of the selected data file is training; the final
  20% is held out for backtesting. Feature standardization uses training-region
  statistics only.

## Requirements

- [Bun](https://bun.sh) >= 1.1
- Data file at `data/EURUSD_D1.json` (gitignored). Expected schema:

```json
{
  "spread": 10,
  "point": 0.00001,
  "time": [ ... ],
  "open": [ ... ],
  "high": [ ... ],
  "low": [ ... ],
  "close": [ ... ],
  "volume": [ ... ]
}
```

The EURUSD JSON files used here come from
`fractal-solutions/paed-framework` (`data/`). D1 (5,014 bars), H4 (25,848), H1
(100,000), and M15 (100,000) are supported; place them in `data/` or pass
`--data <path>`. `bun run audit` automatically picks up every `data/EURUSD_*.json`.

## Install and verify

```bash
bun install
bun run gradcheck     # finite-difference check for LSTM BPTT, Dense, critic, actor
```

## Training

```bash
bun run train --tf D1 --seed 1 --prefix d1_s1     # D1, 20-bar horizon (default)
bun run train --tf H4 --seed 1 --prefix h4_s1     # H4, 120-bar horizon
bun run train --tf H1 --seed 1 --prefix h1_s1     # H1, 480-bar horizon
bun run train --no-pretrain --prefix ddpgenly     # skip supervised pretraining
bun run train --epochs 10 --steps 30 --batch 8 --pretrain-epochs 2 --pretrain-steps 20
```

| Flag | Meaning |
|---|---|
| `--tf D1\|H4\|H1` | Timeframe profile: data file, horizon, and SL/TP range window |
| `--epochs`, `--steps` | Number of DDPG epochs and gradient steps per epoch |
| `--batch` | Replay minibatch size |
| `--hidden` | LSTM / critic hidden size (default 32) |
| `--bars` | Override `barsPredicted` (trade horizon in bars) |
| `--range-window` | Override bars used for the volatility-based SL/TP distance |
| `--features` | Comma-separated override of the feature list |
| `--data-points` | Limit how many trailing bars are used (default: full file) |
| `--pretrain-epochs`, `--pretrain-steps` | Supervised encoder pretraining (0 = skip) |
| `--no-pretrain` | Disable pretraining |
| `--bc-epochs`, `--bc-steps` | Actor behavior cloning (default 5 epochs x 50 steps) |
| `--no-bc` | Disable behavior cloning |
| `--teacher` | Teacher feature for cloning (default per profile) |
| `--teacher-sign` | `-1` contrarian (default), `+1` momentum |
| `--actor-lr`, `--critic-lr`, `--encoder-lr`, `--pretrain-lr` | Learning rates |
| `--sigma`, `--min-sigma`, `--noise-decay` | OU exploration schedule |
| `--gamma` | Discount factor |
| `--seed` | Seed `Math.random` for reproducible run |
| `--prefix` | Prefix for checkpoint names (e.g. `d1_s1_best_model.json`) |
| `--no-save` | Do not write checkpoints |
| `--data` | Alternate data file path |

| Profile | File | Bars | Horizon | Range window |
|---|---|---|---|---|
| `D1` | EURUSD_D1.json | 5,014 | 20 bars (~1 month) | 20 bars |
| `H4` | EURUSD_H4.json | 25,848 | 120 bars (~1 month) | 60 bars |
| `H1` | EURUSD_H1.json | 100,000 | 480 bars (~1 month) | 240 bars |

Pretraining learns to predict the sign of the forward return over the profile's
horizon (binary cross-entropy). It is intentionally cheap; the encoder is then
fine-tuned with a much smaller encoder learning rate.

Checkpoints store weights + metadata (timeframe, feature names, lookback,
hidden size, horizon, range window, state gain, seed, spread) in
`saved_models/`. Checkpoints are rejected if the feature set, horizon, or range
window does not match the current `--tf`/flags.

### Behavior cloning (supervised actor init)

The audit signals are contrarian, so the actor is initialized from a teacher
before RL:

| Profile | Teacher | Teacher sign |
|---|---|---|
| D1 | `ret1z` | contrarian |
| H4 | `bollingerB` | contrarian |
| H1 | `ret20z` | contrarian |

The teacher label is the sign of the last state bar's teacher feature. The
actor's direction output is trained with MSE to that label while the pretrained
encoder is held fixed; size/SL/TP keep their initialized values. The resulting
`<prefix>_bc_model` checkpoint is saved before DDPG starts, so it can be
backtested directly:

```bash
bun run train --tf H1 --epochs 0 --bc-epochs 5 --seed 1 --prefix bc_h1
bun run backtest --tf H1 --model bc_h1_bc_model --folds 3
```

Set `--epochs 0` to skip DDPG and evaluate the cloned policy alone. Training
the encoder on the teacher is intentionally avoided: a fixed, pretrained
encoder keeps the cloning signal clean.

## Backtesting

```bash
bun run backtest --tf D1 --model d1_s2b_best_model --folds 3
bun run backtest --tf H4 --model h4_s2_best_model --folds 3
bun run backtest --random --trades 20            # pipeline smoke test
```

The `--tf` (and matching `--bars`/`--range-window`/`--features`) must match the
checkpoint; the loader rejects mismatches instead of silently simulating a
different horizon.

| Flag | Meaning |
|---|---|
| `--tf D1\|H4\|H1` | Timeframe profile; must match the checkpoint |
| `--model <name>` | Checkpoint name without `.json` (default `final_model`) |
| `--folds <n>` | Split the held-out period into `n` contiguous walk-forward folds |
| `--trades <n>` | Cap the number of trades evaluated |
| `--random` | Use randomly initialized networks (no checkpoint) |
| `--seed` | Seed the random init for `--random` |
| `--no-csv` | Skip writing CSV/JSON results |
| `--data` | Alternate data file path |
| `--bars`, `--range-window`, `--features` | Override profile values (must match checkpoint) |

Results are printed to the terminal and written to `backtest_results/`
(gitignored) as `<model>_trades.csv` and `<model>_summary.json`.

### Metric definitions

- **Trade**: entry at the close of bar `i-1`, state from bars `[i-60, i)`,
  simulated over the profile horizon (`barsPredicted` = 20 D1 / 120 H4 / 480
  H1); non-overlapping windows.
- **R**: price move divided by the initial stop distance (risk unit).
  `netR = pnlR - costR`. Risk-normalization makes different stop sizes comparable.
- **Expectancy**: mean `netR` per trade.
- **Profit factor**: gross winning PnL / gross losing PnL (fractional equity PnL).
- **Total return / CAGR**: compounded equity from position-sized trade PnL minus
  spread; CAGR assumes 252 trading days/year.
- **Max drawdown**: peak-to-trough on that compounded equity curve.
- **Sharpe (per trade)**: mean/std of trade returns, scaled by `sqrt(trades)`
  (not annualized).
- **Walk-forward folds**: the held-out period is split into contiguous blocks;
  each fold's metrics are computed independently so a single lucky regime does
  not dominate the summary.

## Multi-seed sweep

```bash
bun run sweep --seeds 1,2,3 --folds 3
bun run sweep --seeds 1,2 --epochs 10 --steps 30 --pretrain-epochs 2 --pretrain-steps 20
```

Each seed trains with its own `<tf>_seed<SEED>_*` checkpoints and runs the backtest
with folds, then prints a per-seed table and an aggregate line. Use the budget
flags to keep sweeps short; the default 30x50 run takes roughly 90 s per seed on
a typical desktop.

## Signal audit (feature lab)

Train on information that is actually predictive, not on indicator soup. Before
touching DDPG, audit every candidate feature with causal, purged walk-forward
folds:

```bash
bun run audit                                              # all EURUSD_*.json in data/
bun run audit --data data/EURUSD_D1.json --horizons 5,10,20
bun run audit --data data/EURUSD_H4.json --horizons 30,60,120
bun run audit --horizons 5,10,20 --folds 5 --min-ic 0.02 --min-stability 0.7
```

For each feature and forward horizon it prints walk-forward Spearman IC,
fold stability, and directional hit rate, then lists candidates. Per-timeframe
CSV/JSON is written to `signal_audit/` (gitignored). Features are causal and
never normalized with future data:

```
ret1z ret5z ret20z ret60z            vol-normalized momentum
atrRel volRatio rangeRatio shock      volatility / regime
dist20 dist50 slope20                 trend
donchian20 rsi14 bollingerB           range / oscillator
gap volumeZ                           structure / volume
```

Reading it: `|IC| ≈ 0.02–0.03` with stability `>= 0.7` across folds is worth
testing; one high-IC fold is not. A negative IC with hit rate below 50% means
the feature is **contrarian**. Always compare the edge with round-trip cost.

Initial run (5 folds, ~0.9 bps spread, univariate event study; `net edge` is a
rough `(2p−1)·mean|move| − cost` estimate, not a strategy backtest):

| Data | Bars | Best signal at ~20-day horizon | Stability | Hit (contrarian) | Net edge* |
|---|---|---|---|---|---|
| D1 | 5,014 | `ret1z` | 0.80 | 52.2% | **~6.3 bps/trade** |
| H4 | 25,848 | `bollingerB` / `rsi14` | 0.80 | 50.8% / 50.6% | ~2.1 / 1.3 bps |
| H1 | 100,000 | `ret20z` / `bollingerB` | 0.80 | 50.7% / 50.6% | ~1.8 / 1.3 bps |
| M15 | 100,000 | stable mean reversion, but edge < spread | 1.00 | ~48.6% raw | < 1 bp, net negative |

All stable candidates are contrarian/mean-reverting around a ~20-day horizon.
M15 has the most stable short-horizon ICs but its ~0.2 bps edge is smaller than
the 0.9 bps spread, so more bars did not automatically mean a tradable edge.
`volRatio` on D1 has the largest raw |IC| (0.078 at h=20) but is unsigned; use it
for regime/sizing, not direction.

Conclusion: the first credible inputs for the model are **D1 (strongest per
trade) or H4 (5x the samples, ~2 bps edge)**, with horizons near 20 days. H1 is
a useful robustness check; M15 is not a primary training set after costs.

## Current results

First runs with the audited 12-feature set, horizon matching, and behavior
cloning. The held-out tail is the last 20% of each file; buy & hold over those
windows was -8.87% (D1), -8.75% (H4), -8.85% (H1).

**Behavior-cloned policy (5 epochs x 50 steps, DDPG skipped, 3 seeds):**

| Dataset | Seed returns | Avg return | Avg PF | Seeds profitable | Folds+ | Avg Max DD |
|---|---|---|---|---|---|---|
| D1 h20 | +1.01 / -0.32 / +1.67% | +0.79% | 1.117 | 2/3 | 1/3, 1/3, 2/3 | -2.33% |
| H4 h120 | +3.07 / +0.12 / +1.70% | +1.63% | 1.145 | **3/3** | 2/3 each | -4.00% |
| H1 h480 | +4.51 / +2.14 / +5.27% | **+3.97%** | **1.414** | **3/3** | 2/3 each | -4.73% |

Representative detail (H1 seed 1): 39 trades, 23L/16S, 56.4% win rate, PF 1.518,
expectancy +0.121 R, return +4.51%, max DD -4.70%, folds +4.92 / -3.02 / +2.71%.

The cloned policy is the first result that looks like a working baseline rather
than a degenerate one: mixed long/short, modest drawdown, profitable across all
three seeds on H4 and H1 (2/3 seeds on D1), and 2/3 positive folds on H4/H1.
It is still modest (CAGR roughly 0.1–1.4%), D1 is weak, and every run has one
losing fold. Treat it as a promising baseline, not a finished strategy.

**DDPG fine-tuning after BC** is not reliably additive: D1 seed 2 improved to
+2.60% (but all-short), H4 +0.78% on only 13 trades, while D1 seed 1 (-0.25%) and
H1 (-0.97%) degraded and often collapsed to a constant direction. The BC
checkpoint, not the DDPG checkpoint, is currently the better model.

The teacher choice was re-audited using the training region only (no test tail):
D1 `ret1z` IC -0.045 at stability 1.0, H4 `bollingerB` -0.035 at 0.8, H1
`ret20z` -0.024 at 0.8 — so the result is not purely test-tail selection. It is
still three seeds and one split per timeframe; walk-forward retraining and
out-of-sample re-audits are the next validation step.

Reproduce the current best BC result:

```bash
bun run sweep --tf H1 --epochs 0 --seeds 1,2,3 --folds 3
```

## Known limitations / next steps

1. **Policy learning**: behavior cloning is now the best policy and is
   profitable across 3 seeds on H4/H1, but DDPG fine-tuning is unstable and often
   destroys it. Next: anchor/freeze the actor (trust-region or small actor LR),
   optionally freeze the encoder, ensemble seeds, and prefer the `bc_model`
   checkpoint until fine-tuning is proven.
2. **Signal**: the audit candidates are modest and contrarian around a ~20-day
   horizon. The chosen teachers hold on a training-only re-audit, but keep the
   feature/teacher set pre-registered and re-audit per fold before trusting it.
3. **Policy class**: the actor output layer is linear + explicit activations.
   Consider a dedicated "no-trade" gate with its own cost, and a size/target
   parameterization that cannot collapse to zero by default.
4. **Reward**: risk-normalized R is implemented, but portfolio-level drawdown,
   turnover, and slippage beyond the spread are not. Add them if the policy
   learns to over-trade.
5. **Regime robustness**: run sweeps across timeframes (H1/H4) and walk-forward
   retraining, not only a single holdout.
6. **Compute**: pure JS BPTT is correct but slow. Vectorizing the LSTM inner
   loops (or moving to a typed-array/WebGPU path) would enable larger sweeps.
7. **Validation**: model selection currently uses a fixed slice of the training
   region. Prefer rolling walk-forward validation across the training period.
8. **Determinism**: `--seed` covers weight init, replay sampling, and
   exploration, but not OS threading/number formatting edge cases.

## Troubleshooting

| Error / symptom | Cause | Fix |
|---|---|---|
| `Checkpoint ... expects N features per bar` | old checkpoint from a different feature set | retrain with the current `--tf`, or use the matching `d1_*`/`h4_*`/`h1_*` checkpoint |
| `ENOENT ... data/EURUSD_D1.json` | data file missing or wrong working directory | place the file at `data/EURUSD_D1.json` or pass `--data` |
| `Index ... out of bounds` | data too short for the 80/20 split and 60-bar state | use more bars or reduce `lookback` in `DEFAULT_CONFIG` |
| NaN loss or weights | numerical blow-up | run `bun run gradcheck`, lower LRs, keep `clipNorm` |
| All windows flat / 0 trades | exploration noise too small or actor collapsed | raise `--sigma`, lower `--min-sigma`, check `flatPenalty` |
| Very different results per run | no seed set | pass `--seed <n>`; compare means over seeds, not single runs |
| `bun run backtest` uses old `final_model` | legacy checkpoint from git history | train first or pass a compatible tag, e.g. `--tf H1 --model h1_seed1_bc_model` |

## File layout

```
main.js                 # environment, features, pretrain, DDPG loop, checkpoints
backtest.js             # sequential held-out backtest, metrics, folds, CSV/JSON
sweep.js                # multi-seed train + backtest comparison
signalAudit.js          # walk-forward feature IC/stability/hit-rate audit
gradcheck.js            # finite-difference gradient verification
utils/nn.js             # Dense, activations, Adam, soft updates
utils/features.js       # causal feature factory + rank/IC helpers
utils/dataProcessor.js  # normalization, features, windows, price ranges
utils/rng.js            # seeded RNG + Gaussian sampler
models/lstm.js          # recurrent encoder, truncated BPTT
models/ddpg.js          # actor (direction/size/SL/TP head) + critic
saved_models/           # checkpoints (seed1_*, seed2_*, legacy files)
data/                   # input data (gitignored)
backtest_results/       # generated backtest results (gitignored)
signal_audit/           # generated feature-audit results (gitignored)
```

## License

No license file is included in this repository. Add one before distributing or
using commercially.
