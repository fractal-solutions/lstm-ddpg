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

# 3. Train a reproducible model (writes saved_models/first_*)
bun run train --seed 1 --prefix first

# 4. Evaluate the held-out tail with 3 contiguous walk-forward folds
bun run backtest --model first_best_model --folds 3

# 5. Optional: compare seeds under the same protocol
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
| Policy is always one direction | fix features/labels (signal), run more seeds; direction bias is not an RL-only problem |
| Stops never hit / all `close` exits | reduce `--range-window`, cap SL/TP multiples |
| Validation improves then collapses | lower actor LR, increase `--patience`, decay noise faster |
| Over-trades after gains | add/raise spread cost, add turnover/drawdown penalties |
| Results swing wildly by seed | more seeds + ensemble, more data, stronger regularization |
| No result survives the test tail | overfitting or no signal: rework features/horizon, use walk-forward retraining |

## Pipeline

```
OHLCV bars -> log return + relative range -> standardized features
          -> 60-step LSTM (BPTT) -> 32-d hidden state (x fixed state gain)
          -> actor  -> [direction, size, SL fraction, TP fraction]
          -> critic <- (state, action)

SL/TP distances = recent 10-bar price range * (0.5 + k * fraction)
Trade PnL        = simulated over the next 5 bars (SL checked before TP)
Reward           = tanh((PnL_R - cost_R) / 2) * size   (risk-normalized)
```

- The LSTM is genuinely recurrent: 60 bars, one feature vector per step,
  hidden/cell state carried, truncated BPTT from the final hidden state.
- Gradients are exact (checked numerically with `bun run gradcheck`).
- The DDPG actor is trained with the deterministic policy gradient through the
  critic (`-dQ/da * da/dtheta`), and the shared encoder also receives
  `-dQ/ds`. Targets are soft-updated and use TD3-style target smoothing.
- Costs: spread from the data file (`spread * point`, ~1 pip for the bundled
  EURUSD file) is charged on every simulated trade and is also present in the
  training reward.
- Train/test split: first 80% of the normalized window (last 4000 bars) is
  training; the final 20% (~800 D1 bars) is held out for backtesting.
  Feature standardization uses training-region statistics only.

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
`fractal-solutions/paed-framework` (`data/`). Place the desired timeframe at the
path above, or pass `--data <path>`.

## Install and verify

```bash
bun install
bun run gradcheck     # finite-difference check for LSTM BPTT, Dense, critic, actor
```

## Training

```bash
bun run train                                   # defaults: 30 epochs x 50 steps
bun run train --seed 1 --prefix run1            # reproducible, tagged checkpoints
bun run train --no-pretrain --prefix ddpgenly   # skip supervised encoder pretrain
bun run train --epochs 5 --steps 30 --batch 8 --pretrain-epochs 2 --pretrain-steps 20
```

| Flag | Meaning |
|---|---|
| `--epochs`, `--steps` | Number of DDPG epochs and gradient steps per epoch |
| `--batch` | Replay minibatch size |
| `--hidden` | LSTM / critic hidden size (default 32) |
| `--pretrain-epochs`, `--pretrain-steps` | Supervised encoder pretraining (0 = skip) |
| `--no-pretrain` | Disable pretraining |
| `--actor-lr`, `--critic-lr`, `--encoder-lr`, `--pretrain-lr` | Learning rates |
| `--sigma`, `--min-sigma`, `--noise-decay` | OU exploration schedule |
| `--gamma` | Discount factor |
| `--range-window` | Bars used for the volatility-based SL/TP distance |
| `--seed` | Seed `Math.random` for reproducible run |
| `--prefix` | Prefix for checkpoint names (e.g. `seed1_best_model.json`) |
| `--no-save` | Do not write checkpoints |
| `--data` | Alternate data file path |

Pretraining learns to predict the sign of the next 5-bar return (binary
cross-entropy). It is intentionally cheap; the encoder is then fine-tuned with
a much smaller encoder learning rate.

Checkpoints (weights + metadata: lookback, feature size, hidden size,
barsPredicted, state gain, seed, spread) are written to `saved_models/`.
Checkpoints from older versions of this repo are not compatible and are
rejected with an explicit error.

## Backtesting

```bash
bun run backtest --model seed1_best_model
bun run backtest --model seed2_best_model --folds 3
bun run backtest --random --trades 20            # pipeline smoke test
```

| Flag | Meaning |
|---|---|
| `--model <name>` | Checkpoint name without `.json` (default `final_model`) |
| `--folds <n>` | Split the held-out period into `n` contiguous walk-forward folds |
| `--trades <n>` | Cap the number of trades evaluated |
| `--random` | Use randomly initialized networks (no checkpoint) |
| `--seed` | Seed the random init for `--random` |
| `--no-csv` | Skip writing CSV/JSON results |
| `--data` | Alternate data file path |

Results are printed to the terminal and written to `backtest_results/`
(gitignored) as `<model>_trades.csv` and `<model>_summary.json`.

### Metric definitions

- **Trade**: entry at the close of bar `i-1`, state from bars `[i-60, i)`,
  simulated over bars `[i, i+4]`; non-overlapping windows.
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

Each seed trains with its own `seed<SEED>_*` checkpoints and runs the backtest
with folds, then prints a per-seed table and an aggregate line. Use the budget
flags to keep sweeps short; the default 30x50 run takes roughly 90 s per seed on
a typical desktop.

## Current results

Held-out window: 797 D1 bars (~3.16 years), buy & hold +0.04%, 10-point spread.
Default training budget (30 epochs x 50 steps, batch 16, seeded seeds 1 and 2):

| Seed | Trades | Long/Short | Win rate | PF | Expectancy | Return | Max DD | Positive folds |
|---|---|---|---|---|---|---|---|---|
| 1 | 151 | 1 / 150 | 50.99% | 0.827 | -0.0062 R | -0.99% | -1.61% | 1/3 |
| 2 | 133 | 16 / 117 | 48.87% | 0.893 | -0.0275 R | -0.57% | -1.15% | 2/3 |

Both seeds are net negative and the policy is still directionally lopsided
(mostly short, small sizes, stops wide enough that most trades exit at the
5-bar close). This is a real result, not a harness failure:

- The train period is broadly EURUSD down-trending, so a short bias is learned.
  The held-out period is flat/slightly up, so the bias does not generalize.
- 5-bar direction is close to unpredictable at this feature/target choice
  (pretraining accuracy hovers near 50%).
- The DDPG actor collapses toward a low-exposure policy because the flat
  penalty is small and few positions survive risk/cost normalization.

The evaluation pipeline, gradients, costs, and reproducibility are in place;
what is missing is signal, not plumbing.

## Known limitations / next steps

1. **Signal**: absolute-price features were replaced with log returns and range,
   but the model still needs a predictive edge. Try longer horizons, multi-scale
   features (returns over several lookbacks, volatility regime, session effects),
   or a different pretraining target (volatility, direction confidence).
2. **Policy class**: the actor output layer is linear + explicit activations.
   Consider a dedicated "no-trade" gate with its own cost, and a size/target
   parameterization that cannot collapse to zero by default.
3. **Reward**: risk-normalized R is implemented, but portfolio-level drawdown,
   turnover, and slippage beyond the spread are not. Add them if the policy
   learns to over-trade.
4. **Regime robustness**: run sweeps across timeframes (H1/H4) and walk-forward
   retraining, not only a single holdout.
5. **Compute**: pure JS BPTT is correct but slow. Vectorizing the LSTM inner
   loops (or moving to a typed-array/WebGPU path) would enable larger sweeps.
6. **Validation**: model selection currently uses a fixed slice of the training
   region. Prefer rolling walk-forward validation across the training period.
7. **Determinism**: `--seed` covers weight init, replay sampling, and
   exploration, but not OS threading/number formatting edge cases.

## Troubleshooting

| Error / symptom | Cause | Fix |
|---|---|---|
| `Checkpoint ... expects N features per bar` | old checkpoint from the previous architecture | retrain; use `seed1_*`/`seed2_*` or a new `--prefix` |
| `ENOENT ... data/EURUSD_D1.json` | data file missing or wrong working directory | place the file at `data/EURUSD_D1.json` or pass `--data` |
| `Index ... out of bounds` | data too short for the 80/20 split and 60-bar state | use more bars or reduce `lookback` in `DEFAULT_CONFIG` |
| NaN loss or weights | numerical blow-up | run `bun run gradcheck`, lower LRs, keep `clipNorm` |
| All windows flat / 0 trades | exploration noise too small or actor collapsed | raise `--sigma`, lower `--min-sigma`, check `flatPenalty` |
| Very different results per run | no seed set | pass `--seed <n>`; compare means over seeds, not single runs |
| `bun run backtest` uses old `final_model` | legacy checkpoint from git history | train first or pass `--model seed1_best_model` |

## File layout

```
main.js                 # environment, features, pretrain, DDPG loop, checkpoints
backtest.js             # sequential held-out backtest, metrics, folds, CSV/JSON
sweep.js                # multi-seed train + backtest comparison
gradcheck.js            # finite-difference gradient verification
utils/nn.js             # Dense, activations, Adam, soft updates
utils/dataProcessor.js  # normalization, features, windows, price ranges
utils/rng.js            # seeded RNG + Gaussian sampler
models/lstm.js          # recurrent encoder, truncated BPTT
models/ddpg.js          # actor (direction/size/SL/TP head) + critic
saved_models/           # checkpoints (seed1_*, seed2_*, legacy files)
data/                   # input data (gitignored)
backtest_results/       # generated results (gitignored)
```

## License

No license file is included in this repository. Add one before distributing or
using commercially.
