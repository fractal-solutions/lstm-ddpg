// Training-only universe selection across symbols.
//
// For every symbol and walk-forward fold we:
//   1. retrain/evaluate the BC policy (via walkforward.js) on that fold,
//   2. decide eligibility from the TRAINING window only (teacher IC stability
//      plus positive net return in at least 4 of 5 training sub-folds),
//   3. build an equal-weight basket of eligible symbols on the out-of-sample
//      block.
// Eligibility never looks at the test return.

import { mkdir, readdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { buildFeatureMatrix, rank, pearson, FEATURE_META } from './utils/features.js';
import { DEFAULT_CONFIG } from './main.js';
import { loadRates, carryAnnual, dateFromTimestamp, pairCurrencies } from './utils/macro.js';

const ROOT = import.meta.dir;
const RESULTS_DIR = join(ROOT, 'backtest_results');
const DEFAULT_SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF',
    'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY', 'EURCHF', 'EURAUD',
    'EURCAD', 'AUDJPY', 'CADJPY', 'CHFJPY'
];

function parseArgs(argv) {
    const options = {};
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (!arg.startsWith('--')) continue;
        const key = arg.slice(2);
        const value = argv[i + 1];
        if (value === undefined || value.startsWith('--')) {
            options[key] = true;
            continue;
        }
        options[key] = value;
        i++;
    }
    return options;
}

function trimData(data, points) {
    const n = data.close.length;
    if (!points || points <= 0 || n <= points) return data;
    const start = n - points;
    return {
        ...data,
        bars: points,
        time: data.time.slice(start),
        open: data.open.slice(start),
        high: data.high.slice(start),
        low: data.low.slice(start),
        close: data.close.slice(start),
        volume: data.volume.slice(start)
    };
}

function selectTeacher(featureData, horizon, trainEndFraction, allowed, folds = 5) {
    const { matrix, featureNames, warmup, n, close } = featureData;
    const trainEnd = Math.floor(n * trainEndFraction);
    const start = warmup;
    const end = trainEnd - horizon;
    if (end - start < folds * 100) return null;

    const size = Math.floor((end - start) / folds);
    const candidates = featureNames.filter(name => FEATURE_META[name].signed && allowed.has(name));
    const results = [];

    for (const name of candidates) {
        const index = featureNames.indexOf(name);
        const ics = [];
        for (let fold = 0; fold < folds; fold++) {
            const from = start + fold * size;
            const to = fold === folds - 1 ? end : from + size;
            const length = to - from;
            if (length < 50) continue;
            const x = new Float64Array(length);
            const y = new Float64Array(length);
            for (let t = 0; t < length; t++) {
                x[t] = matrix[from + t][index];
                y[t] = Math.log(close[from + t + horizon] / close[from + t]);
            }
            ics.push(pearson(rank(x), rank(y)));
        }
        if (ics.length === 0) continue;
        const mean = ics.reduce((a, b) => a + b, 0) / ics.length;
        const positive = ics.filter(value => value > 0).length;
        const stability = Math.max(positive, ics.length - positive) / ics.length;
        results.push({ feature: name, mean, stability, sign: mean > 0 ? 1 : -1 });
    }

    results.sort((a, b) => Math.abs(b.mean) - Math.abs(a.mean));
    const qualified = results.find(r => r.stability >= 0.6 && Math.abs(r.mean) >= 0.01);
    return qualified ?? results[0] ?? null;
}

// Eligibility uses training data only.
function trainingEligibility(featureData, times, horizon, trainEndFraction, spreadPrice, costs, folds = 5) {
    const teacher = selectTeacher(featureData, horizon, trainEndFraction, new Set(DEFAULT_CONFIG.featureNames), folds);
    if (!teacher) return { eligible: false, reason: 'no-teacher' };

    const { matrix, featureNames, warmup, n, close } = featureData;
    const trainEnd = Math.floor(n * trainEndFraction);
    const start = warmup;
    const end = trainEnd - horizon;
    const teacherIndex = featureNames.indexOf(teacher.feature);
    const length = end - start;
    const size = Math.floor(length / folds);

    const net = new Float64Array(length);
    for (let i = 0; i < length; i++) {
        const t = start + i;
        const direction = teacher.sign * Math.sign(matrix[t][teacherIndex]);
        const forward = Math.log(close[t + horizon] / close[t]);
        const extraCost = 2 * costs.slippagePct + 2 * costs.commissionPct;
        const annual = costs.useCarry
            ? (carryAnnual(costs.rates, costs.base, costs.quote, dateFromTimestamp(times[t])) ?? 0)
            : 0;
        const carryPct = direction * annual * horizon / 252;
        net[i] = direction * forward - spreadPrice / close[t] - extraCost + carryPct;
    }

    let total = 0;
    let positiveSubfolds = 0;
    for (let fold = 0; fold < folds; fold++) {
        const from = fold * size;
        const to = fold === folds - 1 ? length : (fold + 1) * size;
        let sum = 0;
        for (let i = from; i < to; i++) sum += net[i];
        const mean = (to - from) > 0 ? sum / (to - from) : 0;
        total += sum;
        if (mean > 0) positiveSubfolds++;
    }

    const mean = length > 0 ? total / length : 0;
    const eligible = mean > 0 && positiveSubfolds >= folds - 1 && length >= 200;
    return { eligible, teacher, mean, positiveSubfolds, observations: length };
}

const args = parseArgs(process.argv.slice(2));
const symbols = String(args.symbols ?? DEFAULT_SYMBOLS.join(','))
    .split(',')
    .map(value => value.trim().toUpperCase())
    .filter(Boolean);
const folds = Math.max(1, Number(args.folds ?? 5));
const startFraction = Number(args.start ?? 0.4);
const seeds = String(args.seeds ?? '1,2').split(',').map(Number).filter(Number.isFinite);
const dataPoints = Number(args['data-points'] ?? 5000);
const concurrency = Math.max(1, Number(args.concurrency ?? 4));
const tag = args.tag ? String(args.tag) : '';
const featureSet = String(args['feature-set'] ?? 'price');
const featureSetArgs = args['feature-set'] ? ['--feature-set', featureSet] : [];
const modelDir = join(process.env.TEMP ?? process.env.TMPDIR ?? '/tmp', 'lstm-ddpg-universe-models');

const step = (1 - startFraction) / folds;
const modelSuffix = 'bc_model';
const allowed = new Set(DEFAULT_CONFIG.featureNames);
const rates = await loadRates();
const costConfig = {
    slippagePct: DEFAULT_CONFIG.slippagePct,
    commissionPct: DEFAULT_CONFIG.commissionPct,
    useCarry: DEFAULT_CONFIG.useCarry,
    rates
};

await mkdir(RESULTS_DIR, { recursive: true });
for (const name of await readdir(RESULTS_DIR)) {
    if (/^walkforward_Y_/.test(name)) await rm(join(RESULTS_DIR, name), { force: true });
}
await rm(modelDir, { recursive: true, force: true });
await mkdir(modelDir, { recursive: true });

async function runSymbol(symbol) {
    const label = `Y_${symbol}`;
    const summaryPath = join(
        RESULTS_DIR,
        `walkforward_${label}_${modelSuffix}_sel_s${seeds.length}_f${folds}_st${startFraction}.json`
    );

    const command = [
        'bun', 'run', 'walkforward.js',
        '--tf', 'D1',
        '--data', join('data', `Y_${symbol}_D1.json`),
        '--label', label,
        '--folds', String(folds),
        '--start', String(startFraction),
        '--seeds', seeds.join(','),
        '--epochs', '0',
        '--select-teacher',
        '--data-points', String(dataPoints),
        ...featureSetArgs,
        '--model-dir', modelDir
    ];

    const proc = Bun.spawn(command, { cwd: ROOT, stdout: 'pipe', stderr: 'pipe' });
    const code = await proc.exited;
    const stderr = await new Response(proc.stderr).text();
    await new Response(proc.stdout).text();
    if (code !== 0) throw new Error(stderr || `exit ${code}`);

    const summary = await Bun.file(summaryPath).json();
    const data = trimData(await Bun.file(join(ROOT, 'data', `Y_${symbol}_D1.json`)).json(), dataPoints);
    const featureData = buildFeatureMatrix(data);

    const pair = pairCurrencies(symbol);
    const costs = {
        ...costConfig,
        base: pair?.base ?? null,
        quote: pair?.quote ?? null
    };

    const eligibility = [];
    for (let fold = 0; fold < folds; fold++) {
        const trainFraction = startFraction + fold * step;
        eligibility.push(
            trainingEligibility(
                featureData,
                data.time,
                20,
                trainFraction,
                (data.spread ?? 0) * (data.point ?? 0.00001),
                costs,
                5
            )
        );
    }

    console.log(`  ${symbol.padEnd(7)} OOS ${summary.totalReturn.toFixed(2).padStart(7)}% | eligible folds ${eligibility.filter(e => e.eligible).length}/${folds}`);
    return { symbol, summary, eligibility };
}

const results = [];
const queue = [...symbols];
let cursor = 0;

async function worker() {
    while (cursor < queue.length) {
        const symbol = queue[cursor++];
        try {
            results.push(await runSymbol(symbol));
        } catch (error) {
            console.error(`  ${symbol.padEnd(7)} FAILED: ${error.message}`);
        }
    }
}

console.log(`Universe: ${symbols.length} symbols, ${folds} folds, seeds ${seeds.join(',')}, features ${featureSet}, concurrency ${concurrency}\n`);
await Promise.all(Array.from({ length: Math.min(concurrency, symbols.length) }, worker));

results.sort((a, b) => a.symbol.localeCompare(b.symbol));

console.log('\nPer symbol:');
console.log('symbol   oosRet%  posFolds  eligible  foldReturns%');
for (const result of results) {
    const foldsData = result.summary.foldResults;
    const positive = foldsData.filter(fold => fold.ensemble.returnPct > 0).length;
    const eligibleFolds = result.eligibility.filter(e => e.eligible).length;
    const returns = foldsData.map(fold => fold.ensemble.returnPct.toFixed(2)).join(' / ');
    console.log(
        `${result.symbol.padEnd(8)} ${result.summary.totalReturn.toFixed(2).padStart(7)}  ` +
        `${String(positive).padStart(7)}/${folds}  ${String(eligibleFolds).padStart(6)}/${folds}  ${returns}`
    );
}

console.log('\nBasket by fold (equal weight, eligibility from training only):');
console.log('fold  selected  members                          basketRet%  allRet%');

let basketEquity = 1;
let basketPositive = 0;
let allEquity = 1;
const basketReturns = [];
const allReturns = [];
const selectedCounts = [];

for (let fold = 0; fold < folds; fold++) {
    const selected = results.filter(result => result.eligibility[fold]?.eligible);
    const selectedReturns = selected.map(result => result.summary.foldResults[fold].ensemble.returnPct);
    const all = results.map(result => result.summary.foldResults[fold].ensemble.returnPct);
    const basketReturn = selectedReturns.length
        ? selectedReturns.reduce((sum, value) => sum + value, 0) / selectedReturns.length
        : 0;
    const allReturn = all.length ? all.reduce((sum, value) => sum + value, 0) / all.length : 0;

    basketEquity *= 1 + basketReturn / 100;
    allEquity *= 1 + allReturn / 100;
    if (basketReturn > 0) basketPositive++;
    basketReturns.push(basketReturn);
    allReturns.push(allReturn);
    selectedCounts.push(selected.length);

    console.log(
        `${String(fold + 1).padEnd(5)} ${String(selected.length).padStart(8)}  ` +
        `${(selected.map(result => result.symbol).join(',') || '-').padEnd(32)} ` +
        `${basketReturn.toFixed(2).padStart(10)}  ${allReturn.toFixed(2).padStart(7)}`
    );
}

console.log(
    `\nEligible basket: total ${(basketEquity - 1) * 100 > 0 ? '+' : ''}${((basketEquity - 1) * 100).toFixed(2)}%, ` +
    `${basketPositive}/${folds} positive folds`
);
console.log(
    `All-symbol basket: total ${(allEquity - 1) * 100 > 0 ? '+' : ''}${((allEquity - 1) * 100).toFixed(2)}%, ` +
    `${allReturns.filter(value => value > 0).length}/${folds} positive folds`
);

const oracle = results.map(result => result.summary.totalReturn).filter(value => value > 0);
console.log(
    `Test-fitted oracle (not achievable): average of positive-symbol totals ` +
    `${(oracle.reduce((sum, value) => sum + value, 0) / Math.max(1, oracle.length)).toFixed(2)}%`
);

const basketTotal = (basketEquity - 1) * 100;
const lastFoldReturn = basketReturns[basketReturns.length - 1] ?? 0;
const averageSelected = selectedCounts.reduce((sum, value) => sum + value, 0) / Math.max(1, selectedCounts.length);
const acceptFolds = Number(args['accept-folds'] ?? 4);
const reasons = [];
if (basketTotal <= 0) reasons.push('basket total <= 0');
if (basketPositive < acceptFolds) reasons.push(`positive folds ${basketPositive}/${folds} < ${acceptFolds}`);
if (lastFoldReturn < 0) reasons.push(`last fold ${lastFoldReturn.toFixed(2)}% < 0`);
if (averageSelected < 2) reasons.push(`average selected symbols ${averageSelected.toFixed(1)} < 2`);

const verdict = reasons.length === 0 ? 'GO' : 'NO-GO';
console.log(`\nCost assumptions: spread from data, commission=${(DEFAULT_CONFIG.commissionPct * 1e4).toFixed(1)}bps/side, ` +
    `slippage=${(DEFAULT_CONFIG.slippagePct * 1e4).toFixed(1)}bps/side, carry=${DEFAULT_CONFIG.useCarry ? 'on' : 'off'}`);
console.log(`Pre-registered gate: total>0, >=${acceptFolds}/${folds} positive folds, last fold>=0, >=2 symbols selected.`);
console.log(`VERDICT: ${verdict}${reasons.length ? ` (${reasons.join('; ')})` : ''}`);
console.log('Caveat: the final fold has been observed before; treat this as confirmatory, not a pristine holdout.');

const summaryPath = join(
    RESULTS_DIR,
    `universe_${featureSet.replace('+', 'plus')}${tag ? '_' + tag : ''}_s${seeds.length}_f${folds}_st${startFraction}_d${dataPoints}.json`
);
await Bun.write(summaryPath, JSON.stringify({
    symbols,
    folds,
    seeds,
    featureSet,
    tag,
    startFraction,
    dataPoints,
    results: results.map(result => ({
        symbol: result.symbol,
        totalReturn: result.summary.totalReturn,
        foldReturns: result.summary.foldResults.map(fold => fold.ensemble.returnPct),
        eligibility: result.eligibility.map(e => ({
            eligible: e.eligible,
            teacher: e.teacher,
            mean: e.mean,
            positiveSubfolds: e.positiveSubfolds,
            observations: e.observations
        }))
    })),
    basketReturns,
    basketTotal,
    allReturns,
    allTotal: (allEquity - 1) * 100,
    selectedCounts,
    averageSelected,
    lastFoldReturn,
    acceptFolds,
    verdict,
    reasons,
    costs: {
        commissionPct: DEFAULT_CONFIG.commissionPct,
        slippagePct: DEFAULT_CONFIG.slippagePct,
        useCarry: DEFAULT_CONFIG.useCarry
    }
}, null, 2));
console.log(`\nWrote ${summaryPath.replace(ROOT + '\\', '')}`);

await rm(modelDir, { recursive: true, force: true });
