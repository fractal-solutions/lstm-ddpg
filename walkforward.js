// Walk-forward retraining with per-fold seed ensembles.
// For each fold: train `--seeds` models on an expanding window, evaluate each
// on the next contiguous block, then evaluate the seed ensemble on that block.
// No future data is used for training or feature standardization.

import { join, basename, extname } from 'node:path';
import { buildFeatureMatrix, rank, pearson, FEATURE_META } from './utils/features.js';
import { TIMEFRAMES, DEFAULT_CONFIG } from './main.js';

const ROOT = import.meta.dir;

// Select the teacher using only the training window for this fold.
function selectTeacher(featureData, horizon, trainEndFraction, folds = 5, allowed = null) {
    const { matrix, featureNames, warmup, n, close } = featureData;
    const trainEnd = Math.floor(n * trainEndFraction);
    const start = warmup;
    const end = trainEnd - horizon;
    if (end - start < folds * 100) return null;

    const size = Math.floor((end - start) / folds);
    const candidates = featureNames.filter(name => FEATURE_META[name].signed && (!allowed || allowed.has(name)));
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

// Select a regime band using only the training window for this fold.
// Gate thresholds are in standardized feature units (same scale the model sees).
function selectRegimeGate(featureData, teacher, horizon, trainEndFraction, spreadPrice, folds = 5) {
    const { matrix, featureNames, warmup, n, close } = featureData;
    const trainEnd = Math.floor(n * trainEndFraction);
    const start = warmup;
    const end = trainEnd - horizon;
    if (end - start < 1000) return null;

    const teacherIndex = featureNames.indexOf(teacher.feature);
    if (teacherIndex === -1) return null;

    const length = end - start;
    const net = new Float64Array(length);
    for (let i = 0; i < length; i++) {
        const t = start + i;
        const direction = teacher.sign * Math.sign(matrix[t][teacherIndex]);
        const forward = Math.log(close[t + horizon] / close[t]);
        const cost = spreadPrice / close[t];
        net[i] = direction * forward - cost;
    }

    const meanOf = (values) => values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
    const stabilityOf = (indices) => {
        const blockSize = Math.max(1, Math.floor(length / folds));
        let positive = 0;
        let used = 0;
        for (let block = 0; block < folds; block++) {
            const from = block * blockSize;
            const to = block === folds - 1 ? length : (block + 1) * blockSize;
            const values = [];
            for (const i of indices) if (i >= from && i < to) values.push(net[i]);
            if (values.length < 5) continue;
            used++;
            if (meanOf(values) > 0) positive++;
        }
        return used ? positive / used : 0;
    };

    const baseline = meanOf([...net]);
    const candidates = ['atrRel', 'volRatio', 'rangeRatio', 'shock', 'ret60z', 'dist50', 'slope20']
        .filter(name => featureNames.includes(name));

    let best = null;
    for (const name of candidates) {
        const index = featureNames.indexOf(name);
        const trainValues = [];
        for (let t = warmup; t < trainEnd; t++) trainValues.push(matrix[t][index]);
        const mean = meanOf(trainValues);
        const variance = trainValues.reduce((sum, value) => sum + (value - mean) ** 2, 0) / trainValues.length;
        const std = Math.sqrt(variance) || 1e-8;

        const z = new Float64Array(length);
        for (let i = 0; i < length; i++) z[i] = (matrix[start + i][index] - mean) / std;

        const sorted = [...z].sort((a, b) => a - b);
        const edges = [0, 0.2, 0.4, 0.6, 0.8, 1].map(q => sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))]);

        for (let band = 0; band < 5; band++) {
            const min = edges[band];
            const max = edges[band + 1];
            const indices = [];
            for (let i = 0; i < length; i++) {
                if (z[i] >= min && z[i] <= max) indices.push(i);
            }
            if (indices.length < 200) continue;
            const meanNet = meanOf(indices.map(i => net[i]));
            const stability = stabilityOf(indices);
            if (stability < 0.6) continue;
            if (best === null || meanNet > best.meanNet) {
                best = { feature: name, min, max, meanNet, stability, count: indices.length };
            }
        }
    }

    // Selective gating: do not gate a signal that is already positive in training.
    if (baseline > 0) return { active: false, baseline, reason: 'baseline-positive' };
    if (!best || best.meanNet <= baseline) return { active: false, baseline, reason: 'no-improvement' };
    return { active: true, ...best, baseline };
}

function parseArgs(argv) {
    const options = {};
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (!arg.startsWith('--')) continue;
        const key = arg.slice(2);
        if (key === 'no-pretrain' || key === 'no-bc' || key === 'no-freeze-encoder' ||
            key === 'select-teacher' || key === 'regime-gate') {
            options[key] = true;
            continue;
        }
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

function run(command) {
    const result = Bun.spawnSync(command, { cwd: ROOT, stdout: 'pipe', stderr: 'pipe' });
    if (result.exitCode !== 0) {
        console.error(`Command failed: ${command.join(' ')}`);
        console.error(result.stderr.toString());
        process.exit(1);
    }
    return result.stdout.toString();
}

function extract(output, label) {
    for (const line of output.split('\n')) {
        if (line.trimStart().startsWith(label)) {
            return line.split(/\s{2,}/).pop().trim();
        }
    }
    return '';
}

function parseMetrics(output) {
    const pfRaw = extract(output, 'Profit factor');
    return {
        trades: parseInt(extract(output, 'Trades'), 10) || 0,
        returnPct: Number(extract(output, 'Total return').replace('%', '')) || 0,
        pf: pfRaw.includes('inf') ? Infinity : (Number(pfRaw) || 0),
        expectancy: Number(extract(output, 'Expectancy')) || 0,
        maxDrawdown: Number(extract(output, 'Max drawdown').replace('%', '')) || 0
    };
}

const args = parseArgs(process.argv.slice(2));
const tf = String(args.tf ?? 'D1').toUpperCase();
const seeds = String(args.seeds ?? '1,2,3').split(',').map(Number).filter(Number.isFinite);
const folds = Math.max(1, Number(args.folds ?? 3));
const startFraction = Number(args.start ?? 0.5);
const step = (1 - startFraction) / folds;
const epochs = Number(args.epochs ?? 0);
const modelSuffix = epochs === 0 ? 'bc_model' : 'best_model';
const profile = TIMEFRAMES[tf];
if (!profile) throw new Error(`Unknown timeframe "${tf}"`);
const selectPerFold = args['select-teacher'] === true;
const selectGate = args['regime-gate'] === true;
const dataPath = args.data ? String(args.data) : profile.dataPath;
const label = String(args.label ?? (args.data ? basename(dataPath, extname(dataPath)) : tf));
const needsSelection = selectPerFold || selectGate;
const dataPoints = args['data-points'] ? Number(args['data-points']) : null;
const data = needsSelection
    ? trimData(await Bun.file(dataPath).json(), dataPoints)
    : null;
const featureData = needsSelection ? buildFeatureMatrix(data) : null;
const spreadPrice = data ? (data.spread ?? 0) * (data.point ?? 0.00001) : 0;
const dataArgs = args.data ? ['--data', String(args.data)] : [];
const modelDirArgs = args['model-dir'] ? ['--model-dir', String(args['model-dir'])] : [];
const dataPointsArgs = args['data-points'] ? ['--data-points', String(args['data-points'])] : [];
const allowedTeachers = new Set(DEFAULT_CONFIG.featureNames);

const trainArgs = ['bun', 'run', 'main.js', '--tf', tf, ...dataArgs, ...modelDirArgs, ...dataPointsArgs];
for (const key of ['epochs', 'steps', 'batch', 'hidden', 'pretrain-epochs', 'pretrain-steps', 'bc-epochs', 'bc-steps', 'bc-anchor']) {
    if (args[key] !== undefined) trainArgs.push(`--${key}`, String(args[key]));
}
if (args['no-pretrain']) trainArgs.push('--no-pretrain');
if (args['no-bc']) trainArgs.push('--no-bc');
if (args['no-freeze-encoder']) trainArgs.push('--no-freeze-encoder');
const embargoArgs = args.embargo !== undefined ? ['--embargo', String(args.embargo)] : [];

console.log(
    `Walk-forward ${label}: ${folds} folds, ${seeds.length} seeds, start=${startFraction}, ` +
    `model=${modelSuffix}${args.embargo !== undefined ? `, embargo=${args.embargo}` : ''}` +
    `${selectPerFold ? ', select-teacher' : ''}${selectGate ? ', regime-gate' : ''}\n`
);
console.log('fold  train   test    seedRet%  ensRet%   ensPF    ensTrades  ensDD%');

const foldResults = [];
let compounded = 1;
let positiveFolds = 0;
const allSeedReturns = [];

for (let fold = 0; fold < folds; fold++) {
    const trainFraction = startFraction + fold * step;
    const testEnd = startFraction + (fold + 1) * step;
    const prefixes = [];
    const seedReturns = [];

    let teacherArgs = [];
    let teacherSelection = null;
    if (selectPerFold) {
        teacherSelection = selectTeacher(featureData, profile.barsPredicted, trainFraction, 5, allowedTeachers);
        if (teacherSelection) {
            teacherArgs = [
                '--teacher', teacherSelection.feature,
                '--teacher-sign', String(teacherSelection.sign)
            ];
            console.log(
                `  fold ${fold + 1} teacher: ${teacherSelection.feature} ` +
                `sign=${teacherSelection.sign} IC=${teacherSelection.mean.toFixed(4)} ` +
                `stability=${teacherSelection.stability.toFixed(2)}`
            );
        }
    }

    let gateArgs = [];
    let gateSelection = null;
    if (selectGate && featureData) {
        const teacher = teacherSelection ?? { feature: profile.teacherFeature, sign: -1 };
        gateSelection = selectRegimeGate(
            featureData,
            teacher,
            profile.barsPredicted,
            trainFraction,
            spreadPrice
        );
        if (gateSelection?.active) {
            gateArgs = [
                '--gate-feature', gateSelection.feature,
                '--gate-min', String(gateSelection.min),
                '--gate-max', String(gateSelection.max)
            ];
            console.log(
                `  fold ${fold + 1} gate: ${gateSelection.feature} ` +
                `[${gateSelection.min.toFixed(2)}, ${gateSelection.max.toFixed(2)}] ` +
                `trainRet=${(gateSelection.meanNet * 1e4).toFixed(1)}bps ` +
                `stability=${gateSelection.stability.toFixed(2)} n=${gateSelection.count}`
            );
        } else if (gateSelection) {
            console.log(
                `  fold ${fold + 1} gate skipped (${gateSelection.reason}), ` +
                `train baseline=${(gateSelection.baseline * 1e4).toFixed(1)}bps`
            );
        }
    }

    for (const seed of seeds) {
        const prefix = `${label.toLowerCase()}_wf${fold + 1}_s${seed}`;
        prefixes.push(prefix);

        run([
            ...trainArgs,
            ...teacherArgs,
            ...gateArgs,
            ...embargoArgs,
            '--train-fraction', String(trainFraction),
            '--test-end', String(testEnd),
            '--seed', String(seed),
            '--prefix', prefix
        ]);

        const individual = parseMetrics(run([
            'bun', 'run', 'backtest.js',
            '--tf', tf,
            ...dataArgs,
            '--model', `${prefix}_${modelSuffix}`,
            ...modelDirArgs,
            ...dataPointsArgs,
            ...gateArgs,
            ...embargoArgs,
            '--train-fraction', String(trainFraction),
            '--test-end', String(testEnd),
            '--folds', '1',
            '--no-csv'
        ]));
        seedReturns.push(individual.returnPct);
        allSeedReturns.push(individual.returnPct);
    }

    const ensemble = parseMetrics(run([
        'bun', 'run', 'backtest.js',
        '--tf', tf,
        ...dataArgs,
        '--ensemble', prefixes.map(prefix => `${prefix}_${modelSuffix}`).join(','),
        ...modelDirArgs,
        ...dataPointsArgs,
        ...gateArgs,
        ...embargoArgs,
        '--train-fraction', String(trainFraction),
        '--test-end', String(testEnd),
        '--folds', '1',
        '--no-csv'
    ]));

    compounded *= 1 + ensemble.returnPct / 100;
    if (ensemble.returnPct > 0) positiveFolds++;

    const averageSeedReturn = seedReturns.reduce((sum, value) => sum + value, 0) / seedReturns.length;
    console.log(
        `${String(fold + 1).padEnd(5)} ${trainFraction.toFixed(3)}  ${testEnd.toFixed(3)}  ` +
        `${averageSeedReturn.toFixed(2).padStart(8)}  ${ensemble.returnPct.toFixed(2).padStart(8)}  ` +
        `${(ensemble.pf === Infinity ? 'inf' : ensemble.pf.toFixed(2)).padStart(6)}  ` +
        `${String(ensemble.trades).padStart(9)}  ${ensemble.maxDrawdown.toFixed(2).padStart(6)}`
    );

    foldResults.push({
        fold: fold + 1,
        trainFraction,
        testEnd,
        teacher: teacherSelection,
        gate: gateSelection,
        seedReturns,
        ensemble
    });
}

const totalReturn = (compounded - 1) * 100;
const averageSeedReturn = allSeedReturns.reduce((sum, value) => sum + value, 0) / allSeedReturns.length;
console.log(
    `\nAggregate: ensemble total ${totalReturn.toFixed(2)}%, ` +
    `${positiveFolds}/${folds} positive folds, avg seed fold ${averageSeedReturn.toFixed(2)}%`
);

const { mkdir } = await import('node:fs/promises');
await mkdir(join(ROOT, 'backtest_results'), { recursive: true });
const summaryPath = join(
    ROOT,
    'backtest_results',
    `walkforward_${label}_${modelSuffix}${selectPerFold ? '_sel' : ''}${selectGate ? '_gate' : ''}_s${seeds.length}_f${folds}_st${startFraction}.json`
);
await Bun.write(
    summaryPath,
    JSON.stringify({ tf, folds, seeds, startFraction, embargo: args.embargo ?? 'auto', modelSuffix, totalReturn, positiveFolds, foldResults }, null, 2)
);
console.log(`Wrote ${summaryPath.replace(ROOT + '\\', '')}`);
