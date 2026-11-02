// Walk-forward retraining with per-fold seed ensembles.
// For each fold: train `--seeds` models on an expanding window, evaluate each
// on the next contiguous block, then evaluate the seed ensemble on that block.
// No future data is used for training or feature standardization.

import { join } from 'node:path';
import { buildFeatureMatrix, rank, pearson, FEATURE_META } from './utils/features.js';
import { TIMEFRAMES } from './main.js';

const ROOT = import.meta.dir;

// Select the teacher using only the training window for this fold.
function selectTeacher(data, horizon, trainEndFraction, folds = 5) {
    const { matrix, featureNames, warmup, n, close } = buildFeatureMatrix(data);
    const trainEnd = Math.floor(n * trainEndFraction);
    const start = warmup;
    const end = trainEnd - horizon;
    if (end - start < folds * 100) return null;

    const size = Math.floor((end - start) / folds);
    const candidates = featureNames.filter(name => FEATURE_META[name].signed);
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

function parseArgs(argv) {
    const options = {};
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (!arg.startsWith('--')) continue;
        const key = arg.slice(2);
        if (key === 'no-pretrain' || key === 'no-bc' || key === 'no-freeze-encoder') {
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
const data = selectPerFold ? await Bun.file(profile.dataPath).json() : null;

const trainArgs = ['bun', 'run', 'main.js', '--tf', tf];
for (const key of ['epochs', 'steps', 'batch', 'hidden', 'pretrain-epochs', 'pretrain-steps', 'bc-epochs', 'bc-steps', 'bc-anchor']) {
    if (args[key] !== undefined) trainArgs.push(`--${key}`, String(args[key]));
}
if (args['no-pretrain']) trainArgs.push('--no-pretrain');
if (args['no-bc']) trainArgs.push('--no-bc');
if (args['no-freeze-encoder']) trainArgs.push('--no-freeze-encoder');
const embargoArgs = args.embargo !== undefined ? ['--embargo', String(args.embargo)] : [];

console.log(`Walk-forward ${tf}: ${folds} folds, ${seeds.length} seeds, start=${startFraction}, model=${modelSuffix}${args.embargo !== undefined ? `, embargo=${args.embargo}` : ''}\n`);
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
        teacherSelection = selectTeacher(data, profile.barsPredicted, trainFraction);
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

    for (const seed of seeds) {
        const prefix = `${tf.toLowerCase()}_wf${fold + 1}_s${seed}`;
        prefixes.push(prefix);

        run([
            ...trainArgs,
            ...teacherArgs,
            ...embargoArgs,
            '--train-fraction', String(trainFraction),
            '--test-end', String(testEnd),
            '--seed', String(seed),
            '--prefix', prefix
        ]);

        const individual = parseMetrics(run([
            'bun', 'run', 'backtest.js',
            '--tf', tf,
            '--model', `${prefix}_${modelSuffix}`,
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
        '--ensemble', prefixes.map(prefix => `${prefix}_${modelSuffix}`).join(','),
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

    foldResults.push({ fold: fold + 1, trainFraction, testEnd, teacher: teacherSelection, seedReturns, ensemble });
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
    `walkforward_${tf}_${modelSuffix}${selectPerFold ? '_sel' : ''}_s${seeds.length}_f${folds}_st${startFraction}.json`
);
await Bun.write(
    summaryPath,
    JSON.stringify({ tf, folds, seeds, startFraction, embargo: args.embargo ?? 'auto', modelSuffix, totalReturn, positiveFolds, foldResults }, null, 2)
);
console.log(`Wrote ${summaryPath.replace(ROOT + '\\', '')}`);
