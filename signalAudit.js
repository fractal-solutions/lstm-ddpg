// Walk-forward univariate signal audit. For every causal feature and forward
// horizon it reports Spearman IC per fold, stability, and directional hit rate.
// Nothing is trained and no feature is normalized with future data.

import { mkdir, readdir } from 'node:fs/promises';
import { join, basename } from 'node:path';
import { buildFeatureMatrix, rank, pearson } from './utils/features.js';

const ROOT = import.meta.dir;
const DATA_DIR = join(ROOT, 'data');
const OUT_DIR = join(ROOT, 'signal_audit');

function parseArgs(argv) {
    const options = {};
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (!arg.startsWith('--')) continue;
        const key = arg.slice(2);
        if (key === 'no-csv') {
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

function numberList(value, fallback) {
    if (!value) return fallback;
    return String(value).split(',').map(Number).filter(Number.isFinite);
}

async function discoverFiles(explicit) {
    if (explicit) {
        return String(explicit).split(',').map(path => (path.includes('\\') || path.includes('/') ? path : join(DATA_DIR, path)));
    }
    const files = await readdir(DATA_DIR);
    return files
        .filter(name => /^EURUSD_.*\.json$/i.test(name))
        .map(name => join(DATA_DIR, name))
        .sort();
}

async function auditFile(path, options) {
    const data = await Bun.file(path).json();
    const timeframe = basename(path).replace(/^EURUSD_/i, '').replace(/\.json$/i, '');
    const { matrix, featureNames, meta, warmup, n, close } = buildFeatureMatrix(data);
    const featureIndex = new Map(featureNames.map((name, index) => [name, index]));

    const { horizons, folds, minSamples, minIc, minStability } = options;
    const maxHorizon = Math.max(...horizons);
    const start = warmup;
    const end = n - maxHorizon;
    const foldSize = Math.floor((end - start) / folds);

    const results = new Map();
    for (const name of featureNames) {
        results.set(name, { ics: new Map(), hits: new Map(), samples: new Map() });
        for (const horizon of horizons) {
            const entry = results.get(name);
            entry.ics.set(horizon, []);
            entry.hits.set(horizon, { correct: 0, total: 0 });
            entry.samples.set(horizon, 0);
        }
    }

    for (let fold = 0; fold < folds; fold++) {
        const from = start + fold * foldSize;
        const to = fold === folds - 1 ? end : from + foldSize;
        const length = to - from;
        if (length < minSamples) continue;

        const featureRanks = new Map();
        for (const name of featureNames) {
            const index = featureIndex.get(name);
            const values = new Float64Array(length);
            for (let t = 0; t < length; t++) values[t] = matrix[from + t][index];
            featureRanks.set(name, rank(values));
        }

        for (const horizon of horizons) {
            const target = new Float64Array(length);
            for (let t = 0; t < length; t++) {
                target[t] = Math.log(close[from + t + horizon] / close[from + t]);
            }
            const targetRanks = rank(target);

            for (const name of featureNames) {
                const entry = results.get(name);
                const featureRanksFold = featureRanks.get(name);
                entry.ics.get(horizon).push(pearson(featureRanksFold, targetRanks));
                entry.samples.set(horizon, entry.samples.get(horizon) + length);

                if (meta[name].signed) {
                    const index = featureIndex.get(name);
                    let correct = 0;
                    let total = 0;
                    for (let t = 0; t < length; t++) {
                        const targetValue = target[t];
                        if (targetValue === 0) continue;
                        const featureValue = matrix[from + t][index];
                        if (featureValue === 0) continue;
                        total++;
                        if (Math.sign(featureValue) === Math.sign(targetValue)) correct++;
                    }
                    entry.hits.get(horizon).correct += correct;
                    entry.hits.get(horizon).total += total;
                }
            }
        }
    }

    const rows = [];
    const candidates = [];
    for (const name of featureNames) {
        const entry = results.get(name);
        const row = { feature: name, signed: meta[name].signed, group: meta[name].group, horizons: {} };
        let maxAbsIc = 0;
        let bestHorizon = horizons[0];

        for (const horizon of horizons) {
            const ics = entry.ics.get(horizon);
            const k = ics.length;
            const meanIc = k > 0 ? ics.reduce((sum, value) => sum + value, 0) / k : 0;
            const variance = k > 1
                ? ics.reduce((sum, value) => sum + (value - meanIc) ** 2, 0) / (k - 1)
                : 0;
            const stdIc = Math.sqrt(variance);
            const positive = ics.filter(value => value > 0).length;
            const stability = k > 0 ? Math.max(positive, k - positive) / k : 0;
            const hits = entry.hits.get(horizon);
            const hitRate = hits.total > 0 ? hits.correct / hits.total : null;

            row.horizons[horizon] = {
                meanIc, stdIc, folds: k, positiveFolds: positive,
                stability, hitRate, samples: entry.samples.get(horizon)
            };

            if (Math.abs(meanIc) > maxAbsIc) {
                maxAbsIc = Math.abs(meanIc);
                bestHorizon = horizon;
            }
            if (Math.abs(meanIc) >= minIc && stability >= minStability && k > 0) {
                candidates.push({
                    feature: name,
                    horizon,
                    meanIc,
                    stability,
                    hitRate,
                    folds: k,
                    group: meta[name].group
                });
            }
        }
        row.maxAbsIc = maxAbsIc;
        row.bestHorizon = bestHorizon;
        rows.push(row);
    }

    rows.sort((a, b) => b.maxAbsIc - a.maxAbsIc);

    console.log(`\n=== EURUSD ${timeframe} | ${n} bars | ${folds} folds | horizons ${horizons.join(',')} ===`);
    const header = ['feature'.padEnd(14)];
    for (const horizon of horizons) header.push(`IC@${horizon}`.padStart(10));
    header.push('stability'.padStart(12), 'hit@5'.padStart(8));
    console.log(header.join(' '));

    for (const row of rows) {
        const cells = [row.feature.padEnd(14)];
        for (const horizon of horizons) {
            cells.push(row.horizons[horizon].meanIc.toFixed(4).padStart(10));
        }
        const stability = horizons.map(horizon => row.horizons[horizon].stability.toFixed(2)).join('/');
        const hit = row.horizons[horizons[0]]?.hitRate;
        cells.push(stability.padStart(12), (hit === null || hit === undefined ? '-' : (hit * 100).toFixed(1) + '%').padStart(8));
        console.log(cells.join(' '));
    }

    candidates.sort((a, b) => Math.abs(b.meanIc) - Math.abs(a.meanIc));

    console.log(`\nCandidates (|IC| >= ${minIc}, stability >= ${minStability}):`);
    if (candidates.length === 0) {
        console.log('  none');
    } else {
        for (const candidate of candidates) {
            console.log(
                `  ${candidate.feature.padEnd(14)} h=${String(candidate.horizon).padEnd(3)} ` +
                `IC=${candidate.meanIc.toFixed(4)} stab=${candidate.stability.toFixed(2)} ` +
                `hit=${candidate.hitRate === null ? '-' : (candidate.hitRate * 100).toFixed(1) + '%'} (${candidate.group})`
            );
        }
    }

    if (!options.noCsv) {
        await mkdir(OUT_DIR, { recursive: true });
        const csvHeader = 'timeframe,feature,group,horizon,meanIC,stdIC,folds,positiveFolds,stability,hitRate,samples';
        const csvRows = [];
        for (const row of rows) {
            for (const horizon of horizons) {
                const stat = row.horizons[horizon];
                csvRows.push([
                    timeframe, row.feature, row.group, horizon,
                    stat.meanIc, stat.stdIc, stat.folds, stat.positiveFolds,
                    stat.stability, stat.hitRate ?? '', stat.samples
                ].join(','));
            }
        }
        await Bun.write(join(OUT_DIR, `${timeframe}_ic.csv`), [csvHeader, ...csvRows].join('\n'));
        await Bun.write(join(OUT_DIR, `${timeframe}_summary.json`), JSON.stringify({
            timeframe, bars: n, folds, horizons, candidates, rows
        }, null, 2));
    }

    return { timeframe, bars: n, candidates };
}

const args = parseArgs(process.argv.slice(2));
const options = {
    horizons: numberList(args.horizons, [5, 10, 20]),
    folds: Number(args.folds ?? 5),
    minSamples: Number(args['min-samples'] ?? 500),
    minIc: Number(args['min-ic'] ?? 0.02),
    minStability: Number(args['min-stability'] ?? 0.7),
    noCsv: args['no-csv'] === true
};

const files = await discoverFiles(args.data);
if (files.length === 0) {
    console.error('No EURUSD_*.json files found in data/. Download the data first.');
    process.exit(1);
}

const summaries = [];
for (const file of files) {
    try {
        summaries.push(await auditFile(file, options));
    } catch (error) {
        console.error(`Audit failed for ${file}:`, error.message);
    }
}

console.log('\n=== SUMMARY ===');
for (const summary of summaries) {
    const unique = new Set(summary.candidates.map(c => c.feature));
    console.log(`${summary.timeframe.padEnd(4)} ${String(summary.bars).padStart(7)} bars  candidates: ${unique.size} feature(s) -> ${[...unique].join(', ') || 'none'}`);
}
console.log('\nWrote per-timeframe results to signal_audit/ (gitignored).');
