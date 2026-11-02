// Walk-forward IC audit for carry and macro/cross-asset features.
// Features are causal and lagged (previous day / previous month).

import { join } from 'node:path';
import { readdir } from 'node:fs/promises';
import { rank, pearson } from './utils/features.js';
import {
    loadRates, loadMacroSeries, carryAnnual, macroValue,
    dateFromTimestamp, pairCurrencies
} from './utils/macro.js';

const ROOT = import.meta.dir;
const DATA_DIR = join(ROOT, 'data');

function parseArgs(argv) {
    const options = {};
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (!arg.startsWith('--')) continue;
        const key = arg.slice(2);
        if (key === 'no-csv') { options[key] = true; continue; }
        const value = argv[i + 1];
        if (value === undefined || value.startsWith('--')) { options[key] = true; continue; }
        options[key] = value;
        i++;
    }
    return options;
}

function trimData(data, points) {
    const n = data.close.length;
    if (!points || n <= points) return data;
    const start = n - points;
    return {
        ...data,
        time: data.time.slice(start),
        open: data.open.slice(start),
        high: data.high.slice(start),
        low: data.low.slice(start),
        close: data.close.slice(start),
        volume: data.volume.slice(start)
    };
}

function logReturn(a, b) {
    return a > 0 && b > 0 ? Math.log(a / b) : null;
}

const args = parseArgs(process.argv.slice(2));
const horizon = Number(args.horizon ?? 20);
const folds = Number(args.folds ?? 5);
const dataPoints = Number(args['data-points'] ?? 5000);
const minIc = Number(args['min-ic'] ?? 0.02);
const minStability = Number(args['min-stability'] ?? 0.7);

const rates = await loadRates();
const macro = await loadMacroSeries();

const files = (await readdir(DATA_DIR))
    .filter(name => /^Y_[A-Z]+_D1\.json$/.test(name))
    .sort();

const featureNames = [
    'carry', 'carryMom20', 'dxyRet5', 'dxyRet20',
    'vixChange5', 'vixLevel', 'slope2s10', 'oilRet20', 'goldRet20'
];
const perFeature = new Map(featureNames.map(name => [name, []]));
const perSymbol = [];

for (const file of files) {
    const symbol = file.replace(/^Y_/, '').replace(/_D1\.json$/, '');
    const data = trimData(await Bun.file(join(DATA_DIR, file)).json(), dataPoints);
    const pair = pairCurrencies(symbol);
    if (!pair) continue;

    const n = data.close.length;
    const carry = new Array(n).fill(null);
    const features = Object.fromEntries(featureNames.map(name => [name, new Array(n).fill(null)]));

    for (let i = 0; i < n; i++) {
        const date = dateFromTimestamp(data.time[i]);
        carry[i] = carryAnnual(rates, pair.base, pair.quote, date);

        if (i >= 20) {
            const date20 = dateFromTimestamp(data.time[i - 20]);
            const carry20 = carryAnnual(rates, pair.base, pair.quote, date20);
            features.carryMom20[i] = carry[i] !== null && carry20 !== null ? carry[i] - carry20 : null;
        }

        const dxy = macroValue(macro.DXY, date);
        const vix = macroValue(macro.VIX, date);
        const us10 = macroValue(macro.US10, date);
        const us2 = macroValue(macro.US2, date);
        const oil = macroValue(macro.OIL, date);
        const gold = macroValue(macro.GOLD, date);

        features.carry[i] = carry[i];
        features.vixLevel[i] = vix;
        features.slope2s10[i] = us10 !== null && us2 !== null ? us10 - us2 : null;

        if (i >= 5) {
            const date5 = dateFromTimestamp(data.time[i - 5]);
            const dxy5 = macroValue(macro.DXY, date5);
            const vix5 = macroValue(macro.VIX, date5);
            if (dxy !== null && dxy5 !== null) features.dxyRet5[i] = logReturn(dxy, dxy5);
            if (vix !== null && vix5 !== null) features.vixChange5[i] = vix - vix5;
        }
        if (i >= 20) {
            const date20 = dateFromTimestamp(data.time[i - 20]);
            const dxy20 = macroValue(macro.DXY, date20);
            const oil20 = macroValue(macro.OIL, date20);
            const gold20 = macroValue(macro.GOLD, date20);
            if (dxy !== null && dxy20 !== null) features.dxyRet20[i] = logReturn(dxy, dxy20);
            if (oil !== null && oil20 !== null) features.oilRet20[i] = logReturn(oil, oil20);
            if (gold !== null && gold20 !== null) features.goldRet20[i] = logReturn(gold, gold20);
        }
    }

    const symbols = [];
    for (const name of featureNames) {
        const x = [];
        const y = [];
        for (let i = 0; i + horizon < n; i++) {
            const value = features[name][i];
            if (value === null || !Number.isFinite(value)) continue;
            x.push(value);
            y.push(Math.log(data.close[i + horizon] / data.close[i]));
        }
        if (x.length < 300) continue;

        const size = Math.floor(x.length / folds);
        const ics = [];
        for (let fold = 0; fold < folds; fold++) {
            const from = fold * size;
            const to = fold === folds - 1 ? x.length : (fold + 1) * size;
            if (to - from < 50) continue;
            ics.push(pearson(rank(x.slice(from, to)), rank(y.slice(from, to))));
        }
        if (ics.length === 0) continue;

        const mean = ics.reduce((sum, value) => sum + value, 0) / ics.length;
        const positive = ics.filter(value => value > 0).length;
        const stability = Math.max(positive, ics.length - positive) / ics.length;
        const hit = (() => {
            let correct = 0;
            let total = 0;
            for (let i = 0; i < x.length; i++) {
                if (x[i] === 0 || y[i] === 0) continue;
                total++;
                if (Math.sign(x[i]) === Math.sign(y[i])) correct++;
            }
            return total ? correct / total : null;
        })();

        perFeature.get(name).push({ symbol, mean, stability });
        symbols.push({ name, mean, stability, hit });
    }

    symbols.sort((a, b) => Math.abs(b.mean) - Math.abs(a.mean));
    perSymbol.push({ symbol, best: symbols[0] ?? null });
}

console.log(`Macro/carry IC audit: ${files.length} symbols, horizon ${horizon}, ${folds} folds\n`);
console.log('feature       avgIC     avgStab   qualifying symbols   top symbol (IC)');

for (const name of featureNames) {
    const rows = perFeature.get(name);
    if (rows.length === 0) continue;
    const avgIc = rows.reduce((sum, row) => sum + row.mean, 0) / rows.length;
    const avgStability = rows.reduce((sum, row) => sum + row.stability, 0) / rows.length;
    const qualifying = rows.filter(row => Math.abs(row.mean) >= minIc && row.stability >= minStability);
    const best = rows.reduce((a, b) => (Math.abs(b.mean) > Math.abs(a.mean) ? b : a));
    console.log(
        `${name.padEnd(13)} ${avgIc.toFixed(4).padStart(8)} ${avgStability.toFixed(2).padStart(9)} ` +
        `${String(qualifying.length).padStart(8)}/${String(rows.length).padEnd(12)} ` +
        `${best.symbol} (${best.mean.toFixed(3)})`
    );
}

console.log('\nBest macro feature per symbol:');
for (const row of perSymbol.sort((a, b) => a.symbol.localeCompare(b.symbol))) {
    if (!row.best) continue;
    console.log(
        `${row.symbol.padEnd(8)} ${row.best.name.padEnd(13)} IC=${row.best.mean.toFixed(4)} ` +
        `stability=${row.best.stability.toFixed(2)}`
    );
}
