// Per-bar carry and macro/cross-asset features, aligned by date and lagged.
// Rows are null until every macro input for that bar is available.

import {
    loadRates, loadMacroSeries, carryAnnual, macroValue,
    dateFromTimestamp, pairCurrencies
} from './macro.js';

export const MACRO_FEATURE_NAMES = [
    'carry', 'carryMom20', 'vixLevel', 'vixChange5',
    'dxyRet5', 'dxyRet20', 'slope2s10', 'oilRet20', 'goldRet20'
];

export const MACRO_FEATURE_META = {
    carry: { signed: true, group: 'carry' },
    carryMom20: { signed: true, group: 'carry' },
    vixLevel: { signed: false, group: 'macro' },
    vixChange5: { signed: true, group: 'macro' },
    dxyRet5: { signed: true, group: 'macro' },
    dxyRet20: { signed: true, group: 'macro' },
    slope2s10: { signed: true, group: 'macro' },
    oilRet20: { signed: true, group: 'cross-asset' },
    goldRet20: { signed: true, group: 'cross-asset' }
};

function logReturn(a, b) {
    return a > 0 && b > 0 ? Math.log(a / b) : null;
}

export async function buildMacroMatrix(data, symbol = null) {
    const n = data.close.length;
    const time = data.time;
    if (!Array.isArray(time) || time.length !== n || time[0] <= 946684800) {
        return { matrix: null, names: MACRO_FEATURE_NAMES, firstValid: null, n };
    }

    const pair = pairCurrencies(symbol ?? data.symbol);
    if (!pair) {
        return { matrix: null, names: MACRO_FEATURE_NAMES, firstValid: null, n };
    }

    const rates = await loadRates();
    const macro = await loadMacroSeries();
    const matrix = new Array(n).fill(null);
    let firstValid = -1;
    let lastValid = null;

    for (let i = 0; i < n; i++) {
        const date = dateFromTimestamp(time[i]);
        const carry = carryAnnual(rates, pair.base, pair.quote, date);
        const dxy = macroValue(macro.DXY, date);
        const vix = macroValue(macro.VIX, date);
        const us10 = macroValue(macro.US10, date);
        const us2 = macroValue(macro.US2, date);
        const oil = macroValue(macro.OIL, date);
        const gold = macroValue(macro.GOLD, date);

        if (carry === null || dxy === null || vix === null || us10 === null ||
            us2 === null || oil === null || gold === null) {
            continue;
        }

        let carryMom20 = null;
        let dxyRet5 = null;
        let dxyRet20 = null;
        let vixChange5 = null;
        let oilRet20 = null;
        let goldRet20 = null;

        if (i >= 20) {
            const date20 = dateFromTimestamp(time[i - 20]);
            const carry20 = carryAnnual(rates, pair.base, pair.quote, date20);
            const dxy20 = macroValue(macro.DXY, date20);
            const oil20 = macroValue(macro.OIL, date20);
            const gold20 = macroValue(macro.GOLD, date20);
            if (carry20 !== null) carryMom20 = carry - carry20;
            if (dxy20 !== null) dxyRet20 = logReturn(dxy, dxy20);
            // WTI went negative in April 2020; a log return is undefined there.
            // Treat that episode as neutral instead of dropping the bar.
            if (oil20 !== null) oilRet20 = oil > 0 && oil20 > 0 ? logReturn(oil, oil20) : 0;
            if (gold20 !== null) goldRet20 = logReturn(gold, gold20);
        }
        if (i >= 5) {
            const date5 = dateFromTimestamp(time[i - 5]);
            const dxy5 = macroValue(macro.DXY, date5);
            const vix5 = macroValue(macro.VIX, date5);
            if (dxy5 !== null) dxyRet5 = logReturn(dxy, dxy5);
            if (vix5 !== null) vixChange5 = vix - vix5;
        }

        const row = [
            carry, carryMom20, vix, vixChange5,
            dxyRet5, dxyRet20, us10 - us2, oilRet20, goldRet20
        ];
        if (row.some(value => value === null || !Number.isFinite(value))) continue;

        matrix[i] = row;
        if (firstValid === -1) firstValid = i;
        lastValid = row;
    }

    // Forward-fill sparse gaps (holidays, one-off invalid prints) after the
    // first valid bar. Never back-fill before macro coverage starts.
    let previous = null;
    for (let i = 0; i < n; i++) {
        if (matrix[i]) previous = matrix[i];
        else if (previous && i > firstValid) matrix[i] = [...previous];
    }

    return { matrix, names: MACRO_FEATURE_NAMES, firstValid, n };
}
