// Rate/carry and macro/cross-asset helpers. All lookups are lagged to avoid
// using a value that was not published before the bar close.

import { join } from 'node:path';

const ROOT = join(import.meta.dir, '..');
const MACRO_DIR = join(ROOT, 'data', 'macro');

export const RATE_SERIES = {
    USD: 'IR3TIB01USM156N',
    EUR: 'IR3TIB01EZM156N',
    GBP: 'IR3TIB01GBM156N',
    JPY: 'IR3TIB01JPM156N',
    AUD: 'IR3TIB01AUM156N',
    CAD: 'IR3TIB01CAM156N',
    CHF: 'IR3TIB01CHM156N',
    NZD: 'IR3TIB01NZM156N'
};

export const MACRO_SERIES = {
    DXY: 'DTWEXBGS',
    VIX: 'VIXCLS',
    US10: 'DGS10',
    US2: 'DGS2',
    OIL: 'DCOILWTICO',
    GOLD: 'GOLD'
};

const cache = new Map();

async function load(name) {
    if (cache.has(name)) return cache.get(name);
    const promise = Bun.file(join(MACRO_DIR, `${name}.json`)).json();
    cache.set(name, promise);
    return promise;
}

// Binary search: latest observation strictly before `dateStr`.
function latestBefore(series, dateStr) {
    const { dates, values } = series;
    let low = 0;
    let high = dates.length - 1;
    let best = -1;
    while (low <= high) {
        const mid = (low + high) >> 1;
        if (dates[mid] < dateStr) {
            best = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return best >= 0 ? values[best] : null;
}

export async function loadRates() {
    const rates = {};
    for (const [currency, series] of Object.entries(RATE_SERIES)) {
        try {
            rates[currency] = await load(series);
        } catch {
            rates[currency] = null;
        }
    }
    return rates;
}

export async function loadMacroSeries() {
    const macro = {};
    for (const [name, series] of Object.entries(MACRO_SERIES)) {
        try {
            macro[name] = await load(series);
        } catch {
            macro[name] = null;
        }
    }
    return macro;
}

/**
 * Annualized carry for a long base / short quote position at `dateStr`.
 * Uses the previous month's 3-month rate to avoid lookahead.
 */
export function carryAnnual(rates, baseCurrency, quoteCurrency, dateStr) {
    const base = rates?.[baseCurrency];
    const quote = rates?.[quoteCurrency];
    if (!base || !quote) return null;
    // Previous month only: latest observation before the first of this month.
    const cutoff = `${dateStr.slice(0, 7)}-01`;
    const baseRate = latestBefore(base, cutoff);
    const quoteRate = latestBefore(quote, cutoff);
    if (baseRate === null || quoteRate === null) return null;
    return (baseRate - quoteRate) / 100;
}

export function macroValue(series, dateStr, lagDays = 1) {
    if (!series) return null;
    const date = new Date(`${dateStr}T00:00:00Z`);
    date.setUTCDate(date.getUTCDate() - lagDays);
    return latestBefore(series, date.toISOString().slice(0, 10));
}

export function dateFromTimestamp(seconds) {
    return new Date(seconds * 1000).toISOString().slice(0, 10);
}

export function pairCurrencies(symbol) {
    if (!symbol || symbol.length < 6) return null;
    const clean = symbol.toUpperCase().replace(/[^A-Z]/g, '');
    if (clean.length < 6) return null;
    const base = clean.slice(0, 3);
    const quote = clean.slice(3, 6);
    const known = new Set(Object.keys(RATE_SERIES));
    if (!known.has(base) || !known.has(quote)) return null;
    return { base, quote };
}
