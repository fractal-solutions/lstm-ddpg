// Downloads daily FX OHLC from Yahoo Finance into the repo's JSON schema.
// Files are written as data/Y_<SYMBOL>_D1.json so they do not overwrite the
// Dukascopy EURUSD files used by earlier experiments.

import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';

const ROOT = import.meta.dir;
const DATA_DIR = join(ROOT, 'data');

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
        if (key === 'force') {
            options.force = true;
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

async function fetchSymbol(symbol, years) {
    const now = Math.floor(Date.now() / 1000);
    const from = now - Math.round(years * 365.25 * 24 * 3600);
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol + '=X')}` +
        `?period1=${from}&period2=${now}&interval=1d`;

    const response = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const json = await response.json();
    const result = json.chart?.result?.[0];
    if (!result) throw new Error('no chart result');

    const quote = result.indicators.quote[0];
    const timestamps = result.timestamp ?? [];
    const rows = [];

    for (let i = 0; i < timestamps.length; i++) {
        const open = quote.open?.[i];
        const high = quote.high?.[i];
        const low = quote.low?.[i];
        const close = quote.close?.[i];
        if (![open, high, low, close].every(Number.isFinite)) continue;
        if (!(high >= low) || !(close > 0) || !(open > 0)) continue;
        rows.push({ time: timestamps[i], open, high, low, close });
    }

    const deduped = [];
    for (const row of rows) {
        if (deduped.length && deduped[deduped.length - 1].time === row.time) continue;
        deduped.push(row);
    }

    // Drop a possibly incomplete current-day bar.
    if (deduped.length && now - deduped[deduped.length - 1].time < 36 * 3600) {
        deduped.pop();
    }

    if (deduped.length < 500) throw new Error(`too few bars (${deduped.length})`);

    const isJpy = symbol.includes('JPY');
    return {
        ver: 3,
        source: 'Yahoo Finance',
        symbol,
        period: 1440,
        bars: deduped.length,
        spread: 10,
        point: isJpy ? 0.001 : 0.00001,
        time: deduped.map(row => row.time),
        open: deduped.map(row => row.open),
        high: deduped.map(row => row.high),
        low: deduped.map(row => row.low),
        close: deduped.map(row => row.close),
        volume: deduped.map(() => 0)
    };
}

const args = parseArgs(process.argv.slice(2));
const years = Number(args.years ?? 30);
const symbols = String(args.symbols ?? DEFAULT_SYMBOLS.join(','))
    .split(',')
    .map(value => value.trim().toUpperCase())
    .filter(Boolean);

await mkdir(DATA_DIR, { recursive: true });

for (const symbol of symbols) {
    const path = join(DATA_DIR, `Y_${symbol}_D1.json`);
    try {
        const data = await fetchSymbol(symbol, years);
        await Bun.write(path, JSON.stringify(data));
        const first = new Date(data.time[0] * 1000).toISOString().slice(0, 10);
        const last = new Date(data.time[data.time.length - 1] * 1000).toISOString().slice(0, 10);
        console.log(`${symbol.padEnd(7)} ${String(data.bars).padStart(6)} bars  ${first} .. ${last}`);
    } catch (error) {
        console.error(`${symbol.padEnd(7)} FAILED: ${error.message}`);
    }
    await Bun.sleep(250);
}
