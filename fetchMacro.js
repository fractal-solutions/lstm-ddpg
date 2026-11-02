// Downloads rates (for carry/financing) and macro/cross-asset series from FRED.
// Saves JSON under data/macro/ as { dates: [YYYY-MM-DD], values: [number] }.

import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';

const ROOT = import.meta.dir;
const OUT_DIR = join(ROOT, 'data', 'macro');

const RATE_SERIES = {
    USD: 'IR3TIB01USM156N',
    EUR: 'IR3TIB01EZM156N',
    GBP: 'IR3TIB01GBM156N',
    JPY: 'IR3TIB01JPM156N',
    AUD: 'IR3TIB01AUM156N',
    CAD: 'IR3TIB01CAM156N',
    CHF: 'IR3TIB01CHM156N',
    NZD: 'IR3TIB01NZM156N'
};

const MACRO_SERIES = {
    DXY: 'DTWEXBGS',
    VIX: 'VIXCLS',
    US10: 'DGS10',
    US2: 'DGS2',
    OIL: 'DCOILWTICO'
};

async function fetchFred(series) {
    const response = await fetch(`https://fred.stlouisfed.org/graph/fredgraph.csv?id=${series}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await response.text();
    const lines = text.trim().split('\n').slice(1);

    const dates = [];
    const values = [];
    for (const line of lines) {
        const [date, raw] = line.split(',');
        if (!date || raw === undefined || raw === '.' || raw === '') continue;
        const value = Number(raw);
        if (!Number.isFinite(value)) continue;
        dates.push(date);
        values.push(value);
    }
    if (dates.length === 0) throw new Error('no observations');
    return { series, dates, values };
}

async function fetchYahooGold(years = 30) {
    const now = Math.floor(Date.now() / 1000);
    const from = now - Math.round(years * 365.25 * 24 * 3600);
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/GC=F?period1=${from}&period2=${now}&interval=1d`;
    const response = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const json = await response.json();
    const result = json.chart?.result?.[0];
    const quote = result?.indicators?.quote?.[0];
    const timestamps = result?.timestamp ?? [];
    const dates = [];
    const values = [];
    for (let i = 0; i < timestamps.length; i++) {
        const close = quote?.close?.[i];
        if (!Number.isFinite(close)) continue;
        dates.push(new Date(timestamps[i] * 1000).toISOString().slice(0, 10));
        values.push(close);
    }
    if (dates.length === 0) throw new Error('no observations');
    return { series: 'GOLD', dates, values };
}

await mkdir(OUT_DIR, { recursive: true });

for (const [currency, series] of Object.entries(RATE_SERIES)) {
    try {
        const data = await fetchFred(series);
        await Bun.write(join(OUT_DIR, `${series}.json`), JSON.stringify(data));
        console.log(`RATE ${currency.padEnd(4)} ${series.padEnd(18)} ${data.dates.length} obs ${data.dates[0]} .. ${data.dates[data.dates.length - 1]}`);
    } catch (error) {
        console.error(`RATE ${currency} ${series} FAILED: ${error.message}`);
    }
    await Bun.sleep(150);
}

for (const [name, series] of Object.entries(MACRO_SERIES)) {
    try {
        const data = await fetchFred(series);
        await Bun.write(join(OUT_DIR, `${series}.json`), JSON.stringify(data));
        console.log(`MACRO ${name.padEnd(4)} ${series.padEnd(18)} ${data.dates.length} obs ${data.dates[0]} .. ${data.dates[data.dates.length - 1]}`);
    } catch (error) {
        console.error(`MACRO ${name} ${series} FAILED: ${error.message}`);
    }
    await Bun.sleep(150);
}

try {
    const gold = await fetchYahooGold();
    await Bun.write(join(OUT_DIR, 'GOLD.json'), JSON.stringify(gold));
    console.log(`MACRO GOLD ${gold.dates.length} obs ${gold.dates[0]} .. ${gold.dates[gold.dates.length - 1]}`);
} catch (error) {
    console.error(`GOLD FAILED: ${error.message}`);
}
