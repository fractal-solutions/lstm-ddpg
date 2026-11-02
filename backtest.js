import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { DEFAULT_CONFIG, TradingSystem } from './main.js';
import { setSeed } from './utils/rng.js';

const ROOT = import.meta.dir;
const RESULTS_DIR = join(ROOT, 'backtest_results');

function parseArgs(argv) {
    const options = {};
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (!arg.startsWith('--')) continue;
        const key = arg.slice(2);
        if (key === 'random' || key === 'no-csv') {
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

function computeStats(trades) {
    const n = trades.length;
    const wins = trades.filter(t => t.netPnl > 0);
    const losses = trades.filter(t => t.netPnl <= 0);
    const grossProfit = wins.reduce((sum, t) => sum + t.netPnl, 0);
    const grossLoss = Math.abs(losses.reduce((sum, t) => sum + t.netPnl, 0));

    let equity = 1;
    let peak = 1;
    let maxDrawdown = 0;
    const equityCurve = [1];
    for (const trade of trades) {
        equity *= 1 + trade.netPnl;
        equityCurve.push(equity);
        peak = Math.max(peak, equity);
        maxDrawdown = Math.min(maxDrawdown, equity / peak - 1);
    }

    const mean = n > 0 ? trades.reduce((sum, t) => sum + t.netPnl, 0) / n : 0;
    const meanR = n > 0 ? trades.reduce((sum, t) => sum + t.netR, 0) / n : 0;
    const variance = n > 1
        ? trades.reduce((sum, t) => sum + (t.netPnl - mean) ** 2, 0) / (n - 1)
        : 0;
    const stdDev = Math.sqrt(variance);

    return {
        trades: n,
        longs: trades.filter(t => t.direction > 0).length,
        shorts: trades.filter(t => t.direction < 0).length,
        wins: wins.length,
        losses: losses.length,
        winRate: n > 0 ? wins.length / n : 0,
        grossProfit,
        grossLoss,
        profitFactor: grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? Infinity : 0),
        totalReturn: equity - 1,
        equityCurve,
        avgTrade: mean,
        avgR: meanR,
        stdDev,
        sharpePerTrade: stdDev > 0 ? (mean / stdDev) * Math.sqrt(n) : 0,
        maxDrawdown,
        avgBarsHeld: n > 0 ? trades.reduce((sum, t) => sum + t.barsHeld, 0) / n : 0,
        slCount: trades.filter(t => t.reason === 'sl').length,
        tpCount: trades.filter(t => t.reason === 'tp').length,
        closeCount: trades.filter(t => t.reason === 'close').length,
        avgSize: n > 0 ? trades.reduce((sum, t) => sum + t.size, 0) / n : 0
    };
}

function formatRow(label, value) {
    return `${label.padEnd(22)} ${value}`;
}

function formatNumber(value, digits = 3) {
    if (value === Infinity) return 'inf';
    if (value === -Infinity) return '-inf';
    return value.toFixed(digits);
}

async function run() {
    const args = parseArgs(process.argv.slice(2));
    const modelName = typeof args.model === 'string' ? args.model : 'final_model';
    const useRandom = args.random === true;
    const foldCount = Math.max(1, Number(args.folds ?? 1));
    const config = { ...DEFAULT_CONFIG };
    if (args.data) config.dataPath = args.data;
    if (args.seed !== undefined) {
        config.seed = Number(args.seed);
        setSeed(config.seed);
    }

    const trader = new TradingSystem(config);
    await trader.loadData();

    if (useRandom) {
        const gain = await trader.calibrateStateGain();
        console.log(`No model loaded: using randomly initialized networks (stateGain=${gain.toFixed(3)}).`);
    } else {
        const modelData = await trader.loadModels(modelName);
        console.log(`Loaded model "${modelName}"`);
        if (modelData.meta) {
            const { savedAt, lookback, featureSize, hiddenSize, barsPredicted, stateGain, seed, epoch, stage } = modelData.meta;
            console.log(`  meta: saved=${savedAt} lookback=${lookback} features=${featureSize} hidden=${hiddenSize} bars=${barsPredicted} gain=${stateGain?.toFixed?.(3) ?? stateGain} seed=${seed ?? '-'} epoch=${epoch ?? '-'} stage=${stage ?? 'ddpg'}`);
        }
    }

    const dp = trader.dataProcessor;
    const start = Math.max(trader.testStart, dp.lookback + 1);
    const last = dp.normalized.length - trader.barsPredicted;
    const maxTrades = Number.isFinite(Number(args.trades)) ? Number(args.trades) : Infinity;

    let flatWindows = 0;
    let invalidWindows = 0;
    const trades = [];

    for (let index = start; index <= last; index += trader.barsPredicted) {
        if (trades.length >= maxTrades) break;

        let state;
        try {
            state = await dp.getState(index);
        } catch {
            invalidWindows++;
            continue;
        }

        const action = trader.predict(state, false, true);
        if (action.positionSize === 0) {
            flatWindows++;
            continue;
        }

        const trade = trader.simulateTrade(action, index);
        if (!trade) {
            invalidWindows++;
            continue;
        }

        const netPnl = trade.size * (trade.pnlPct - trade.costPct);
        trades.push({
            index,
            direction: trade.direction,
            size: trade.size,
            entry: trade.entry,
            stopLoss: trade.stopLoss,
            takeProfit: trade.takeProfit,
            slMultiple: trade.slMultiple,
            tpMultiple: trade.tpMultiple,
            exit: trade.exit,
            exitIndex: trade.exitIndex,
            reason: trade.reason,
            barsHeld: trade.barsHeld,
            pnlPct: trade.pnlPct,
            pnlR: trade.pnlR,
            netR: trade.netR,
            netPnl
        });
    }

    const stats = computeStats(trades);
    const barsInTest = last - (start - 1) + 1;
    const years = barsInTest / 252;
    const startPrice = dp.getBar(start - 1).close;
    const endPrice = dp.getBar(last).close;
    const buyHold = endPrice / startPrice - 1;
    const equity = 1 + stats.totalReturn;
    const cagr = equity > 0 ? Math.pow(equity, 252 / barsInTest) - 1 : -1;

    console.log('\n=== BACKTEST ===');
    console.log(formatRow('Model', useRandom ? 'random-init' : modelName));
    console.log(formatRow('Test bars', `${barsInTest} (${start}..${last}, ~${years.toFixed(2)}y)`));
    console.log(formatRow('Spread cost/trade', `${trader.spreadPrice.toExponential(2)} price`));
    console.log(formatRow('Trades', `${stats.trades} (long ${stats.longs} / short ${stats.shorts})`));
    console.log(formatRow('Skipped flat windows', flatWindows));
    console.log(formatRow('Invalid windows', invalidWindows));
    console.log('');
    console.log(formatRow('Win rate', `${(stats.winRate * 100).toFixed(2)}%`));
    console.log(formatRow('Profit factor', formatNumber(stats.profitFactor)));
    console.log(formatRow('Expectancy', `${formatNumber(stats.avgR, 4)} R / trade`));
    console.log(formatRow('Total return', `${(stats.totalReturn * 100).toFixed(2)}%`));
    console.log(formatRow('CAGR', `${(cagr * 100).toFixed(2)}%`));
    console.log(formatRow('Buy & hold', `${(buyHold * 100).toFixed(2)}%`));
    console.log(formatRow('Max drawdown', `${(stats.maxDrawdown * 100).toFixed(2)}%`));
    console.log(formatRow('Avg trade', `${(stats.avgTrade * 100).toFixed(4)}%`));
    console.log(formatRow('Std dev / trade', `${(stats.stdDev * 100).toFixed(4)}%`));
    console.log(formatRow('Sharpe (per trade)', stats.sharpePerTrade.toFixed(3)));
    console.log(formatRow('Avg bars held', stats.avgBarsHeld.toFixed(2)));
    console.log(formatRow('Avg position size', stats.avgSize.toFixed(3)));
    console.log(formatRow('Exits (SL/TP/close)', `${stats.slCount} / ${stats.tpCount} / ${stats.closeCount}`));

    if (foldCount > 1) {
        console.log(`\nWalk-forward folds (${foldCount})`);
        console.log('fold   bars        trades  win%   PF      expR     ret%     maxDD%');
        const span = (last - start + 1) / foldCount;
        for (let fold = 0; fold < foldCount; fold++) {
            const from = start + Math.floor(fold * span);
            const to = fold === foldCount - 1 ? last + 1 : start + Math.floor((fold + 1) * span);
            const foldTrades = trades.filter(t => t.index >= from && t.index < to);
            const foldStats = computeStats(foldTrades);
            console.log(
                `${String(fold + 1).padEnd(6)} ${String(to - from).padEnd(11)} ` +
                `${String(foldStats.trades).padEnd(7)} ${(foldStats.winRate * 100).toFixed(1).padEnd(6)} ` +
                `${formatNumber(foldStats.profitFactor, 2).padEnd(7)} ${formatNumber(foldStats.avgR, 3).padEnd(8)} ` +
                `${(foldStats.totalReturn * 100).toFixed(2).padEnd(8)} ${(foldStats.maxDrawdown * 100).toFixed(2)}`
            );
        }
    }

    if (!args['no-csv']) {
        await mkdir(RESULTS_DIR, { recursive: true });
        const tag = useRandom ? 'random' : modelName.replace(/[^a-zA-Z0-9_-]/g, '_');

        const csvHeader = 'index,direction,size,entry,stopLoss,takeProfit,slMultiple,tpMultiple,exit,exitIndex,reason,barsHeld,pnlPct,pnlR,netR,netPnl';
        const csvRows = trades.map(t => [
            t.index, t.direction, t.size, t.entry, t.stopLoss, t.takeProfit,
            t.slMultiple, t.tpMultiple, t.exit, t.exitIndex, t.reason, t.barsHeld,
            t.pnlPct, t.pnlR, t.netR, t.netPnl
        ].join(','));
        await Bun.write(join(RESULTS_DIR, `${tag}_trades.csv`), [csvHeader, ...csvRows].join('\n'));

        const summary = {
            model: useRandom ? 'random-init' : modelName,
            test: { start, last, bars: barsInTest, years, startPrice, endPrice },
            spreadPrice: trader.spreadPrice,
            skippedFlatWindows: flatWindows,
            invalidWindows,
            buyHold,
            cagr,
            stats
        };
        await Bun.write(join(RESULTS_DIR, `${tag}_summary.json`), JSON.stringify(summary, null, 2));
        console.log(`\nWrote ${join('backtest_results', `${tag}_trades.csv`)} and ${join('backtest_results', `${tag}_summary.json`)}`);
    }
}

run().catch(error => {
    console.error(error);
    process.exit(1);
});
