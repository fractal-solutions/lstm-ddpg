// Causal feature factory for the signal audit and for model inputs.
// Every feature at bar i uses only bars <= i. No global normalization.

export const FEATURE_META = {
    ret1z: { signed: true, group: 'momentum' },
    ret5z: { signed: true, group: 'momentum' },
    ret20z: { signed: true, group: 'momentum' },
    ret60z: { signed: true, group: 'momentum' },
    atrRel: { signed: false, group: 'volatility' },
    volRatio: { signed: false, group: 'volatility' },
    rangeRatio: { signed: false, group: 'volatility' },
    shock: { signed: false, group: 'volatility' },
    dist20: { signed: true, group: 'trend' },
    dist50: { signed: true, group: 'trend' },
    slope20: { signed: true, group: 'trend' },
    donchian20: { signed: true, group: 'range' },
    rsi14: { signed: true, group: 'oscillator' },
    bollingerB: { signed: true, group: 'oscillator' },
    gap: { signed: true, group: 'structure' },
    volumeZ: { signed: true, group: 'volume' }
};

export const FEATURE_NAMES = Object.keys(FEATURE_META);

function prefix(values) {
    const out = new Float64Array(values.length + 1);
    for (let i = 0; i < values.length; i++) out[i + 1] = out[i] + values[i];
    return out;
}

function prefixSquares(values) {
    const out = new Float64Array(values.length + 1);
    for (let i = 0; i < values.length; i++) out[i + 1] = out[i] + values[i] * values[i];
    return out;
}

function windowMean(prefixSum, i, window) {
    return (prefixSum[i + 1] - prefixSum[i + 1 - window]) / window;
}

function windowStd(prefixSum, prefixSquare, i, window) {
    const mean = windowMean(prefixSum, i, window);
    const variance = (prefixSquare[i + 1] - prefixSquare[i + 1 - window]) / window - mean * mean;
    return Math.sqrt(Math.max(0, variance));
}

export function buildFeatureMatrix(data, { warmup = 60 } = {}) {
    const n = data.close.length;
    const { open, high, low, close, volume } = data;

    const logClose = new Float64Array(n);
    const ret1 = new Float64Array(n);
    const trueRange = new Float64Array(n);
    const logVolume = new Float64Array(n);
    const gains = new Float64Array(n);
    const losses = new Float64Array(n);

    for (let i = 0; i < n; i++) {
        logClose[i] = Math.log(close[i]);
        logVolume[i] = Math.log(Math.max(1, volume[i]));
    }

    for (let i = 1; i < n; i++) {
        ret1[i] = logClose[i] - logClose[i - 1];
        const prevClose = close[i - 1];
        trueRange[i] = Math.max(
            high[i] - low[i],
            Math.abs(high[i] - prevClose),
            Math.abs(low[i] - prevClose)
        );
        const diff = close[i] - prevClose;
        gains[i] = Math.max(diff, 0);
        losses[i] = Math.max(-diff, 0);
    }
    trueRange[0] = high[0] - low[0];

    const pClose = prefix(close);
    const pClose2 = prefixSquares(close);
    const pRet = prefix(ret1);
    const pRet2 = prefixSquares(ret1);
    const pTR = prefix(trueRange);
    const pGain = prefix(gains);
    const pLoss = prefix(losses);
    const pLogVol = prefix(logVolume);
    const pLogVol2 = prefixSquares(logVolume);

    const matrix = new Array(n);
    for (let i = warmup; i < n; i++) {
        const atr14 = windowMean(pTR, i, 14);
        const atrRel = atr14 / close[i] || 1e-12;
        const vol10 = windowStd(pRet, pRet2, i, 10);
        const vol60 = windowStd(pRet, pRet2, i, 60) || 1e-12;
        const tr5 = windowMean(pTR, i, 5);
        const tr20 = windowMean(pTR, i, 20) || 1e-12;
        const sma20 = windowMean(pClose, i, 20);
        const sma50 = windowMean(pClose, i, 50);
        const sma20Past = windowMean(pClose, i - 5, 20);
        const std20 = windowStd(pClose, pClose2, i, 20) || 1e-12;

        let high20 = -Infinity;
        let low20 = Infinity;
        for (let j = i - 19; j <= i; j++) {
            if (high[j] > high20) high20 = high[j];
            if (low[j] < low20) low20 = low[j];
        }
        const range20 = high20 - low20 || 1e-12;

        const avgGain = windowMean(pGain, i, 14);
        const avgLoss = windowMean(pLoss, i, 14);
        const rsi = avgGain + avgLoss > 0 ? (100 * avgGain) / (avgGain + avgLoss) : 50;

        const meanLogVol = windowMean(pLogVol, i, 20);
        const stdLogVol = windowStd(pLogVol, pLogVol2, i, 20) || 1e-12;

        matrix[i] = [
            ret1[i] / vol60,                                    // ret1z
            (logClose[i] - logClose[i - 5]) / (vol60 * Math.sqrt(5)),   // ret5z
            (logClose[i] - logClose[i - 20]) / (vol60 * Math.sqrt(20)), // ret20z
            (logClose[i] - logClose[i - 60]) / (vol60 * Math.sqrt(60)), // ret60z
            atrRel,                                             // atrRel
            vol10 / vol60,                                      // volRatio
            tr5 / tr20,                                         // rangeRatio
            Math.abs(ret1[i]) / atrRel,                         // shock
            (close[i] / sma20 - 1) / atrRel,                    // dist20
            (close[i] / sma50 - 1) / atrRel,                    // dist50
            (sma20 / sma20Past - 1) / atrRel,                   // slope20
            2 * ((close[i] - low20) / range20 - 0.5),           // donchian20
            (rsi - 50) / 50,                                    // rsi14
            (close[i] - sma20) / (2 * std20),                   // bollingerB
            Math.log(open[i] / close[i - 1]),                   // gap
            (logVolume[i] - meanLogVol) / stdLogVol             // volumeZ
        ];
    }

    return { matrix, featureNames: FEATURE_NAMES, meta: FEATURE_META, warmup, n, close };
}

/** Forward log return over `horizon` bars, labelled at bar t. */
export function forwardReturn(close, t, horizon) {
    return Math.log(close[t + horizon] / close[t]);
}

/** Average-rank transform (handles ties), used for Spearman correlation. */
export function rank(values) {
    const n = values.length;
    const indices = Array.from({ length: n }, (_, i) => i);
    indices.sort((a, b) => values[a] - values[b]);

    const ranks = new Float64Array(n);
    let i = 0;
    while (i < n) {
        let j = i;
        while (j + 1 < n && values[indices[j + 1]] === values[indices[i]]) j++;
        const averageRank = (i + j) / 2 + 1;
        for (let k = i; k <= j; k++) ranks[indices[k]] = averageRank;
        i = j + 1;
    }
    return ranks;
}

export function pearson(a, b) {
    const n = a.length;
    if (n < 2) return 0;
    let meanA = 0;
    let meanB = 0;
    for (let i = 0; i < n; i++) {
        meanA += a[i];
        meanB += b[i];
    }
    meanA /= n;
    meanB /= n;

    let cov = 0;
    let varA = 0;
    let varB = 0;
    for (let i = 0; i < n; i++) {
        const da = a[i] - meanA;
        const db = b[i] - meanB;
        cov += da * db;
        varA += da * da;
        varB += db * db;
    }
    if (varA === 0 || varB === 0) return 0;
    return cov / Math.sqrt(varA * varB);
}
