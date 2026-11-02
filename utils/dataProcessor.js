import { clamp } from './rng.js';
import { buildFeatureMatrix } from './features.js';
import { buildMacroMatrix, MACRO_FEATURE_NAMES } from './macroFeatures.js';

export class DataProcessor {
    constructor(data, { lookback = 60, dataPoints = null, symbol = null } = {}) {
        this.data = data;
        this.lookback = lookback;
        this.dataPoints = dataPoints;
        this.symbol = symbol ?? data.symbol ?? null;
        this.features = ['open', 'high', 'low', 'close', 'volume'];
        this.normalized = null;
        this.combined = null;
        this.priceScale = { min: 0, max: 1 };
        this.volumeScale = { min: 0, max: 1 };
        this.featureMatrix = null;
        this.featureNames = null;
        this.featureStats = null;
        this.warmup = 60;
        this.stateStart = lookback + 60;
    }

    setScales() {
        const n = this.data.close.length;
        const start = Math.max(0, n - this.dataPoints);
        let low = Infinity;
        let high = -Infinity;
        let volLow = Infinity;
        let volHigh = -Infinity;
        for (let i = start; i < n; i++) {
            if (this.data.low[i] < low) low = this.data.low[i];
            if (this.data.high[i] > high) high = this.data.high[i];
            if (this.data.volume[i] < volLow) volLow = this.data.volume[i];
            if (this.data.volume[i] > volHigh) volHigh = this.data.volume[i];
        }
        this.priceScale = { min: low, max: high };
        this.volumeScale = { min: volLow, max: volHigh };
    }

    minMaxNormalize(array, { min, max } = this.priceScale) {
        const window = this.dataPoints >= array.length ? array : array.slice(-this.dataPoints);
        if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
            return window.map(() => 0.5);
        }
        return window.map(value => (value - min) / (max - min));
    }

    toPrice(normalizedValue) {
        const { min, max } = this.priceScale;
        return normalizedValue * (max - min) + min;
    }

    getBar(index) {
        const bar = this.normalized[index];
        if (!bar) return null;
        return {
            open: this.toPrice(bar[0]),
            high: this.toPrice(bar[1]),
            low: this.toPrice(bar[2]),
            close: this.toPrice(bar[3])
        };
    }

    combineData(normalizedData) {
        const { open, high, low, close, volume } = normalizedData;
        this.combined = open.map((_, i) => [open[i], high[i], low[i], close[i], volume[i]]);
        return this.combined;
    }

    normalizeData(dataPoints = this.dataPoints) {
        this.dataPoints = dataPoints ?? this.data.close.length;
        this.setScales();
        this.normalized = this.combineData({
            open: this.minMaxNormalize(this.data.open, this.priceScale),
            high: this.minMaxNormalize(this.data.high, this.priceScale),
            low: this.minMaxNormalize(this.data.low, this.priceScale),
            close: this.minMaxNormalize(this.data.close, this.priceScale),
            volume: this.minMaxNormalize(this.data.volume, this.volumeScale)
        });
        return this.normalized;
    }

    trimmedData() {
        const n = this.data.close.length;
        const points = this.dataPoints ?? n;
        if (points >= n) return this.data;
        const start = n - points;
        return {
            ...this.data,
            bars: points,
            time: this.data.time?.slice(start),
            open: this.data.open.slice(start),
            high: this.data.high.slice(start),
            low: this.data.low.slice(start),
            close: this.data.close.slice(start),
            volume: this.data.volume.slice(start)
        };
    }

    /**
     * Builds standardized model inputs from the audited causal feature set.
     * Price and macro features are built on the same trimmed window so state
     * and execution indices always line up. Statistics use the training region.
     */
    async buildFeatures(trainEnd, featureNames) {
        const trimmed = this.trimmedData();
        const price = buildFeatureMatrix(trimmed);
        const needsMacro = featureNames.some(name => MACRO_FEATURE_NAMES.includes(name));

        let macro = null;
        if (needsMacro) {
            macro = await buildMacroMatrix(trimmed, this.symbol);
            if (!macro.matrix) {
                throw new Error('Macro features requested, but macro data or valid bar timestamps are unavailable');
            }
            if (macro.firstValid === -1) {
                throw new Error('No bar has complete macro coverage; shorten the history with --data-points');
            }
        }

        const allNames = [...price.featureNames, ...(macro ? macro.names : [])];
        const n = trimmed.close.length;
        const full = new Array(n);

        for (let i = 0; i < n; i++) {
            if (macro) {
                if (!price.matrix[i] || !macro.matrix[i]) {
                    full[i] = null;
                } else {
                    full[i] = [...price.matrix[i], ...macro.matrix[i]];
                }
            } else {
                full[i] = price.matrix[i] ?? null;
            }
        }

        const firstValid = Math.max(price.warmup, macro ? macro.firstValid : 0);
        this.warmup = firstValid;
        this.stateStart = firstValid + this.lookback;
        this.featureNames = featureNames;
        this.macroFirstValid = macro ? macro.firstValid : null;

        const indices = featureNames.map(name => {
            const index = allNames.indexOf(name);
            if (index === -1) throw new Error(`Unknown feature "${name}"`);
            return index;
        });

        const end = Math.min(trainEnd, n);
        if (end <= firstValid + 2) {
            throw new Error(`Not enough bars after macro coverage (firstValid=${firstValid}, trainEnd=${end})`);
        }

        const means = new Array(featureNames.length).fill(0);
        const stds = new Array(featureNames.length).fill(0);
        let count = 0;
        for (let i = firstValid; i < end; i++) {
            if (!full[i]) continue;
            for (let f = 0; f < indices.length; f++) means[f] += full[i][indices[f]];
            count++;
        }
        if (count === 0) throw new Error('Not enough valid bars to build features');
        for (let f = 0; f < means.length; f++) means[f] /= count;

        for (let i = firstValid; i < end; i++) {
            if (!full[i]) continue;
            for (let f = 0; f < indices.length; f++) {
                stds[f] += (full[i][indices[f]] - means[f]) ** 2;
            }
        }
        for (let f = 0; f < stds.length; f++) stds[f] = Math.sqrt(stds[f] / count) || 1e-8;

        this.featureStats = { means, stds, warmup: firstValid, trainEnd: end };
        this.featureMatrix = new Array(n);
        for (let i = 0; i < n; i++) {
            if (i < firstValid || !full[i]) {
                this.featureMatrix[i] = null;
                continue;
            }
            this.featureMatrix[i] = indices.map((sourceIndex, f) =>
                clamp((full[i][sourceIndex] - means[f]) / stds[f], -5, 5)
            );
        }

        return this.featureMatrix;
    }

    getState(index) {
        if (!this.featureMatrix) {
            throw new Error('Features not built. Call buildFeatures() first.');
        }
        if (index < this.stateStart || index > this.featureMatrix.length) {
            throw new Error(`Index ${index} out of bounds`);
        }
        return this.featureMatrix.slice(index - this.lookback, index);
    }

    /** Actual (non-normalized) high-low range over `window` bars ending at index-1. */
    priceRangeAt(index, window = this.lookback) {
        let high = -Infinity;
        let low = Infinity;
        for (let i = index - window; i < index; i++) {
            const bar = this.getBar(i);
            if (!bar) continue;
            if (bar.high > high) high = bar.high;
            if (bar.low < low) low = bar.low;
        }
        return Number.isFinite(high) && Number.isFinite(low) ? high - low : 0;
    }
}
