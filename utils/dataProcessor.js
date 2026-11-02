import { clamp } from './rng.js';

export class DataProcessor {
    constructor(data, { lookback = 60, dataPoints = 4000 } = {}) {
        this.data = data;
        this.lookback = lookback;
        this.dataPoints = dataPoints;
        this.features = ['open', 'high', 'low', 'close', 'volume'];
        this.normalized = null;
        this.combined = null;
        this.priceScale = { min: 0, max: 1 };
        this.volumeScale = { min: 0, max: 1 };
        this.featureMatrix = null;
        this.featureStats = null;
    }

    setScales() {
        const start = -this.dataPoints;
        this.priceScale = {
            min: Math.min(...this.data.low.slice(start)),
            max: Math.max(...this.data.high.slice(start))
        };
        this.volumeScale = {
            min: Math.min(...this.data.volume.slice(start)),
            max: Math.max(...this.data.volume.slice(start))
        };
    }

    minMaxNormalize(array, { min, max } = this.priceScale) {
        const window = array.slice(-this.dataPoints);
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
        this.dataPoints = dataPoints;
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

    /**
     * Builds the model input: per-bar log return + relative range, standardized
     * with statistics from the training region only (no test leakage).
     */
    buildFeatures(trainEnd) {
        const n = this.normalized.length;
        const raw = new Array(n);

        for (let i = 0; i < n; i++) {
            const bar = this.getBar(i);
            const prevClose = i > 0 ? this.getBar(i - 1).close : bar.close;
            const logReturn = i > 0 ? Math.log(bar.close / prevClose) : 0;
            const relativeRange = bar.close > 0 ? (bar.high - bar.low) / bar.close : 0;
            raw[i] = [logReturn, relativeRange];
        }

        const end = Math.max(2, Math.min(trainEnd ?? n, n));
        const means = [0, 0];
        for (let i = 0; i < end; i++) {
            means[0] += raw[i][0];
            means[1] += raw[i][1];
        }
        means[0] /= end;
        means[1] /= end;

        const variances = [0, 0];
        for (let i = 0; i < end; i++) {
            variances[0] += (raw[i][0] - means[0]) ** 2;
            variances[1] += (raw[i][1] - means[1]) ** 2;
        }
        const stds = [
            Math.sqrt(variances[0] / end) || 1e-8,
            Math.sqrt(variances[1] / end) || 1e-8
        ];

        this.featureStats = { means, stds, trainEnd: end };
        this.featureMatrix = raw.map(([logReturn, relativeRange]) => [
            clamp((logReturn - means[0]) / stds[0], -5, 5),
            clamp((relativeRange - means[1]) / stds[1], -5, 5)
        ]);

        return this.featureMatrix;
    }

    getState(index) {
        if (!this.featureMatrix) {
            throw new Error('Features not built. Call buildFeatures() first.');
        }
        if (index < this.lookback || index > this.featureMatrix.length) {
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

    calculateRange(window) {
        if (!Array.isArray(window) || window.length === 0) return 0;
        return Math.max(...window) - Math.min(...window);
    }
}
