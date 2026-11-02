import { join } from 'node:path';
import { LSTM } from './models/lstm.js';
import { DDPG } from './models/ddpg.js';
import { DataProcessor } from './utils/dataProcessor.js';
import { Adam, Dense, softUpdateParams, copyParams } from './utils/nn.js';
import { randn, clamp, setSeed } from './utils/rng.js';

const ROOT = import.meta.dir;

export const DEFAULT_CONFIG = {
    lookback: 60,
    // Audited causal features (see signalAudit.js / README)
    featureNames: [
        'ret1z', 'ret5z', 'ret20z', 'ret60z',
        'atrRel', 'volRatio',
        'dist20', 'dist50', 'slope20',
        'donchian20', 'rsi14', 'bollingerB'
    ],
    hiddenSize: 32,
    actionSize: 4,          // [direction, size, SL fraction, TP fraction]
    barsPredicted: 20,
    rangeWindow: 20,        // bars used for the volatility-based SL/TP distance
    dataPoints: null,       // null => use the full file
    gamma: 0.99,
    tau: 0.001,
    flatPenalty: 0.01,      // small cost for staying flat so it is not a free escape
    batchSize: 16,
    replaySize: 20000,
    trainFraction: 0.8,
    testEndFraction: null,  // walk-forward: end of the evaluation block (<1)
    embargoBars: null,      // gap between train and test; default = barsPredicted
    epochs: 30,
    stepsPerEpoch: 50,
    patience: 15,
    validationSteps: 100,
    minValidationTrades: 10,
    logEvery: 25,
    pretrainEpochs: 3,
    pretrainStepsPerEpoch: 50,
    pretrainBatchSize: 16,
    // Supervised actor initialization (behavior cloning) from the audit signal
    bcEpochs: 5,
    bcStepsPerEpoch: 50,
    bcBatchSize: 16,
    bcLr: 1e-3,
    teacherFeature: 'ret1z',
    teacherSign: -1,        // -1 = contrarian, +1 = momentum
    // Anchored fine-tuning: regularize the actor toward the BC policy and
    // keep the pretrained/BC encoder frozen by default.
    bcAnchor: 1.0,
    freezeEncoderAfterBc: true,
    // Optional regime gate on a standardized feature: trade only inside the band.
    gateFeature: null,
    gateMin: null,
    gateMax: null,
    // Fine-tuning the pretrained encoder slowly avoids destroying it.
    encoderLr: 1e-4,
    actorLr: 1e-3,
    criticLr: 1e-3,
    pretrainLr: 1e-3,
    clipNorm: 1.0,
    // Exploration / regularization
    noiseSigma: 0.2,
    minNoiseSigma: 0.05,
    noiseDecay: 0.98,
    targetNoise: 0.1,
    targetNoiseClip: 0.2,
    // Learning-rate schedule
    lrDecay: 0.99,
    minLr: 1e-5,
    // Costs
    spreadPoints: null,     // null => read from the data file
    seed: null,
    dataPath: join(ROOT, 'data', 'EURUSD_D1.json'),
    modelDir: join(ROOT, 'saved_models')
};

// Horizon-matched profiles: ~20 trading days per timeframe.
export const TIMEFRAMES = {
    D1: { dataPath: join(ROOT, 'data', 'EURUSD_D1.json'), barsPredicted: 20, rangeWindow: 20, teacherFeature: 'ret1z' },
    H4: { dataPath: join(ROOT, 'data', 'EURUSD_H4.json'), barsPredicted: 120, rangeWindow: 60, teacherFeature: 'bollingerB' },
    H1: { dataPath: join(ROOT, 'data', 'EURUSD_H1.json'), barsPredicted: 480, rangeWindow: 240, teacherFeature: 'ret20z' }
};

export function resolveConfig(args = {}) {
    const config = { ...DEFAULT_CONFIG };
    const timeframe = typeof args.tf === 'string' ? args.tf.toUpperCase() : null;
    if (timeframe) {
        const profile = TIMEFRAMES[timeframe];
        if (!profile) throw new Error(`Unknown timeframe "${args.tf}" (use D1, H4, or H1)`);
        Object.assign(config, profile);
        config.timeframe = timeframe;
    }
    if (args.data) config.dataPath = args.data;
    if (args.bars) config.barsPredicted = Number(args.bars);
    if (args['range-window']) config.rangeWindow = Number(args['range-window']);
    if (args['data-points']) config.dataPoints = Number(args['data-points']);
    if (args.features) config.featureNames = String(args.features).split(',');
    if (args['train-fraction']) config.trainFraction = Number(args['train-fraction']);
    if (args['test-end']) config.testEndFraction = Number(args['test-end']);
    if (args.embargo !== undefined) config.embargoBars = Number(args.embargo);
    if (args['gate-feature']) config.gateFeature = String(args['gate-feature']);
    if (args['gate-min'] !== undefined) config.gateMin = Number(args['gate-min']);
    if (args['gate-max'] !== undefined) config.gateMax = Number(args['gate-max']);
    if (args['model-dir']) config.modelDir = String(args['model-dir']);
    return config;
}

export class ReplayBuffer {
    constructor(maxSize = 100000) {
        this.buffer = [];
        this.maxSize = maxSize;
    }

    add(state, action, reward, nextState, done) {
        if (this.buffer.length >= this.maxSize) this.buffer.shift();
        this.buffer.push({ state, action, reward, nextState, done });
    }

    sample(batchSize) {
        if (this.buffer.length === 0) return [];
        const samples = [];
        for (let i = 0; i < Math.min(batchSize, this.buffer.length); i++) {
            samples.push(this.buffer[Math.floor(Math.random() * this.buffer.length)]);
        }
        return samples;
    }
}

export class TradingSystem {
    constructor(config = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        const cfg = this.config;

        const featureSize = cfg.featureNames.length;
        this.lstm = new LSTM(featureSize, cfg.hiddenSize);
        this.ddpg = new DDPG(cfg.hiddenSize, cfg.actionSize);
        this.targetLstm = new LSTM(featureSize, cfg.hiddenSize);
        this.targetDdpg = new DDPG(cfg.hiddenSize, cfg.actionSize);

        this.replayBuffer = new ReplayBuffer(cfg.replaySize);
        this.barsPredicted = cfg.barsPredicted;
        this.gamma = cfg.gamma;
        this.tau = cfg.tau;
        this.batchSize = cfg.batchSize;

        this.encoderOptimizer = new Adam({ lr: cfg.encoderLr, clipNorm: cfg.clipNorm });
        this.actorOptimizer = new Adam({ lr: cfg.actorLr, clipNorm: cfg.clipNorm });
        this.criticOptimizer = new Adam({ lr: cfg.criticLr, clipNorm: cfg.clipNorm });
        this.pretrainOptimizer = new Adam({ lr: cfg.pretrainLr, clipNorm: 5 });
        this.bcOptimizer = new Adam({ lr: cfg.bcLr, clipNorm: cfg.clipNorm });

        this.encoderHead = null;
        this.anchorActor = null;
        this.freezeEncoder = false;
        this.gate = this.config.gateFeature
            ? { feature: this.config.gateFeature, min: this.config.gateMin, max: this.config.gateMax }
            : null;
        this.stateGain = 1;
        this.noiseSigma = cfg.noiseSigma;
        this.ouState = new Array(cfg.actionSize).fill(0);
        this.spreadPrice = 0;

        this.dataProcessor = null;
        this.testStart = null;

        this.updateTargetNetworks(1.0);
    }

    get inputSize() {
        return this.config.lookback;
    }

    async loadData(path = this.config.dataPath) {
        const data = await Bun.file(path).json();
        this.dataProcessor = new DataProcessor(data, {
            lookback: this.config.lookback,
            dataPoints: this.config.dataPoints
        });
        this.dataProcessor.normalizeData(this.config.dataPoints ?? data.close.length);
        this.initializeSplit();
        this.dataProcessor.buildFeatures(this.trainEnd, this.config.featureNames);

        const point = Number.isFinite(data.point) ? data.point : 0.00001;
        const spreadPoints = this.config.spreadPoints ?? (Number.isFinite(data.spread) ? data.spread : 0);
        this.spreadPrice = spreadPoints * point;

        return data;
    }

    initializeSplit() {
        const length = this.dataProcessor.normalized.length;
        this.trainEnd = Math.floor(length * this.config.trainFraction);
        // Embargo prevents training labels from overlapping the evaluation block.
        this.embargo = Number.isFinite(this.config.embargoBars)
            ? this.config.embargoBars
            : this.barsPredicted;
        this.testStart = this.trainEnd + this.embargo;
        this.testEnd = Math.floor(length * (this.config.testEndFraction ?? 1));

        const minRequired = this.dataProcessor.lookback + this.barsPredicted + 1;
        if (this.trainEnd <= minRequired) {
            throw new Error(`Not enough training data (trainEnd=${this.trainEnd}, need > ${minRequired})`);
        }
        if (this.testEnd <= this.testStart + this.barsPredicted) {
            throw new Error(
                `Empty evaluation block: testStart=${this.testStart}, testEnd=${this.testEnd}, ` +
                `embargo=${this.embargo}`
            );
        }
    }

    get trainRange() {
        return {
            min: this.dataProcessor.stateStart,
            max: this.trainEnd - this.barsPredicted
        };
    }

    encode(state) {
        const { hidden } = this.lstm.forward(state);
        return hidden.map(value => value * this.stateGain);
    }

    applyExplorationNoise(vector) {
        const theta = 0.15;
        return vector.map((value, i) => {
            this.ouState[i] += -theta * this.ouState[i] + this.noiseSigma * randn();
            const noisy = value + this.ouState[i];
            return i === 0 ? clamp(noisy, -1, 1) : clamp(noisy, 0, 1);
        });
    }

    validateState(state) {
        if (!Array.isArray(state) || state.length !== this.inputSize) {
            throw new Error(`State must be an array of length ${this.inputSize}, received ${Array.isArray(state) ? state.length : typeof state}`);
        }
        for (const bar of state) {
            if (!Array.isArray(bar) || bar.length !== this.config.featureNames.length) {
                throw new Error(`Each state bar must have ${this.config.featureNames.length} features`);
            }
            if (bar.some(value => !Number.isFinite(value))) {
                throw new Error('Invalid state values detected');
            }
        }
    }

    isGateActive(state) {
        if (!this.gate) return true;
        const index = this.config.featureNames.indexOf(this.gate.feature);
        if (index === -1) return true;
        const value = state[state.length - 1][index];
        return value >= this.gate.min && value <= this.gate.max;
    }

    predict(state, addNoise = false, quiet = false) {
        this.validateState(state);

        const features = this.encode(state);
        if (this.gate && !addNoise && !this.isGateActive(state)) {
            return {
                positionSize: 0,
                direction: 0,
                size: 0,
                stopLossFrac: 0.5,
                takeProfitFrac: 0.5,
                actionVector: [0, 0.5, 0.5, 0.5],
                features,
                gated: true
            };
        }
        const policy = this.ddpg.policyForward(features);
        const vector = addNoise ? this.applyExplorationNoise(policy.vector) : [...policy.vector];
        const [direction, size, stopLossFrac, takeProfitFrac] = vector;

        let positionSize = direction * size;
        if (Math.abs(positionSize) < 0.02) {
            positionSize = 0;
        } else {
            positionSize = Number(positionSize.toFixed(2));
        }

        if (!quiet) {
            console.log('[dir] [size] [SLx] [TPx]', direction.toFixed(2), size.toFixed(2), stopLossFrac.toFixed(2), takeProfitFrac.toFixed(2));
        }

        return {
            positionSize,
            direction: Math.sign(positionSize),
            size: Math.abs(positionSize),
            stopLossFrac,
            takeProfitFrac,
            actionVector: vector,
            features
        };
    }

    simulateTrade(action, index) {
        const position = action?.positionSize ?? 0;
        if (!Number.isFinite(position) || Math.abs(position) < 0.02) return null;

        const dp = this.dataProcessor;
        const lastIndex = index + this.barsPredicted - 1;
        if (index - 1 < 0 || lastIndex >= dp.normalized.length) return null;

        const direction = Math.sign(position);
        const size = Math.abs(position);
        const entryBar = dp.getBar(index - 1);
        if (!entryBar) return null;
        const entry = entryBar.close;
        const range = dp.priceRangeAt(index, this.config.rangeWindow);
        if (!(entry > 0) || !(range > 0)) return null;

        // Stops/targets are a multiple of the recent price range, so risk is
        // always proportional to volatility instead of an arbitrary small value.
        const slMultiple = 0.5 + 3 * Math.abs(action.stopLossFrac);
        const tpMultiple = 0.5 + 4 * Math.abs(action.takeProfitFrac);
        const slDistance = range * slMultiple;
        const tpDistance = range * tpMultiple;
        if (!(slDistance > 0) || !(tpDistance > 0)) return null;

        const stopLoss = entry - direction * slDistance;
        const takeProfit = entry + direction * tpDistance;

        let exitPrice = null;
        let exitIndex = lastIndex;
        let reason = 'close';

        for (let i = index; i <= lastIndex; i++) {
            const bar = dp.getBar(i);
            if (!bar) break;

            // Pessimistic intrabar ordering: stop first, then target.
            const hitSl = direction > 0 ? bar.low <= stopLoss : bar.high >= stopLoss;
            if (hitSl) {
                exitPrice = stopLoss;
                exitIndex = i;
                reason = 'sl';
                break;
            }

            const hitTp = direction > 0 ? bar.high >= takeProfit : bar.low <= takeProfit;
            if (hitTp) {
                exitPrice = takeProfit;
                exitIndex = i;
                reason = 'tp';
                break;
            }
        }

        if (exitPrice === null) exitPrice = dp.getBar(lastIndex).close;

        const pnlPct = direction * (exitPrice - entry) / entry;
        const pnlR = direction * (exitPrice - entry) / slDistance;
        const costPct = this.spreadPrice / entry;
        const costR = this.spreadPrice / slDistance;

        return {
            entry,
            exit: exitPrice,
            stopLoss,
            takeProfit,
            direction,
            size,
            pnlPct,
            pnlR,
            costPct,
            costR,
            netR: pnlR - costR,
            exitIndex,
            barsHeld: exitIndex - index + 1,
            reason,
            range,
            slMultiple,
            tpMultiple
        };
    }

    rewardFromTrade(trade) {
        // Risk-normalized return minus trading cost, scaled by position size.
        // Bounded, no loophole for wide stops or oversized positions.
        return clamp(Math.tanh(trade.netR * 0.5) * trade.size, -1, 1);
    }

    calculateReward(action, index) {
        if (!action || action.positionSize === 0) return -this.config.flatPenalty;
        const trade = this.simulateTrade(action, index);
        if (!trade) return 0;
        return this.rewardFromTrade(trade);
    }

    getRandomIndex() {
        const { min, max } = this.trainRange;
        if (max <= min) throw new Error(`Invalid training range [${min}, ${max}]`);
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }

    getValidationIndices(steps) {
        const { min, max } = this.trainRange;
        const start = min + this.barsPredicted;
        const end = max;
        if (end <= start) return [];
        const stride = Math.max(1, Math.floor((end - start) / steps));
        const indices = [];
        for (let i = start; i <= end; i += stride) indices.push(i);
        return indices;
    }

    async calibrateStateGain(samples = 100) {
        const indices = this.getValidationIndices(samples);
        const values = [];
        for (const index of indices) {
            try {
                const state = await this.dataProcessor.getState(index);
                const { hidden } = this.lstm.forward(state);
                values.push(...hidden);
            } catch {
                continue;
            }
        }
        if (values.length === 0) {
            this.stateGain = 1;
            return this.stateGain;
        }
        const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
        const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
        const std = Math.sqrt(variance) || 1;
        this.stateGain = clamp(1 / std, 1, 100);
        return this.stateGain;
    }

    async pretrain({ epochs = this.config.pretrainEpochs, stepsPerEpoch = this.config.pretrainStepsPerEpoch } = {}) {
        if (epochs <= 0) {
            await this.calibrateStateGain();
            console.log(`Skipping pretrain. State gain calibrated to ${this.stateGain.toFixed(3)}`);
            return;
        }

        this.encoderHead = new Dense(this.config.hiddenSize, 1, 'linear', { scale: 0.1 });
        const { min, max } = this.trainRange;
        const batchSize = this.config.pretrainBatchSize;
        const dp = this.dataProcessor;

        console.log(`Pretraining encoder (${epochs} epochs x ${stepsPerEpoch} steps) on ${this.barsPredicted}-bar direction`);

        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;
            let correct = 0;
            let count = 0;

            for (let step = 0; step < stepsPerEpoch; step++) {
                this.lstm.zeroGrad();
                this.encoderHead.zeroGrad();
                let used = 0;

                for (let b = 0; b < batchSize; b++) {
                    const index = min + 1 + Math.floor(Math.random() * (max - min));
                    try {
                        const state = await dp.getState(index);
                        const entry = dp.getBar(index - 1).close;
                        const exit = dp.getBar(index + this.barsPredicted - 1).close;
                        if (!(entry > 0) || !(exit > 0)) continue;

                        const target = exit > entry ? 1 : 0;
                        const { hidden } = this.lstm.forward(state);
                        const logit = this.encoderHead.forward(hidden)[0];
                        const probability = 1 / (1 + Math.exp(-clamp(logit, -30, 30)));

                        totalLoss += -(target * Math.log(Math.max(1e-7, probability)) + (1 - target) * Math.log(Math.max(1e-7, 1 - probability)));
                        if ((probability > 0.5 ? 1 : 0) === target) correct++;
                        count++;
                        used++;

                        const dLogit = (probability - target) / batchSize;
                        const dHidden = this.encoderHead.backward([dLogit], true);
                        this.lstm.backward(dHidden);
                    } catch {
                        continue;
                    }
                }

                if (used === 0) continue;
                this.pretrainOptimizer.step(
                    [...this.lstm.parameters(), ...this.encoderHead.parameters()],
                    [...this.lstm.gradients(), ...this.encoderHead.gradients()]
                );
            }

            const averageLoss = count > 0 ? totalLoss / count : 0;
            const accuracy = count > 0 ? correct / count : 0;
            console.log(`[pretrain ${epoch + 1}/${epochs}] bce=${averageLoss.toFixed(4)} acc=${(accuracy * 100).toFixed(1)}%`);
        }

        await this.calibrateStateGain();
        console.log(`State gain calibrated to ${this.stateGain.toFixed(3)}`);
    }

    async behaviorClone({ epochs = this.config.bcEpochs, steps = this.config.bcStepsPerEpoch } = {}) {
        if (epochs <= 0) return;

        const teacherIndex = this.config.featureNames.indexOf(this.config.teacherFeature);
        if (teacherIndex === -1) {
            throw new Error(`Teacher feature "${this.config.teacherFeature}" is not in the feature set`);
        }

        const { min, max } = this.trainRange;
        const batchSize = this.config.bcBatchSize;
        const sign = this.config.teacherSign;

        console.log(`Behavior cloning actor (${epochs} epochs x ${steps} steps), teacher=${this.config.teacherFeature} sign=${sign}`);

        for (let epoch = 0; epoch < epochs; epoch++) {
            let correct = 0;
            let total = 0;
            let squaredError = 0;

            for (let step = 0; step < steps; step++) {
                this.ddpg.zeroActorGrad();
                let used = 0;

                for (let b = 0; b < batchSize; b++) {
                    const index = min + 1 + Math.floor(Math.random() * (max - min));
                    try {
                        const state = await this.dataProcessor.getState(index);
                        const teacherValue = state[state.length - 1][teacherIndex];
                        const active = this.isGateActive(state);
                        let target;
                        let countAccuracy = false;

                        if (active) {
                            target = sign * Math.sign(teacherValue);
                            if (target === 0) continue;
                            countAccuracy = true;
                        } else {
                            // Outside the regime gate the cloned policy learns to stay flat.
                            if (!this.gate) continue;
                            target = 0;
                        }

                        const policy = this.ddpg.policyForward(this.encode(state));
                        const predicted = policy.direction;

                        if (countAccuracy) {
                            if (Math.sign(predicted) === target) correct++;
                            total++;
                        }
                        squaredError += (predicted - target) ** 2;

                        const dDirection = (predicted - target) / batchSize;
                        this.ddpg.policyBackward([dDirection, 0, 0, 0], true);
                        used++;
                    } catch {
                        continue;
                    }
                }

                if (used === 0) continue;
                this.bcOptimizer.step(this.ddpg.actorParameters(), this.ddpg.actorGradients());
            }

            const accuracy = total > 0 ? correct / total : 0;
            const mse = total > 0 ? squaredError / total : 0;
            console.log(`[bc ${epoch + 1}/${epochs}] teacherAcc=${(accuracy * 100).toFixed(1)}% mse=${mse.toFixed(4)}`);
        }

        const evaluation = await this.evaluateTeacher();
        console.log(`Behavior-clone validation: teacherAcc=${(evaluation.accuracy * 100).toFixed(1)}% over ${evaluation.total} windows`);
    }

    async evaluateTeacher(steps = this.config.validationSteps) {
        const teacherIndex = this.config.featureNames.indexOf(this.config.teacherFeature);
        const sign = this.config.teacherSign;
        let correct = 0;
        let total = 0;

        for (const index of this.getValidationIndices(steps)) {
            try {
                const state = await this.dataProcessor.getState(index);
                const target = sign * Math.sign(state[state.length - 1][teacherIndex]);
                if (target === 0) continue;
                const policy = this.ddpg.policyForward(this.encode(state));
                if (Math.sign(policy.direction) === target) correct++;
                total++;
            } catch {
                continue;
            }
        }

        return { accuracy: total > 0 ? correct / total : 0, total };
    }

    async validate(steps = this.config.validationSteps) {
        let sumR = 0;
        let sumPnl = 0;
        let grossProfit = 0;
        let grossLoss = 0;
        let trades = 0;
        let equity = 1;
        let peak = 1;
        let maxDrawdown = 0;
        const windows = this.getValidationIndices(steps);

        for (const index of windows) {
            try {
                const state = await this.dataProcessor.getState(index);
                const action = this.predict(state, false, true);
                const trade = this.simulateTrade(action, index);
                if (!trade) continue;

                const netPnl = trade.size * (trade.pnlPct - trade.costPct);
                equity *= 1 + netPnl;
                peak = Math.max(peak, equity);
                maxDrawdown = Math.min(maxDrawdown, equity / peak - 1);

                sumR += trade.netR;
                sumPnl += netPnl;
                if (netPnl > 0) grossProfit += netPnl;
                else grossLoss += Math.abs(netPnl);
                trades++;
            } catch {
                continue;
            }
        }

        const avgR = trades > 0 ? sumR / trades : 0;
        // Expectancy is the objective; drawdown is only a light tie-breaker.
        const score = trades >= this.config.minValidationTrades
            ? avgR - 0.1 * Math.abs(maxDrawdown)
            : -Infinity;

        return {
            score,
            avgR,
            avgPnl: trades > 0 ? sumPnl / trades : 0,
            profitFactor: grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? Infinity : 0),
            maxDrawdown,
            trades,
            windows: windows.length
        };
    }

    async train({ epochs = this.config.epochs, stepsPerEpoch = this.config.stepsPerEpoch, save = true, prefix = '' } = {}) {
        const modelName = (name) => (prefix ? `${prefix}_${name}` : name);

        await this.loadData();
        const { min, max } = this.trainRange;
        console.log(`Loaded ${this.dataProcessor.data.close.length} bars (normalized ${this.dataProcessor.normalized.length})`);
        console.log(`Train index range [${min}, ${max}], test starts at ${this.testStart}`);
        console.log(`Features: ${this.config.featureNames.length} (${this.config.featureNames.join(',')})`);
        console.log(`Horizon: ${this.barsPredicted} bars, range window: ${this.config.rangeWindow}, spread cost: ${this.spreadPrice.toExponential(2)}`);

        const didPretrain = this.config.pretrainEpochs > 0;
        await this.pretrain();

        if (save && didPretrain) {
            await this.saveModels(modelName('pretrain_model'), { stage: 'pretrain' });
        }

        this.anchorActor = null;
        this.freezeEncoder = false;
        if (this.config.bcEpochs > 0) {
            await this.behaviorClone();
            if (save) {
                await this.saveModels(modelName('bc_model'), { stage: 'behavior-clone' });
            }
            // Freeze a copy of the cloned actor as the anchor for fine-tuning.
            this.anchorActor = new DDPG(this.config.hiddenSize, this.config.actionSize);
            copyParams(this.anchorActor.actorParameters(), this.ddpg.actorParameters());
            this.freezeEncoder = this.config.freezeEncoderAfterBc === true;
        }

        console.log(`Training DDPG ${epochs} epochs x ${stepsPerEpoch} steps, batch ${this.batchSize}`);

        let bestScore = -Infinity;
        let bestValidation = null;
        let noImprovementCount = 0;
        let updates = 0;

        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalReward = 0;
            let validSteps = 0;
            let tradedSteps = 0;

            for (let step = 0; step < stepsPerEpoch; step++) {
                try {
                    const startIdx = this.getRandomIndex();
                    const currentState = await this.dataProcessor.getState(startIdx);
                    const action = this.predict(currentState, true, true);
                    const reward = this.calculateReward(action, startIdx);

                    if (!Number.isFinite(reward)) continue;

                    totalReward += reward;
                    validSteps++;
                    if (action.positionSize !== 0) tradedSteps++;

                    const nextState = await this.dataProcessor.getState(startIdx + this.barsPredicted - 1);
                    this.replayBuffer.add(
                        currentState,
                        action.actionVector,
                        reward,
                        nextState,
                        false
                    );

                    if (this.replayBuffer.buffer.length >= this.batchSize) {
                        const loss = await this.trainStep();
                        updates++;
                        if (this.config.logEvery > 0 && (step + 1) % this.config.logEvery === 0) {
                            console.log(`  epoch ${epoch} step ${step + 1}: criticLoss=${loss.toFixed(4)} buffer=${this.replayBuffer.buffer.length} sigma=${this.noiseSigma.toFixed(3)}`);
                        }
                    }
                } catch (error) {
                    console.error(`Error at epoch ${epoch}, step ${step}:`, error.message);
                }
            }

            const validation = await this.validate();
            const averageReward = validSteps > 0 ? totalReward / validSteps : 0;

            if (validation.score > bestScore) {
                bestScore = validation.score;
                bestValidation = validation;
                noImprovementCount = 0;
                if (save) await this.saveModels(modelName('best_model'), { epoch: epoch + 1, validation });
            } else {
                noImprovementCount++;
            }

            console.log(
                `[epoch ${epoch + 1}/${epochs}] reward=${averageReward.toFixed(4)} ` +
                `valid=${validSteps} traded=${tradedSteps} updates=${updates} ` +
                `valR=${validation.avgR.toFixed(4)} valPF=${formatNumber(validation.profitFactor)} ` +
                `valDD=${(validation.maxDrawdown * 100).toFixed(2)}% valTrades=${validation.trades} ` +
                `score=${formatNumber(validation.score)} best=${formatNumber(bestScore)}`
            );

            if (save && (epoch + 1) % 10 === 0) {
                await this.saveModels(modelName(`models_epoch_${epoch + 1}`), { epoch: epoch + 1, validation });
            }

            // Exploration and learning-rate schedules
            this.noiseSigma = Math.max(this.config.minNoiseSigma, this.noiseSigma * this.config.noiseDecay);
            this.actorOptimizer.lr = Math.max(this.config.minLr, this.actorOptimizer.lr * this.config.lrDecay);
            this.criticOptimizer.lr = Math.max(this.config.minLr, this.criticOptimizer.lr * this.config.lrDecay);
            this.encoderOptimizer.lr = Math.max(this.config.minLr, this.encoderOptimizer.lr * this.config.lrDecay);

            if (noImprovementCount >= this.config.patience) {
                console.log(`Early stopping at epoch ${epoch + 1}`);
                break;
            }
        }

        if (save) await this.saveModels(modelName('final_model'), { validation: bestValidation });
        return { bestScore, bestValidation };
    }

    async trainStep() {
        const batch = this.replayBuffer.sample(this.batchSize);
        if (batch.length === 0) return 0;

        const N = batch.length;
        const gain = this.stateGain;
        let criticLoss = 0;
        let used = 0;

        // Pass 1: critic regression to the (smoothed) target Q
        this.ddpg.zeroCriticGrad();
        for (const experience of batch) {
            const { state, action, reward, nextState, done } = experience;
            try {
                const nextFeatures = this.targetLstm.forward(nextState).hidden.map(v => v * gain);
                const nextPolicy = this.targetDdpg.policyForward(nextFeatures);

                // Target policy smoothing (TD3-style regularizer / entropy proxy)
                const smoothAction = nextPolicy.vector.map((value, i) => {
                    const noise = clamp(randn() * this.config.targetNoise, -this.config.targetNoiseClip, this.config.targetNoiseClip);
                    return i === 0 ? clamp(value + noise, -1, 1) : clamp(value + noise, 0, 1);
                });
                const nextQ = this.targetDdpg.criticForward(nextFeatures, smoothAction);
                const targetQ = reward + (done ? 0 : this.gamma * Math.max(-10, Math.min(10, nextQ)));
                if (!Number.isFinite(targetQ)) continue;

                const features = this.lstm.forward(state).hidden.map(v => v * gain);
                const currentQ = this.ddpg.criticForward(features, action);
                if (!Number.isFinite(currentQ)) continue;

                const tdError = targetQ - currentQ;
                const dQ = clamp(2 * tdError / N, -1, 1);

                this.ddpg.criticBackward(dQ, true);
                criticLoss += tdError * tdError;
                used++;
            } catch (error) {
                console.error('Critic step error:', error.message);
            }
        }

        if (used === 0) {
            this.ddpg.zeroCriticGrad();
            return 0;
        }

        this.criticOptimizer.step(this.ddpg.criticParameters(), this.ddpg.criticGradients());
        this.ddpg.zeroCriticGrad();

        // Pass 2: deterministic policy gradient through the critic + BPTT
        this.ddpg.zeroActorGrad();
        this.lstm.zeroGrad();
        for (const experience of batch) {
            const { state } = experience;
            try {
                const { hidden } = this.lstm.forward(state);
                const features = hidden.map(v => v * gain);
                const policy = this.ddpg.policyForward(features);
                const actionVector = policy.vector;
                this.ddpg.criticForward(features, actionVector);

                const dQ = -1 / N; // maximise Q
                const { dState, dAction } = this.ddpg.criticBackward(dQ, false);

                // Anchor the policy to the behavior-cloned actor.
                let dVector = dAction;
                if (this.anchorActor) {
                    const anchorVector = this.anchorActor.actorForward(features);
                    dVector = dAction.map((value, i) =>
                        value + 2 * this.config.bcAnchor * (actionVector[i] - anchorVector[i]) / N
                    );
                }

                const dFeaturesActor = this.ddpg.policyBackward(dVector, true);
                if (!this.freezeEncoder) {
                    // dL/dh = gain * (-dQ/ds (direct) + path through actor)
                    const dHidden = hidden.map((_, i) => gain * (-dState[i] + dFeaturesActor[i]));
                    this.lstm.backward(dHidden);
                }
            } catch (error) {
                console.error('Actor step error:', error.message);
            }
        }

        this.actorOptimizer.step(this.ddpg.actorParameters(), this.ddpg.actorGradients());
        this.ddpg.zeroActorGrad();

        if (!this.freezeEncoder) {
            this.encoderOptimizer.step(this.lstm.parameters(), this.lstm.gradients());
            this.lstm.zeroGrad();
        }

        this.updateTargetNetworks(this.tau);

        return criticLoss / used;
    }

    updateTargetNetworks(tau) {
        softUpdateParams(this.targetLstm.parameters(), this.lstm.parameters(), tau);
        softUpdateParams(this.targetDdpg.allParameters(), this.ddpg.allParameters(), tau);
    }

    async saveModels(filename, meta = null) {
        const modelData = {
            lstm: this.lstm.toJSON(),
            ddpg: this.ddpg.toJSON(),
            meta: {
                savedAt: new Date().toISOString(),
                timeframe: this.config.timeframe ?? null,
                lookback: this.config.lookback,
                featureNames: this.config.featureNames,
                hiddenSize: this.config.hiddenSize,
                actionSize: this.config.actionSize,
                barsPredicted: this.barsPredicted,
                rangeWindow: this.config.rangeWindow,
                dataPoints: this.config.dataPoints,
                trainFraction: this.config.trainFraction,
                testEndFraction: this.config.testEndFraction,
                trainEnd: this.trainEnd,
                embargo: this.embargo,
                freezeEncoder: this.freezeEncoder,
                bcAnchor: this.config.bcAnchor,
                gate: this.gate,
                testStart: this.testStart,
                stateGain: this.stateGain,
                seed: this.config.seed,
                spreadPrice: this.spreadPrice,
                ...meta
            }
        };

        await Bun.write(join(this.config.modelDir, `${filename}.json`), JSON.stringify(modelData));
    }

    async loadModels(filename) {
        const path = join(this.config.modelDir, `${filename}.json`);
        const modelData = await Bun.file(path).json();

        if (modelData.lstm.inputSize !== this.config.featureNames.length) {
            throw new Error(
                `Checkpoint "${filename}" expects ${modelData.lstm.inputSize} features per bar, ` +
                `config uses ${this.config.featureNames.length}. Retrain for this feature set.`
            );
        }
        if (modelData.meta?.featureNames && modelData.meta.featureNames.join(',') !== this.config.featureNames.join(',')) {
            throw new Error(
                `Checkpoint "${filename}" uses a different feature set. ` +
                `Pass the matching --tf/--features or retrain.`
            );
        }
        if (modelData.meta?.lookback && modelData.meta.lookback !== this.config.lookback) {
            throw new Error(
                `Checkpoint "${filename}" expects lookback=${modelData.meta.lookback}, config uses ${this.config.lookback}.`
            );
        }
        if (modelData.meta?.barsPredicted && modelData.meta.barsPredicted !== this.barsPredicted) {
            throw new Error(
                `Checkpoint "${filename}" expects barsPredicted=${modelData.meta.barsPredicted}, ` +
                `config uses ${this.barsPredicted}. Pass the matching --tf/--bars.`
            );
        }
        if (modelData.meta?.rangeWindow && modelData.meta.rangeWindow !== this.config.rangeWindow) {
            throw new Error(
                `Checkpoint "${filename}" expects rangeWindow=${modelData.meta.rangeWindow}, ` +
                `config uses ${this.config.rangeWindow}. Pass the matching --tf/--range-window.`
            );
        }
        if (modelData.ddpg.stateSize !== this.config.hiddenSize || modelData.ddpg.actionSize !== this.config.actionSize) {
            throw new Error(
                `Checkpoint sizes (state=${modelData.ddpg.stateSize}, action=${modelData.ddpg.actionSize}) do not match ` +
                `config (state=${this.config.hiddenSize}, action=${this.config.actionSize}).`
            );
        }

        this.lstm = LSTM.fromJSON(modelData.lstm);
        this.ddpg = DDPG.fromJSON(modelData.ddpg);
        this.targetLstm = LSTM.fromJSON(modelData.lstm);
        this.targetDdpg = DDPG.fromJSON(modelData.ddpg);
        copyParams(this.targetLstm.parameters(), this.lstm.parameters());
        copyParams(this.targetDdpg.allParameters(), this.ddpg.allParameters());
        this.stateGain = modelData.meta?.stateGain ?? 1;
        if (!this.config.gateFeature && modelData.meta?.gate) {
            this.gate = modelData.meta.gate;
        }
        if (Number.isFinite(modelData.meta?.spreadPrice)) this.spreadPrice = modelData.meta.spreadPrice;

        return modelData;
    }
}

function formatNumber(value) {
    if (value === Infinity) return 'inf';
    if (value === -Infinity) return '-inf';
    return value.toFixed(3);
}

function parseArgs(argv) {
    const options = {};
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (!arg.startsWith('--')) continue;
        const key = arg.slice(2);
        if (key === 'no-save' || key === 'no-pretrain') {
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

async function main() {
    const args = parseArgs(process.argv.slice(2));
    const config = resolveConfig(args);
    if (args.epochs) config.epochs = Number(args.epochs);
    if (args.steps) config.stepsPerEpoch = Number(args.steps);
    if (args.batch) config.batchSize = Number(args.batch);
    if (args.hidden) config.hiddenSize = Number(args.hidden);
    if (args['pretrain-epochs'] !== undefined) config.pretrainEpochs = Number(args['pretrain-epochs']);
    if (args['pretrain-steps'] !== undefined) config.pretrainStepsPerEpoch = Number(args['pretrain-steps']);
    if (args['no-pretrain']) config.pretrainEpochs = 0;
    if (args['bc-epochs'] !== undefined) config.bcEpochs = Number(args['bc-epochs']);
    if (args['bc-steps'] !== undefined) config.bcStepsPerEpoch = Number(args['bc-steps']);
    if (args['no-bc']) config.bcEpochs = 0;
    if (args.teacher) config.teacherFeature = String(args.teacher);
    if (args['teacher-sign'] !== undefined) config.teacherSign = Number(args['teacher-sign']);
    if (args['bc-anchor'] !== undefined) config.bcAnchor = Number(args['bc-anchor']);
    if (args['no-freeze-encoder']) config.freezeEncoderAfterBc = false;
    if (args['actor-lr']) config.actorLr = Number(args['actor-lr']);
    if (args['critic-lr']) config.criticLr = Number(args['critic-lr']);
    if (args['encoder-lr']) config.encoderLr = Number(args['encoder-lr']);
    if (args['pretrain-lr']) config.pretrainLr = Number(args['pretrain-lr']);
    if (args.sigma) config.noiseSigma = Number(args.sigma);
    if (args['min-sigma']) config.minNoiseSigma = Number(args['min-sigma']);
    if (args['noise-decay']) config.noiseDecay = Number(args['noise-decay']);
    if (args.gamma) config.gamma = Number(args.gamma);
    if (args.seed !== undefined) {
        config.seed = Number(args.seed);
        setSeed(config.seed);
    }

    const trader = new TradingSystem(config);
    await trader.train({
        epochs: config.epochs,
        stepsPerEpoch: config.stepsPerEpoch,
        save: args.save !== false,
        prefix: typeof args.prefix === 'string' ? args.prefix : ''
    });
}

if (import.meta.main) {
    main().catch(error => {
        console.error(error);
        process.exit(1);
    });
}
