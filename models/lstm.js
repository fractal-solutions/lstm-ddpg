import { zeros, matVec, matTVec, outer, clipVector } from '../utils/nn.js';

const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, x))));

// Recurrent LSTM encoder. `forward` consumes a sequence (one vector per bar)
// and `backward` performs truncated backpropagation through time from the
// final hidden state. Gradients accumulate until `zeroGrad()` is called.
export class LSTM {
    constructor(inputSize = 1, hiddenSize = 32) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        const inputScale = Math.sqrt(1 / inputSize);
        const recurrentScale = Math.sqrt(1 / hiddenSize);
        const createMatrix = (rows, cols, scale) =>
            Array.from({ length: rows }, () =>
                Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale));

        // Input weights
        this.Wf = createMatrix(hiddenSize, inputSize, inputScale);
        this.Wi = createMatrix(hiddenSize, inputSize, inputScale);
        this.Wc = createMatrix(hiddenSize, inputSize, inputScale);
        this.Wo = createMatrix(hiddenSize, inputSize, inputScale);

        // Recurrent weights
        this.Uf = createMatrix(hiddenSize, hiddenSize, recurrentScale);
        this.Ui = createMatrix(hiddenSize, hiddenSize, recurrentScale);
        this.Uc = createMatrix(hiddenSize, hiddenSize, recurrentScale);
        this.Uo = createMatrix(hiddenSize, hiddenSize, recurrentScale);

        // Biases (forget gate starts open)
        this.bf = new Array(hiddenSize).fill(1);
        this.bi = zeros(hiddenSize);
        this.bc = zeros(hiddenSize);
        this.bo = zeros(hiddenSize);

        this.gradWf = createMatrix(hiddenSize, inputSize, 0);
        this.gradWi = createMatrix(hiddenSize, inputSize, 0);
        this.gradWc = createMatrix(hiddenSize, inputSize, 0);
        this.gradWo = createMatrix(hiddenSize, inputSize, 0);
        this.gradUf = createMatrix(hiddenSize, hiddenSize, 0);
        this.gradUi = createMatrix(hiddenSize, hiddenSize, 0);
        this.gradUc = createMatrix(hiddenSize, hiddenSize, 0);
        this.gradUo = createMatrix(hiddenSize, hiddenSize, 0);
        this.gradBf = zeros(hiddenSize);
        this.gradBi = zeros(hiddenSize);
        this.gradBc = zeros(hiddenSize);
        this.gradBo = zeros(hiddenSize);

        this.steps = null;
    }

    forward(sequence) {
        if (!Array.isArray(sequence) || sequence.length === 0) {
            throw new Error('LSTM.forward expects a non-empty sequence');
        }

        let h = zeros(this.hiddenSize);
        let c = zeros(this.hiddenSize);
        const steps = [];

        for (const rawInput of sequence) {
            const x = Array.isArray(rawInput) ? rawInput : [rawInput];
            if (x.length !== this.inputSize) {
                throw new Error(`LSTM expected input of size ${this.inputSize}, got ${x.length}`);
            }

            const hPrev = h;
            const cPrev = c;

            const preF = this.gatePreActivation(this.Wf, this.Uf, this.bf, x, hPrev);
            const f = preF.map(sigmoid);
            const preI = this.gatePreActivation(this.Wi, this.Ui, this.bi, x, hPrev);
            const i = preI.map(sigmoid);
            const preG = this.gatePreActivation(this.Wc, this.Uc, this.bc, x, hPrev);
            const g = preG.map(Math.tanh);
            const preO = this.gatePreActivation(this.Wo, this.Uo, this.bo, x, hPrev);
            const o = preO.map(sigmoid);

            c = f.map((fv, idx) => fv * cPrev[idx] + i[idx] * g[idx]);
            const tanhC = c.map(Math.tanh);
            h = o.map((ov, idx) => ov * tanhC[idx]);

            steps.push({ x, hPrev, cPrev, f, i, g, o, c, tanhC, h });
        }

        this.steps = steps;
        return {
            hidden: h,
            cell: c,
            hiddenStates: steps.map(step => step.h),
            sequenceLength: steps.length
        };
    }

    gatePreActivation(W, U, b, x, hPrev) {
        const wx = matVec(W, x);
        const uh = matVec(U, hPrev);
        return wx.map((v, idx) => v + uh[idx] + b[idx]);
    }

    backward(dHiddenFinal) {
        if (!this.steps) throw new Error('LSTM.backward called before forward');
        if (dHiddenFinal.length !== this.hiddenSize) {
            throw new Error(`dHiddenFinal must have length ${this.hiddenSize}`);
        }

        const T = this.steps.length;
        let dh = clipVector(dHiddenFinal, 10);
        let dc = zeros(this.hiddenSize);

        for (let t = T - 1; t >= 0; t--) {
            const s = this.steps[t];
            const { x, hPrev, cPrev, f, i, g, o, tanhC } = s;

            // Output gate
            const dO = dh.map((dv, idx) => dv * tanhC[idx] * o[idx] * (1 - o[idx]));

            // Cell state (from h_t plus the future cell gradient)
            const dCFromH = dh.map((dv, idx) => dv * o[idx] * (1 - tanhC[idx] * tanhC[idx]));
            dc = dc.map((dv, idx) => dv + dCFromH[idx]);

            const dF = dc.map((dv, idx) => dv * cPrev[idx] * f[idx] * (1 - f[idx]));
            const dI = dc.map((dv, idx) => dv * g[idx] * i[idx] * (1 - i[idx]));
            const dG = dc.map((dv, idx) => dv * i[idx] * (1 - g[idx] * g[idx]));

            this.accumulateGateGradients(this.gradWf, this.gradUf, this.gradBf, dF, x, hPrev);
            this.accumulateGateGradients(this.gradWi, this.gradUi, this.gradBi, dI, x, hPrev);
            this.accumulateGateGradients(this.gradWc, this.gradUc, this.gradBc, dG, x, hPrev);
            this.accumulateGateGradients(this.gradWo, this.gradUo, this.gradBo, dO, x, hPrev);

            // Propagate to the previous time step
            const dhPrev = zeros(this.hiddenSize);
            this.addTransposedProduct(dhPrev, this.Uf, dF);
            this.addTransposedProduct(dhPrev, this.Ui, dI);
            this.addTransposedProduct(dhPrev, this.Uc, dG);
            this.addTransposedProduct(dhPrev, this.Uo, dO);
            dh = clipVector(dhPrev, 10);
            dc = dc.map((dv, idx) => dv * f[idx]);
        }
    }

    accumulateGateGradients(gradW, gradU, gradB, delta, x, hPrev) {
        for (let i = 0; i < this.hiddenSize; i++) {
            gradB[i] += delta[i];
            const rowW = gradW[i];
            for (let j = 0; j < this.inputSize; j++) rowW[j] += delta[i] * x[j];
            const rowU = gradU[i];
            for (let j = 0; j < this.hiddenSize; j++) rowU[j] += delta[i] * hPrev[j];
        }
    }

    addTransposedProduct(target, matrix, vector) {
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[i].length; j++) {
                target[j] += matrix[i][j] * vector[i];
            }
        }
    }

    zeroGrad() {
        for (const grad of this.gradients()) {
            if (Array.isArray(grad[0])) {
                for (const row of grad) row.fill(0);
            } else {
                grad.fill(0);
            }
        }
    }

    parameters() {
        return [
            this.Wf, this.Wi, this.Wc, this.Wo,
            this.Uf, this.Ui, this.Uc, this.Uo,
            this.bf, this.bi, this.bc, this.bo
        ];
    }

    gradients() {
        return [
            this.gradWf, this.gradWi, this.gradWc, this.gradWo,
            this.gradUf, this.gradUi, this.gradUc, this.gradUo,
            this.gradBf, this.gradBi, this.gradBc, this.gradBo
        ];
    }

    toJSON() {
        return {
            inputSize: this.inputSize,
            hiddenSize: this.hiddenSize,
            Wf: this.Wf, Wi: this.Wi, Wc: this.Wc, Wo: this.Wo,
            Uf: this.Uf, Ui: this.Ui, Uc: this.Uc, Uo: this.Uo,
            bf: this.bf, bi: this.bi, bc: this.bc, bo: this.bo
        };
    }

    static fromJSON(json) {
        const lstm = new LSTM(json.inputSize, json.hiddenSize);
        for (const key of ['Wf', 'Wi', 'Wc', 'Wo', 'Uf', 'Ui', 'Uc', 'Uo', 'bf', 'bi', 'bc', 'bo']) {
            if (json[key] !== undefined) {
                lstm[key] = Array.isArray(json[key][0])
                    ? json[key].map(row => [...row])
                    : [...json[key]];
            }
        }
        return lstm;
    }
}
