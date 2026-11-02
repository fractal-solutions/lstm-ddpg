// Minimal manual-backprop primitives (dense layers + Adam) used by the
// LSTM encoder and the DDPG actor/critic. Everything is plain JS arrays so
// gradients can be checked numerically (see gradcheck.js).

export const zeros = (n) => new Array(n).fill(0);

export const zerosLike = (p) =>
    Array.isArray(p[0]) ? p.map(row => row.map(() => 0)) : p.map(() => 0);

export function matVec(matrix, vector) {
    return matrix.map(row => {
        let sum = 0;
        for (let i = 0; i < row.length; i++) sum += row[i] * vector[i];
        return sum;
    });
}

export function matTVec(matrix, vector) {
    const cols = matrix[0].length;
    const out = zeros(cols);
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < cols; j++) out[j] += matrix[i][j] * vector[i];
    }
    return out;
}

export function outer(a, b) {
    return a.map(ai => b.map(bj => ai * bj));
}

export function vectorNorm(v) {
    let sum = 0;
    for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
    return Math.sqrt(sum);
}

export function clipVector(v, maxNorm = 10) {
    const norm = vectorNorm(v);
    if (!Number.isFinite(norm) || norm <= maxNorm) return v;
    const scale = maxNorm / norm;
    return v.map(x => x * scale);
}

const clampExp = (x) => Math.max(-60, Math.min(60, x));

export const activations = {
    linear: {
        forward: (z) => z,
        backward: (dOut) => dOut
    },
    relu: {
        forward: (z) => z.map(v => (v > 0 ? v : 0)),
        backward: (dOut, z) => dOut.map((d, i) => (z[i] > 0 ? d : 0))
    },
    tanh: {
        forward: (z) => z.map(v => Math.tanh(v)),
        backward: (dOut, z, a) => dOut.map((d, i) => d * (1 - a[i] * a[i]))
    },
    sigmoid: {
        forward: (z) => z.map(v => 1 / (1 + Math.exp(-clampExp(v)))),
        backward: (dOut, z, a) => dOut.map((d, i) => d * a[i] * (1 - a[i]))
    }
};

export class Dense {
    constructor(inSize, outSize, activation = 'linear', { scale = null } = {}) {
        if (!activations[activation]) throw new Error(`Unknown activation: ${activation}`);
        this.inSize = inSize;
        this.outSize = outSize;
        this.activation = activation;

        const initScale = scale ?? (activation === 'relu'
            ? Math.sqrt(2 / inSize)
            : Math.sqrt(1 / inSize));

        this.W = Array.from({ length: outSize }, () =>
            Array.from({ length: inSize }, () => (Math.random() * 2 - 1) * initScale));
        this.b = zeros(outSize);

        this.gradW = Array.from({ length: outSize }, () => zeros(inSize));
        this.gradB = zeros(outSize);
        this.cache = null;
    }

    forward(x) {
        if (x.length !== this.inSize) {
            throw new Error(`Dense expected input of size ${this.inSize}, got ${x.length}`);
        }
        const z = zeros(this.outSize);
        for (let i = 0; i < this.outSize; i++) {
            let sum = this.b[i];
            const row = this.W[i];
            for (let j = 0; j < this.inSize; j++) sum += row[j] * x[j];
            z[i] = sum;
        }
        const a = activations[this.activation].forward(z);
        this.cache = { x, z, a };
        return a;
    }

    backward(dOut, accumulate = true) {
        if (!this.cache) throw new Error('Dense.backward called before forward');
        const { x, z, a } = this.cache;
        const dz = activations[this.activation].backward(dOut, z, a);

        if (accumulate) {
            for (let i = 0; i < this.outSize; i++) {
                this.gradB[i] += dz[i];
                const row = this.gradW[i];
                for (let j = 0; j < this.inSize; j++) row[j] += dz[i] * x[j];
            }
        }

        return matTVec(this.W, dz);
    }

    zeroGrad() {
        for (let i = 0; i < this.outSize; i++) {
            this.gradB[i] = 0;
            this.gradW[i].fill(0);
        }
    }

    parameters() {
        return [this.W, this.b];
    }

    gradients() {
        return [this.gradW, this.gradB];
    }

    toJSON() {
        return {
            inSize: this.inSize,
            outSize: this.outSize,
            activation: this.activation,
            W: this.W,
            b: this.b
        };
    }

    static fromJSON(json) {
        const layer = new Dense(json.inSize, json.outSize, json.activation);
        layer.W = json.W.map(row => [...row]);
        layer.b = [...json.b];
        return layer;
    }
}

export class Adam {
    constructor({ lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, clipNorm = 1.0 } = {}) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.clipNorm = clipNorm;
        this.t = 0;
        this.state = new Map();
    }

    step(params, grads) {
        if (params.length !== grads.length) {
            throw new Error('Adam: params and grads length mismatch');
        }

        let normSquares = 0;
        for (const g of grads) {
            for (let i = 0; i < g.length; i++) {
                if (Array.isArray(g[i])) {
                    for (let j = 0; j < g[i].length; j++) normSquares += g[i][j] * g[i][j];
                } else {
                    normSquares += g[i] * g[i];
                }
            }
        }
        const norm = Math.sqrt(normSquares);
        const clipScale = this.clipNorm > 0 && norm > this.clipNorm ? this.clipNorm / norm : 1;

        this.t += 1;
        const bias1 = 1 - Math.pow(this.beta1, this.t);
        const bias2 = 1 - Math.pow(this.beta2, this.t);

        for (let k = 0; k < params.length; k++) {
            const p = params[k];
            const g = grads[k];

            let st = this.state.get(p);
            if (!st) {
                st = { m: zerosLike(p), v: zerosLike(p) };
                this.state.set(p, st);
            }

            if (Array.isArray(p[0])) {
                for (let i = 0; i < p.length; i++) {
                    for (let j = 0; j < p[i].length; j++) {
                        const gi = g[i][j] * clipScale;
                        st.m[i][j] = this.beta1 * st.m[i][j] + (1 - this.beta1) * gi;
                        st.v[i][j] = this.beta2 * st.v[i][j] + (1 - this.beta2) * gi * gi;
                        const mHat = st.m[i][j] / bias1;
                        const vHat = st.v[i][j] / bias2;
                        p[i][j] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
                    }
                }
            } else {
                for (let i = 0; i < p.length; i++) {
                    const gi = g[i] * clipScale;
                    st.m[i] = this.beta1 * st.m[i] + (1 - this.beta1) * gi;
                    st.v[i] = this.beta2 * st.v[i] + (1 - this.beta2) * gi * gi;
                    const mHat = st.m[i] / bias1;
                    const vHat = st.v[i] / bias2;
                    p[i] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
                }
            }
        }
    }
}

export function scaleGrads(grads, factor) {
    for (const g of grads) {
        if (Array.isArray(g[0])) {
            for (let i = 0; i < g.length; i++) {
                for (let j = 0; j < g[i].length; j++) g[i][j] *= factor;
            }
        } else {
            for (let i = 0; i < g.length; i++) g[i] *= factor;
        }
    }
}

export function softUpdateParams(targetParams, sourceParams, tau) {
    if (targetParams.length !== sourceParams.length) {
        throw new Error('softUpdateParams: parameter count mismatch');
    }
    for (let k = 0; k < targetParams.length; k++) {
        const target = targetParams[k];
        const source = sourceParams[k];
        if (Array.isArray(target[0])) {
            for (let i = 0; i < target.length; i++) {
                for (let j = 0; j < target[i].length; j++) {
                    target[i][j] = (1 - tau) * target[i][j] + tau * source[i][j];
                }
            }
        } else {
            for (let i = 0; i < target.length; i++) {
                target[i] = (1 - tau) * target[i] + tau * source[i];
            }
        }
    }
}

export function copyParams(targetParams, sourceParams) {
    softUpdateParams(targetParams, sourceParams, 1);
}
