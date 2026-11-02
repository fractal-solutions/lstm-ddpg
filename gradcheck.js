import { LSTM } from './models/lstm.js';
import { DDPG } from './models/ddpg.js';
import { Dense } from './utils/nn.js';

const EPS = 1e-6;
const TOLERANCE = 1e-4;

function relativeError(analytic, numeric) {
    return Math.abs(analytic - numeric) / Math.max(1e-8, Math.abs(analytic) + Math.abs(numeric));
}

function numericGradient(lossFn, param, i, j = null) {
    const isMatrix = Array.isArray(param[0]);
    const get = () => (isMatrix ? param[i][j] : param[i]);
    const set = (value) => {
        if (isMatrix) param[i][j] = value;
        else param[i] = value;
    };

    const original = get();
    set(original + EPS);
    const plus = lossFn();
    set(original - EPS);
    const minus = lossFn();
    set(original);
    return (plus - minus) / (2 * EPS);
}

function checkParams(name, params, grads, lossFn, picks) {
    const errors = [];
    for (const [paramIndex, row, col] of picks) {
        const param = params[paramIndex];
        const grad = grads[paramIndex];
        const analytic = Array.isArray(param[0]) ? grad[row][col] : grad[row];
        const numeric = numericGradient(lossFn, param, row, col);
        const error = relativeError(analytic, numeric);
        errors.push(error);
        console.log(
            `  ${name}[${paramIndex}]${Array.isArray(param[0]) ? `[${row}][${col}]` : `[${row}]`} ` +
            `analytic=${analytic.toExponential(4)} numeric=${numeric.toExponential(4)} relErr=${error.toExponential(2)}`
        );
    }
    return Math.max(...errors);
}

function checkLstm() {
    console.log('\nLSTM BPTT');
    const lstm = new LSTM(2, 3);
    const sequence = [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2], [0.6, 0.1], [0.2, -0.3]];
    const dHidden = [0.3, -0.2, 0.5];
    const lossFn = () => {
        const { hidden } = lstm.forward(sequence);
        return hidden.reduce((sum, value, i) => sum + value * dHidden[i], 0);
    };

    lstm.zeroGrad();
    lstm.forward(sequence);
    lstm.backward(dHidden);

    return checkParams('LSTM', lstm.parameters(), lstm.gradients(), () => lossFn(), [
        [0, 0, 1],   // Wf
        [1, 2, 0],   // Wi
        [4, 1, 2],   // Uf
        [6, 0, 1],   // Uc
        [8, 2],      // bf
        [11, 1]      // bo
    ]);
}

function checkDense() {
    console.log('\nDense layer');
    const dense = new Dense(3, 4, 'tanh');
    const x = [0.2, -0.4, 0.7];
    const dOut = [0.1, -0.2, 0.3, -0.4];
    const lossFn = () => {
        const out = dense.forward(x);
        return out.reduce((sum, value, i) => sum + value * dOut[i], 0);
    };

    dense.zeroGrad();
    dense.forward(x);
    dense.backward(dOut, true);

    return checkParams('Dense', dense.parameters(), dense.gradients(), lossFn, [
        [0, 0, 0],
        [0, 3, 2],
        [1, 1]
    ]);
}

function checkDdpg() {
    console.log('\nDDPG critic');
    const ddpg = new DDPG(4, 4);
    const state = [0.1, -0.2, 0.3, 0.4];
    const action = [0.5, 0.6, 0.4, 0.7];
    const dQ = 0.7;
    const criticLoss = () => dQ * ddpg.criticForward(state, action);

    ddpg.zeroCriticGrad();
    ddpg.criticForward(state, action);
    ddpg.criticBackward(dQ, true);

    const criticError = checkParams(
        'critic',
        ddpg.criticParameters(),
        ddpg.criticGradients(),
        criticLoss,
        [
            [0, 0, 1],
            [1, 3],
            [4, 2, 5],
            [7, 0]
        ]
    );

    console.log('\nDDPG actor');
    const g = [0.3, -0.4, 0.2, -0.1];
    const actorLoss = () => {
        const out = ddpg.policyForward(state).vector;
        return out.reduce((sum, value, i) => sum + value * g[i], 0);
    };

    ddpg.zeroActorGrad();
    ddpg.policyForward(state);
    ddpg.policyBackward(g, true);

    const actorError = checkParams(
        'actor',
        ddpg.actorParameters(),
        ddpg.actorGradients(),
        actorLoss,
        [
            [0, 0, 0],
            [1, 2],
            [4, 1, 1],
            [5, 0]
        ]
    );

    return Math.max(criticError, actorError);
}

const maxError = Math.max(checkLstm(), checkDense(), checkDdpg());

console.log(`\nMax relative error: ${maxError.toExponential(3)} (tolerance ${TOLERANCE.toExponential(1)})`);
if (maxError > TOLERANCE) {
    console.error('GRADIENT CHECK FAILED');
    process.exit(1);
}
console.log('GRADIENT CHECK PASSED');
