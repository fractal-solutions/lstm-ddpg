import { Dense } from '../utils/nn.js';

const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, x))));

// Actor: state -> 64 (relu) -> 32 (relu) -> 4 linear outputs
//   [direction (tanh), size (sigmoid), SL fraction (sigmoid), TP fraction (sigmoid)]
// Critic: state -> 64 (relu), action -> 64 (relu), concat -> 32 (relu) -> 1
export class DDPG {
    constructor(stateSize, actionSize = 4, { actorHidden = [64, 32], criticHidden = [64, 64, 32] } = {}) {
        if (!Number.isInteger(stateSize) || !Number.isInteger(actionSize)) {
            throw new Error('State size and action size must be integers');
        }
        if (actionSize !== 4) {
            throw new Error('DDPG expects actionSize=4 ([direction, size, SL, TP])');
        }
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.actorHidden = actorHidden;
        this.criticHidden = criticHidden;

        this.actorLayers = this.buildActor();
        const [stateOut, actionOut, hiddenOut] = criticHidden;
        this.criticStateLayer = new Dense(stateSize, stateOut, 'relu');
        this.criticActionLayer = new Dense(actionSize, actionOut, 'relu');
        this.criticHiddenLayer = new Dense(stateOut + actionOut, hiddenOut, 'relu');
        this.criticOutputLayer = new Dense(hiddenOut, 1, 'linear');
        this.criticLayers = [
            this.criticStateLayer,
            this.criticActionLayer,
            this.criticHiddenLayer,
            this.criticOutputLayer
        ];

        this.lastPolicy = null;
    }

    buildActor() {
        const [hidden1, hidden2] = this.actorHidden;
        const layers = [
            new Dense(this.stateSize, hidden1, 'relu'),
            new Dense(hidden1, hidden2, 'relu'),
            new Dense(hidden2, this.actionSize, 'linear', { scale: 0.1 })
        ];
        // Start with a mildly directional policy instead of a perfectly flat one.
        layers[layers.length - 1].b[0] = 0.5;
        return layers;
    }

    policyForward(state) {
        let x = state;
        for (let i = 0; i < this.actorLayers.length - 1; i++) {
            x = this.actorLayers[i].forward(x);
        }
        const raw = this.actorLayers[this.actorLayers.length - 1].forward(x);

        const policy = {
            raw,
            direction: Math.tanh(raw[0]),
            size: sigmoid(raw[1]),
            stopLossFrac: sigmoid(raw[2]),
            takeProfitFrac: sigmoid(raw[3])
        };
        policy.vector = [
            policy.direction,
            policy.size,
            policy.stopLossFrac,
            policy.takeProfitFrac
        ];
        this.lastPolicy = policy;
        return policy;
    }

    actorForward(state) {
        return this.policyForward(state).vector;
    }

    // dVector is dLoss/d(actionVector). Backpropagates through the output
    // activations (tanh/sigmoid) and the actor MLP.
    policyBackward(dVector, accumulate = true) {
        const policy = this.lastPolicy;
        if (!policy) throw new Error('policyBackward called before policyForward');

        const dRaw = [
            dVector[0] * (1 - policy.direction * policy.direction),
            dVector[1] * policy.size * (1 - policy.size),
            dVector[2] * policy.stopLossFrac * (1 - policy.stopLossFrac),
            dVector[3] * policy.takeProfitFrac * (1 - policy.takeProfitFrac)
        ];

        let grad = this.actorLayers[this.actorLayers.length - 1].backward(dRaw, accumulate);
        for (let i = this.actorLayers.length - 2; i >= 0; i--) {
            grad = this.actorLayers[i].backward(grad, accumulate);
        }
        return grad;
    }

    criticForward(state, action) {
        const stateFeatures = this.criticStateLayer.forward(state);
        const actionFeatures = this.criticActionLayer.forward(action);
        const combined = [...stateFeatures, ...actionFeatures];
        const hidden = this.criticHiddenLayer.forward(combined);
        const value = this.criticOutputLayer.forward(hidden)[0];
        return Math.max(-10, Math.min(10, value));
    }

    criticBackward(dQ, accumulate = true) {
        const dHidden = this.criticOutputLayer.backward([dQ], accumulate);
        const dCombined = this.criticHiddenLayer.backward(dHidden, accumulate);
        const stateOut = this.criticStateLayer.outSize;
        const dStateFeatures = dCombined.slice(0, stateOut);
        const dActionFeatures = dCombined.slice(stateOut);
        const dState = this.criticStateLayer.backward(dStateFeatures, accumulate);
        const dAction = this.criticActionLayer.backward(dActionFeatures, accumulate);
        return { dState, dAction };
    }

    zeroActorGrad() {
        for (const layer of this.actorLayers) layer.zeroGrad();
    }

    zeroCriticGrad() {
        for (const layer of this.criticLayers) layer.zeroGrad();
    }

    zeroGrad() {
        this.zeroActorGrad();
        this.zeroCriticGrad();
    }

    actorParameters() {
        return this.actorLayers.flatMap(layer => layer.parameters());
    }

    actorGradients() {
        return this.actorLayers.flatMap(layer => layer.gradients());
    }

    criticParameters() {
        return this.criticLayers.flatMap(layer => layer.parameters());
    }

    criticGradients() {
        return this.criticLayers.flatMap(layer => layer.gradients());
    }

    allParameters() {
        return [...this.actorParameters(), ...this.criticParameters()];
    }

    toJSON() {
        return {
            stateSize: this.stateSize,
            actionSize: this.actionSize,
            actorHidden: this.actorHidden,
            criticHidden: this.criticHidden,
            actor: this.actorLayers.map(layer => layer.toJSON()),
            critic: {
                state: this.criticStateLayer.toJSON(),
                action: this.criticActionLayer.toJSON(),
                hidden: this.criticHiddenLayer.toJSON(),
                output: this.criticOutputLayer.toJSON()
            }
        };
    }

    static fromJSON(json) {
        const ddpg = new DDPG(json.stateSize, json.actionSize, {
            actorHidden: json.actorHidden ?? [64, 32],
            criticHidden: json.criticHidden ?? [64, 64, 32]
        });
        ddpg.actorLayers = json.actor.map(Dense.fromJSON);
        ddpg.criticStateLayer = Dense.fromJSON(json.critic.state);
        ddpg.criticActionLayer = Dense.fromJSON(json.critic.action);
        ddpg.criticHiddenLayer = Dense.fromJSON(json.critic.hidden);
        ddpg.criticOutputLayer = Dense.fromJSON(json.critic.output);
        ddpg.criticLayers = [
            ddpg.criticStateLayer,
            ddpg.criticActionLayer,
            ddpg.criticHiddenLayer,
            ddpg.criticOutputLayer
        ];
        return ddpg;
    }
}
