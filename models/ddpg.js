export class DDPG {
    constructor(stateSize, actionSize) {
        if (!Number.isInteger(stateSize) || !Number.isInteger(actionSize)) {
            throw new Error('State size and action size must be integers');
        }
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        
        // Initialize actor and critic networks
        this.initializeNetworks();
    }
    
    initializeNetworks() {
        // Actor network weights
        this.actorWeights = {
            layer1: this.createMatrix(64, this.stateSize),
            layer2: this.createMatrix(32, 64),
            output: this.createMatrix(this.actionSize, 32)
        };
        
        // Critic network weights
        this.criticWeights = {
            stateLayer: this.createMatrix(64, this.stateSize),
            actionLayer: this.createMatrix(64, this.actionSize),
            hidden: this.createMatrix(32, 128),
            output: this.createMatrix(1, 32)
        };
    }
    
    createMatrix(rows, cols) {
        return Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () =>
                (Math.random() * 2 - 1) * Math.sqrt(2 / (rows + cols))
            )
        );
    }
    
    relu(x) {
        return Math.max(0, x);
    }
    
    tanh(x) {
        return Math.tanh(x);
    }
    
    matrixMultiply(matrix, vector) {
        if (!matrix || !vector) {
            throw new Error('Matrix and vector must be defined');
        }
        
        if (!Array.isArray(matrix) || !Array.isArray(vector)) {
            throw new Error('Matrix and vector must be arrays');
        }
        
        // Convert vector to array if it's not already
        const inputVector = Array.isArray(vector) ? vector : Array.from(vector);
        
        if (matrix[0].length !== inputVector.length) {
            throw new Error(`Matrix columns (${matrix[0].length}) must match vector length (${inputVector.length})`);
        }
        
        return matrix.map(row =>
            row.reduce((sum, val, i) => sum + val * inputVector[i], 0)
        );
    }
    
    actorForward(state) {
        if (!this.actorWeights || !this.actorWeights.layer1) {
            this.initializeNetworks();
        }

        const flatState = Array.isArray(state[0]) ? state.flat() : state;
        
        if (flatState.length !== this.stateSize) {
            throw new Error(`State must have length ${this.stateSize}, received ${flatState.length}`);
        }

        // Layer 1
        const layer1 = this.matrixMultiply(this.actorWeights.layer1, flatState)
            .map(x => this.relu(x));
        
        // Layer 2
        const layer2 = this.matrixMultiply(this.actorWeights.layer2, layer1)
            .map(x => this.relu(x));
        
        // Output layer
        const output = this.matrixMultiply(this.actorWeights.output, layer2)
            .map(x => this.tanh(x));
        
        return output;
    }
    
    criticForward(state, action) {
        // Ensure state is an array and has the correct length
        if (!Array.isArray(state)) {
            throw new Error(`State must be an array, received ${typeof state}`);
        }

        // If state is a nested array (e.g., from LSTM), flatten it
        const flatState = Array.isArray(state[0]) ? state.flat() : state;
        
        if (flatState.length !== this.stateSize) {
            throw new Error(`State must have length ${this.stateSize}, received ${flatState.length}`);
        }
        if (!Array.isArray(action) || action.length !== this.actionSize) {
            throw new Error(`Action must be an array of length ${this.actionSize}`);
        }
        
        // Process state
        const stateFeatures = this.matrixMultiply(this.criticWeights.stateLayer, flatState)
            .map(x => this.relu(x));
        
        // Process action
        const actionFeatures = this.matrixMultiply(this.criticWeights.actionLayer, action)
            .map(x => this.relu(x));
        
        // Concatenate features
        const combined = [...stateFeatures, ...actionFeatures];
        
        // Hidden layer
        const hidden = this.matrixMultiply(this.criticWeights.hidden, combined)
            .map(x => this.relu(x));
        
        // Output layer
        const output = this.matrixMultiply(this.criticWeights.output, hidden)[0];
        
        return output;
    }
    
    updateCritic(tdError) {
        const learningRate = 0.001;
        
        // Update critic weights using TD error
        Object.keys(this.criticWeights).forEach(layer => {
            this.criticWeights[layer] = this.criticWeights[layer].map(row =>
                row.map(weight => weight + learningRate * tdError)
            );
        });
    }
    
    getActorGradient(state) {
        // Ensure state is an array and has the correct length
        if (!Array.isArray(state)) {
            throw new Error(`State must be an array, received ${typeof state}`);
        }

        // If state is a nested array (e.g., from LSTM), flatten it
        const flatState = Array.isArray(state[0]) ? state.flat() : state;
        
        if (flatState.length !== this.stateSize) {
            throw new Error(`State must have length ${this.stateSize}, received ${flatState.length}`);
        }
        
        // Calculate actor gradient using deterministic policy gradient
        const actions = this.actorForward(flatState);
        const criticValue = this.criticForward(flatState, actions);
        
        // Calculate gradient of critic with respect to actions
        const actionGradient = actions.map(action => {
            return criticValue * (1 - action * action); // derivative of tanh
        });
        
        return actionGradient;
    }
    
    updateActor(gradient) {
        if (!Array.isArray(gradient)) {
            throw new Error('Gradient must be an array');
        }
        
        const learningRate = 0.0001;
        
        // Update actor weights using policy gradient
        Object.keys(this.actorWeights).forEach(layer => {
            this.actorWeights[layer] = this.actorWeights[layer].map(row =>
                row.map(weight => weight + learningRate * gradient)
            );
        });
    }

    toJSON() {
        return {
            stateSize: this.stateSize,
            actionSize: this.actionSize,
            actorWeights: this.actorWeights,
            criticWeights: this.criticWeights
        };
    }

    static fromJSON(json) {
        const ddpg = new DDPG(json.stateSize, json.actionSize);
        ddpg.actorWeights = json.actorWeights;
        ddpg.criticWeights = json.criticWeights;
        return ddpg;
    }
}