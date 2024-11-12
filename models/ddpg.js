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
        // Actor network weights with smaller initial values
        this.actorWeights = {
            layer1: this.createMatrix(64, this.stateSize, 0.1),
            layer2: this.createMatrix(32, 64, 0.1),
            output: this.createMatrix(this.actionSize, 32, 0.1)
        };
        
        // Critic network weights with smaller initial values
        this.criticWeights = {
            stateLayer: this.createMatrix(64, this.stateSize, 0.1),
            actionLayer: this.createMatrix(64, this.actionSize, 0.1),
            hidden: this.createMatrix(32, 128, 0.1),
            output: this.createMatrix(1, 32, 0.1)
        };
    }
    
    createMatrix(rows, cols, scale = 0.1) {
        return Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () =>
                (Math.random() * 2 - 1) * scale
            )
        );
    }
    
    relu(x) {
        return Math.max(0, Math.min(10, x)); // Clipped ReLU to prevent explosion
    }
    
    tanh(x) {
        // Clip input to prevent NaN
        const clipped = Math.max(-20, Math.min(20, x));
        return Math.tanh(clipped);
    }
    
    matrixMultiply(matrix, vector) {
        if (!matrix || !vector) {
            throw new Error('Matrix and vector must be defined');
        }
        
        if (!Array.isArray(matrix) || !Array.isArray(vector)) {
            throw new Error('Matrix and vector must be arrays');
        }
        
        const inputVector = Array.isArray(vector) ? vector : Array.from(vector);
        
        if (matrix[0].length !== inputVector.length) {
            throw new Error(`Matrix columns (${matrix[0].length}) must match vector length (${inputVector.length})`);
        }
        
        return matrix.map(row => {
            const sum = row.reduce((sum, val, i) => {
                const product = val * inputVector[i];
                return sum + (isNaN(product) ? 0 : product); // Handle NaN values
            }, 0);
            return Math.max(-10, Math.min(10, sum)); // Clip output
        });
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

        // Validate output
        if (output.some(isNaN)) {
            console.error('NaN detected in actor output');
            return Array(this.actionSize).fill(0); // Safe fallback
        }            
        
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
        
        return isNaN(output) ? 0 : Math.max(-10, Math.min(10, output));
    }
    
    updateCritic(tdError) {
        const learningRate = 0.0001; // Reduced learning rate
        const clippedError = Math.max(-1, Math.min(1, tdError));
        
        // Update critic weights using clipped TD error
        Object.keys(this.criticWeights).forEach(layer => {
            this.criticWeights[layer] = this.criticWeights[layer].map(row =>
                row.map(weight => {
                    const update = weight + learningRate * clippedError;
                    return Math.max(-1, Math.min(1, update)); // Clip weights
                })
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
        
        // Calculate gradient with clipping
        const actionGradient = actions.map(action => {
            const grad = criticValue * (1 - action * action); // derivative of tanh
            return Math.max(-1, Math.min(1, grad)); // Clip gradient
        });
        
        return actionGradient;
    }
    
    updateActor(gradient) {
        if (!Array.isArray(gradient)) {
            throw new Error('Gradient must be an array');
        }
        
        const learningRate = 0.00001; // Reduced learning rate
        // Update actor weights using clipped policy gradient
        Object.keys(this.actorWeights).forEach(layer => {
            this.actorWeights[layer] = this.actorWeights[layer].map(row =>
                row.map(weight => {
                    const clippedGrad = Math.max(-1, Math.min(1, gradient[0]));
                    const update = weight + learningRate * clippedGrad;
                    return Math.max(-1, Math.min(1, update)); // Clip weights
                })
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