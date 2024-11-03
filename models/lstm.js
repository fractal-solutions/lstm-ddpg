export class LSTM {
    constructor(inputSize, hiddenSize) {
      this.inputSize = inputSize;
      this.hiddenSize = hiddenSize;
      
      // Initialize gates weights and biases
      this.initializeWeights();
      // Store activations for backward pass
      this.clearActivations();
    }
  
    initializeWeights() {
      // Helper function to create weight matrices
      const createMatrix = (rows, cols) => 
        Array.from({ length: rows }, () => 
          Array.from({ length: cols }, () => 
            (Math.random() * 2 - 1) * Math.sqrt(2 / (rows + cols))
          )
        );
  
      // Input gates
      this.Wf = createMatrix(this.hiddenSize, this.inputSize);
      this.Wi = createMatrix(this.hiddenSize, this.inputSize);
      this.Wc = createMatrix(this.hiddenSize, this.inputSize);
      this.Wo = createMatrix(this.hiddenSize, this.inputSize);
  
      // Hidden gates
      this.Uf = createMatrix(this.hiddenSize, this.hiddenSize);
      this.Ui = createMatrix(this.hiddenSize, this.hiddenSize);
      this.Uc = createMatrix(this.hiddenSize, this.hiddenSize);
      this.Uo = createMatrix(this.hiddenSize, this.hiddenSize);
  
      // Biases
      this.bf = new Array(this.hiddenSize).fill(0);
      this.bi = new Array(this.hiddenSize).fill(0);
      this.bc = new Array(this.hiddenSize).fill(0);
      this.bo = new Array(this.hiddenSize).fill(0);
    }

    clearActivations() {
        this.activations = {
            x: null,
            prevHidden: null,
            prevCell: null,
            f_gate: null,
            i_gate: null,
            c_tilde: null,
            cellState: null,
            o_gate: null,
            hiddenState: null
        };
    }
  
    sigmoid(x) {
      const clampedX = Math.max(-709, Math.min(709, x)); // Prevent exp overflow
      return 1 / (1 + Math.exp(-clampedX));
    }
  
    tanh(x) {
      return Math.tanh(x);
    }

    sigmoidDerivative(x) {
        const sx = this.sigmoid(x);
        return sx * (1 - sx);
    }

    tanhDerivative(x) {
        const sx = Math.tanh(x);
        const tx = sx * (1 - sx);
        return 1 - tx * tx;
    }
  
    matrixVectorMultiply(matrix, vector) {
      if (!Array.isArray(vector)) {
        throw new Error("Input vector must be an array");
      }
      
      return matrix.map(row => 
        row.reduce((sum, val, i) => sum + val * vector[i], 0)
      );
    }
  
    elementWiseAdd(arr1, arr2) {
      return arr1.map((val, i) => val + arr2[i]);
    }
  
    elementWiseMultiply(arr1, arr2) {
      return arr1.map((val, i) => val * arr2[i]);
    }
  
    forward(x, prevHidden = null, prevCell = null) {
        // Store inputs for backward pass
        this.activations.x = x;
        this.activations.prevHidden = prevHidden || new Array(this.hiddenSize).fill(0);
        this.activations.prevCell = prevCell || new Array(this.hiddenSize).fill(0);

        // Forget gate
        const Wf_x = this.matrixVectorMultiply(this.Wf, x);
        const Uf_h = this.matrixVectorMultiply(this.Uf, this.activations.prevHidden);
        this.activations.f_gate = this.elementWiseAdd(
            this.elementWiseAdd(Wf_x, Uf_h),
            this.bf
        ).map(this.sigmoid);

        // Input gate
        const Wi_x = this.matrixVectorMultiply(this.Wi, x);
        const Ui_h = this.matrixVectorMultiply(this.Ui, this.activations.prevHidden);
        this.activations.i_gate = this.elementWiseAdd(
            this.elementWiseAdd(Wi_x, Ui_h),
            this.bi
        ).map(this.sigmoid);

        // Candidate cell state
        const Wc_x = this.matrixVectorMultiply(this.Wc, x);
        const Uc_h = this.matrixVectorMultiply(this.Uc, this.activations.prevHidden);
        this.activations.c_tilde = this.elementWiseAdd(
            this.elementWiseAdd(Wc_x, Uc_h),
            this.bc
        ).map(this.tanh);

        // Cell state
        const forgotten = this.elementWiseMultiply(this.activations.f_gate, this.activations.prevCell);
        const input = this.elementWiseMultiply(this.activations.i_gate, this.activations.c_tilde);
        this.activations.cellState = this.elementWiseAdd(forgotten, input);

        // Output gate
        const Wo_x = this.matrixVectorMultiply(this.Wo, x);
        const Uo_h = this.matrixVectorMultiply(this.Uo, this.activations.prevHidden);
        this.activations.o_gate = this.elementWiseAdd(
            this.elementWiseAdd(Wo_x, Uo_h),
            this.bo
        ).map(this.sigmoid);

        // Hidden state
        this.activations.hiddenState = this.elementWiseMultiply(
            this.activations.o_gate,
            this.activations.cellState.map(this.tanh)
        );

        return {
            hiddenState: this.activations.hiddenState,
            cellState: this.activations.cellState
        };
    }

    //gradient clipping
    clipGradients(gradients, clipValue = 0.9) {
        const clipGradient = (value) => Math.max(-clipValue, Math.min(clipValue, value));
        
        return {
            ...gradients,
            Wf: gradients.Wf.map(row => row.map(clipGradient)),
            Wi: gradients.Wi.map(row => row.map(clipGradient)),
            Wc: gradients.Wc.map(row => row.map(clipGradient)),
            Wo: gradients.Wo.map(row => row.map(clipGradient))
        };
    }

    backward(tdError, learningRate = 0.0001) {
        // Initialize gradients for all weights and biases
        const gradients = {
            Wf: Array(this.hiddenSize).fill().map(() => Array(this.inputSize).fill(0)),
            Wi: Array(this.hiddenSize).fill().map(() => Array(this.inputSize).fill(0)),
            Wc: Array(this.hiddenSize).fill().map(() => Array(this.inputSize).fill(0)),
            Wo: Array(this.hiddenSize).fill().map(() => Array(this.inputSize).fill(0)),
            
            Uf: Array(this.hiddenSize).fill().map(() => Array(this.hiddenSize).fill(0)),
            Ui: Array(this.hiddenSize).fill().map(() => Array(this.hiddenSize).fill(0)),
            Uc: Array(this.hiddenSize).fill().map(() => Array(this.hiddenSize).fill(0)),
            Uo: Array(this.hiddenSize).fill().map(() => Array(this.hiddenSize).fill(0)),

            bf: new Array(this.hiddenSize).fill(0),
            bi: new Array(this.hiddenSize).fill(0),
            bc: new Array(this.hiddenSize).fill(0),
            bo: new Array(this.hiddenSize).fill(0)
        };

        // Backpropagate through time
        let dh_next = Array.isArray(tdError) ? tdError : new Array(this.hiddenSize).fill(tdError);
        let dc_next = this.activations.cellState.map(this.tanhDerivative);

        // Calculate gradients
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.inputSize; j++) {
                gradients.Wf[i][j] = dh_next[i] * this.activations.x[j];
                gradients.Wi[i][j] = dh_next[i] * this.activations.x[j];
                gradients.Wc[i][j] = dh_next[i] * this.activations.x[j];
                gradients.Wo[i][j] = dh_next[i] * this.activations.x[j];
            }
            
            for (let j = 0; j < this.hiddenSize; j++) {
                gradients.Uf[i][j] = dh_next[i] * this.activations.prevHidden[j];
                gradients.Ui[i][j] = dh_next[i] * this.activations.prevHidden[j];
                gradients.Uc[i][j] = dh_next[i] * this.activations.prevHidden[j];
                gradients.Uo[i][j] = dh_next[i] * this.activations.prevHidden[j];
            }

            gradients.bf[i] = dh_next[i];
            gradients.bi[i] = dh_next[i];
            gradients.bc[i] = dh_next[i];
            gradients.bo[i] = dh_next[i];
        }
        //gradients = this.clipGradients(gradients);
        return this.clipGradients(gradients);
    }

    updateWeights(gradients, learningRate = 0.0001) {
        // Update all weights and biases using the calculated gradients
        const updateMatrix = (target, gradient) => {
            for (let i = 0; i < target.length; i++) {
                for (let j = 0; j < target[i].length; j++) {
                    target[i][j] -= learningRate * gradient[i][j];
                }
            }
        };

        const updateVector = (target, gradient) => {
            for (let i = 0; i < target.length; i++) {
                target[i] -= learningRate * gradient[i];
            }
        };

        // Update input weights
        updateMatrix(this.Wf, gradients.Wf);
        updateMatrix(this.Wi, gradients.Wi);
        updateMatrix(this.Wc, gradients.Wc);
        updateMatrix(this.Wo, gradients.Wo);

        // Update recurrent weights
        updateMatrix(this.Uf, gradients.Uf);
        updateMatrix(this.Ui, gradients.Ui);
        updateMatrix(this.Uc, gradients.Uc);
        updateMatrix(this.Uo, gradients.Uo);

        // Update biases
        updateVector(this.bf, gradients.bf);
        updateVector(this.bi, gradients.bi);
        updateVector(this.bc, gradients.bc);
        updateVector(this.bo, gradients.bo);

        // Clear activations after update
        this.clearActivations();
    }
}