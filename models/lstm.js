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

    sigmoidDerivative(x) {
        const sx = this.sigmoid(x);
        return sx * (1 - sx);
    }

    tanhDerivative(x) {
        const tx = this.tanh(x);
        return 1 - tx * tx;
    }

    backward(tdError, learningRate = 0.001) {
        // Initialize gradients
        const dWf = Array(this.hiddenSize).fill().map(() => Array(this.inputSize).fill(0));
        const dWi = Array(this.hiddenSize).fill().map(() => Array(this.inputSize).fill(0));
        const dWc = Array(this.hiddenSize).fill().map(() => Array(this.inputSize).fill(0));
        const dWo = Array(this.hiddenSize).fill().map(() => Array(this.inputSize).fill(0));
        
        const dUf = Array(this.hiddenSize).fill().map(() => Array(this.hiddenSize).fill(0));
        const dUi = Array(this.hiddenSize).fill().map(() => Array(this.hiddenSize).fill(0));
        const dUc = Array(this.hiddenSize).fill().map(() => Array(this.hiddenSize).fill(0));
        const dUo = Array(this.hiddenSize).fill().map(() => Array(this.hiddenSize).fill(0));

        const dbf = new Array(this.hiddenSize).fill(0);
        const dbi = new Array(this.hiddenSize).fill(0);
        const dbc = new Array(this.hiddenSize).fill(0);
        const dbo = new Array(this.hiddenSize).fill(0);

        // Backpropagate through time
        let dh_next = tdError;
        let dc_next = this.activations.cellState.map(this.tanhDerivative);

        // Output gate
        const do_gate = this.elementWiseMultiply(
            dh_next,
            this.activations.cellState.map(this.tanh)
        );

        // Cell state
        const dc = this.elementWiseAdd(
            this.elementWiseMultiply(
                dh_next,
                this.elementWiseMultiply(
                    this.activations.o_gate,
                    dc_next
                )
            ),
            dc_next
        );

        // Input gate
        const di_gate = this.elementWiseMultiply(
            dc,
            this.activations.c_tilde
        );

        // Forget gate
        const df_gate = this.elementWiseMultiply(
            dc,
            this.activations.prevCell
        );

        // Update weights
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.inputSize; j++) {
                // Input weights
                dWf[i][j] = df_gate[i] * this.activations.x[j];
                dWi[i][j] = di_gate[i] * this.activations.x[j];
                dWc[i][j] = dc[i] * this.activations.x[j];
                dWo[i][j] = do_gate[i] * this.activations.x[j];

                // Apply updates
                this.Wf[i][j] -= learningRate * dWf[i][j];
                this.Wi[i][j] -= learningRate * dWi[i][j];
                this.Wc[i][j] -= learningRate * dWc[i][j];
                this.Wo[i][j] -= learningRate * dWo[i][j];
            }

            // Update biases
            this.bf[i] -= learningRate * df_gate[i];
            this.bi[i] -= learningRate * di_gate[i];
            this.bc[i] -= learningRate * dc[i];
            this.bo[i] -= learningRate * do_gate[i];
        }

        // Update recurrent weights
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                dUf[i][j] = df_gate[i] * this.activations.prevHidden[j];
                dUi[i][j] = di_gate[i] * this.activations.prevHidden[j];
                dUc[i][j] = dc[i] * this.activations.prevHidden[j];
                dUo[i][j] = do_gate[i] * this.activations.prevHidden[j];

                this.Uf[i][j] -= learningRate * dUf[i][j];
                this.Ui[i][j] -= learningRate * dUi[i][j];
                this.Uc[i][j] -= learningRate * dUc[i][j];
                this.Uo[i][j] -= learningRate * dUo[i][j];
            }
        }

        this.clearActivations();
        return dh_next;
    }
}
  