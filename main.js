import { LSTM } from './models/lstm.js';
import { DDPG } from './models/ddpg.js';
import { DataProcessor } from './utils/dataProcessor.js';

const INPUT_SIZE = 5;  // OHLCV
const HIDDEN_SIZE = 64;
const ACTION_SIZE = 3; // [position, sl, tp]

//ReplayBuffer class to store experiences:

export class ReplayBuffer {
    constructor(maxSize = 100000) {
      this.buffer = [];
      this.maxSize = maxSize;
    }
  
    add(state, action, reward, nextState, done) {
      if (this.buffer.length >= this.maxSize) {
        this.buffer.shift();
      }
      this.buffer.push({state, action, reward, nextState, done});
    }
  
    sample(batchSize) {
      const samples = [];
      for(let i = 0; i < batchSize; i++) {
        const index = Math.floor(Math.random() * this.buffer.length);
        samples.push(this.buffer[index]);
      }
      return samples;
    }
  }
  

  export class TradingSystem {
    constructor() {
      this.lstm = new LSTM(INPUT_SIZE, HIDDEN_SIZE);
      this.ddpg = new DDPG(HIDDEN_SIZE, ACTION_SIZE);
      this.replayBuffer = new ReplayBuffer();
      this.batchSize = 32;
      this.gamma = 0.99; // Discount factor
      this.tau = 0.001; // Soft update parameter
      
      // Create target networks
      this.targetLSTM = new LSTM(INPUT_SIZE, HIDDEN_SIZE);
      this.targetDDPG = new DDPG(HIDDEN_SIZE, ACTION_SIZE);
      
      // Copy initial weights
      this.updateTargetNetworks(1.0);
    }
  
    async loadData() {
      const data = await Bun.file('./data/EURUSD_H1.json').json();
      this.dataProcessor = new DataProcessor(data);
      return data;
    }
  
    predict(state, addNoise = false) {
      const { hiddenState } = this.lstm.forward(state);
      let actions = this.ddpg.actorForward(hiddenState);
      
      if (addNoise) {
        // Add exploration noise during training
        actions = actions.map(a => {
          const noise = (Math.random() * 2 - 1) * 0.1; // Gaussian noise
          return Math.max(-1, Math.min(1, a + noise));
        });
      }
      
      const [positionSize, stopLoss, takeProfit] = actions;
      return {
        positionSize,
        stopLoss,
        takeProfit,
        hiddenState
      };
    }
  
    calculateReward(action, nextPrice, currentPrice) {
      const pnl = (nextPrice - currentPrice) * action.positionSize;
      
      // Add penalties for excessive risk
      const slPenalty = Math.abs(action.stopLoss) > 0.5 ? -0.1 : 0;
      const tpPenalty = Math.abs(action.takeProfit) < 0.2 ? -0.1 : 0;
      
      return pnl + slPenalty + tpPenalty;
    }
  
    async train(epochs = 100, stepsPerEpoch = 1000) {
      const data = await this.loadData();
      
      for (let epoch = 0; epoch < epochs; epoch++) {
        let totalReward = 0;
        
        for (let step = 0; step < stepsPerEpoch; step++) {
          // Get random starting point
          const startIdx = Math.floor(Math.random() * (data.length - this.dataProcessor.lookback - 2));
          
          // Get current state
          const currentState = this.dataProcessor.getState(startIdx);
          
          // Get action with noise for exploration
          const action = this.predict(currentState[currentState.length-1], true);
          
          // Get next state
          const nextState = this.dataProcessor.getState(startIdx + 1);
          
          // Calculate reward
          const reward = this.calculateReward(
            action,
            data.close[startIdx + 1],
            data.close[startIdx]
          );
          
          totalReward += reward;
          
          // Store experience
          this.replayBuffer.add(
            currentState[currentState.length-1],
            [action.positionSize, action.stopLoss, action.takeProfit],
            reward,
            nextState[nextState.length-1],
            false
          );
          
          // Train if we have enough samples
          if (this.replayBuffer.buffer.length >= this.batchSize) {
            await this.trainStep();
          }
        }
        
        console.log(`Epoch ${epoch + 1}/${epochs}, Average Reward: ${totalReward / stepsPerEpoch}`);
        
        // Save model weights periodically
        if ((epoch + 1) % 10 === 0) {
          await this.saveModels(`models_epoch_${epoch + 1}`);
        }
      }
    }
  
    async trainStep() {
      const batch = this.replayBuffer.sample(this.batchSize);
      
      // Calculate target Q-values
      batch.forEach(experience => {
        const { state, action, reward, nextState, done } = experience;
        
        // Get next action from target networks
        const nextHiddenState = this.targetLSTM.forward(nextState).hiddenState;
        const nextAction = this.targetDDPG.actorForward(nextHiddenState);
        
        // Calculate target Q-value
        const targetQ = reward + (done ? 0 : this.gamma * 
          this.targetDDPG.criticForward(nextHiddenState, nextAction));
        
        // Update networks
        this.updateNetworks(state, action, targetQ);
      });
      
      // Soft update target networks
      this.updateTargetNetworks(this.tau);
    }
  
    updateNetworks(state, action, targetQ) {
      // Get current Q-value
      const { hiddenState } = this.lstm.forward(state);
      const currentQ = this.ddpg.criticForward(hiddenState, action);
      
      // Calculate TD error
      const tdError = targetQ - currentQ;
      
      // Update critic
      this.ddpg.updateCritic(tdError);
      
      // Update actor using policy gradient
      const actorGradient = this.ddpg.getActorGradient(hiddenState);
      this.ddpg.updateActor(actorGradient);
      
      // Update LSTM
      const lstmGradient = this.lstm.backward(tdError);
      this.lstm.updateWeights(lstmGradient);
    }
  

    updateTargetNetworks(tau) {
        // Deep copy the weights instead of shallow copying
        this.targetDDPG.actorWeights = JSON.parse(JSON.stringify(this.ddpg.actorWeights));
        this.targetDDPG.criticWeights = JSON.parse(JSON.stringify(this.ddpg.criticWeights));
        
        // Apply soft update
        Object.keys(this.targetDDPG.actorWeights).forEach(key => {
            const target = this.targetDDPG.actorWeights[key];
            const source = this.ddpg.actorWeights[key];
            
            for (let i = 0; i < target.length; i++) {
                for (let j = 0; j < target[i].length; j++) {
                    target[i][j] = (1 - tau) * target[i][j] + tau * source[i][j];
                }
            }
        });

        Object.keys(this.targetDDPG.criticWeights).forEach(key => {
            const target = this.targetDDPG.criticWeights[key];
            const source = this.ddpg.criticWeights[key];
            
            for (let i = 0; i < target.length; i++) {
                for (let j = 0; j < target[i].length; j++) {
                    target[i][j] = (1 - tau) * target[i][j] + tau * source[i][j];
                }
            }
        });
    }
    // updateTargetNetworks(tau) {
    //   // Soft update target networks
    //   this.softUpdate(this.targetLSTM, this.lstm, tau);
    //   this.softUpdate(this.targetDDPG, this.ddpg, tau);
    // }

  
    softUpdate(target, source, tau) {
        // Implement soft update for network weights
        Object.keys(source).forEach(key => {
          if (Array.isArray(source[key])) {
            if (Array.isArray(source[key][0])) {
              // Handle 2D arrays (matrices)
              target[key] = target[key].map((row, i) =>
                row.map((val, j) => 
                  (1 - tau) * val + tau * source[key][i][j]
                )
              );
            } else {
              // Handle 1D arrays (vectors)
              target[key] = target[key].map((val, i) =>
                (1 - tau) * val + tau * source[key][i]
              );
            }
          } else {
            // Handle scalar values
            target[key] = (1 - tau) * target[key] + tau * source[key];
          }
        });
    }
  
    async saveModels(filename) {
        const modelData = {
            lstm: this.lstm,
            ddpg: this.ddpg.toJSON()
        };
        
        await Bun.write(`./saved_models/${filename}.json`, JSON.stringify(modelData));
    }
    
    async loadModels(filename) {
        const modelData = await Bun.file(`./saved_models/${filename}.json`).json();
        this.lstm = Object.assign(new LSTM(INPUT_SIZE, HIDDEN_SIZE), modelData.lstm);
        this.ddpg = DDPG.fromJSON(modelData.ddpg);
    }
  }

// Main execution
async function main() {
    const trader = new TradingSystem();
    
    // Training phase
    console.log("Starting training...");
    await trader.train(100, 1000); // 100 epochs, 1000 steps per epoch
    
    // Save final model
    await trader.saveModels('final_model');
    
    // Testing phase
    console.log("Testing model...");
    const data = await trader.loadData();
    
    // Example prediction
    const currentState = trader.dataProcessor.getState(25);
    const prediction = trader.predict(currentState[20]);
    
    console.log('Trading Decision:', prediction);
}
  
main().catch(console.error);