import { LSTM } from './models/lstm.js';
import { DDPG } from './models/ddpg.js';
import { DataProcessor } from './utils/dataProcessor.js';

const INPUT_SIZE = 20;  // OHLCV
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
        this.batchSize = 1000;
        this.gamma = 0.99; // Discount factor
        this.tau = 0.001; // Soft update parameter
        this.barsPredicted = 5;
        
        // Create target networks
        this.targetLSTM = new LSTM(INPUT_SIZE, HIDDEN_SIZE);
        this.targetDDPG = new DDPG(HIDDEN_SIZE, ACTION_SIZE);
        
        // Copy initial weights
        this.updateTargetNetworks(1.0);
    }
  
    async loadData() {
      const data = await Bun.file('./data/EURUSD_D1.json').json();
      this.dataProcessor = new DataProcessor(data);
      return data;
    }
  
    predict(state, addNoise = false) {
        // Ensure state is properly normalized
        //console.log(state);
        console.warn('PREDICTION')
        if (!Array.isArray(state) || state.some(val => isNaN(val))) {
          console.log("state: ", state);
          throw new Error('Invalid state values detected');
        }
  
        const { hiddenState } = this.lstm.forward(state);
        let actions = this.ddpg.actorForward(hiddenState);
        //console.log("state ",state, "\nactions ",actions);
        
        if (addNoise) {
          // Use Ornstein-Uhlenbeck noise for better exploration
          actions = actions.map(a => {
            const theta = 0.15;
            const sigma = 0.2;
            const noise = -theta * a + sigma * (Math.random() * 2 - 1);
            return Math.max(-1, Math.min(1, a + noise)); // Clip between -1 and 1
          });
        }
        
        // Ensure actions are within valid ranges
       // Custom position size logic keep between 0.01 and 1
        let positionSize = actions[0]*50;
        if (positionSize < 0.01 && positionSize >= 0) {
            positionSize = 0.01;
        } else if (positionSize > -0.01 && positionSize < 0) {
            positionSize = -0.01;
        } else {
            positionSize = Math.max(-1, Math.min(1, positionSize));
            positionSize = positionSize.toFixed(2);
        }
        
        const stopLoss = Math.max(-1, Math.min(1.0, (actions[1]) )); // Convert to 0.1-0.5 range
        const takeProfit = Math.max(-1, Math.min(1.0, (actions[2]) )); // Convert to 0.2-1.0 range
        
        console.log("[position] [SL] [TP]", positionSize,stopLoss,takeProfit);

        return {
          positionSize,
          stopLoss,
          takeProfit,
          hiddenState
        };
      }
  
    calculateReward(action, nextPrice, currentPrice, index, currentState) {
        if (isNaN(nextPrice) || isNaN(currentPrice) || isNaN(action.positionSize)) {
            console.error('Invalid inputs in calculateReward:', { nextPrice, currentPrice, action });
            return 0;
        }
        //lower than TP or Entry stoploss for Sell BLOCK
        else if(action.positionSize < 0 && (action.stopLoss < action.takeProfit || action.stopLoss < 0 || action.takeProfit > 0)) {
            console.log("INVALID TRADE");
            return 0;
        }
        
        //Higher than TP or Entry stoploss for Buy BLOCK
        else if(action.positionSize > 0 && (action.stopLoss > action.takeProfit || action.stopLoss > 0 || action.takeProfit < 0)) {
            console.log("INVALID TRADE");
            return 0;
        }
       
        else {
            const checkStoploss = (indx) => {
                const newCandle = this.dataProcessor.normalized[indx + this.barsPredicted - 1]; //o,h,l,c,v
                const range = this.dataProcessor.calculateRange(currentState);
                console.log("range ",range)
                const sl = action.positionSize > 0 ? 
                            currentPrice - (range * Math.sqrt(action.stopLoss*action.stopLoss)) : 
                            currentPrice + (range * Math.sqrt(action.stopLoss*action.stopLoss));
                const tp = action.positionSize > 0 ? 
                            currentPrice + (range * Math.sqrt(action.takeProfit*action.takeProfit)) : 
                            currentPrice - (range * Math.sqrt(action.takeProfit*action.takeProfit));

                let tradeHighs= [], tradeLows = [], tradeHigh, tradeLow;
                for (let i = indx; i < indx + this.barsPredicted; i++) {
                  tradeHighs.push(this.dataProcessor.normalized[i][1] );
                  tradeLows.push(this.dataProcessor.normalized[i][2]);
                  
                }
                console.log("trade bounds(H,L): \n", tradeHighs,"\n", tradeLows);
                console.log("Entry: ",currentPrice," LastClose: ", newCandle ? newCandle[3]: 0," SL: ",sl," TP: ",tp);
                if (action.positionSize > 0 && newCandle) { //BUY
                    if (/*newCandle[2]*/ Math.min(...tradeLows) < sl) {
                        console.log("SL HIT");
                        const priceChange = (sl - currentPrice) / currentPrice;
                        const scaledPriceChange = Math.tanh(priceChange * 10); // Scale and bound between -1 and 1
                        const reward = (scaledPriceChange) * action.positionSize;
                        return reward;
                        //return -1* Math.sqrt(action.positionSize*action.positionSize);
                    }//if newCandle low is lower than SL
                    if (/*newCandle[1]*/ Math.max(...tradeHighs) > tp || newCandle[3] > tp ) {//TP is hit
                        console.log("TP HIT");
                        const priceChange = (tp - currentPrice) / currentPrice;
                        const scaledPriceChange = Math.tanh(priceChange * 10); // Scale and bound between -1 and 1
                        const reward = (scaledPriceChange) * action.positionSize;
                        return reward;
                    }
                } else if (action.positionSize < 0 && newCandle) { //SELL
                    if (Math.max(...tradeHighs) > sl) {
                        console.log("SL HIT");
                        const priceChange = (sl - currentPrice) / currentPrice;
                        const scaledPriceChange = Math.tanh(priceChange * 10); // Scale and bound between -1 and 1
                        const reward = (scaledPriceChange) * action.positionSize;
                        return reward;
                        // return -1* Math.sqrt(action.positionSize*action.positionSize);
                    } //if newCandle High is higher than SL
                    if (Math.min(...tradeLows) < tp || newCandle[3] < tp ) {//TP is Hit
                        console.log("TP HIT");
                        const priceChange = (currentPrice - tp) / currentPrice;
                        const scaledPriceChange = Math.tanh(priceChange * 10); // Scale and bound between -1 and 1
                        const reward = (scaledPriceChange) * action.positionSize;
                        return reward;
                    }
                } else {
                    // Scale price change to reasonable range
                    console.log("FLOATING PROFIT");
                    const priceChange = (currentPrice - nextPrice) / currentPrice;
                    const scaledPriceChange = Math.tanh(priceChange * 10); // Scale and bound between -1 and 1
                    
                    // Calculate scaled PnL
                    const reward = (scaledPriceChange) * action.positionSize;
                    return reward;
                }
            }
            const pnl = checkStoploss(index);
            
            
            // Risk management penalties (scaled)
            const slPenalty = action.stopLoss < 0.03 ? -0.1 : 
                            action.stopLoss > 0.5 ? -0.1 : 0;
            
            const tpPenalty = action.takeProfit < 0.05 ? -0.1 :
                            action.takeProfit > 1.0 ? -0.1 : 0;
            
            const sizePenalty = Math.abs(action.positionSize) > 0.8 ? -0.05 : 0;
            
            // Combine rewards with appropriate scaling
            const totalReward = pnl + slPenalty + tpPenalty + sizePenalty;
            
            return Math.max(-1, Math.min(2, totalReward));
        }
    }
  

    async train(epochs = 100, stepsPerEpoch = 1000) {
        const data = await this.loadData();
        this.dataProcessor.normalizeData();
        console.log(this.dataProcessor.normalized.length);
        //this.dataProcessor.combineData(this.dataProcessor.normalized);

        
        let bestReward = -Infinity;
        let noImprovementCount = 0;
        const patience = 20; // Number of epochs without improvement before stopping
    
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalReward = 0;
            let validSteps = 0;
            
            for (let step = 0; step < stepsPerEpoch; step++) {
                try {
                    const startIdx = this.getRandomIndex();//Math.floor(Math.random() * (this.dataProcessor.normalized.length - this.dataProcessor.lookback - 2));
                    const currentState = await this.dataProcessor.getState(startIdx);
                    console.log("\nSTEP\nstartIdx ", startIdx); //, "currentState ", currentState.length);

                    const action = this.predict(currentState, true);
                    const nextState = await this.dataProcessor.getState(startIdx + this.barsPredicted - 1);

                    
                    //console.log("price change: ",nextState[nextState.length - 1] - currentState[currentState.length-1]);
                    const reward = this.calculateReward(
                        action,
                        nextState[nextState.length - 1], 
                        currentState[currentState.length-1],
                        startIdx,
                        currentState
                    );
                    console.log('step: ',step," reward: ",reward, "\n\n");

                    if (!isNaN(reward)) {
                        totalReward += reward;
                        validSteps++;
                        
                        this.replayBuffer.add(
                            currentState,
                            [action.positionSize, action.stopLoss, action.takeProfit],
                            reward,
                            nextState,
                            false
                        );
                    }


                    
                    if (this.replayBuffer.buffer.length >= this.batchSize) {
                        this.replayBuffer.buffer = this.replayBuffer.buffer.slice(-this.batchSize);
                        const loss = await this.trainStep();
                        if (step % 1 === 0) {
                            console.log(`Epoch ${epoch}, Step ${step}, Loss: ${loss.toFixed(4)}`);
                            //await Bun.sleep(2000);
                        }
                    }
                } catch (error) {
                    console.error(`Error at epoch ${epoch}, step ${step}:`, error);
                    
                    continue;
                }
            }
            
            // Validation phase
            const validationReward = await this.validate();
            
            // Early stopping check
            if (validationReward > bestReward) {
                bestReward = validationReward;
                noImprovementCount = 0;
                await this.saveModels(`best_model`);
            } else {
                noImprovementCount++;
                if (noImprovementCount >= patience) {
                    console.log(`\n\n~Early stopping at epoch ${epoch}\n\n`);
                    break;
                }
            }
            const averageReward = validSteps > 0 ? totalReward / validSteps : 0;
            console.log(`Epoch ${epoch + 1}/${epochs}, Average Reward: ${averageReward.toFixed(4)}, Valid Steps: ${validSteps}`);
            
            //await Bun.sleep(2000);
            // Save model weights periodically
            if ((epoch + 1) % 10 === 0) {
                await this.saveModels(`models_epoch_${epoch + 1}`);
            }
        }
    }

    async validate(steps = 100) {
        let totalReward = 0;
        
        for (let i = 0; i < steps; i++) {
            const startIdx = this.getRandomIndex();
            const state = await this.dataProcessor.getState(startIdx);
            const action = this.predict(state, false); // No noise during validation
            
            const reward = this.calculateReward(
                action,
                this.dataProcessor.normalized[startIdx + this.barsPredicted - 1][4],
                this.dataProcessor.normalized[startIdx][4]
            );
            
            if (!isNaN(reward)) {
                totalReward += reward;
            }
        }
        
        return totalReward / steps;
    }

    getRandomIndex() {
        const min = this.dataProcessor.lookback + this.barsPredicted;
        const max = this.dataProcessor.normalized.length - 2*(min) - this.barsPredicted;
        const randomNumber = Math.floor(Math.random() * (max - min + 1)) + min;
        return randomNumber;
      }
  
    async trainStep() {
        const batch = this.replayBuffer.sample(this.batchSize);
        let totalLoss = 0;
        
        for (const experience of batch) {
          const { state, action, reward, nextState, done } = experience;
          try {
            // Get next action from target networks
            //console.log("trainingStep next state: ", nextState);
            const nextHiddenState = this.targetLSTM.forward(nextState).hiddenState;
            const nextAction = this.targetDDPG.actorForward(nextHiddenState);
            
            // Calculate target Q-value with clipping
            const nextQ = this.targetDDPG.criticForward(nextHiddenState, nextAction);
            const targetQ = reward + (done ? 0 : this.gamma * Math.max(-10, Math.min(10, nextQ)));
            
            // Update networks if targetQ is valid
            if (!isNaN(targetQ)) {
              // Get current Q-value
              const { hiddenState } = this.lstm.forward(state);
              const currentQ = this.ddpg.criticForward(hiddenState, action);
              
              // Calculate TD error with clipping
              const tdError = Math.max(-1, Math.min(1, targetQ - currentQ));
              
              // Update networks
              this.ddpg.updateCritic(tdError);
              const actorGradient = this.ddpg.getActorGradient(hiddenState);
              
              // Clip gradients
              const clippedGradient = actorGradient.map(g => Math.max(-1, Math.min(1, g)));
              this.ddpg.updateActor(clippedGradient);
              
              // Update LSTM
              const lstmGradient = this.lstm.backward(tdError);
              this.lstm.updateWeights(lstmGradient);
              
              totalLoss += Math.abs(tdError);
            }
          } catch (error) {
            console.error('Error in training step:', error);
            continue;
          }
        }
        
        // Soft update target networks
        this.updateTargetNetworks(this.tau);
        
        return totalLoss / batch.length;
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
    console.log("\nStarting training...");
    await trader.train(100, 100); // 100 epochs, 1000 steps per epoch
    
    // Save final model
    await trader.saveModels('final_model');
    
    // Testing phase
    console.log("\nTesting model...");
    const data = await trader.loadData();
    console.log("data length ", data.open.length);
    await Bun.sleep(3000);
    
    // Example prediction
    for(let i = 24; i < 96; i++) {
        const currentState = await trader.dataProcessor.getState(i);
        const prediction = trader.predict(currentState);
        const nextState = await trader.dataProcessor.getState(i+1);

        console.log('Trading Decision: Position/Dir[', prediction.positionSize,'] SL[', prediction.stopLoss,'] TP[', prediction.takeProfit,']');
        console.log('price change: ', nextState[nextState.length-1] - currentState[currentState.length-1],'\n');
    }
}
  
main().catch(console.error);