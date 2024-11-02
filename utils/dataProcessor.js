export class DataProcessor {
    constructor(data) {
      this.data = data;
      this.lookback = 20; // Previous days for range calculation
      this.dataPoints = 4000;
      this.features = ['open', 'high', 'low', 'close', 'volume'];
      this.normalized;
      this.combined;

    }

    minMaxNormalize(array) {
        //slice array according to required data points before normalizing
        let min = Math.min(...this.data.low.slice(-this.dataPoints));
        let max = Math.max(...this.data.high.slice(-this.dataPoints));
        // Handle edge case where min equals max
        if (min === max) {
            return array.slice(-this.dataPoints).map(() => 0.5); // Return mid-point
        }
        return array.slice(-this.dataPoints).map(value => (value - min) / (max - min));
    }

    combineData(normalizedData) {
        const { open, high, low, close, volume } = normalizedData;
        //console.log(normalizedData);
        this.combined = open.map((_, i) => [open[i], high[i], low[i], close[i], volume[i]]);
        return this.combined
    }
  
    normalizeData(dataPoints = 4000) {
        this.dataPoints = dataPoints;
        this.normalized = this.combineData({
            open: this.minMaxNormalize(this.data.open),
            high: this.minMaxNormalize(this.data.high),
            low: this.minMaxNormalize(this.data.low),
            close: this.minMaxNormalize(this.data.close),
            volume: this.minMaxNormalize(this.data.volume)
        });
        
        //console.log(normalized2);
        return this.normalized;
    }
  
    calculateRange(window) {
      //console.log(window);
      //const highs = window.map(d => d.high);
      //const lows = window.map(d => d.low);
      if (window)return Math.max(...window) - Math.min(...window);
      else return 0;
    }
  
    calculateVolumeMA(window) {
      const volumes = window.map(d => d.volume);
      return volumes.reduce((a, b) => a + b, 0) / window.length;
    }
  
    async getState(index) {
        //const normalized = this.normalizeData();
        // Ensure index is within bounds
        if (index < this.lookback + 5 || index >= this.normalized.length - this.lookback - 5 ) {
            throw new Error(`Index ${index} out of bounds`);
        }
        const sliced = this.normalized.slice(Math.max(0, index - this.lookback), index);
        let closePrices = [];
        
        for (let i = 0; i < this.lookback; i++) {
            closePrices.push(sliced[i][3]);
        }
        //console.log(closePrices);
        return closePrices;
    }
  }