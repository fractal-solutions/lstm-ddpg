export class DataProcessor {
    constructor(data) {
      this.data = data;
      this.lookback = 20; // Previous days for range calculation
      this.dataPoints = 2000;
      this.features = ['open', 'high', 'low', 'close', 'volume'];
      this.normalized;
      this.combined;

    }

    minMaxNormalize(array) {
        let min = Math.min(...array.slice(-this.dataPoints));
        let max = Math.max(...array.slice(-this.dataPoints));
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
  
    normalizeData(dataPoints = 2000) {
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
      const highs = window.map(d => d.high);
      const lows = window.map(d => d.low);
      return Math.max(...highs) - Math.min(...lows);
    }
  
    calculateVolumeMA(window) {
      const volumes = window.map(d => d.volume);
      return volumes.reduce((a, b) => a + b, 0) / window.length;
    }
  
    getState(index) {
        const normalized = this.normalizeData();
        // Ensure index is within bounds
        if (index < this.lookback || index >= normalized.length - this.lookback ) {
            throw new Error(`Index ${index} out of bounds`);
        }
        const sliced = normalized.slice(Math.max(0, index - this.lookback), index);
        let closePrices = [];
        for (let i = 0; i < this.lookback; i++) {
            closePrices.push(sliced[i][4]);
        }
        return closePrices;
    }
  }