export class DataProcessor {
    constructor(data) {
      this.data = data;
      this.lookback = 20; // Previous days for range calculation
      this.dataPoints = 2000;
      this.features = ['open', 'high', 'low', 'close', 'volume'];

    }

    minMaxNormalize(array) {
        let min = Math.min(...array.slice(-this.dataPoints));
        let max = Math.max(...array.slice(-this.dataPoints));

        // Apply min-max normalization to each value
        return array.slice(-this.dataPoints).map(value => (value - min) / (max - min));
    }

    combineData(normalizedData) {
        const { open, high, low, close, volume } = normalizedData;
        return open.map((_, i) => [open[i], high[i], low[i], close[i], volume[i]]);
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
      const points = normalized.slice(normalized.length-1-index,normalized.length-1);
      const point = normalized[index];
      //console.log(points);
      
      return points
    //   return [
    //     point[0],//open
    //     point[1],//high
    //     point[2],//low
    //     point[3],//close
    //     point[4]//volume
    //   ];
    }
  }