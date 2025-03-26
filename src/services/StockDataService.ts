import { StockData, CandlestickData } from '../types';

export class StockDataService {
  // Cache for generated demo data
  private static demoDataCache: Record<string, CandlestickData[]> = {};

  static readonly API_BASE_URL = 'http://localhost:5000/api';

  static async fetchStockList(): Promise<StockData[]> {
    try {
      console.log('Attempting to fetch stock list from CSV...');
      
      try {
        // First check if file exists and is accessible
        const checkResponse = await fetch('/stock_data_fixed.csv', { method: 'HEAD' });
        if (!checkResponse.ok) {
          console.warn(`CSV file check failed: ${checkResponse.status}. Falling back to demo data.`);
          return this.generateDemoStockList();
        }
        
        const response = await fetch('/stock_data_fixed.csv');
        if (!response.ok) {
          throw new Error(`Failed to fetch stock data: ${response.status}`);
        }
        
        const csv = await response.text();
        console.log('CSV data received, length:', csv.length);
        
        if (csv.trim().length < 10) {
          console.warn('CSV file is empty or too small. Using demo data.');
          return this.generateDemoStockList();
        }
        
        // Parse CSV
        const rows = csv.trim().split('\n');
        if (rows.length < 2) {
          console.warn('CSV has insufficient rows. Using demo data.');
          return this.generateDemoStockList();
        }
        
        const headers = rows[0].split(',');
        
        // Find column indices
        const tickerIndex = headers.indexOf('Ticker');
        const openIndex = headers.indexOf('Open');
        const closeIndex = headers.indexOf('Close');
        const volumeIndex = headers.indexOf('Volume');
        
        if (tickerIndex === -1 || openIndex === -1 || closeIndex === -1 || volumeIndex === -1) {
          console.error('CSV is missing required columns. Found headers:', headers);
          return this.generateDemoStockList();
        }
        
        // Process by ticker
        const latestByTicker = {};
        
        // Process each row
        for (let i = 1; i < rows.length; i++) {
          const values = rows[i].split(',');
          
          // Skip rows with wrong column count
          if (values.length !== headers.length) continue;
          
          const ticker = values[tickerIndex];
          if (!ticker) continue; // Skip if no ticker
          
          // For each ticker, keep only the latest data
          if (!latestByTicker[ticker] || i > latestByTicker[ticker].rowIndex) {
            latestByTicker[ticker] = {
              rowIndex: i,
              open: parseFloat(values[openIndex]),
              close: parseFloat(values[closeIndex]),
              volume: parseFloat(values[volumeIndex])
            };
          }
        }
        
        // Convert to StockData array
        const stocks: StockData[] = Object.entries(latestByTicker).map(([symbol, data]) => {
          const change = ((data.close - data.open) / data.open) * 100;
          
          return {
            symbol,
            price: data.close,
            change: change,
            volume: data.volume
          };
        });
        
        if (stocks.length === 0) {
          console.warn('No stocks extracted from CSV. Using demo data.');
          return this.generateDemoStockList();
        }
        
        console.log(`Loaded ${stocks.length} stocks from CSV`);
        return stocks;
      } catch (e) {
        console.error('Error parsing CSV:', e);
        return this.generateDemoStockList();
      }
    } catch (error) {
      console.error('Error fetching stock list:', error);
      return this.generateDemoStockList();
    }
  }
  
  static async fetchCandlestickData(symbol: string, timeRange: string): Promise<CandlestickData[]> {
    if (!symbol) {
      console.error('Symbol is undefined or empty');
      return [];
    }
    
    try {
      // Determine which CSV file to use based on time range
      const csvFile = timeRange === '1d' ? '/stock_data_intraday.csv' : '/stock_data_fixed.csv';
      console.log(`Fetching ${timeRange} data for ${symbol} from ${csvFile}`);
      
      try {
        // First check if file exists
        const checkResponse = await fetch(csvFile, { method: 'HEAD' });
        if (!checkResponse.ok) {
          console.warn(`File check failed for ${csvFile}: ${checkResponse.status}. Using demo data.`);
          return this.generateDemoCandlestickData(symbol, timeRange);
        }
        
        const response = await fetch(csvFile);
        if (!response.ok) {
          throw new Error(`Failed to fetch ${csvFile}: ${response.status}`);
        }
        
        const csv = await response.text();
        console.log(`CSV data received from ${csvFile}, length: ${csv.length}`);
        
        if (csv.trim().length < 10) {
          console.warn('CSV file is empty or too small. Using demo data.');
          return this.generateDemoCandlestickData(symbol, timeRange);
        }
        
        // Parse CSV
        const rows = csv.trim().split('\n');
        if (rows.length < 2) {
          console.warn('CSV has insufficient rows. Using demo data.');
          return this.generateDemoCandlestickData(symbol, timeRange);
        }
        
        // Special handling for intraday CSV format which has different column structure
        if (timeRange === '1d') {
          console.log('Processing intraday CSV format...');
          
          // For intraday CSV: [Date/Time,Open,High,Low,Close,Volume,Ticker] 
          // First column might be unnamed
          const headers = rows[0].split(',');
          console.log('Intraday CSV headers:', headers);
          
          // Find column indices (accounting for possible blank first column)
          const dateIndex = 0;  // First column is always date/time in intraday CSV
          const openIndex = 1;  // Open is second column
          const highIndex = 2;  // High is third column
          const lowIndex = 3;   // Low is fourth column
          const closeIndex = 4; // Close is fifth column
          const volumeIndex = 5; // Volume is sixth column
          const tickerIndex = 6; // Ticker is seventh column
          
          // Filter and parse the data
          const filteredData: CandlestickData[] = [];
          
          for (let i = 1; i < rows.length; i++) {
            const values = rows[i].split(',');
            
            // Skip rows with wrong column count
            if (values.length < 7) {
              continue;
            }
            
            // Process only rows for the requested symbol
            if (values[tickerIndex] === symbol) {
              // For intraday, format timestamp for chart library
              const timestamp = new Date(values[dateIndex]).getTime() / 1000;
              
              filteredData.push({
                time: timestamp,
                open: parseFloat(values[openIndex]),
                high: parseFloat(values[highIndex]),
                low: parseFloat(values[lowIndex]),
                close: parseFloat(values[closeIndex]),
                volume: parseInt(values[volumeIndex], 10)
              });
            }
          }
          
          console.log(`Found ${filteredData.length} intraday candles for ${symbol}`);
          return filteredData;
        } else {
          // Original CSV parsing logic for daily data
          const headers = rows[0].split(',');
        
          // Find column indices
          const dateIndex = headers.indexOf('Date');
          const tickerIndex = headers.indexOf('Ticker');
          const openIndex = headers.indexOf('Open');
          const highIndex = headers.indexOf('High');
          const lowIndex = headers.indexOf('Low');
          const closeIndex = headers.indexOf('Close');
          const volumeIndex = headers.indexOf('Volume');
          
          // Check if all required columns are present
          if (dateIndex === -1 || tickerIndex === -1 || openIndex === -1 || 
              highIndex === -1 || lowIndex === -1 || closeIndex === -1 || volumeIndex === -1) {
            console.error('CSV is missing required columns. Found headers:', headers);
            return this.generateDemoCandlestickData(symbol, timeRange);
          }
          
          // Filter and parse the data
          const filteredData: CandlestickData[] = [];
          
          for (let i = 1; i < rows.length; i++) {
            const values = rows[i].split(',');
            
            // Skip rows with wrong column count
            if (values.length !== headers.length) {
              console.warn(`Row ${i} has incorrect column count, skipping`);
              continue;
            }
            
            // Process only rows for the requested symbol
            if (values[tickerIndex] === symbol) {
              filteredData.push({
                time: values[dateIndex],
                open: parseFloat(values[openIndex]),
                high: parseFloat(values[highIndex]),
                low: parseFloat(values[lowIndex]),
                close: parseFloat(values[closeIndex]),
                volume: parseInt(values[volumeIndex], 10)
              });
            }
          }
          
          console.log(`Found ${filteredData.length} data points for ${symbol}`);
          
          if (filteredData.length === 0) {
            console.warn(`No data found for symbol ${symbol}. Generating demo data.`);
            return this.generateDemoCandlestickData(symbol, timeRange);
          }
          
          // Sort by date ascending
          filteredData.sort((a, b) => {
            return new Date(a.time).getTime() - new Date(b.time).getTime();
          });
          
          return filteredData;
        }
      } catch (e) {
        console.error('Error parsing CSV:', e);
        return this.generateDemoCandlestickData(symbol, timeRange);
      }
    } catch (error) {
      console.error('Error fetching candlestick data:', error);
      return this.generateDemoCandlestickData(symbol, timeRange);
    }
  }
  
  // Generate demo stock list when CSV loading fails
  static generateDemoStockList(): StockData[] {
    console.log('Generating demo stock list');
    return [
      { symbol: 'AAPL', price: 220.45, change: 1.25, volume: 67904500 },
      { symbol: 'MSFT', price: 425.63, change: 0.62, volume: 25371900 },
      { symbol: 'GOOGL', price: 175.98, change: -0.47, volume: 21563200 },
      { symbol: 'AMZN', price: 178.75, change: 2.14, volume: 32584100 },
      { symbol: 'TSLA', price: 177.82, change: -1.65, volume: 125687400 },
      { symbol: 'META', price: 485.92, change: 3.27, volume: 15789200 },
      { symbol: 'NVDA', price: 925.75, change: 4.53, volume: 35462800 },
      { symbol: 'JPM', price: 196.82, change: 0.28, volume: 8796540 },
      { symbol: 'NFLX', price: 610.34, change: 1.92, volume: 5467890 },
      { symbol: 'IBM', price: 168.76, change: -0.36, volume: 3764290 }
    ];
  }
  
  // Generate realistic candlestick data for demo purposes
  static generateDemoCandlestickData(symbol: string, timeRange: string): CandlestickData[] {
    // Check cache first
    const cacheKey = `${symbol}_${timeRange}`;
    if (this.demoDataCache[cacheKey]) {
      console.log(`Using cached demo data for ${symbol}`);
      return this.demoDataCache[cacheKey];
    }
    
    console.log(`Generating demo candlestick data for ${symbol}`);
    
    // Set base parameters based on the symbol
    let basePrice = 100;
    let volatility = 0.02; // 2% daily volatility by default
    
    switch (symbol) {
      case 'AAPL':
        basePrice = 176.54;
        volatility = 0.015;
        break;
      case 'MSFT':
        basePrice = 382.77;
        volatility = 0.018;
        break;
      case 'GOOGL':
        basePrice = 142.84;
        volatility = 0.017;
        break;
      case 'AMZN':
        basePrice = 178.75;
        volatility = 0.022;
        break;
      case 'TSLA':
        basePrice = 177.82;
        volatility = 0.035;
        break;
      case 'META':
        basePrice = 485.92;
        volatility = 0.025;
        break;
      case 'NVDA':
        basePrice = 925.75;
        volatility = 0.03;
        break;
      case 'JPM':
        basePrice = 196.82;
        volatility = 0.014;
        break;
      case 'NFLX':
        basePrice = 610.34;
        volatility = 0.028;
        break;
      case 'IBM':
        basePrice = 168.76;
        volatility = 0.012;
        break;
    }
    
    // Determine data points based on time range
    let days = 30; // default for 1m
    switch (timeRange) {
      case '1d':
        days = 1;
        break;
      case '1w':
        days = 7;
        break;
      case '1m':
        days = 30;
        break;
      case '1y':
        days = 252; // Trading days in a year
        break;
      case '5y':
        days = 252 * 5;
        break;
    }
    
    const result: CandlestickData[] = [];
    const now = new Date();
    let currentPrice = basePrice;
    
    // For intraday, generate minute data
    if (timeRange === '1d') {
      // Trading hours: 9:30 AM - 4:00 PM
      const marketOpen = new Date(now);
      marketOpen.setHours(9, 30, 0, 0);
      
      const marketClose = new Date(now);
      marketClose.setHours(16, 0, 0, 0);
      
      // Handle weekends and after-hours
      const currentTime = new Date();
      const day = currentTime.getDay();
      if (day === 0) { // Sunday
        marketOpen.setDate(marketOpen.getDate() - 2);
        marketClose.setDate(marketClose.getDate() - 2);
      } else if (day === 6) { // Saturday
        marketOpen.setDate(marketOpen.getDate() - 1);
        marketClose.setDate(marketClose.getDate() - 1);
      }
      
      // Generate data every 5 minutes - ensure we have multiple candles
      let time = new Date(marketOpen);
      
      // Make sure base price is close to the stock's current price
      const stockInfo = this.generateDemoStockList().find(s => s.symbol === symbol);
      if (stockInfo) {
        currentPrice = stockInfo.price;
      }
      
      // If it's the first data point, set a slightly different open
      let previousClose = currentPrice * (1 + (Math.random() * 0.01 - 0.005));
      
      while (time <= marketClose) {
        // Random price movement based on volatility and direction bias
        let bias = 0.2; // Slight upward bias
        if (symbol === 'TSLA' || symbol === 'GOOGL' || symbol === 'MSFT') {
          bias = -0.1; // Slight downward bias for certain stocks
        }
        
        const minuteVolatility = volatility / Math.sqrt(252 * 78); // Scale daily vol to 5-min vol
        const change = currentPrice * minuteVolatility * ((Math.random() * 2 - 1) + bias);
        currentPrice += change;
        
        // For the first candle, use previousClose as the open price
        const open = (time.getTime() === marketOpen.getTime()) ? previousClose : currentPrice * (1 + (Math.random() * 0.002 - 0.001));
        
        // Make close slightly different from open
        const close = currentPrice;
        
        // High and low calculated from open/close
        const high = Math.max(open, close) * (1 + Math.random() * 0.002);
        const low = Math.min(open, close) * (1 - Math.random() * 0.002);
        
        // Format time as ISO string with seconds for proper chart display
        const timeStr = time.toISOString().replace('T', ' ').substring(0, 19);
        
        result.push({
          time: timeStr,
          open,
          high,
          low,
          close,
          volume: Math.floor(50000 + Math.random() * 450000) // Volume between 50k-500k
        });
        
        // Add 5 minutes
        time = new Date(time.getTime() + 5 * 60000);
      }
      
      // Ensure we have at least 78 data points (6.5 hours × 12 5-min intervals)
      if (result.length < 78) {
        console.warn(`Generated only ${result.length} intraday candles, expected 78+`);
      }
    } else {
      // For daily data, go back the required number of days
      for (let i = days; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        
        // Skip weekends
        if (date.getDay() === 0 || date.getDay() === 6) {
          continue;
        }
        
        // Random daily price movement
        const dailyChange = currentPrice * volatility * (Math.random() * 2 - 1);
        currentPrice += dailyChange;
        
        const open = currentPrice;
        const close = currentPrice * (1 + (Math.random() * 0.016 - 0.008)); // ±0.8%
        const high = Math.max(open, close) * (1 + Math.random() * 0.01); // Up to 1% higher
        const low = Math.min(open, close) * (1 - Math.random() * 0.01); // Up to 1% lower
        
        result.push({
          time: date.toISOString().split('T')[0], // YYYY-MM-DD
          open,
          high,
          low,
          close,
          volume: Math.floor(500000 + Math.random() * 4500000) // Volume between 500k-5M
        });
      }
    }
    
    // Sort by date (ascending)
    result.sort((a, b) => {
      return new Date(a.time).getTime() - new Date(b.time).getTime();
    });
    
    // Cache the result
    this.demoDataCache[cacheKey] = result;
    
    console.log(`Generated ${result.length} demo candlesticks for ${symbol}`);
    return result;
  }

  static async getPrediction(symbol: string): Promise<any> {
    console.log(`Getting prediction for ${symbol}...`);
    
    try {
      // ONLY use the ML prediction from summary CSV - no fallbacks
      console.log("Fetching ML prediction from prediction_summary.csv...");
      const response = await fetch(`${this.API_BASE_URL}/prediction-summary/${symbol}`);
      
      if (!response.ok) {
        throw new Error(`No prediction data found for ${symbol} in prediction_summary.csv`);
      }
      
      const data = await response.json();
      console.log("ML prediction response:", data);
      
      // Map CSV data to UI format
      return this.mapCsvPredictionToUi(symbol, data);
    } catch (error) {
      console.error("Error fetching ML prediction:", error);
      
      // Instead of fallbacks, throw an error that will be displayed to the user
      throw new Error(`No ML prediction available for ${symbol}. Please check prediction_summary.csv file.`);
    }
  }
  
  private static mapCsvPredictionToUi(symbol: string, csvData: any): any {
    try {
      console.log("Raw CSV data from prediction_summary.csv:", csvData);
      
      // Extract values directly from the CSV
      const currentPrice = parseFloat(csvData['Current Price'] || 0);
      const prediction = csvData['Prediction'] || "";
      const probability = parseFloat(csvData['Probability (%)'] || 0);
      const confidence = parseFloat(csvData['Confidence (%)'] || 0);
      const recentChange = parseFloat(csvData['Recent Change (%)'] || 0);
      const factorsRaw = csvData['Factors'] || "";
      
      // Parse factors without modifying them
      const factors = factorsRaw.split(',').map(f => f.trim());
      
      // FIX HERE: Check for exact match of "Likely profitable"
      // This fixes the issue where "Likely unprofitable" was incorrectly detected as profitable
      const isProfitable = prediction.toLowerCase() === 'likely profitable';
      
      // Generate recommendation directly from prediction
      const recommendation = isProfitable ? "Buy" : "Sell";
      
      console.log(`${symbol} prediction: "${prediction}" → isProfitable: ${isProfitable} → recommendation: ${recommendation}`);
      
      return {
        isProfitable: isProfitable,
        confidence: confidence,
        factors: factors,
        prediction: prediction,
        recommendation: recommendation,
        targetPrice: null,
        stopLoss: null,
        potentialReturn: null,
        timeframe: null,
        riskLevel: null,
        currentPrice: currentPrice,
        probabilityScore: probability
      };
    } catch (e) {
      console.error("Error mapping CSV prediction data:", e);
      throw new Error(`Cannot process prediction data for ${symbol}: ${e.message}`);
    }
  }

  private static getFallbackPrediction(symbol: string): any {
    console.log("Using fallback prediction for", symbol);
    
    // Create a deterministic but "random-seeming" prediction based on symbol name
    const hash = symbol.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const isBuy = hash % 2 === 0;
    
    return {
        isProfitable: isBuy,
        confidence: 65 + (hash % 20),
        factors: [
            "Synthetic prediction (API unavailable)",
            "Based on historical patterns",
            "Market sentiment analysis"
        ],
        prediction: isBuy ? "Price likely to increase" : "Price may decrease",
        recommendation: isBuy ? "Buy" : "Sell",
        targetPrice: 100 + (hash % 50),
        stopLoss: 100 - (hash % 15),
        potentialReturn: 5 + (hash % 10),
        timeframe: "Medium-term (1-3 months)",
        riskLevel: "Medium"
    };
  }

  private static runPredictionAlgorithm(symbol: string, data: CandlestickData[]): {
    isProfitable: boolean;
    confidence: number;
    factors: string[];
    prediction: string;
  } {
    // Seed with symbol to make predictions consistent for same stock
    const hash = symbol.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const seed = hash % 100;
    
    // Calculate technical indicators
    const prices = data.map(d => d.close);
    
    // 1. Price trend analysis
    const recentPrices = prices.slice(-20);
    const priceChange = (recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentPrices[0] * 100;
    
    // 2. Volatility analysis
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i-1]) / prices[i-1]);
    }
    const volatility = Math.sqrt(
      returns.reduce((sum, ret) => sum + ret * ret, 0) / returns.length
    ) * Math.sqrt(252) * 100; // Annualized volatility
    
    // 3. Volume trend
    const volumes = data.map(d => d.volume);
    const recentVolumes = volumes.slice(-10);
    const volumeChange = (
      recentVolumes.reduce((sum, vol) => sum + vol, 0) / recentVolumes.length
    ) / (
      volumes.slice(-20, -10).reduce((sum, vol) => sum + vol, 0) / 10
    ) - 1;
    
    // 4. Moving averages
    const ma50 = prices.slice(-50).reduce((sum, price) => sum + price, 0) / 50;
    const ma200 = prices.length >= 200 
      ? prices.slice(-200).reduce((sum, price) => sum + price, 0) / 200
      : ma50;
    
    // Calculate probability of profit
    let profitProbability = 50; // Base probability
    
    // Adjust based on price trend
    profitProbability += priceChange * 2;
    
    // Adjust based on moving average relationship
    if (ma50 > ma200) {
      profitProbability += 10; // Golden cross is bullish
    } else {
      profitProbability -= 10; // Death cross is bearish
    }
    
    // Adjust based on volume
    if (volumeChange > 0.05) {
      profitProbability += 5; // Increasing volume supports trend
    } else if (volumeChange < -0.05) {
      profitProbability -= 5; // Decreasing volume suggests weakness
    }
    
    // Adjust based on volatility
    if (volatility > 30) {
      profitProbability -= 5; // High volatility increases risk
    } else if (volatility < 15) {
      profitProbability += 5; // Low volatility suggests stability
    }
    
    // Adjust with symbol-specific bias based on seed
    profitProbability += (seed - 50) * 0.2;
    
    // Clamp probability between 0-100
    profitProbability = Math.min(98, Math.max(2, profitProbability));
    
    // Determine if profitable
    const isProfitable = profitProbability > 50;
    
    // Generate factors that influenced decision
    const factors = [];
    
    if (Math.abs(priceChange) > 2) {
      factors.push(`${priceChange.toFixed(1)}% price ${priceChange > 0 ? 'increase' : 'decrease'} in last 20 days`);
    }
    
    if (ma50 > ma200) {
      factors.push('50-day MA above 200-day MA (bullish)');
    } else {
      factors.push('50-day MA below 200-day MA (bearish)');
    }
    
    if (Math.abs(volumeChange) > 0.05) {
      factors.push(`${Math.abs(volumeChange * 100).toFixed(1)}% ${volumeChange > 0 ? 'increase' : 'decrease'} in trading volume`);
    }
    
    if (volatility > 25) {
      factors.push(`High volatility (${volatility.toFixed(1)}%)`);
    } else if (volatility < 15) {
      factors.push(`Low volatility (${volatility.toFixed(1)}%)`);
    }
    
    // Generate prediction text
    const prediction = isProfitable
      ? `${symbol} is likely to be profitable with ${profitProbability.toFixed(1)}% confidence. Factors include ${factors.join(', ')}.`
      : `${symbol} is likely to underperform with ${(100-profitProbability).toFixed(1)}% confidence. Factors include ${factors.join(', ')}.`;
    
    return {
      isProfitable,
      confidence: isProfitable ? profitProbability : 100 - profitProbability,
      factors,
      prediction
    };
  }
}