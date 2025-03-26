import { StockData, CandlestickData } from '../types';

export const generateFallbackStocks = (): StockData[] => {
  return [
    { symbol: 'AAPL', price: 180.25, change: 1.25, volume: 45000000 },
    { symbol: 'MSFT', price: 420.75, change: -0.5, volume: 25000000 },
    { symbol: 'GOOGL', price: 175.50, change: 0.75, volume: 15000000 },
    { symbol: 'TSLA', price: 220.80, change: 3.2, volume: 80000000 },
    { symbol: 'AMZN', price: 178.90, change: -1.1, volume: 35000000 },
    { symbol: 'META', price: 485.30, change: 2.1, volume: 22000000 },
    { symbol: 'NVDA', price: 920.15, change: 4.3, volume: 55000000 },
    { symbol: 'JPM', price: 195.75, change: -0.3, volume: 12000000 },
  ];
};

export const generateCandlestickData = (symbol: string, timeRange: string): CandlestickData[] => {
  // Base price for each symbol
  const basePrices: Record<string, number> = {
    'AAPL': 180,
    'MSFT': 420,
    'GOOGL': 175,
    'TSLA': 220,
    'AMZN': 178,
    'META': 485,
    'NVDA': 920,
    'JPM': 195,
  };
  
  const basePrice = basePrices[symbol] || 100;
  const volatility = symbol === 'TSLA' ? 0.03 : 0.015;
  const numPoints = timeRange === '1d' ? 78 : // 5-minute candles for 6.5 hours
                   timeRange === '1w' ? 7 :
                   timeRange === '1m' ? 30 :
                   timeRange === '1y' ? 252 : 
                   500; // 5y
  
  const result: CandlestickData[] = [];
  let currentPrice = basePrice;
  let today = new Date();
  
  // For intraday data
  if (timeRange === '1d') {
    const marketOpen = new Date();
    marketOpen.setHours(9, 30, 0, 0);
    
    for (let i = 0; i < numPoints; i++) {
      const change = (Math.random() - 0.5) * volatility * currentPrice;
      currentPrice += change;
      
      const pointTime = new Date(marketOpen);
      pointTime.setMinutes(marketOpen.getMinutes() + (i * 5));
      
      const open = currentPrice;
      const close = currentPrice + (Math.random() - 0.5) * volatility * currentPrice;
      const high = Math.max(open, close) + Math.random() * volatility * currentPrice;
      const low = Math.min(open, close) - Math.random() * volatility * currentPrice;
      
      result.push({
        time: pointTime.toISOString(),
        open,
        high,
        high,
        low,
        close,
        volume: Math.floor(Math.random() * 500000) + 100000
      });
    }
  } 
  // For daily data
  else {
    for (let i = 0; i < numPoints; i++) {
      const change = (Math.random() - 0.5) * volatility * currentPrice;
      currentPrice += change;
      
      const pointDate = new Date(today);
      pointDate.setDate(today.getDate() - (numPoints - i));
      
      const open = currentPrice;
      const close = currentPrice + (Math.random() - 0.5) * volatility * currentPrice;
      const high = Math.max(open, close) + Math.random() * volatility * currentPrice;
      const low = Math.min(open, close) - Math.random() * volatility * currentPrice;
      
      result.push({
        time: pointDate.toISOString().split('T')[0],
        open,
        high,
        low,
        close,
        volume: Math.floor(Math.random() * 10000000) + 1000000
      });
    }
  }
  
  return result;
};