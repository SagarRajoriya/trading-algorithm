export interface StockData {
  symbol: string;
  price: number;
  change: number;
  volume: number;
}

export interface ChartData {
  time: string;
  value: number;
}

export interface CandlestickData {
  time: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PredictionData {
  isProfitable: boolean;
  confidence: number;
  factors: string[];
  prediction: string;
}