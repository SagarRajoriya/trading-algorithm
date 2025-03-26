import { StockData, CandlestickData } from '../types';

export class LocalStorageService {
  private static readonly STOCKS_KEY = 'hft_dashboard_stocks';
  private static readonly CANDLESTICK_PREFIX = 'hft_dashboard_candlestick_';
  private static readonly DATA_EXPIRY_KEY = 'hft_dashboard_data_expiry';
  private static readonly EXPIRY_TIME = 15 * 60 * 1000; // 15 minutes
  
  // Check if data has expired
  private static hasDataExpired(): boolean {
    const expiry = localStorage.getItem(this.DATA_EXPIRY_KEY);
    if (!expiry) return true;
    
    const expiryTime = parseInt(expiry, 10);
    return Date.now() > expiryTime;
  }
  
  // Update expiry time
  private static updateExpiry(): void {
    const newExpiryTime = Date.now() + this.EXPIRY_TIME;
    localStorage.setItem(this.DATA_EXPIRY_KEY, newExpiryTime.toString());
  }
  
  // Store stocks
  static saveStocks(stocks: StockData[]): void {
    try {
      localStorage.setItem(this.STOCKS_KEY, JSON.stringify(stocks));
      this.updateExpiry();
    } catch (error) {
      console.error("Error saving stocks to localStorage:", error);
    }
  }
  
  // Get stocks
  static getStocks(): StockData[] | null {
    try {
      if (this.hasDataExpired()) return null;
      
      const stocks = localStorage.getItem(this.STOCKS_KEY);
      return stocks ? JSON.parse(stocks) : null;
    } catch (error) {
      console.error("Error retrieving stocks from localStorage:", error);
      return null;
    }
  }
  
  // Store candlestick data for a symbol
  static saveCandlestickData(symbol: string, data: CandlestickData[]): void {
    try {
      localStorage.setItem(`${this.CANDLESTICK_PREFIX}${symbol}`, JSON.stringify(data));
      this.updateExpiry();
    } catch (error) {
      console.error(`Error saving candlestick data for ${symbol}:`, error);
    }
  }
  
  // Get candlestick data for a symbol
  static getCandlestickData(symbol: string): CandlestickData[] | null {
    try {
      if (this.hasDataExpired()) return null;
      
      const data = localStorage.getItem(`${this.CANDLESTICK_PREFIX}${symbol}`);
      return data ? JSON.parse(data) : null;
    } catch (error) {
      console.error(`Error retrieving candlestick data for ${symbol}:`, error);
      return null;
    }
  }
  
  // Clear all stored data
  static clearAll(): void {
    try {
      localStorage.removeItem(this.STOCKS_KEY);
      localStorage.removeItem(this.DATA_EXPIRY_KEY);
      
      // Find and remove all candlestick data
      Object.keys(localStorage).forEach(key => {
        if (key.startsWith(this.CANDLESTICK_PREFIX)) {
          localStorage.removeItem(key);
        }
      });
    } catch (error) {
      console.error("Error clearing localStorage:", error);
    }
  }
}