import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { format } from 'date-fns';
import Chart from './src/components/Chart';
import type { CandlestickData } from './src/types';

// Define API endpoint - update this to point to your Flask backend
const API_BASE_URL = 'http://localhost:5000/api';

// Service to fetch stock data from backend
const fetchStockData = async (
  symbol: string,
  period: string = '1mo',
  interval: string = '1d',
  forceRefresh: boolean = false
): Promise<CandlestickData[]> => {
  try {
    // Call your Flask backend API with refresh flag if needed
    const response = await axios.get(
      `${API_BASE_URL}/stocks/${symbol}?period=${period}&interval=${interval}&refresh=${forceRefresh}`
    );

    console.log(`Received data for ${symbol}:`, response.data);

    // Convert API response to CandlestickData format
    const candlestickData: CandlestickData[] = response.data.map((item: any) => ({
      time: format(new Date(item.date), 'yyyy-MM-dd'),
      open: item.open,
      high: item.high,
      low: item.low,
      close: item.close
    }));

    // Store the data locally
    storeStockData(symbol, candlestickData);

    return candlestickData;
  } catch (error) {
    console.error('Error fetching stock data:', error);
    // Return empty array instead of mock data
    return [];
  }
};

// Store stock data in local storage
const storeStockData = (symbol: string, data: CandlestickData[]): void => {
  try {
    localStorage.setItem(`stockData_${symbol}`, JSON.stringify(data));
    localStorage.setItem(`stockData_${symbol}_timestamp`, new Date().toString());
  } catch (error) {
    console.error('Error storing stock data:', error);
  }
};

// Retrieve stored stock data - reduce cache time to 1 hour
const getStoredStockData = (symbol: string): CandlestickData[] | null => {
  try {
    const data = localStorage.getItem(`stockData_${symbol}`);
    const timestamp = localStorage.getItem(`stockData_${symbol}_timestamp`);

    // Only use cached data if less than 1 hour old
    if (data && timestamp) {
      const cachedTime = new Date(timestamp);
      const now = new Date();
      const hoursDiff = (now.getTime() - cachedTime.getTime()) / (1000 * 60 * 60);

      if (hoursDiff < 1) {
        return JSON.parse(data);
      }
    }
    return null;
  } catch (error) {
    console.error('Error retrieving stored stock data:', error);
    return null;
  }
};

// Component to display a stock candlestick chart with real data
const StockCandlestickChart: React.FC<{
  symbol: string;
  period?: string;
  interval?: string;
  forceRefresh?: boolean;
}> = ({ symbol, period = '1mo', interval = '1d', forceRefresh = false }) => {
  const [data, setData] = useState<CandlestickData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>('');

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);

      // Skip cache if forceRefresh is true
      const storedData = forceRefresh ? null : getStoredStockData(symbol);

      if (storedData && storedData.length > 0) {
        setData(storedData);
        setLoading(false);
        setLastUpdated('from cache');
      } else {
        try {
          const fetchedData = await fetchStockData(symbol, period, interval, forceRefresh);
          setData(fetchedData);
          setLastUpdated(new Date().toLocaleString());
        } catch (err) {
          setError(`Failed to load data for ${symbol}`);
        } finally {
          setLoading(false);
        }
      }
    };

    loadData();
  }, [symbol, period, interval, forceRefresh]);

  const handleRefresh = async () => {
    setLoading(true);
    try {
      const fetchedData = await fetchStockData(symbol, period, interval, true);
      setData(fetchedData);
      setLastUpdated(new Date().toLocaleString());
    } catch (err) {
      setError(`Failed to refresh data for ${symbol}`);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="p-4">Loading {symbol} data...</div>;
  if (error) return <div className="p-4 text-red-500">{error}</div>;

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <div className="flex justify-between items-center mb-2">
        <h3 className="font-bold">{symbol}</h3>
        <button
          onClick={handleRefresh}
          className="px-2 py-1 bg-blue-500 text-white rounded text-xs"
        >
          Refresh
        </button>
      </div>
      <Chart
        data={data}
        type="candlestick"
        title={`${symbol} Candlestick Chart`}
      />
      <div className="mt-4 text-xs text-gray-500">
        Data sourced from backend • Last updated: {lastUpdated}
      </div>
    </div>
  );
};

export { fetchStockData, storeStockData, getStoredStockData, StockCandlestickChart };