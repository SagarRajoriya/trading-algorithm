import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, CrosshairMode, ColorType } from 'lightweight-charts';
import { CandlestickData } from '../types';

// Company descriptions for information display
const companyInfo = {
  AAPL: {
    name: 'Apple Inc.',
    description: 'Designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.',
    sector: 'Technology',
  },
  MSFT: {
    name: 'Microsoft Corporation',
    description: 'Develops, licenses, and supports software, services, devices, and solutions worldwide.',
    sector: 'Technology',
  },
  TSLA: {
    name: 'Tesla, Inc.',
    description: 'Designs, develops, manufactures, and sells electric vehicles, energy generation and storage systems.',
    sector: 'Automotive',
  },
  AMZN: {
    name: 'Amazon.com, Inc.',
    description: 'Engages in the retail sale of consumer products, advertising, and subscription services through online and physical stores.',
    sector: 'Consumer Cyclical',
  },
  GOOGL: {
    name: 'Alphabet Inc.',
    description: 'Provides various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America.',
    sector: 'Technology',
  },
  META: {
    name: 'Meta Platforms, Inc.',
    description: 'Develops products that enable people to connect through mobile devices, personal computers, virtual reality headsets, and wearables.',
    sector: 'Technology',
  },
  NVDA: {
    name: 'NVIDIA Corporation',
    description: 'Provides graphics, computing and networking solutions in the United States, Taiwan, China, and internationally.',
    sector: 'Technology',
  },
  JPM: {
    name: 'JPMorgan Chase & Co.',
    description: 'Operates as a financial services company worldwide providing investment banking, financial services and asset management.',
    sector: 'Financial Services',
  },
  NFLX: {
    name: 'Netflix, Inc.',
    description: 'Provides entertainment services offering TV series, documentaries, feature films, and mobile games across various genres and languages.',
    sector: 'Communication Services',
  },
  IBM: {
    name: 'International Business Machines',
    description: 'Provides integrated solutions and services worldwide in cloud computing, artificial intelligence, and hybrid cloud environments.',
    sector: 'Technology',
  }
};

// Helper function to get company info with case-insensitive lookup
const getCompanyInfo = (symbol: string) => {
  if (!symbol) return null;

  // Try direct lookup first
  if (companyInfo[symbol]) {
    return companyInfo[symbol];
  }

  // Try case-insensitive lookup
  const upperSymbol = symbol.toUpperCase();
  if (companyInfo[upperSymbol]) {
    return companyInfo[upperSymbol];
  }

  // If still not found, use default values
  return {
    name: symbol,
    description: `Trading data for ${symbol}`,
    sector: 'Market Data'
  };
};

interface ChartProps {
  data: CandlestickData[];
  symbol: string;
  timeRange: string;
}

const Chart: React.FC<ChartProps> = ({ data, symbol, timeRange }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<IChartApi | null>(null);
  const isMountedRef = useRef(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [priceChange, setPriceChange] = useState<{ value: number, percent: number } | null>(null);

  useEffect(() => {
    if (data && data.length > 0) {
      const latest = data[data.length - 1];
      setCurrentPrice(latest.close);

      if (data.length > 1) {
        const previous = data[data.length - 2];
        const change = latest.close - previous.close;
        const percentChange = (change / previous.close) * 100;
        setPriceChange({
          value: change,
          percent: percentChange
        });
      }
    }
  }, [data]);

  useEffect(() => {
    // Set mounted flag
    isMountedRef.current = true;

    // Clean up function that safely removes chart
    const cleanupChart = () => {
      if (chartInstanceRef.current) {
        try {
          chartInstanceRef.current.remove();
        } catch (e) {
          console.warn("Error removing chart:", e);
        }
        chartInstanceRef.current = null;
      }
    };

    // Skip if no data or container
    if (!chartContainerRef.current) {
      return cleanupChart;
    }

    if (!data || data.length === 0) {
      setError("No data available for chart");
      return cleanupChart;
    }

    // Clean up any existing chart first
    cleanupChart();

    // Create chart with delay
    const timeoutId = setTimeout(() => {
      if (!isMountedRef.current || !chartContainerRef.current) return;

      try {
        // Create the chart instance
        const chart = createChart(chartContainerRef.current, {
          width: chartContainerRef.current.clientWidth,
          height: 500,
          layout: {
            background: { color: '#1f2937' },
            textColor: '#d1d5db',
          },
          grid: {
            vertLines: { color: 'rgba(42, 46, 57, 0.5)' },
            horzLines: { color: 'rgba(42, 46, 57, 0.5)' },
          },
          crosshair: {
            mode: CrosshairMode.Normal,
          },
          timeScale: {
            timeVisible: timeRange === '1d',
            secondsVisible: timeRange === '1d',
          },
        });

        // Save reference only if still mounted
        if (isMountedRef.current) {
          chartInstanceRef.current = chart;

          const candlestickSeries = chart.addCandlestickSeries({
            upColor: '#22c55e',
            downColor: '#ef4444',
            borderUpColor: '#22c55e',
            borderDownColor: '#ef4444',
            wickUpColor: '#22c55e',
            wickDownColor: '#ef4444',
          });

          // Process data for the chart
          try {
            candlestickSeries.setData(data);
            chart.timeScale().fitContent();
            setError(null);
          } catch (e) {
            console.error('Error setting chart data:', e);
            setError(`Failed to render chart: ${e instanceof Error ? e.message : String(e)}`);
          }
        } else {
          // If component unmounted during creation, remove the chart
          chart.remove();
        }
      } catch (err) {
        console.error("Error creating candlestick chart:", err);
        if (isMountedRef.current) {
          setError(`Chart error: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    }, 100);

    // Return a cleanup function
    return () => {
      isMountedRef.current = false;
      clearTimeout(timeoutId);
      cleanupChart();
    };
  }, [data, symbol, timeRange]); // Dependencies: recreate chart when these change

  return (
    <div className="relative">
      {/* Company Information Panel */}
      <div className="mb-4 bg-gray-800 border border-gray-700 rounded-md p-4">
        <div className="flex justify-between items-start">
          <div className="flex-1">
            <h3 className="text-lg font-bold text-white">
              {getCompanyInfo(symbol)?.name || symbol || 'Select a Stock'}
            </h3>
            <p className="text-sm text-gray-300">
              {getCompanyInfo(symbol)?.sector || 'Market Data'} • {timeRange} Chart
            </p>
            <p className="mt-2 text-sm text-gray-400">
              {getCompanyInfo(symbol)?.description || `Trading data for ${symbol || 'selected stock'}`}
            </p>
          </div>

          <div className="text-right">
            {currentPrice !== null ? (
              <div className="text-xl font-bold text-white">
                ${currentPrice.toFixed(2)}
              </div>
            ) : (
              <div className="text-xl font-bold text-gray-500">--</div>
            )}

            {priceChange ? (
              <div className={`text-sm font-medium ${priceChange.value >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {priceChange.value >= 0 ? '+' : ''}{priceChange.value.toFixed(2)}
                ({priceChange.value >= 0 ? '+' : ''}{priceChange.percent.toFixed(2)}%)
              </div>
            ) : (
              <div className="text-sm font-medium text-gray-500">--</div>
            )}
          </div>
        </div>
      </div>

      {/* Chart Container */}
      <div
        ref={chartContainerRef}
        className="w-full h-[500px] bg-gray-900/50 rounded-lg"
        style={{ minHeight: "500px" }}
      />

      {error && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-gray-400">
            No chart data available
          </div>
        </div>
      )}
    </div>
  );
};

export default Chart;