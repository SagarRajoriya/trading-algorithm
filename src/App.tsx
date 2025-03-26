import React, { useState, useEffect } from 'react';
import { Activity, RefreshCw, ChevronDown, AlertCircle } from 'lucide-react';
import Chart from './components/Chart';
import StockList from './components/StockList';
import { StockDataService } from './services/StockDataService';
import { LocalStorageService } from './services/LocalStorageService';
import type { StockData, CandlestickData } from './types';
import Prediction from './components/Prediction';

// Define the available time ranges
type TimeRange = '1d' | '1w' | '1m' | '1y' | '5y';

const timeRangeOptions: { value: TimeRange; label: string }[] = [
  { value: '1d', label: '1 Day' },
  { value: '1w', label: '1 Week' },
  { value: '1m', label: '1 Month' },
  { value: '1y', label: '1 Year' },
  { value: '5y', label: '5 Years' }
];

function App() {
  const [stocks, setStocks] = useState<StockData[]>([]);
  const [selectedStock, setSelectedStock] = useState<string>('');
  const [timeRange, setTimeRange] = useState<TimeRange>('1d');
  const [candlestickData, setCandlestickData] = useState<CandlestickData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [dropdownOpen, setDropdownOpen] = useState<boolean>(false);
  const [lastUpdated, setLastUpdated] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  // Replace your initialize function with this more robust version
  const initialize = async () => {
    setLoading(true);
    setError(null);

    try {
      // First set a default stock to avoid undefined issues
      const defaultStock = 'AAPL';
      setSelectedStock(defaultStock);

      // Try to load stock list
      const stockList = await StockDataService.fetchStockList();
      console.log('Loaded stock list:', stockList);

      if (stockList && stockList.length > 0) {
        setStocks(stockList);
        // Only update selectedStock if we have a list and haven't set it yet
        setSelectedStock(stockList[0].symbol);
        console.log('Set default stock to:', stockList[0].symbol);
      } else {
        console.warn('No stocks loaded from CSV, using fallback data');
        // Use fallback data if no stocks were loaded
        const fallbackStocks = [
          { symbol: 'AAPL', price: 220.45, change: 1.25, volume: 80000000 },
          { symbol: 'MSFT', price: 425.63, change: -0.5, volume: 25000000 },
          { symbol: 'TSLA', price: 177.82, change: 3.5, volume: 100000000 }
        ];
        setStocks(fallbackStocks);
      }

      // Show demo data initially instead of trying to load CSV right away
      loadDemoData();
    } catch (error) {
      console.error('Failed to initialize app:', error);
      setError('Could not load initial data. Using fallback data.');

      // Set fallback stocks
      const fallbackStocks = [
        { symbol: 'AAPL', price: 220.45, change: 1.25, volume: 80000000 },
        { symbol: 'MSFT', price: 425.63, change: -0.5, volume: 25000000 },
        { symbol: 'TSLA', price: 177.82, change: 3.5, volume: 100000000 }
      ];
      setStocks(fallbackStocks);
      setSelectedStock('AAPL'); // Always set a default stock

      // Show demo data
      loadDemoData();
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    console.log('App mounted - initializing');
    initialize();
  }, []);

  // Load chart data when stock or time range changes
  useEffect(() => {
    // Guard clause - don't proceed if no stock is selected
    if (!selectedStock) {
      console.log('No stock selected yet, skipping chart data load');
      return;
    }

    console.log(`Loading chart data for ${selectedStock} (${timeRange})`);

    const loadChart = async () => {
      setLoading(true);
      setError(null);

      try {
        const data = await StockDataService.fetchCandlestickData(selectedStock, timeRange);
        if (data && data.length > 0) {
          console.log(`Loaded ${data.length} candlesticks for ${selectedStock}`);
          setCandlestickData(data);
        } else {
          console.warn(`No chart data found for ${selectedStock}, loading demo data`);
          loadDemoData(); // Fallback to demo data when no data is available
        }
      } catch (error) {
        console.error('Failed to load chart data:', error);
        setError(`Could not load chart data for ${selectedStock}`);
        // Don't clear the chart data here - let's keep any previous data
      } finally {
        setLoading(false);
      }
    };

    loadChart();
  }, [selectedStock, timeRange]);

  const handleRefresh = async () => {
    setRefreshing(true);
    setError(null);

    try {
      LocalStorageService.clearAll();
      await loadStocks(true);
      await loadStockData(true);
    } catch (error) {
      console.error('Error during refresh:', error);
      setError('Refresh failed. Check console for details.');
    } finally {
      setRefreshing(false);
    }
  };

  const toggleDropdown = () => setDropdownOpen(!dropdownOpen);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownOpen && !(event.target as Element).closest('.dropdown-container')) {
        setDropdownOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [dropdownOpen]);

  // Replace your existing time range handler with this one
  const handleTimeRangeSelect = (range: TimeRange) => {
    setTimeRange(range);
    setDropdownOpen(false);

    // Force a data refresh when time range changes
    if (selectedStock) {
      console.log(`Time range changed to ${range}, forcing data refresh...`);
      setLoading(true);

      StockDataService.fetchCandlestickData(selectedStock, range)
        .then(data => {
          console.log(`Loaded ${data.length} candlesticks for ${selectedStock} with range ${range}`);
          setCandlestickData(data);
          setLastUpdated(new Date().toLocaleTimeString());
        })
        .catch(error => {
          console.error(`Error loading data for range ${range}:`, error);
          loadDemoData();
        })
        .finally(() => {
          setLoading(false);
        });
    }
  };

  // Add this function inside your App component, before the return statement
  const loadDemoData = () => {
    // Creates different demo data based on the timeRange
    console.log(`Generating demo data for ${timeRange}...`);

    const now = new Date();
    const syntheticData: CandlestickData[] = [];

    // Base values for the selected stock
    const stockPrices = {
      'AAPL': 220.45,
      'MSFT': 425.63,
      'TSLA': 177.82,
      'AMZN': 178.75,
      'GOOGL': 175.98,
      'META': 485.92,
      'NVDA': 925.75,
      'JPM': 196.82
    };

    const basePrice = stockPrices[selectedStock] || 100;
    let currentPrice = basePrice;
    const volatility = selectedStock === 'TSLA' ? 0.03 : 0.015;

    // Generate different amounts of data based on the time range
    let numPoints = 30; // Default
    let intervalSeconds = 86400; // Default to daily

    switch (timeRange) {
      case '1d':
        // 5-minute intervals for 6.5 hours (78 candles)
        numPoints = 78;
        intervalSeconds = 300; // 5 minutes
        break;
      case '1w':
        // 1-hour intervals for 1 week
        numPoints = 7 * 24;
        intervalSeconds = 3600; // 1 hour
        break;
      case '1m':
        // 6-hour intervals for 1 month
        numPoints = 30 * 4;
        intervalSeconds = 21600; // 6 hours
        break;
      case '1y':
        // Daily for 1 year
        numPoints = 252; // Trading days in a year
        intervalSeconds = 86400; // 1 day
        break;
      case '5y':
        // Weekly for 5 years
        numPoints = 260; // ~52 weeks * 5 years
        intervalSeconds = 604800; // 1 week
        break;
    }

    // Base timestamp (seconds)
    let baseTimestamp = Math.floor(now.getTime() / 1000);

    // For non-intraday, align to day boundary
    if (timeRange !== '1d') {
      const dayStart = new Date(now);
      dayStart.setHours(0, 0, 0, 0);
      baseTimestamp = Math.floor(dayStart.getTime() / 1000);
    }

    // Generate data points going backwards in time
    for (let i = 0; i < numPoints; i++) {
      // Calculate the timestamp for this point
      const pointTimestamp = baseTimestamp - (numPoints - i - 1) * intervalSeconds;

      // Add some random price movement
      const priceChange = (Math.random() - 0.5) * volatility * currentPrice;
      currentPrice += priceChange;

      // Generate OHLC values
      const open = currentPrice;
      const close = currentPrice * (1 + (Math.random() - 0.5) * 0.01);
      const high = Math.max(open, close) * (1 + Math.random() * 0.01);
      const low = Math.min(open, close) * (1 - Math.random() * 0.01);
      const volume = Math.floor(Math.random() * 1000000) + 100000;

      syntheticData.push({
        time: pointTimestamp,
        open: open,
        high: high,
        low: low,
        close: close,
        volume: volume
      });
    }

    console.log(`Generated ${syntheticData.length} demo data points`);
    setCandlestickData(syntheticData);
    setError(null);
  };

  // Update your loadStocks function to handle errors without demo data
  async function loadStocks(forceRefresh = false) {
    if (refreshing) return;

    if (forceRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }

    setError(null);

    try {
      const fetchedStocks = await StockDataService.fetchStockList();
      setStocks(fetchedStocks);

      // If no stock is selected yet, select the first one
      if (!selectedStock && fetchedStocks.length > 0) {
        setSelectedStock(fetchedStocks[0].symbol);
      }

      setLastUpdated(new Date().toLocaleTimeString());
    } catch (error) {
      console.error('Error loading stocks:', error);
      setError(`Could not load stock data: ${error.message}. Please ensure the CSV files are in the correct location.`);

      // Don't clear existing stocks on error
    } finally {
      setRefreshing(false);
      setLoading(false);
    }
  }

  // Update your loadStockData function to handle the forceRefresh parameter
  async function loadStockData(forceRefresh = false) {
    if (!selectedStock) return;

    if (forceRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }

    setError(null);

    try {
      // Pass the timeRange explicitly to ensure fresh data is fetched for this range
      const candleData = await StockDataService.fetchCandlestickData(selectedStock, timeRange);
      console.log(`Fetched ${candleData?.length || 0} data points for ${selectedStock} (${timeRange})`);
      setCandlestickData(candleData);

      // Update last updated timestamp
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (error) {
      console.error('Error loading stock data:', error);
      setCandlestickData([]);
      setError(`Could not load chart data for ${selectedStock}: ${error.message}`);

      // Fall back to demo data if real data fetch fails
      console.log('Falling back to demo data');
      loadDemoData();
    } finally {
      if (forceRefresh) {
        setRefreshing(false);
      }
      setLoading(false);
    }
  }

  // Add this new useEffect after your existing ones
  useEffect(() => {
    // Skip initial render
    if (!selectedStock) return;

    console.log(`Time range changed to ${timeRange}, reloading data...`);

    // Force reload data when time range changes
    const fetchNewRangeData = async () => {
      // Use the same loading state as in loadStockData
      setLoading(true);

      try {
        // Make sure we're fetching fresh data for this time range
        LocalStorageService.removeItem(`candlestick_${selectedStock}_${timeRange}`);
        const data = await StockDataService.fetchCandlestickData(selectedStock, timeRange);

        if (data && data.length > 0) {
          console.log(`Loaded ${data.length} candlesticks for ${selectedStock} (${timeRange})`);
          setCandlestickData(data);
        } else {
          console.warn(`No data received for ${selectedStock} (${timeRange}), using demo data`);
          loadDemoData();
        }
      } catch (error) {
        console.error(`Error loading ${timeRange} data:`, error);
        loadDemoData();
      } finally {
        setLoading(false); // FIXED: Was incorrectly setting to true
      }
    };

    fetchNewRangeData();
  }, [timeRange]); // Only trigger when timeRange changes

  // This effect loads data when selectedStock changes
  useEffect(() => {
    if (!selectedStock) return;

    console.log(`Selected stock changed to ${selectedStock}, loading data...`);
    setLoading(true);

    StockDataService.fetchCandlestickData(selectedStock, timeRange)
      .then(data => {
        console.log(`Loaded ${data.length} candlesticks for ${selectedStock}`);
        setCandlestickData(data);
        setLastUpdated(new Date().toLocaleTimeString());
      })
      .catch(error => {
        console.error('Failed to load chart data:', error);
        setError(`Could not load chart data for ${selectedStock}`);
        loadDemoData();
      })
      .finally(() => {
        setLoading(false);
      });
  }, [selectedStock]);

  return (
    <div className="min-h-screen bg-gray-900">
      <header className="bg-gray-800 shadow-md border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Activity className="w-8 h-8 text-blue-400" />
              <div>
                <h1 className="text-2xl font-bold text-white">ML Trading Analytics</h1>
                {lastUpdated && (
                  <span className="text-xs text-gray-400">Last updated: {lastUpdated}</span>
                )}
              </div>
            </div>
            <button
              onClick={handleRefresh}
              disabled={refreshing || loading}
              className="flex items-center px-3 py-2 bg-blue-600 rounded-md text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
              Refresh Data
            </button>
          </div>
        </div>
      </header>

      {/* Error alert */}
      {error && (
        <div className="max-w-7xl mx-auto mt-4 px-4 sm:px-6 lg:px-8">
          <div className="bg-red-900/50 border border-red-800 rounded-md p-4 flex items-start">
            <AlertCircle className="w-5 h-5 text-red-300 mr-3 mt-0.5 flex-shrink-0" />
            <div>
              <h3 className="text-sm font-medium text-red-300">Error</h3>
              <p className="mt-1 text-sm text-red-200">{error}</p>
              <p className="mt-2 text-sm text-red-300">
                Make sure the CSV files (stock_data_fixed.csv and stock_data_intraday.csv) are in the public folder.
              </p>
            </div>
          </div>
        </div>
      )}

      <main className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Stock List - Left Sidebar */}
          <div className="lg:col-span-2">
            <StockList
              stocks={stocks}
              onSelect={setSelectedStock}
              selectedSymbol={selectedStock}
            />
          </div>

          {/* Main Content Area - Chart */}
          <div className="lg:col-span-7">
            <div className="bg-gray-800 rounded-lg shadow-lg p-4 h-full">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-white">
                  {selectedStock ? `${selectedStock} Candlestick Chart` : 'Select a Stock'}
                </h2>

                {/* Time Range Selector */}
                <div className="flex space-x-1">
                  {timeRangeOptions.map((option) => (
                    <button
                      key={option.value}
                      onClick={() => setTimeRange(option.value)}
                      className={`px-3 py-1 text-sm rounded-md transition-colors ${timeRange === option.value
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }`}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>

              {!selectedStock ? (
                <div className="flex items-center justify-center h-64">
                  <div className="text-center">
                    <p className="text-gray-400">Select a stock from the list to view chart</p>
                    <button
                      onClick={() => {
                        // Ensure we have a selected stock
                        if (stocks.length > 0) {
                          setSelectedStock(stocks[0].symbol);
                        } else {
                          setStocks([
                            { symbol: 'AAPL', price: 220.45, change: 1.25, volume: 80000000 }
                          ]);
                          setSelectedStock('AAPL');
                        }
                        loadDemoData();
                      }}
                      className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                    >
                      Load Demo Stock
                    </button>
                  </div>
                </div>
              ) : loading ? (
                <div className="flex items-center justify-center h-[500px]">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
                    <p className="mt-4 text-gray-300">Loading chart data...</p>
                  </div>
                </div>
              ) : candlestickData.length > 0 ? (
                <Chart
                  data={candlestickData}
                  symbol={selectedStock}
                  timeRange={timeRangeOptions.find(option => option.value === timeRange)?.label || ''}
                />
              ) : (
                <div className="flex items-center justify-center h-64">
                  <div className="text-center">
                    <p className="text-gray-300">No data available for {selectedStock}</p>
                    <button
                      onClick={loadDemoData}
                      className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                    >
                      Load Demo Data
                    </button>
                  </div>
                </div>
              )}
              {error && (
                <div className="mt-4">
                  <button
                    onClick={loadDemoData}
                    className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                  >
                    Load Demo Data
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Prediction Card - Right Side */}
          <div className="lg:col-span-3">
            {selectedStock && (
              <div className="bg-gray-800 rounded-lg shadow-lg border border-blue-900/50 h-full">
                <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 p-3 rounded-t-lg border-b border-gray-700">
                  <h2 className="text-lg font-semibold text-white flex items-center">
                    <Activity className="w-5 h-5 mr-2 text-blue-400" />
                    AI Prediction
                  </h2>
                </div>
                <div className="p-4">
                  {/* Add a console log here */}
                  {console.log("About to render Prediction with symbol:", selectedStock)}
                  <Prediction symbol={selectedStock} />
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;