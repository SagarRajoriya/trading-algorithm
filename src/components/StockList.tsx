import React from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';
import type { StockData } from '../types';

interface StockListProps {
  stocks: StockData[];
  onSelect: (symbol: string) => void;
  selectedSymbol?: string;
}

const StockList: React.FC<StockListProps> = ({ stocks, onSelect, selectedSymbol }) => {
  return (
    <div className="bg-gray-800 rounded-lg shadow-lg p-4">
      <h2 className="text-xl font-semibold mb-4 text-white">Stock List</h2>
      {stocks.length === 0 ? (
        <div className="text-center py-6">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-2 text-gray-300">Loading stocks...</p>
        </div>
      ) : (
        <div className="space-y-2">
          {stocks.map((stock) => (
            <div
              key={stock.symbol}
              className={`flex items-center justify-between p-3 rounded-lg cursor-pointer ${selectedSymbol === stock.symbol
                  ? 'bg-gray-700 border border-blue-500'
                  : 'hover:bg-gray-700'
                }`}
              onClick={() => onSelect(stock.symbol)}
            >
              <div>
                <h3 className="font-semibold text-white">{stock.symbol}</h3>
                <p className="text-sm text-gray-400">Vol: {stock.volume.toLocaleString()}</p>
              </div>
              <div className="text-right">
                <p className="font-medium text-white">${stock.price.toFixed(2)}</p>
                <p className={`text-sm flex items-center ${stock.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {stock.change >= 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
                  {stock.change.toFixed(2)}%
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default StockList;