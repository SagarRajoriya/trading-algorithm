import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, BarChart, ArrowUpCircle, ArrowDownCircle, Info } from 'lucide-react';
import { StockDataService } from '../services/StockDataService';

interface PredictionProps {
    symbol: string;
}

interface PredictionData {
    isProfitable: boolean;
    confidence: number;
    factors: string[];
    prediction: string;
    recommendation: string;
    targetPrice: number;
    stopLoss: number;
    potentialReturn: number;
    timeframe: string;
    riskLevel: string;
    probabilityScore: number;
    currentPrice: number;
}

const Prediction: React.FC<PredictionProps> = ({ symbol }) => {
    console.log("Prediction component rendering for symbol:", symbol);
    const [prediction, setPrediction] = useState<PredictionData | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        console.log("Prediction useEffect running for symbol:", symbol);

        const loadPrediction = async () => {
            if (!symbol) return;

            setLoading(true);
            setError(null);
            setPrediction(null); // Reset prediction

            try {
                console.log("Attempting to fetch ML prediction for:", symbol);

                // Debug: Check the endpoint URL and raw response
                const response = await fetch(`http://localhost:5000/api/prediction-summary/${symbol}`);
                const rawData = await response.json();
                console.log("Raw API response:", rawData);

                // Now get the processed prediction
                const predictionData = await StockDataService.getPrediction(symbol);
                console.log("Processed prediction data:", predictionData);

                setPrediction(predictionData);
            } catch (err) {
                console.error('Failed to load prediction:', err);
                setError(err.message || 'Could not load ML prediction data.');
            } finally {
                setLoading(false);
            }
        };

        loadPrediction();
    }, [symbol]);

    if (loading) {
        return (
            <div className="bg-gray-800 rounded-lg shadow-lg p-4 animate-pulse">
                <div className="flex items-center mb-3">
                    <BarChart className="w-5 h-5 text-blue-400 mr-2" />
                    <h3 className="text-lg font-semibold text-white">AI Profit Prediction</h3>
                </div>
                <div className="h-32 bg-gray-700 rounded-md"></div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-gray-800 rounded-lg shadow-lg p-4">
                <div className="flex items-center mb-3">
                    <BarChart className="w-5 h-5 text-blue-400 mr-2" />
                    <h3 className="text-lg font-semibold text-white">AI Profit Prediction</h3>
                </div>
                <div className="bg-red-900/30 rounded-md p-3 flex items-center">
                    <AlertTriangle className="w-5 h-5 text-red-400 mr-2" />
                    <span className="text-red-300">{error}</span>
                </div>
            </div>
        );
    }

    // No prediction data
    if (!prediction) {
        return (
            <div className="bg-gray-800 rounded-lg shadow-lg p-4">
                <div className="flex items-center mb-3">
                    <BarChart className="w-5 h-5 text-blue-400 mr-2" />
                    <h3 className="text-lg font-semibold text-white">AI Profit Prediction</h3>
                </div>
                <div className="bg-gray-700/30 rounded-md p-3 flex items-center">
                    <AlertTriangle className="w-5 h-5 text-yellow-400 mr-2" />
                    <span className="text-gray-300">No prediction data available for {symbol}</span>
                </div>
            </div>
        );
    }

    // We have prediction data
    const isBuy = prediction.recommendation.toLowerCase().includes('buy');

    return (
        <div className="bg-gray-800 rounded-lg overflow-hidden">
            {/* PROMINENT BUY/SELL INDICATOR */}
            <div className={`p-4 ${isBuy ? 'bg-green-900/50' : 'bg-red-900/50'} border-b border-gray-700`}>
                <div className="flex justify-between items-center">
                    {/* Left side with icon and recommendation */}
                    <div className="flex items-center">
                        {isBuy ? (
                            <ArrowUpCircle className="w-10 h-10 text-green-400 mr-3" />
                        ) : (
                            <ArrowDownCircle className="w-10 h-10 text-red-400 mr-3" />
                        )}
                        <div>
                            <div className="text-2xl font-bold text-white">
                                {isBuy ? "BUY" : "SELL"}
                            </div>
                            <div className="text-sm text-gray-300">
                                ML Prediction: {prediction.prediction}
                            </div>
                        </div>
                    </div>

                    {/* Right side with confidence */}
                    <div>
                        <div className="text-sm text-gray-400">Confidence</div>
                        <div className={`text-xl font-bold ${isBuy ? 'text-green-400' : 'text-red-400'}`}>
                            {prediction.confidence.toFixed(2)}%
                        </div>
                    </div>
                </div>
            </div>

            {/* Rest of prediction details */}
            <div className="p-4">
                {/* Confidence bar */}
                <div className="mb-4">
                    <div className="flex justify-between items-center mb-1">
                        <span className="text-sm text-gray-400">Probability Score</span>
                        <span className="text-sm font-medium text-white">{prediction.probabilityScore.toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2.5">
                        <div
                            className={`h-2.5 rounded-full ${prediction.probabilityScore > 80 ? 'bg-green-500' :
                                prediction.probabilityScore > 40 ? 'bg-yellow-500' : 'bg-red-500'
                                }`}
                            style={{ width: `${prediction.probabilityScore}%` }}
                        ></div>
                    </div>
                </div>

                {/* Key metrics in a grid */}
                <div className="bg-gray-700/50 p-3 rounded-lg mb-4">
                    <div className="text-sm text-gray-400">Current Price</div>
                    <div className="text-lg font-medium text-white">${prediction.currentPrice}</div>
                </div>

                {/* Note that explains the data is directly from ML */}
                <div className="bg-blue-900/20 border border-blue-800/20 rounded-lg p-3 mb-4">
                    <div className="flex items-center text-xs text-blue-300 mb-1">
                        <Info className="w-3 h-3 mr-1" />
                        <span>ML Model Output</span>
                    </div>
                    <p className="text-xs text-gray-400">
                        This prediction comes directly from our machine learning model without any additional calculations or adjustments.
                    </p>
                </div>

                {/* Key Factors section - if factors are available */}
                {prediction.factors && prediction.factors.length > 0 && (
                    <div className="mb-4">
                        <h4 className="font-medium text-white mb-2">Key Factors:</h4>
                        <ul className="space-y-1">
                            {prediction.factors.map((factor, index) => (
                                <li key={index} className="text-sm flex items-start">
                                    <span className="text-gray-400 mr-2">•</span>
                                    <span className="text-gray-300">{factor}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}

                {/* Bottom section */}
                <div className="bg-gray-700/30 p-3 rounded-lg mt-4">
                    <div className="flex items-center mb-2">
                        <TrendingUp className="w-4 h-4 mr-2 text-blue-400" />
                        <div className="text-sm font-medium text-white">AI Trading Strategy</div>
                    </div>
                    <p className="text-sm text-gray-300">
                        {isBuy
                            ? `Buy ${symbol} with target price of $${prediction.targetPrice}. Set stop loss at $${prediction.stopLoss}.`
                            : `Sell ${symbol} with target price of $${prediction.targetPrice}. Set stop loss at $${prediction.stopLoss}.`
                        }
                    </p>
                    <div className="mt-2 text-xs text-gray-400">{prediction.timeframe} • {prediction.riskLevel} Risk</div>
                </div>
            </div>
        </div>
    );
};

export default Prediction;