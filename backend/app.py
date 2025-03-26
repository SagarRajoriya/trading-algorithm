from flask import Flask, jsonify, request
import yfinance as yf
from flask_cors import CORS
import pandas as pd
from datetime import datetime, timedelta
import time
import random
import traceback

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)  # Enhanced CORS

# Cache to store recent API requests and reduce redundant calls
data_cache = {}
CACHE_DURATION = 300  # Reduced to 5 minutes
STOCK_LIST_CACHE_DURATION = 10  # 10 seconds for stock list (for real-time updates)

@app.route('/api/stocks', methods=['GET'])
def get_stock_list():
    try:
        # Force refresh if requested
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        # Check if we have a cached stock list and if it's still valid
        cache_key = "stock_list"
        current_time = time.time()
        
        if not force_refresh and cache_key in data_cache:
            cached_data, timestamp = data_cache[cache_key]
            # If cache is still valid (less than STOCK_LIST_CACHE_DURATION seconds old)
            if current_time - timestamp < STOCK_LIST_CACHE_DURATION:
                print("Returning cached stock list")
                return jsonify(cached_data)
        
        # Default symbols if none provided
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD']
        
        print(f"Fetching fresh stock data for: {', '.join(symbols)}")
        result = []
        
        for symbol in symbols:
            try:
                # Create a fresh session for each request to avoid caching
                ticker = yf.Ticker(symbol)
                
                # Get latest price data with no caching
                hist = ticker.history(period="1d", prepost=True)
                
                # Get info with no caching
                info = ticker.info
                
                # Get the most accurate price
                current_price = None
                
                # First try fast_info for real-time data
                try:
                    quote = ticker.fast_info
                    if hasattr(quote, 'last_price') and quote.last_price:
                        current_price = quote.last_price
                    elif hasattr(quote, 'regular_market_price') and quote.regular_market_price:
                        current_price = quote.regular_market_price
                except:
                    pass
                
                # If fast_info didn't work, try info dict
                if not current_price:
                    current_price = info.get('regularMarketPrice', None) or info.get('currentPrice', None)
                
                # Finally try history data
                if not current_price and not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                
                # Previous close for percentage calculation
                prev_close = None
                try:
                    prev_close = ticker.fast_info.previous_close
                except:
                    prev_close = info.get('previousClose', None)
                
                if not prev_close and not hist.empty and len(hist) > 1:
                    prev_close = hist['Close'].iloc[-2]
                
                # Calculate percentage change
                change_percent = 0
                if current_price and prev_close:
                    change_percent = ((current_price - prev_close) / prev_close) * 100
                
                # Get volume
                volume = info.get('volume', 0) or info.get('averageVolume', 0)
                if volume == 0 and not hist.empty:
                    volume = hist['Volume'].iloc[-1]
                
                # Add to result
                result.append({
                    "symbol": symbol,
                    "price": float(current_price) if current_price else 0,
                    "change": float(change_percent),
                    "volume": int(volume) if volume else 0,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                print(f"Updated {symbol}: ${float(current_price) if current_price else 0:.2f}")
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                traceback.print_exc()  # Print full stack trace
                
                # Add placeholder data so UI doesn't break
                result.append({
                    "symbol": symbol,
                    "price": 0,
                    "change": 0,
                    "volume": 0,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "error": str(e)
                })
        
        # Cache the result with current timestamp
        data_cache[cache_key] = (result, current_time)
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in get_stock_list: {str(e)}")
        traceback.print_exc()
        return jsonify([])

@app.route('/api/stocks/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    try:
        # Get parameters
        period = request.args.get('period', '1mo')
        interval = request.args.get('interval', '1d')
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        # Create cache key
        cache_key = f"{symbol}_{period}_{interval}"
        current_time = time.time()
        
        # Check if we have cached data and it's not a forced refresh
        if not refresh and cache_key in data_cache:
            cached_data, timestamp = data_cache[cache_key]
            # If cache is still valid
            if current_time - timestamp < CACHE_DURATION:
                print(f"Returning cached data for {symbol}")
                return jsonify(cached_data)
        
        # Fetch fresh data
        print(f"Fetching fresh data for {symbol} with period={period}, interval={interval}")
        ticker = yf.Ticker(symbol)
        
        # Get data with prepost=True to include pre/post market data
        hist = ticker.history(period=period, interval=interval, prepost=True, actions=False)
        
        # For debugging
        if not hist.empty:
            print(f"Latest data for {symbol} is from: {hist.index[-1]}")
            print(f"Latest close price: {hist['Close'].iloc[-1]}")
        else:
            print(f"No data returned for {symbol}")
        
        # Convert to desired format
        result = []
        for date, row in hist.iterrows():
            # Format date properly
            if isinstance(date, pd.Timestamp):
                date_str = date.strftime('%Y-%m-%d')
                if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
                    date_str = date.strftime('%Y-%m-%d %H:%M:%S')
            else:
                date_str = str(date)
            
            result.append({
                "date": date_str,
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": float(row['Volume'])
            })
        
        # Cache the result
        data_cache[cache_key] = (result, current_time)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        traceback.print_exc()
        return jsonify([])

# Make this a consistent endpoint pattern with the other endpoint
@app.route('/api/stocks/<symbol>/candlestick', methods=['GET'])
def get_candlestick_data(symbol):
    try:
        # Get the time period from the request query parameters
        time_period = request.args.get('period', '1m')
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        # Create cache key
        cache_key = f"{symbol}_{time_period}_candlestick"
        current_time = time.time()
        
        # Check if we have cached data and it's not a forced refresh
        if not refresh and cache_key in data_cache:
            cached_data, timestamp = data_cache[cache_key]
            # If cache is still valid
            if current_time - timestamp < CACHE_DURATION:
                print(f"Returning cached candlestick data for {symbol}")
                return jsonify(cached_data)
        
        # Convert time period to yfinance period parameter
        period_map = {
            '1d': '1d',
            '1w': '1wk',
            '1m': '1mo',
            '1y': '1y',
            '5y': '5y'
        }
        
        # Set interval based on time period for reasonable data points
        interval_map = {
            '1d': '5m',     # 5-minute intervals for 1 day
            '1w': '1h',     # 1-hour intervals for 1 week
            '1m': '1d',     # 1-day intervals for 1 month
            '1y': '1d',     # 1-day intervals for 1 year
            '5y': '1wk'     # 1-week intervals for 5 years
        }
        
        period = period_map.get(time_period, '1mo')
        interval = interval_map.get(time_period, '1d')
        
        # Get data from yfinance with the specified period and interval
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval, prepost=True)
        
        result = []
        for date, row in hist.iterrows():
            # Convert timestamp to string in the format expected by lightweight-charts
            time_str = date.strftime('%Y-%m-%d')
            if interval in ['1m', '5m', '15m', '30m', '1h']:
                # Include time for intraday data
                time_str = date.strftime('%Y-%m-%d %H:%M')
            
            result.append({
                "time": time_str,
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close'])
            })
        
        # Cache the result
        data_cache[cache_key] = (result, current_time)
        
        return jsonify(result)
    except Exception as e:
        print(f"Error fetching candlestick data for {symbol}: {str(e)}")
        traceback.print_exc()
        return jsonify([])

@app.route('/api/stocks/fixed-data', methods=['GET'])
def get_stock_data_from_csv():
    """
    Return 5-year stock data for 10 companies from stock_data_fixed.csv
    """
    import pandas as pd
    import os
    
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'public', 'stock_data_fixed.csv'  # Adjust path as needed
    )
    
    if not os.path.exists(csv_path):
        return jsonify({"error": "CSV file not found"}), 404
    
    df = pd.read_csv(csv_path)
    
    # Convert DataFrame to a list of dicts (one for each row)
    records = df.to_dict(orient='records')
    
    return jsonify(records)

# Add endpoint that clears cache
@app.route('/api/clear-cache', methods=['GET'])
def clear_cache():
    global data_cache
    data_cache = {}
    return jsonify({"status": "success", "message": "Cache cleared"})

@app.route('/api/trained-predictions', methods=['GET'])
def get_trained_predictions():
    """Return the stored model predictions as JSON."""
    import os
    import json
    
    json_path = 'prediction_api_data.json'  # Or wherever you saved
    if not os.path.exists(json_path):
        return jsonify({'error': 'No prediction data found'}), 404
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return jsonify(data)

@app.route('/api/prediction-summary', methods=['GET'])
def get_prediction_summary():
    """Return all predictions from the ML-generated CSV file"""
    try:
        import pandas as pd
        
        csv_path = 'prediction_summary.csv'
        
        if not os.path.exists(csv_path):
            return jsonify({'error': 'Prediction summary file not found'}), 404
            
        # Read the prediction summary CSV
        df = pd.read_csv(csv_path)
        
        # Convert to a dictionary where ticker symbols are keys
        predictions = {}
        for _, row in df.iterrows():
            symbol = row.get('Symbol') or row.get('Ticker')  # Check both common column names
            if symbol:
                predictions[symbol] = row.to_dict()
                
        return jsonify(predictions)
    except Exception as e:
        print(f"Error loading prediction summary: {str(e)}")
        return jsonify({'error': 'Failed to load prediction data'}), 500

@app.route('/api/prediction-summary/<symbol>', methods=['GET'])
def get_prediction_for_symbol(symbol):
    """Return prediction for a specific stock symbol from the ML-generated CSV"""
    try:
        import pandas as pd
        import os
        
        csv_path = 'prediction_summary.csv'
        
        if not os.path.exists(csv_path):
            print(f"CSV file not found at: {os.path.abspath(csv_path)}")
            return jsonify({'error': 'Prediction summary file not found'}), 404
            
        # Print details about the file
        print(f"Reading CSV at: {os.path.abspath(csv_path)}")
        
        # Read the prediction summary CSV
        df = pd.read_csv(csv_path)
        print(f"CSV loaded, found columns: {list(df.columns)}")
        print(f"Available symbols: {list(df['Symbol'])}")
        
        # Look for symbol in Symbol column
        symbol = symbol.upper()
        symbol_data = df[df['Symbol'] == symbol]
            
        if len(symbol_data) == 0:
            print(f"No data found for symbol: {symbol}")
            return jsonify({'error': f'No prediction found for {symbol}'}), 404
        
        row_dict = symbol_data.iloc[0].to_dict()
        print(f"Found prediction for {symbol}: {row_dict}")
            
        # Return the first matching row
        return jsonify(row_dict)
    except Exception as e:
        import traceback
        print(f"Error loading prediction for {symbol}: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to load prediction data', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)