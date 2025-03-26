import os
import time
import random
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#########################################
# 1. DATA FETCHING & STORAGE FUNCTIONS  #
#########################################

def fetch_stock_data(ticker, period="2y", retries=3, delay=2):
    """Fetch stock data for a given ticker using yFinance."""
    print(f"Fetching data for {ticker}...")
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df.empty or len(df) < 30:
                print(f"Attempt {attempt+1}/{retries}: Not enough data for {ticker}")
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                return None
            print(f"Retrieved {len(df)} days of data for {ticker}")
            return df
        except Exception as e:
            print(f"Attempt {attempt+1}/{retries}: Error fetching {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None

def generate_and_save_stock_data():
    """Fetch and store historical data for a list of companies."""
    tickers = {
        "AAPL": 220.45, 
        "MSFT": 425.63, 
        "TSLA": 177.82, 
        "AMZN": 178.75, 
        "GOOGL": 175.98,
        "META": 485.92, 
        "NVDA": 925.75, 
        "JPM": 196.82, 
        "NFLX": 610.34, 
        "IBM": 168.76
    }
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    for ticker in tickers.keys():
        df = fetch_stock_data(ticker, period="5y")
        if df is not None:
            file_path = os.path.join(data_dir, f"{ticker}_data.csv")
            df.to_csv(file_path)
            print(f"Saved {ticker} data to {file_path}")
        else:
            print(f"⚠️ Failed to fetch data for {ticker}")

#########################################
# 2. FEATURE ENGINEERING & MODEL UTILS  #
#########################################

def prepare_features(df):
    data = df.copy()
    
    # Force keep only numeric columns
    # so we never pass any string/datetime column to the scaler.
    data = data.select_dtypes(include=[np.number])
    
    # Recompute your additional features (make sure each new feature is numeric)
    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Return'].rolling(window=5).std()
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA5_Cross'] = (data['MA5'] > data['MA10']).astype(int)
    data['MA_Ratio'] = data['MA5'] / data['MA10']
    data['Volume_MA5'] = data['Volume'].rolling(window=5).mean()
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(window=10).mean()
    data['Price_Momentum'] = data['Close'].pct_change(periods=5)
    data['Price_Acceleration'] = data['Return'].diff()
    data['HL_Ratio'] = (data['High'] - data['Low']) / data['Close']
    data['OC_Ratio'] = (data['Close'] - data['Open']) / data['Open']
    
    data.dropna(inplace=True)
    return data

def create_sequences(data, target_col, window_size=20):
    """Turn processed data into sequences for DRPO training."""
    features = data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume']).values
    targets = data[target_col].values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    X, y = [], []
    for i in range(len(features_scaled) - window_size):
        X.append(features_scaled[i:i + window_size])
        # Binary target: 1 if price increases at next step, else 0
        y.append(1 if targets[i + window_size] > targets[i + window_size - 1] else 0)
    return np.array(X), np.array(y), scaler

#########################################
# 3. DRPO NETWORK & TRAINER DEFINITIONS  #
#########################################

class DRPONetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DRPONetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        # Use the last time step for prediction
        output = self.fc(lstm_out[:, -1, :])
        return output

class TradingEnvironment:
    def __init__(self, features, targets, window_size=20):
        self.features = features  # Should be shape (n_samples, window_size, n_features)
        self.targets = targets
        self.window_size = window_size
        self.current_step = 0
        self.max_steps = len(features)
        
    def get_state(self, step):
        """Return a single sequence at the current step"""
        if step >= len(self.features):
            # Return the last available sequence if we're out of bounds
            return torch.FloatTensor(self.features[-1]).unsqueeze(0).to(device)
        return torch.FloatTensor(self.features[step]).unsqueeze(0).to(device)
    
    def reset(self):
        self.current_step = 0
        return self.get_state(self.current_step)
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        state = self.get_state(self.current_step)
        if self.current_step < len(self.targets):
            actual = self.targets[self.current_step]
            action_val = action.item()
            predicted_up = action_val > 0.5
            correct = (predicted_up and actual == 1) or (not predicted_up and actual == 0)
            confidence = abs(action_val - 0.5) * 2
            reward = confidence if correct else -confidence
        else:
            reward = 0
        return state, reward, done

class DRPOTrainer:
    def __init__(self, model, env, learning_rate=0.001, gamma=0.99, batch_size=32):
        self.model = model.to(device)
        self.env = env
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = []
        self.training_history = []
        
    def select_action(self, state):
        """State should already be properly shaped as [1, seq_len, features]"""
        with torch.no_grad():
            # Don't unsqueeze again - it should already have batch dimension
            logits = self.model(state)  
            action = torch.sigmoid(logits)
        return action
        
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)
            
    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Process each experience individually to avoid dimension issues
        total_loss = 0
        for i in range(len(states)):
            state = states[i]
            next_state = next_states[i]
            reward = rewards[i]
            done = dones[i]
            
            # Forward pass
            current_q = self.model(state)
            with torch.no_grad():
                next_q = self.model(next_state)
                max_next_q = torch.sigmoid(next_q)
            
            # Calculate target and loss
            target_q = reward + (1 - done) * self.gamma * max_next_q
            loss = self.criterion(current_q, target_q)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / self.batch_size
        
    def train(self, epochs=10):
        epoch_rewards = []
        epoch_losses = []
        for epoch in range(epochs):
            state = self.env.reset()
            done = False
            total_reward = 0
            loss_sum = 0
            steps = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                self.store_experience(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                steps += 1
                loss = self.learn()
                loss_sum += loss
            avg_loss = loss_sum / max(1, steps)
            epoch_rewards.append(total_reward)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Reward: {total_reward:.2f}")
        self.training_history.append({
            'epoch_rewards': epoch_rewards,
            'epoch_losses': epoch_losses
        })
        return epoch_rewards, epoch_losses

def save_model(model, ticker):
    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", f"{ticker}_model.pth")
    torch.save(model.state_dict(), path)
    print(f"Model for {ticker} saved to {path}")

def load_model(model, ticker):
    path = os.path.join("models", f"{ticker}_model.pth")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model for {ticker} loaded from {path}")
        return True
    return False

def train_company_model_with_data(
    ticker,
    df,
    hidden_size=32,    # was 64 before
    window_size=10,    # was 20 before
    epochs=3,          # was 10 before
    retrain=False
):
    print(f"\nTraining model for {ticker}")

    data = prepare_features(df)
    if len(data) < window_size + 30:
        print(f"Not enough data for {ticker}")
        return None

    feature_cols = [c for c in data.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    input_size = len(feature_cols)
    model = DRPONetwork(input_size, hidden_size, 1)

    if not retrain and load_model(model, ticker):
        print(f"Loaded existing model for {ticker}")
        return model

    X, y, _ = create_sequences(data, 'Close', window_size)
    if len(X) == 0:
        return None

    env = TradingEnvironment(X, y, window_size)
    trainer = DRPOTrainer(model, env, learning_rate=0.001, gamma=0.95, batch_size=16)
    trainer.train(epochs)
    save_model(model, ticker)
    return model

def get_fallback_data(ticker):
    """Generate synthetic data in case yFinance fails."""
    print(f"Generating fallback data for {ticker}")
    base_params = {
        'AAPL': {'price': 175, 'volatility': 0.015, 'trend': 0.0002},
        'MSFT': {'price': 380, 'volatility': 0.014, 'trend': 0.0003},
        'GOOGL': {'price': 140, 'volatility': 0.016, 'trend': 0.0001},
        'AMZN': {'price': 180, 'volatility': 0.018, 'trend': 0.0002},
        'TSLA': {'price': 180, 'volatility': 0.03, 'trend': -0.0001},
        'META': {'price': 480, 'volatility': 0.02, 'trend': 0.0004},
        'NVDA': {'price': 900, 'volatility': 0.025, 'trend': 0.0006},
        'JPM': {'price': 195, 'volatility': 0.013, 'trend': 0.0001},
        'NFLX': {'price': 615, 'volatility': 0.022, 'trend': 0.0000}
    }
    params = base_params.get(ticker, {'price': 100, 'volatility': 0.02, 'trend': 0.0001})
    days = 500
    dates = pd.date_range(end=datetime.now(), periods=days)
    price = params['price']
    prices = [price]
    for _ in range(1, days):
        daily_return = np.random.normal(params['trend'], params['volatility'])
        price *= (1 + daily_return)
        prices.append(price)
    df = pd.DataFrame({
        'Open': prices,
        'Close': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.006))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.006))) for p in prices],
        'Volume': [int(np.random.normal(5000000, 2000000)) for _ in range(days)]
    }, index=dates)
    return df

def predict_stock(ticker, model, window_size=20):
    """Make a prediction for a company using the trained DRPO model."""
    # Use local CSV data instead of Yahoo Finance
    df = load_stock_data_from_csv(ticker)
    
    if df is None or len(df) < 30:
        print(f"Not enough data for {ticker} prediction")
        return None
        
    data = prepare_features(df)
    features = data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume']).values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    if len(features_scaled) < window_size:
        print(f"Insufficient window size for {ticker}")
        return None
    recent_window = features_scaled[-window_size:]
    recent_window_tensor = torch.FloatTensor(recent_window).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(recent_window_tensor)
        probability = torch.sigmoid(prediction).item()
    confidence = abs(probability - 0.5) * 2 * 100
    price_data = df['Close'].values
    recent_price_change = ((price_data[-1] / price_data[-5]) - 1) * 100
    factors = []
    if recent_price_change > 2:
        factors.append(f"Upward momentum (+{recent_price_change:.1f}%)")
    elif recent_price_change < -2:
        factors.append(f"Downward momentum ({recent_price_change:.1f}%)")
    ma5 = data['MA5'].values[-1]
    ma20 = data['MA20'].values[-1]
    factors.append("Bullish" if ma5 > ma20 else "Bearish")
    vol_ratio = data['Volume_Ratio'].values[-1]
    if (vol_ratio > 1.5):
        factors.append(f"High volume ({vol_ratio:.1f}x)")
    volatility = data['Volatility'].values[-1] * 100
    if volatility > 2:
        factors.append(f"High volatility ({volatility:.1f}%)")
    return {
        'symbol': ticker,
        'isProfitable': probability > 0.5,
        'confidence': confidence,
        'probability': probability * 100,
        'recent_change': recent_price_change,
        'factors': factors,
        'current_price': df['Close'].values[-1],
        'prediction': "Likely profitable" if probability > 0.5 else "Likely unprofitable"
    }

#########################################
# 4. MAIN EXECUTION: TRAINING & PREDICTION #
#########################################

def load_stock_data_from_csv(ticker):
    """Load stock data from local CSV files."""
    try:
        # Try multiple possible paths for the CSV files
        possible_paths = [
            os.path.join("backend", "stock_data_fixed.csv"),  # When running from project root
            "stock_data_fixed.csv",                          # When running from backend folder
            os.path.abspath("stock_data_fixed.csv"),         # Absolute path
            os.path.join("..", "stock_data_fixed.csv")       # One directory up
        ]
        
        # Try each possible path for fixed data
        fixed_df = None
        fixed_path = None
        for path in possible_paths:
            if os.path.exists(path):
                fixed_path = path
                fixed_df = pd.read_csv(path)
                print(f"Found fixed data file at: {path}")
                print(f"Columns: {fixed_df.columns.tolist()}")
                break
                
        if fixed_df is not None:
            # Check different possible formats of the CSV
            if ticker in fixed_df.columns:
                # Format 1: Each ticker is a column
                ticker_df = fixed_df[['Date', ticker]].rename(columns={ticker: 'Close'})
                ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
                ticker_df.set_index('Date', inplace=True)
                
                # Fill in missing OHLC data based on Close
                ticker_df['Open'] = ticker_df['Close'] * 0.998
                ticker_df['High'] = ticker_df['Close'] * 1.005
                ticker_df['Low'] = ticker_df['Close'] * 0.995
                ticker_df['Volume'] = 1000000  # Default volume
                
                print(f"Found {len(ticker_df)} records for {ticker}")
                return ticker_df
                
            elif 'Symbol' in fixed_df.columns:
                # Format 2: Rows with a Symbol column
                ticker_df = fixed_df[fixed_df['Symbol'] == ticker].copy()
                if len(ticker_df) > 0:
                    # Convert Date column to datetime and set as index
                    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
                    ticker_df.set_index('Date', inplace=True)
                    print(f"Found {len(ticker_df)} records for {ticker}")
                    return ticker_df
            
            # Format 3: Maybe all data is for one ticker
            if 'Date' in fixed_df.columns and all(col in fixed_df.columns for col in ['Open', 'High', 'Low', 'Close']):
                print(f"CSV appears to be for a single stock, assuming it's {ticker}")
                ticker_df = fixed_df.copy()
                ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
                ticker_df.set_index('Date', inplace=True)
                print(f"Found {len(ticker_df)} records")
                return ticker_df
                
            elif 'Ticker' in fixed_df.columns and all(col in fixed_df.columns for col in ['Open','High','Low','Close','Volume']):
                # Filter rows matching the desired ticker
                ticker_df = fixed_df[fixed_df['Ticker'] == ticker].copy()
                if len(ticker_df) > 0:
                    # If you have a Date column or can create one, set as index
                    if 'Date' in ticker_df.columns:
                        ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
                        ticker_df.set_index('Date', inplace=True)
                    else:
                        # If no Date in CSV, you might add a dummy daily date index
                        ticker_df['Date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(ticker_df))
                        ticker_df.set_index('Date', inplace=True)
                    print(f"Found {len(ticker_df)} records for {ticker} in Ticker column")
                    return ticker_df
                
        # Try same logic for intraday file
        possible_intraday_paths = [
            os.path.join("backend", "stock_data_intraday.csv"),
            "stock_data_intraday.csv",
            os.path.abspath("stock_data_intraday.csv"),
            os.path.join("..", "stock_data_intraday.csv")
        ]
        
        for path in possible_intraday_paths:
            if (os.path.exists(path)):
                print(f"Loading intraday data from {path}")
                intraday_df = pd.read_csv(path)
                
                # Apply same logic to intraday file
                if ticker in intraday_df.columns:
                    ticker_df = intraday_df[['Date', ticker]].rename(columns={ticker: 'Close'})
                    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
                    ticker_df.set_index('Date', inplace=True)
                    
                    # Fill in missing OHLC data
                    ticker_df['Open'] = ticker_df['Close'] * 0.998
                    ticker_df['High'] = ticker_df['Close'] * 1.003
                    ticker_df['Low'] = ticker_df['Close'] * 0.997
                    ticker_df['Volume'] = 500000
                    
                    print(f"Found {len(ticker_df)} intraday records for {ticker}")
                    return ticker_df
                    
                elif 'Symbol' in intraday_df.columns:
                    ticker_df = intraday_df[intraday_df['Symbol'] == ticker].copy()
                    if len(ticker_df) > 0:
                        ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
                        ticker_df.set_index('Date', inplace=True)
                        print(f"Found {len(ticker_df)} intraday records for {ticker}")
                        return ticker_df
                        
                # Check if intraday is for single stock
                if 'Date' in intraday_df.columns and all(col in intraday_df.columns for col in ['Open', 'High', 'Low', 'Close']):
                    print(f"Intraday CSV appears to be for a single stock, assuming it's {ticker}")
                    ticker_df = intraday_df.copy()
                    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
                    ticker_df.set_index('Date', inplace=True)
                    print(f"Found {len(ticker_df)} intraday records")
                    return ticker_df
        
        print(f"Could not find data for {ticker} in CSV files")
        return None
        
    except Exception as e:
        print(f"Error loading CSV data for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None

def store_predictions_in_json(predictions, output_path="model_predictions.json"):
    """Save model predictions to a JSON file for frontend access."""
    import json
    with open(output_path, "w") as outfile:
        json.dump(predictions, outfile, indent=2)
    print(f"Predictions saved to {output_path}")

def main():
    # Step 1: Check for data files and generate if they don't exist
    required_files = [
        "stock_data_fixed.csv",
        "stock_data_intraday.csv"
    ]
    
    # Check if data files exist
    all_files_exist = True
    for file in required_files:
        if not os.path.exists(file) and not os.path.exists(os.path.join("backend", file)):
            all_files_exist = False
            print(f"Required data file {file} not found")
    
    # If data files don't exist, try to generate them
    if not all_files_exist:
        print("Attempting to generate sample data files...")
        try:
            # Import the generate function from our helper script
            from generate_sample_data import generate_sample_stock_data
            generate_sample_stock_data()
            print("Successfully generated sample data!")
        except Exception as e:
            print(f"Error generating sample data: {e}")
            print("Please run generate_sample_data.py first to create the necessary CSV files.")
            return
    
    # Step 1: Fetch and store data if not already done.
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # List of companies with primary and alternative tickers.
    companies = [
        {'primary': 'AAPL', 'alternatives': ['AAPL.US']},
        {'primary': 'MSFT', 'alternatives': ['MSFT.US']},
        {'primary': 'GOOGL', 'alternatives': ['GOOG', 'GOOGL.US']},
        {'primary': 'AMZN', 'alternatives': ['AMZN.US']},
        {'primary': 'TSLA', 'alternatives': ['TSLA.US']},
        {'primary': 'META', 'alternatives': ['FB', 'META.US']},
        {'primary': 'NVDA', 'alternatives': ['NVDA.US']},
        {'primary': 'JPM', 'alternatives': ['JPM.US']},
        {'primary': 'NFLX', 'alternatives': ['NFLX.US', 'NFLX.MX']},
        {'primary': 'IBM', 'alternatives': ['IBM.US']}
    ]
    hidden_size = 64
    window_size = 20
    epochs = 10
    retrain = False

    trained_models = {}
    predictions = {}
    for company in companies:
        primary = company['primary']
        alternatives = company['alternatives']
        print(f"\nProcessing {primary}...")
        
        # Load data from local CSV files instead of Yahoo Finance
        df = load_stock_data_from_csv(primary)
        
        # If primary ticker not found, try alternatives
        if df is None:
            for alt in alternatives:
                df = load_stock_data_from_csv(alt)
                if df is not None:
                    print(f"Using alternative ticker {alt} for {primary}")
                    break
        
        if df is not None:
            model = train_company_model_with_data(primary, df, hidden_size, window_size, epochs, retrain)
            if model:
                trained_models[primary] = model
                pred = predict_stock(primary, model, window_size)
                if pred:
                    predictions[primary] = pred
                    print(f"Prediction for {primary}: {pred['prediction']}, Confidence: {pred['confidence']:.2f}%")
        else:
            print(f"WARNING! Failed to retrieve data for {primary}")
    
    # Save prediction results to CSV and JSON for frontend use.
    if predictions:
        summary_data = []
        for ticker, pred in predictions.items():
            summary_data.append({
                'Symbol': ticker,
                'Current Price': pred['current_price'],
                'Prediction': pred['prediction'],
                'Probability (%)': pred['probability'],
                'Confidence (%)': pred['confidence'],
                'Recent Change (%)': pred['recent_change'],
                'Factors': ", ".join(pred['factors'])
            })
        summary_df = pd.DataFrame(summary_data)
        summary_csv = "prediction_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Prediction summary saved to '{summary_csv}'")
        
        api_data = {}
        for ticker, pred in predictions.items():
            api_data[ticker] = {
                'recommendation': pred['prediction'],
                'confidence': pred['confidence'],
                'current_price': pred['current_price'],
                'predicted_probability': pred['probability'],
                'investment_suggestion': 5000 if pred['prediction'] == "Likely profitable" else 2000
            }
        api_json = "prediction_api_data.json"
        with open(api_json, "w") as f:
            json.dump(api_data, f, indent=2)
        print(f"API data saved to '{api_json}'")
    else:
        print("No predictions were made.")

if __name__ == '__main__':
    main()
