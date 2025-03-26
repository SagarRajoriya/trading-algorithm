# Stock Trading Algorithm with ML Predictions

A full-stack trading dashboard application that uses LSTM and Deep Reinforcement Policy Optimization (DRPO) to predict stock movements and generate trading signals.

## Features

- Interactive stock price charts with candlestick patterns
- ML-powered price prediction and trading recommendations
- Historical stock data visualization
- Portfolio performance tracking

## Tech Stack

- **Frontend**: React, TypeScript, Vite, TailwindCSS
- **Backend**: Python, PyTorch, Pandas
- **ML Models**: LSTM networks, Deep Reinforcement Learning

## Project Structure

- `/models` - Trained PyTorch models for stock prediction
- `/data` - Stock price datasets
- `/trading-algo/project/src` - React frontend application
- `/trading-algo/project/ML.py` - Machine learning model implementation

## Getting Started

### Prerequisites

- Node.js 16+ and npm
- Python 3.8+
- PyTorch 1.8+

### Installation

1. Clone the repository
   ```
   git remote add origin https://github.com/SagarRajoriya/trading-algorithm.git
   cd trading-algorithm
   ```

2. Install frontend dependencies
   ```
   cd trading-algo/project
   npm install
   ```

3. Install Python dependencies
   ```
   pip install -r requirements.txt
   ```

4. Start the development server
   ```
   npm run dev
   ```

5. Run the ML prediction script
   ```
   python ML.py
   ```

## Usage

- Select a stock from the list to view its price chart
- View AI predictions and trading recommendations
- Adjust time ranges to see different periods of data

## License

[MIT License](LICENSE)
