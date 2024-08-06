import torch
import sys

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')

if not torch.cuda.is_available():
    print("GPU is not available. Falling back to CPU.")

import sys
import subprocess
import platform
import os

def install_libraries():
    libraries = [
        'requests', 'yfinance', 'pandas', 'pandas-ta', 'numpy', 'scikit-learn', 'joblib', 'torch',
        'cudf-cu12', 'cuml-cu12', 'cupy-cuda12x'
    ]
    for lib in libraries:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

def run_trading_algorithm():
    # Your existing code goes here
    import requests
    import yfinance as yf
    import pandas as pd
    import pandas_ta as ta
    import numpy as np
    from datetime import datetime
    import time
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib

    SERVER_IP = '139.162.143.65'
    SERVER_PORT = 5000

    device_uuid = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU"

    def get_task():
        response = requests.get(f'http://{SERVER_IP}:{SERVER_PORT}/get_task', params={'device_uuid': device_uuid})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting task: {response.status_code}")
            return None

    def submit_result(task_id, win_rate, profit):
        data = {
            'task_id': task_id,
            'win_rate': win_rate,
            'profit': profit
        }
        response = requests.post(f'http://{SERVER_IP}:{SERVER_PORT}/submit_result', json=data)
        if response.status_code == 200:
            print("Result submitted successfully")
        else:
            print(f"Error submitting result: {response.status_code}")

    def calculate_indicators(data):
        data['SMA_20'] = ta.sma(data['Close'], length=20)
        data['SMA_50'] = ta.sma(data['Close'], length=50)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        macd = ta.macd(data['Close'])
        data['MACD'] = macd['MACD_12_26_9']
        data['MACD_Signal'] = macd['MACDs_12_26_9']
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        return data

    def process_task(task):
        symbol = task['symbol']
        start_date = datetime.fromisoformat(task['start_date'])
        end_date = datetime.fromisoformat(task['end_date'])

        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data available for {symbol} in the specified date range")

            data = calculate_indicators(data)
            data.dropna(inplace=True)

            if len(data) < 2:
                raise ValueError(f"Insufficient data points for {symbol} in the specified date range")

            data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ATR']
            X = data[features].dropna()
            y = data['Target'].dropna()

            if len(X) != len(y) or len(X) == 0:
                raise ValueError(f"Mismatched feature and target data for {symbol}")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if torch.cuda.is_available():
                from cuml.ensemble import RandomForestClassifier as cuRFClassifier
                model = cuRFClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            # Simulate trading
            data['Position'] = model.predict(X)
            data['Returns'] = data['Close'].pct_change().fillna(0)
            data['Strategy_Returns'] = data['Position'].shift(1).fillna(0) * data['Returns']
            
            cumulative_returns = (1 + data['Strategy_Returns']).cumprod()
            total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0

            return accuracy, total_return

        except Exception as e:
            print(f"Error processing task: {e}")
            return None, None

    def main():
        while True:
            task = get_task()
            if task and 'id' in task:
                print(f"Processing task: {task['id']}")
                win_rate, profit = process_task(task)
                if win_rate is not None and profit is not None:
                    submit_result(task['id'], win_rate, profit)
                else:
                    print(f"Error processing task: {task['id']}")
            elif task and 'message' in task:
                print(f"Server message: {task['message']}")
                if task['message'] == 'No tasks available':
                    time.sleep(60)  # Wait for 60 seconds before checking again
            else:
                print("Incomplete task data:", task)
            time.sleep(1)  # Small delay to prevent hammering the server

    main()

if __name__ == "__main__":
    print("Checking and installing required libraries...")
    install_libraries()
    print("Libraries installed successfully.")
    print("Running trading algorithm...")
    run_trading_algorithm()
