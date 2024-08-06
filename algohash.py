import os
import sys
import subprocess
import platform
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_python_installed():
    try:
        subprocess.check_call([sys.executable, "--version"])
        return True
    except:
        return False

def install_python():
    system = platform.system().lower()
    if system == "linux":
        subprocess.check_call(["sudo", "apt-get", "update"])
        subprocess.check_call(["sudo", "apt-get", "install", "-y", "python3", "python3-pip"])
    elif system == "windows":
        # Download and install Python for Windows
        subprocess.check_call(["powershell", "-Command", 
                               "Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe -OutFile python_installer.exe"])
        subprocess.check_call(["python_installer.exe", "/quiet", "InstallAllUsers=1", "PrependPath=1"])
        os.remove("python_installer.exe")
    else:
        raise OSError("Unsupported operating system")

def install_libraries():
    libraries = [
        'requests', 'yfinance', 'pandas', 'pandas-ta', 'numpy', 'scikit-learn', 'joblib'
    ]
    for lib in libraries:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

def install_gpu_libraries():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cudf-cu11", "cuml-cu11", "cupy-cuda11x"])
        return True
    except:
        logging.warning("Failed to install GPU libraries. Falling back to CPU.")
        return False

def check_gpu():
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except:
        return False

def run_trading_algorithm():
    SERVER_IP = '139.162.143.65'
    SERVER_PORT = 5000

    # Rest of the imports
    import requests
    import yfinance as yf
    import pandas as pd
    import pandas_ta as ta
    import numpy as np
    from datetime import datetime
    import time
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import uuid
    import joblib

    use_gpu = check_gpu()
    if use_gpu:
        import cudf
        import cupy as cp
        from cuml.ensemble import RandomForestClassifier as cuRFClassifier
        logging.info("GPU acceleration enabled")
    else:
        logging.info("Running on CPU")

    device_uuid = str(uuid.uuid4())

    # Cache for downloaded data
    data_cache = {}

    def get_task():
        try:
            response = requests.get(f'http://{SERVER_IP}:{SERVER_PORT}/get_task', params={'device_uuid': device_uuid})
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Error getting task: {e}")
            return None

    def submit_result(task_id, win_rate, profit):
        data = {
            'task_id': task_id,
            'win_rate': win_rate,
            'profit': profit
        }
        try:
            response = requests.post(f'http://{SERVER_IP}:{SERVER_PORT}/submit_result', json=data)
            response.raise_for_status()
            logging.info("Result submitted successfully")
        except requests.RequestException as e:
            logging.error(f"Error submitting result: {e}")

    def get_data(symbol, start_date, end_date):
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if cache_key in data_cache:
            return data_cache[cache_key]
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            data_cache[cache_key] = data
            return data
        except Exception as e:
            logging.error(f"Error downloading data: {e}")
            return None

    def calculate_indicators(data):
        try:
            if use_gpu:
                gpu_data = cudf.DataFrame(data)
                gpu_data['SMA_20'] = gpu_data['Close'].rolling(window=20).mean()
                gpu_data['SMA_50'] = gpu_data['Close'].rolling(window=50).mean()
                # Some indicators might not be available in cuDF, fallback to CPU for these
                cpu_data = gpu_data.to_pandas()
                cpu_data['RSI'] = ta.rsi(cpu_data['Close'], length=14)
                macd = ta.macd(cpu_data['Close'])
                cpu_data['MACD'] = macd['MACD_12_26_9']
                cpu_data['MACD_Signal'] = macd['MACDs_12_26_9']
                cpu_data['ATR'] = ta.atr(cpu_data['High'], cpu_data['Low'], cpu_data['Close'], length=14)
                cpu_data['EMA_10'] = ta.ema(cpu_data['Close'], length=10)
                bb = ta.bbands(cpu_data['Close'], length=20)
                cpu_data['BB_upper'] = bb['BBU_20_2.0']
                cpu_data['BB_middle'] = bb['BBM_20_2.0']
                cpu_data['BB_lower'] = bb['BBL_20_2.0']
                return cudf.DataFrame(cpu_data)
            else:
                data['SMA_20'] = ta.sma(data['Close'], length=20)
                data['SMA_50'] = ta.sma(data['Close'], length=50)
                data['RSI'] = ta.rsi(data['Close'], length=14)
                macd = ta.macd(data['Close'])
                data['MACD'] = macd['MACD_12_26_9']
                data['MACD_Signal'] = macd['MACDs_12_26_9']
                data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
                data['EMA_10'] = ta.ema(data['Close'], length=10)
                bb = ta.bbands(data['Close'], length=20)
                data['BB_upper'] = bb['BBU_20_2.0']
                data['BB_middle'] = bb['BBM_20_2.0']
                data['BB_lower'] = bb['BBL_20_2.0']
                return data
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            raise

    def create_features(data):
        try:
            if use_gpu:
                data['Price_Change'] = data['Close'] - data['Open']
                data['High_Low_Range'] = data['High'] - data['Low']
                data['SMA_Cross'] = cp.where(data['SMA_20'] > data['SMA_50'], 1, 0)
            else:
                data['Price_Change'] = data['Close'] - data['Open']
                data['High_Low_Range'] = data['High'] - data['Low']
                data['SMA_Cross'] = np.where(data['SMA_20'] > data['SMA_50'], 1, 0)
            return data
        except Exception as e:
            logging.error(f"Error creating features: {e}")
            raise

    def train_model(X, y):
        if use_gpu:
            model = cuRFClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
        else:
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
            best_model = None
            best_score = 0
            for name, model in models.items():
                score = cross_val_score(model, X, y, cv=5).mean()
                if score > best_score:
                    best_score = score
                    best_model = model
            
            best_model.fit(X, y)
            model = best_model

        return model

    def implement_strategy(data, model, features, confidence_threshold=0.6):
        try:
            if use_gpu:
                probabilities = model.predict_proba(data[features])
                data['Position'] = cp.where(probabilities[:, 1] > confidence_threshold, 1, 
                                            cp.where(probabilities[:, 1] < 1 - confidence_threshold, -1, 0))
                data['Returns'] = data['Close'].pct_change()
                data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
                return data.to_pandas()
            else:
                probabilities = model.predict_proba(data[features])
                data['Position'] = np.where(probabilities[:, 1] > confidence_threshold, 1, 
                                            np.where(probabilities[:, 1] < 1 - confidence_threshold, -1, 0))
                data['Returns'] = data['Close'].pct_change()
                data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
                return data
        except Exception as e:
            logging.error(f"Error implementing strategy: {e}")
            raise

    def process_task(task):
        try:
            symbol = task['symbol']
            start_date = datetime.fromisoformat(task['start_date'])
            end_date = datetime.fromisoformat(task['end_date'])

            logging.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
            data = get_data(symbol, start_date, end_date)
            if data is None or data.empty:
                raise ValueError(f"No data available for {symbol}")

            logging.info("Calculating indicators")
            data = calculate_indicators(data)
            data = create_features(data)
            data.dropna(inplace=True)

            if len(data) < 2:
                raise ValueError(f"Insufficient data points for {symbol}")

            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'EMA_10', 'BB_upper', 'BB_lower', 'Price_Change', 'High_Low_Range', 'SMA_Cross']
            
            # Ensure all features are present
            missing_features = [f for f in features if f not in data.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")

            X = data[features]
            y = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

            if len(X) != len(y):
                raise ValueError(f"Mismatch in feature and target data lengths: X={len(X)}, y={len(y)}")

            logging.info("Training model")
            model = train_model(X[:-1], y[:-1])  # Exclude the last row for training
            
            logging.info("Implementing strategy")
            data = implement_strategy(data, model, features)
            
            win_rate = (data['Strategy_Returns'] > 0).mean()
            total_return = (1 + data['Strategy_Returns']).prod() - 1

            logging.info(f"Task completed. Win rate: {win_rate}, Total return: {total_return}")
            return win_rate, total_return

        except Exception as e:
            logging.error(f"Error processing task: {e}")
            return None, None

    def main():
        while True:
            try:
                task = get_task()
                if task and 'id' in task:
                    logging.info(f"Processing task: {task['id']}")
                    win_rate, profit = process_task(task)
                    if win_rate is not None and profit is not None:
                        submit_result(task['id'], win_rate, profit)
                    else:
                        logging.error(f"Failed to process task: {task['id']}")
                elif task and 'message' in task:
                    logging.info(f"Server message: {task['message']}")
                    if task['message'] == 'No tasks available':
                        time.sleep(60)  # Wait for 60 seconds before checking again
                else:
                    logging.warning(f"Incomplete task data: {task}")
                time.sleep(1)  # Small delay to prevent hammering the server
            except Exception as e:
                logging.error(f"Unexpected error in main loop: {e}")
                time.sleep(5)  # Wait a bit before trying again

    main()

if __name__ == "__main__":
    if not is_python_installed():
        print("Python is not installed. Installing Python...")
        install_python()
    
    print("Checking and installing required libraries...")
    install_libraries()
    
    print("Checking for GPU support...")
    if install_gpu_libraries():
        print("GPU libraries installed successfully.")
    else:
        print("GPU libraries not available. Using CPU.")
    
    print("Libraries installed successfully.")
    print("Running trading algorithm...")
    run_trading_algorithm()