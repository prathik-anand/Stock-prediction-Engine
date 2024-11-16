import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load environment variables
load_dotenv()
TICKERS = os.getenv("TICKERS").split(',')
START_DATE = os.getenv("START_DATE")
# Set END_DATE to the last trading day
END_DATE = datetime.now().strftime('%Y-%m-%d')  # Default to today
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")

# 1. Data Collection
def download_stock_data(ticker, start_date, end_date):
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    return data

# 2. Feature Engineering
def add_technical_indicators(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Calculate RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    data['RSI'] = 100 - (100 / (1 + rs))

    # Fix for P/E Ratio Calculation
    pe_ratio = (data['Close'] / (data['Return'] + 1e-9)).rolling(window=10).mean()
    if isinstance(pe_ratio, pd.DataFrame):
        pe_ratio = pe_ratio.iloc[:, 0]
    data['P/E'] = pe_ratio.fillna(0)

    # Fill any remaining NaN values
    data.fillna(0, inplace=True)
    return data

# 3. Data Preprocessing
def preprocess_data(data):
    features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'P/E']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i, 0])  # Predicting the next closing price
    
    X, y = np.array(X), np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42), scaler

# 4. LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Predicting the next price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 5. Model Training and Evaluation
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Test Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    return model

# 6. Future Price Prediction
def predict_future_prices(model, data, scaler, days=365):
    last_60_days = data[-60:][['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'P/E']].values
    future_prices = []

    for _ in range(days):
        last_60_days_scaled = scaler.transform(last_60_days)
        X_input = np.expand_dims(last_60_days_scaled, axis=0)
        predicted_price = model.predict(X_input)
        future_prices.append(predicted_price[0][0])

        # Update last_60_days with the predicted price
        # Create a new row with the predicted price and zeros for other features
        new_row = np.array([[predicted_price[0][0], 0, 0, 0, 0, 0]])
        last_60_days = np.append(last_60_days, new_row, axis=0)[-60:]

    # Inverse transform to get actual prices
    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    return future_prices

# Main Function
if __name__ == "__main__":
    try:
        for ticker in TICKERS:
            # Step 1: Data Collection
            data = download_stock_data(ticker, START_DATE, END_DATE)
            
            # Step 2: Feature Engineering
            data = add_technical_indicators(data)
            
            # Step 3: Data Preprocessing
            (X_train, X_test, y_train, y_test), scaler = preprocess_data(data)

            # Step 4: Build LSTM Model
            model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

            # Step 5: Train and Evaluate Model
            model = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)

            # Save the model
            model.save(MODEL_SAVE_PATH.replace('.h5', f'_{ticker}.h5'))
            print(f"Model for {ticker} saved as {MODEL_SAVE_PATH.replace('.h5', f'_{ticker}.h5')}")

            # Step 6: Future Price Prediction
            future_prices = predict_future_prices(model, data, scaler, days=365)
            print(f"Predicted prices for the next year for {ticker}: {future_prices.flatten()}")

            # Calculate predicted high and low prices
            predicted_high = np.max(future_prices)
            predicted_low = np.min(future_prices)
            print(f"Predicted High for {ticker}: {predicted_high:.2f}")
            print(f"Predicted Low for {ticker}: {predicted_low:.2f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
