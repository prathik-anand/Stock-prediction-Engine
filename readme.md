# Real-Time Stock Prediction Engine

This project implements a real-time stock prediction engine using machine learning techniques, specifically Long Short-Term Memory (LSTM) networks. The model predicts whether to buy or sell a stock based on historical price data and technical indicators.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [License](#license)

## Features
- Downloads historical stock data using Yahoo Finance.
- Calculates technical indicators such as Simple Moving Averages (SMA), Relative Strength Index (RSI), and Price-to-Earnings (P/E) ratio.
- Preprocesses data for LSTM model training.
- Trains an LSTM model to predict stock price movements.
- Provides buy/sell recommendations based on model predictions.

## Installation

To run this project, you need to have Python installed on your machine. Follow these steps to set up the environment:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. Install the required packages:
   ```bash
   pip install pandas numpy yfinance scikit-learn tensorflow matplotlib
   ```

## Usage

1. Open the `main.py` file.
2. Modify the `ticker`, `start_date`, and `end_date` variables in the `__main__` section to specify the stock you want to analyze.
3. Run the script:
   ```bash
   python main.py
   ```

The script will download the stock data, process it, train the LSTM model, and output a buy/sell recommendation based on the latest data.

## Code Structure

- `main.py`: The main script that contains all the functionality for downloading data, feature engineering, model building, training, and evaluation.
  - **Functions**:
    - `download_stock_data(ticker, start_date, end_date)`: Downloads historical stock data.
    - `add_technical_indicators(data)`: Adds technical indicators to the dataset.
    - `preprocess_data(data)`: Prepares the data for LSTM training.
    - `build_lstm_model(input_shape)`: Constructs the LSTM model.
    - `train_and_evaluate_model(model, X_train, y_train, X_test, y_test)`: Trains the model and evaluates its performance.
    - `stock_recommendation(model, data, scaler)`: Provides a buy/sell recommendation based on the model's prediction.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
