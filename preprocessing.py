import yfinance as yf
import sys
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

@dataclass
class ProcessTickerData:
    
    ticker : str
    
    def _generate(self):
        """
        Generate historical data and derived features for the given ticker.
        """
        try:
            ticker = yf.Ticker(self.ticker)
            historical_data = ticker.history(
                start=str((datetime.today() - timedelta(365*20)).date()), 
                end=str(datetime.today().date())
            )
            # Derived columns
            historical_data['close_dir'] = np.where(
                historical_data['Close'].shift(1) > historical_data['Close'], 1, 0
            )
            # Lagged features
            historical_data['lag1_Close'] = historical_data['Close'].shift(1)
            historical_data['lag1_Open'] = historical_data['Open'].shift(1)
            historical_data['lag1_High'] = historical_data['High'].shift(1)
            historical_data['lag1_Low'] = historical_data['Low'].shift(1)
            historical_data['lag1_Volume'] = historical_data['Volume'].shift(1)
            historical_data['lag1_Dividends'] = historical_data['Dividends'].shift(1)
            historical_data['lag1_Stock_Splits'] = historical_data['Stock Splits'].shift(1)

            # Percentage changes and other features
            historical_data['pct_change_Close'] = historical_data['Close'].pct_change(1)
            historical_data['pct_change_Open'] = historical_data['Open'].pct_change(1)
            historical_data['pct_change_High'] = historical_data['High'].pct_change(1)
            historical_data['pct_change_Low'] = historical_data['Low'].pct_change(1)
            historical_data['pct_change_Volume'] = historical_data['Volume'].pct_change(1)

            # Rolling windows for volatility, SMA, EMA
            window = 5
            historical_data['volatility_5'] = historical_data['Close'].pct_change().rolling(window).std()
            historical_data['SMA_5'] = historical_data['Close'].rolling(window=5).mean()
            historical_data['SMA_10'] = historical_data['Close'].rolling(window=10).mean()
            historical_data['EMA_5'] = historical_data['Close'].ewm(span=5, adjust=False).mean()

            # RSI calculation
            delta = historical_data['Close'].diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            historical_data['RSI'] = 100 - (100 / (1 + rs))

            # Additional features
            historical_data['price_range'] = historical_data['High'] - historical_data['Low']
            historical_data['vol_to_close'] = historical_data['Volume'] / historical_data['Close']

            # Handle missing or infinite values
            historical_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            historical_data.dropna(inplace=True)

            # Select features and target
            input_columns = [
                'lag1_Close', 'lag1_Open', 'lag1_High', 'lag1_Low', 'lag1_Volume', 'lag1_Dividends', 
                'lag1_Stock_Splits', 'pct_change_Close', 'pct_change_Open', 'pct_change_High', 
                'pct_change_Low', 'pct_change_Volume', 'volatility_5', 'SMA_5', 'SMA_10', 'EMA_5', 
                'RSI', 'price_range', 'vol_to_close'
            ]
            X = historical_data[input_columns]
            y = historical_data['close_dir']

            return X, y

        except Exception as e:
            raise e

    def _train_test_split(self, x_shape, X, y):
        """
        Split the data into training and testing sets.
        """
        try:
            train_till_index = int((80 * x_shape) / 100)
            x_train_set = X.iloc[:train_till_index, :]
            x_test_set = X.iloc[train_till_index:, :]

            y_train_set = y.iloc[:train_till_index]
            y_test_set = y.iloc[train_till_index:]

            return x_train_set, x_test_set, y_train_set, y_test_set

        except Exception as e:
            raise e

    def _scale(self, x_train, x_test, scaler):
        """
        Scale the training and testing data using the provided scaler.
        """
        try:
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            return x_train_scaled, x_test_scaled

        except Exception as e:
            raise e


def preprocess_main(ticker):
    """
    Main function to handle data preprocessing.
    """
    try:
        scaler = StandardScaler()
        ticker_data = ProcessTickerData(ticker=ticker)

        # Generate data
        X, y = ticker_data._generate()

        # Split into training and testing sets
        x_train_set, x_test_set, y_train_set, y_test_set = ticker_data._train_test_split(
            x_shape=X.shape[0], X=X, y=y
        )

        # Scale the features
        x_train_scaled, x_test_scaled = ticker_data._scale(
            x_train=x_train_set, x_test=x_test_set, scaler=scaler
        )
        return x_train_scaled, x_test_scaled, y_train_set, y_test_set

    except Exception as e:
        raise e