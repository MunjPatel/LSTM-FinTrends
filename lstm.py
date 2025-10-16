import pandas as pd
import numpy as np
import os
import joblib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dataclasses import dataclass
from preprocessing import preprocess_main
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

@dataclass
class LSTMClassifier:
    
    ticker  : str
    x_train : np.array
    x_test  : np.array
    y_train : pd.DataFrame
    y_test  : pd.DataFrame

    # def save_model(self, model):
    #     model_path = f"./models/{self.ticker}.pkl"
    #     os.makedirs(os.path.dirname(model_path), exist_ok=True)
    #     joblib.dump(model, model_path)
    #     print(f"Model saved for ticker: {self.ticker}")
        
    def save_results(self, forecast_accuracy, conf_matrix, class_report, y_test, y_pred):
        results = {
            "forecast_accuracy": forecast_accuracy,
            "confusion_matrix": conf_matrix.tolist(),  # Convert to list for JSON compatibility
            "classification_report": class_report,
            "y_test": y_test.tolist(),  # Convert to list
            "y_pred": y_pred.tolist()   # Convert to list
        }
        result_path = f"./results/{self.ticker}.json"
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved for ticker: {self.ticker}")
        
    def _train(self):
        # Reshape input to 3D for LSTM [samples, timesteps, features]
        x_train_reshaped = self.x_train.reshape((self.x_train.shape[0], 1, self.x_train.shape[1]))
        # x_test_reshaped = self.x_test.reshape((self.x_test.shape[0], 1, self.x_test.shape[1]))
        
        lstm_model = Sequential()
        lstm_model.add(LSTM(100, input_shape=(x_train_reshaped.shape[1], x_train_reshaped.shape[2]), activation='relu', return_sequences=True))
        lstm_model.add(Dropout(0.6))
        lstm_model.add(LSTM(50, activation='relu', return_sequences=True))
        lstm_model.add(Dropout(0.6))
        lstm_model.add(LSTM(25, activation='relu'))
        lstm_model.add(Dropout(0.6))
        lstm_model.add(Dense(1, activation='sigmoid'))
        
        lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        lstm_history = lstm_model.fit(
            x_train_reshaped, self.y_train, epochs=100, batch_size=32,
            validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=5)], verbose=1
        )
        lstm_metrics = pd.DataFrame(lstm_history.history)
        # self.save_model(lstm_model)
        return lstm_model, lstm_metrics
    
    def _test(self, lstm_model):
        x_test_reshaped = self.x_test.reshape((self.x_test.shape[0], 1, self.x_test.shape[1]))
        y_predicted = lstm_model.predict(x_test_reshaped)
        y_pred = np.where(y_predicted > 0.99, 1, 0)  # Threshold for binary classification
        forecast_accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # Save results to JSON
        self.save_results(forecast_accuracy, conf_matrix, class_report, self.y_test, y_pred)
        
        return forecast_accuracy, conf_matrix, class_report

def process_ticker(ticker):
    # Preprocess data for the specific ticker
    x_train, x_test, y_train, y_test = preprocess_main(ticker)
    
    # Initialize and train classifier
    classifier = LSTMClassifier(ticker=ticker, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    lstm_model, lstm_metrics = classifier._train()
    forecast_accuracy, conf_matrix, class_report = classifier._test(lstm_model)
    
    return ticker, forecast_accuracy

if __name__ == "__main__":
    # Load ticker symbols from JSON file
    with open("tickers.json", 'r') as tickers_syms:
        tickers = json.load(tickers_syms).keys()

    # Run in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_ticker, ticker): ticker for ticker in tickers}
        
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                ticker, forecast_accuracy = future.result()
                print(f"Processing complete for ticker: {ticker}, Forecast Accuracy: {forecast_accuracy}")
            except Exception as e:
                print(f"Error processing ticker {ticker}: {e}")
