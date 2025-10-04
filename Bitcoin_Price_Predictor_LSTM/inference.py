import numpy as np
from preprocessing import load_data, create_dataset
from model import build_lstm

def sample_predictions(model, X_test, y_test, scaler, n_samples=5, time_step=60):
    for i in range(n_samples):
        input_sequence = X_test[i].reshape(1, time_step, 1)
        prediction = model.predict(input_sequence, verbose=0)

        predicted_price = scaler.inverse_transform(prediction)[0][0]
        actual_price = scaler.inverse_transform(y_test[i].reshape(-1, 1))[0][0]
        input_prices = scaler.inverse_transform(X_test[i].reshape(-1, 1)).flatten()

        print(f"\nSample {i+1}:")
        print("Input closing prices (last 5 of 60):", np.round(input_prices[-5:], 2))
        print("Predicted next closing price       :", round(predicted_price, 2))
        print("Actual next closing price          :", round(actual_price, 2))
