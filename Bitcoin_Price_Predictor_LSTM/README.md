# Bitcoin Price Prediction with LSTM 📈

This project trains a **Long Short-Term Memory (LSTM)** neural network to predict Bitcoin closing prices from historical data.

## Structure
- `preprocessing.py` — Data loading, normalization, dataset creation
- `model.py` — LSTM model definition
- `train.py` — Model training & evaluation
- `inference.py` — Predict next-day prices with sample outputs
- `data/btc.csv` — Historical Bitcoin price data

## Setup
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib plotly

## Sample Output
Example predictions from the model are stored in [`output/sample_output.txt`](output/sample_output.txt).

![Prediction Plot](output/pred_vs_actual.png)
