# Time Series Forecasting — BTC Price Prediction

## Overview
Uses an LSTM-based RNN to forecast Bitcoin (BTC) close price one hour ahead,
trained on minute-level data from the Coinbase and Bitstamp exchanges.

## Files
- `preprocess_data.py` — cleans, merges, normalizes, and sequences the raw datasets
- `forecast_btc.py` — builds, trains, and validates the Keras LSTM model
- `README.md` — this file

## Datasets
Download before running:
- `coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv`
- `bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv`

Each row = one 60-second window with columns:
`Timestamp, Open, High, Low, Close, Volume_BTC, Volume_Currency, Weighted_Price`

## Usage

### Step 1 — Preprocess
```bash
./preprocess_data.py
```
Outputs: `X_btc.npy`, `y_btc.npy`, `btc_mins.npy`, `btc_maxs.npy`

### Step 2 — Train & Validate
```bash
./forecast_btc.py
```
Outputs: `btc_forecast_model.keras`

## Preprocessing Details
- **Filtering**: Only data from 2017 onwards is kept (pre-2017 data is sparse)
- **Missing values**: Forward-filled (carry last known price forward)
- **Exchange merging**: Timestamps present in both exchanges are averaged
- **Features used**: `Close, High, Low, Open, Volume_BTC, Weighted_Price`
- **Normalization**: Min-max scaling per feature over the full dataset
- **Sequences**: Sliding window of 1440 steps (24 hours) → predict Close 60 steps (1 hour) ahead

## Model Architecture
```
Input: (1440, 6)
→ LSTM(64, return_sequences=True)
→ Dropout(0.2)
→ LSTM(32)
→ Dropout(0.2)
→ Dense(16, relu)
→ Dense(1)  ← predicted normalized close price
```

- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr=1e-3)
- **Train/Val split**: 80/20 chronological (no shuffling of time order)
- **Callbacks**: EarlyStopping (patience=5), ReduceLROnPlateau (patience=3)
