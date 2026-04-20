#!/usr/bin/env python3
"""
Preprocesses coinbase and bitstamp BTC datasets for time series forecasting.
Saves cleaned, merged, normalized data as a numpy file.
"""
import numpy as np
import pandas as pd


def load_and_clean(filepath):
    """
    Load a BTC CSV dataset and clean it.

    Args:
        filepath: path to the CSV file

    Returns:
        DataFrame with cleaned BTC data
    """
    cols = [
        'Timestamp', 'Open', 'High', 'Low', 'Close',
        'Volume_BTC', 'Volume_Currency', 'Weighted_Price'
    ]
    df = pd.read_csv(filepath, names=cols, header=0)
    df = df.replace('NaN', np.nan)
    df[cols[1:]] = df[cols[1:]].apply(pd.to_numeric, errors='coerce')

    # Forward fill NaN values (carry last known price forward)
    df = df.ffill()
    df = df.dropna()

    # Only keep data from 2017 onwards (pre-2017 is sparse/less relevant)
    df = df[df['Timestamp'] >= 1483228800]

    df = df.sort_values('Timestamp').reset_index(drop=True)
    return df


def merge_datasets(coinbase_path, bitstamp_path):
    """
    Load, clean, and merge coinbase and bitstamp datasets.

    Args:
        coinbase_path: path to coinbase CSV file
        bitstamp_path: path to bitstamp CSV file

    Returns:
        Merged DataFrame averaged across both exchanges
    """
    cb = load_and_clean(coinbase_path)
    bs = load_and_clean(bitstamp_path)

    # Merge on Timestamp, average price columns across exchanges
    merged = pd.merge(cb, bs, on='Timestamp', suffixes=('_cb', '_bs'))
    price_cols = ['Open', 'High', 'Low', 'Close', 'Weighted_Price']
    for col in price_cols:
        merged[col] = (merged[f'{col}_cb'] + merged[f'{col}_bs']) / 2
    merged['Volume_BTC'] = (
        merged['Volume_BTC_cb'] + merged['Volume_BTC_bs']
    )
    merged['Volume_Currency'] = (
        merged['Volume_Currency_cb'] + merged['Volume_Currency_bs']
    )
    keep = ['Timestamp'] + price_cols + ['Volume_BTC', 'Volume_Currency']
    merged = merged[keep]
    return merged


def normalize(df):
    """
    Normalize feature columns using min-max scaling.

    Args:
        df: DataFrame with BTC data

    Returns:
        Tuple of (normalized numpy array, min values, max values)
    """
    features = ['Close', 'High', 'Low', 'Open',
                'Volume_BTC', 'Weighted_Price']
    data = df[features].values.astype(np.float32)
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    # Avoid division by zero
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normalized = (data - mins) / ranges
    return normalized, mins, maxs


def create_sequences(data, window=1440):
    """
    Create sliding window sequences for time series forecasting.

    Uses past 24 hours (1440 minutes) to predict next hour close price.
    The target is the Close price (index 0) 60 steps ahead.

    Args:
        data: normalized numpy array of shape (n, features)
        window: number of time steps to look back (default 1440 = 24 hours)

    Returns:
        Tuple of (X sequences, y targets) as numpy arrays
    """
    X, y = [], []
    # Predict close price 60 steps (1 hour) ahead
    horizon = 60
    for i in range(len(data) - window - horizon):
        X.append(data[i:i + window])
        # Target: Close price (index 0) at t + window + horizon
        y.append(data[i + window + horizon, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


if __name__ == '__main__':
    coinbase_path = 'coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    bitstamp_path = 'bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv'

    print("Loading and cleaning datasets...")
    merged = merge_datasets(coinbase_path, bitstamp_path)
    print(f"Merged dataset shape: {merged.shape}")

    print("Normalizing...")
    normalized, mins, maxs = normalize(merged)

    print("Creating sequences (window=1440, horizon=60)...")
    X, y = create_sequences(normalized, window=1440)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Save preprocessed data
    np.save('X_btc.npy', X)
    np.save('y_btc.npy', y)
    np.save('btc_mins.npy', mins)
    np.save('btc_maxs.npy', maxs)
    print("Saved: X_btc.npy, y_btc.npy, btc_mins.npy, btc_maxs.npy")
