# Forecasting Bitcoin Prices with LSTM: A Deep Dive into Time Series Forecasting

*From raw exchange data to a working RNN model — every step explained.*

---

![Bitcoin Price Chart](https://s3.tradingview.com/x/XUphZ4qP_big.png)

---

## Introduction to Time Series Forecasting

Time series forecasting is the art of predicting future values from a sequence
of historically ordered observations. Unlike standard regression, **the order
of data points matters** — tomorrow's Bitcoin price depends on what happened
yesterday, last week, and last month, not just on arbitrary features.

Financial time series like BTC have several defining characteristics:

- **Non-stationarity** — the mean and variance shift dramatically over time
- **Volatility clustering** — large price swings tend to cluster together
- **Temporal dependencies** — recent minutes are far more informative than
  data from years ago

Classical methods like ARIMA handle linear dependencies well, but Bitcoin's
wild price swings and complex patterns demand something more powerful. That's
where **Recurrent Neural Networks (RNNs)** — specifically **Long Short-Term
Memory (LSTM)** networks — shine. LSTMs learn which past information to
remember and which to forget, making them ideal for volatile financial
sequences.

**The goal:** use the past 24 hours of minute-level BTC data to predict the
closing price one hour into the future.

---

## The Data

Two exchange datasets were used:

- **Coinbase**: minute-level BTC/USD data from 2014–2019
- **Bitstamp**: minute-level BTC/USD data from 2012–2021

Each row represents a 60-second window with 8 fields:

```
Timestamp, Open, High, Low, Close, Volume_BTC, Volume_Currency, Weighted_Price
```

---

## Preprocessing: Turning Raw Data Into Model Inputs

Raw financial data is messy. Here's every decision made in
`preprocess_data.py`:

### 1. Filter to 2017+

Pre-2017 Coinbase data has massive gaps — hundreds of consecutive missing
rows. Including it would poison the model with forward-filled artifacts over
multi-hour stretches. Cutting to `2017-01-01` (Unix timestamp `1483228800`)
keeps only the reliable, dense portion of the dataset.

```python
df = df[df['Timestamp'] >= 1483228800]
```

### 2. Handle Missing Values with Forward Fill

Exchanges occasionally go offline. Rather than dropping entire gaps (which
breaks temporal continuity), missing values are **forward-filled** — the last
known price is carried forward. This is standard practice in high-frequency
financial data and keeps sequences intact.

```python
df = df.ffill()
```

### 3. Merge Both Exchanges by Averaging

Using timestamps present in **both** exchanges, price columns are averaged and
volumes are summed. This reduces exchange-specific noise and gives a more
representative market picture.

### 4. Feature Selection

Not all 8 columns are equally useful:

| Column | Decision | Reason |
|---|---|---|
| `Timestamp` | ❌ Dropped | Positional index, not a feature |
| `Volume_Currency` | ❌ Dropped | Linearly dependent on `Volume_BTC × Close` |
| `Close` | ✅ Kept | Primary prediction target |
| `High`, `Low`, `Open` | ✅ Kept | Intrawindow price context |
| `Volume_BTC` | ✅ Kept | Market activity signal |
| `Weighted_Price` | ✅ Kept | Volume-adjusted price signal |

### 5. Min-Max Normalization

LSTMs are sensitive to input scale. Values in the thousands (BTC price)
alongside values in the single digits (BTC volume per minute) would make the
gradient landscape extremely uneven. Min-max scaling maps every feature into
`[0, 1]`:

```python
normalized = (data - mins) / (maxs - mins)
```

The min/max values are saved separately so predictions can be
inverse-transformed back to USD.

### 6. Sliding Window Sequences

A window of **1440 timesteps (24 hours)** of 6 features each becomes one
input sequence `X`. The target `y` is the **Close price 60 steps (1 hour)
later**. This sliding window moves one minute at a time:

```python
X[i] = data[i : i + 1440]          # shape: (1440, 6)
y[i] = data[i + 1440 + 60, 0]      # Close price 1 hour after window ends
```

---

## Setting Up `tf.data.Dataset`

Feeding numpy arrays directly to `model.fit()` works, but it's slow and
memory-inefficient for large datasets. A `tf.data.Dataset` pipeline enables
**lazy loading, shuffling, and batching** in a single chain:

```python
def make_dataset(X, y, batch_size=64, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
```

Key choices:

- **`shuffle=True` only for training** — the validation set must stay in
  chronological order to simulate real inference
- **`buffer_size=10000`** — shuffles among the next 10,000 samples, breaking
  temporal correlation between adjacent training batches
- **`.prefetch(AUTOTUNE)`** — prepares the next batch on CPU while the GPU
  processes the current one, eliminating pipeline idle time

---

## Model Architecture

The model uses two stacked LSTM layers followed by dense projection:

```
Input shape:  (1440, 6)
              ↓
LSTM(64, return_sequences=True)   # learns temporal patterns, passes full sequence forward
Dropout(0.2)
              ↓
LSTM(32, return_sequences=False)  # summarizes the sequence into a single vector
Dropout(0.2)
              ↓
Dense(16, activation='relu')      # non-linear feature combination
              ↓
Dense(1)                          # predicted normalized close price
```

**Why two LSTM layers?** The first layer learns lower-level temporal features
(minute-to-minute momentum). The second layer sees the full sequence of these
abstractions and learns higher-order patterns (hourly trends, volatility
regimes).

**Why Dropout?** BTC data is inherently noisy. Dropout at 20% regularizes the
model and prevents it from memorizing exchange-specific quirks.

**Loss function: MSE** — directly penalizes large prediction errors, which is
appropriate when the cost of a large miss (buying or selling at the wrong
price) is disproportionately high.

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse',
    metrics=['mae']
)
```

Training used:

- **EarlyStopping** (patience=5) — stops when validation loss stops
  improving, restoring best weights automatically
- **ReduceLROnPlateau** (patience=3, factor=0.5) — halves the learning rate
  when loss plateaus

---

## Results

The model was trained on **80% of the data chronologically** (no shuffling
the train/val split — doing so would constitute data leakage) and validated
on the remaining 20%.

| Metric | Value (normalized space) |
|---|---|
| Validation MSE | ~0.0012 |
| Validation MAE | ~0.024 |

In USD terms, at 2018 BTC price levels, an MAE of ~0.024 in normalized space
translates to roughly **$240–$480 in absolute price error** depending on the
price range in the validation period. This is acceptable for directional
trading signals but far from precise enough for high-frequency arbitrage.

The training curve shows validation loss closely tracking training loss with
no significant divergence — dropout and early stopping successfully prevented
overfitting.

---

## Conclusion

Building an LSTM-based BTC forecaster taught several hard lessons:

**What worked:** The LSTM architecture picked up clear short-term price
momentum signals. When the previous hour trended strongly in one direction,
the model predicted the continuation reasonably well.

**What didn't:** BTC is subject to sudden, news-driven shocks — exchange
hacks, regulatory announcements, macroeconomic events — that no
pattern-based model can anticipate. The model systematically under-predicted
extreme moves, unsurprisingly since those extremes are rare in training data.

**Would I trade on this?** Not without significant further work: incorporating
sentiment data, on-chain metrics (active addresses, hash rate), and ensemble
methods. The model is a useful baseline, not an oracle.

Time series forecasting with deep learning is a fascinating problem where the
hardest challenges are almost never the model architecture — they're data
quality, feature engineering, and resisting the urge to overfit on historical
regimes that no longer apply.

**Full code:** [GitHub Repository](https://github.com/abfabs/holbertonschool-machine_learning/tree/main/supervised_learning/time_series)