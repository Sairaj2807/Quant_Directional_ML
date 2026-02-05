# Quant Directional ML – Time Series Price Movement Prediction

## 1. Problem Understanding & Assumptions

### Objective
Design and implement a **time-series aware machine learning pipeline** to predict the **directional movement (UP/DOWN)** of a stock’s closing price over the **next 15 trading days**.

Formally:
> For each day *t*, predict whether  
> **Close[t+15] > Close[t]**

This is framed as a **binary classification problem** with strong emphasis on:
- Avoiding data leakage
- Financially meaningful evaluation
- Robust backtesting rather than raw ML accuracy

### Assumptions
- Trades are executed at the **next day’s open** after signal generation
- No transaction costs, slippage, or liquidity constraints (explicitly discussed later)
- Risk-free rate assumed to be **0** for Sharpe calculation
- Strategy is **long-only**: long when prediction = 1, flat otherwise

---

## 2. Data Preparation

### Data Source
- Historical daily OHLCV data downloaded using **Yahoo Finance (`yfinance`)**
- Tickers used:
  - `RELIANCE.NS`
  - `HDFCBANK.NS`
  - `INFY.NS`

### Steps
- Download data starting from **2015-01-01**
- Cache locally to avoid repeated downloads
- Ensure:
  - Sorted by date
  - No duplicate timestamps
  - Proper datetime handling

### Returns
- Computed **percentage returns** and **log returns**
- Returns are used both as:
  - Predictive features
  - Inputs for volatility estimation
  - Backtesting performance calculations

Missing values created by rolling features are **not forward-filled** to avoid leakage.  
Rows with insufficient historical context are naturally dropped.

---

## 3. Target Definition (Binary Classification)

For each timestamp *t*:

    target = 1 if Close[t+15] > Close[t]
    target = 0 otherwise

    
Key precautions:
- Target is **shifted backward** to align with features at time *t*
- Last 15 rows are dropped (no future label available)
- Ensures **strict no-lookahead bias**

### Class Imbalance Check
- Up/down distribution is close to **50/50** for most tickers
- Mild imbalance handled using:
  - `class_weight="balanced"` for Logistic Regression
  - Tree-based robustness in XGBoost

---

## 4. Feature Engineering

A total of **20+ features** are created using only data available up to time *t*.

### Feature Categories

#### 1. Lagged Returns
- `ret_1`, `ret_5`, `ret_10`, `ret_20`

#### 2. Trend & Moving Averages
- SMA: 5, 10, 20, 50, 100, 200


#### 3. Momentum
- Rate of Change (ROC 10, 20)
- Price relative to medium-term trend

#### 4. Volatility
- Rolling standard deviation of returns:
  - 10, 20, 60 days

#### 5. Technical Indicators
(using `pandas-ta`)
- RSI (14)
- MACD + Signal line
- Bollinger Band width (volatility proxy)

#### 6. Calendar Effects
- Day of week
- Month

> **Important:**  
> All features are strictly causal — no future information is used.

---

## 5. Train/Test Split & Cross-Validation

### Why NOT Random K-Fold?
Random shuffling causes **severe leakage** in time-series data because future information can leak into training.

### Strategy Used
1. **Chronological split**
   - Train: first ~80%
   - Test: last ~20%

2. **Walk-forward / Expanding Window CV**
   - Implemented using `TimeSeriesSplit`
   - Ensures:
     - Training data always occurs before validation data
     - Realistic simulation of live trading conditions

Optional improvements (not implemented here):
- Purged CV
- Embargo periods to avoid overlapping labels

---

## 6. Modeling

### Models Implemented

#### 1. Logistic Regression (Baseline)
- Interpretable
- Strong baseline for directional prediction
- Handles class imbalance via class weights

#### 2. XGBoost Classifier
- Captures non-linear relationships
- Robust to feature interactions
- Generally superior performance in financial data

Hyperparameters were chosen conservatively to reduce overfitting.

---

## 7. Evaluation Metrics

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1-score
(Focus on **precision** for long signals)

### Financial Metrics (Primary Focus)
- Cumulative return
- Annualized Sharpe ratio
- Maximum drawdown
- Hit rate (directional accuracy)

### Benchmark
- Compared strategy returns against **buy-and-hold**

---

## 8. Backtest Simulation

### Trading Logic
- Signal is **shifted by 1 day** to simulate realistic execution
- Strategy:
    If prediction == 1 → Long
    If prediction == 0 → Flat


### Outputs
- Equity curve
- Daily strategy returns
- Drawdown profile

Backtest results are saved per ticker for:
- XGBoost
- Logistic Regression

---

## 9. Model Comparison

A consolidated comparison table (`model_comparison.csv`) evaluates:

| Ticker | Model | Sharpe | Max Drawdown | Total Return |
|------|------|--------|--------------|--------------|

Key observations:
- XGBoost generally outperforms Logistic Regression
- Some tickers show weak or negative Sharpe → highlights regime dependency
- Directional accuracy does not always translate to profitability

---

## 10. Key Learnings

- Time-series leakage is **the #1 hidden risk** in ML trading systems
- High accuracy ≠ profitable strategy
- Simpler models can outperform complex ones in certain regimes
- Walk-forward validation is mandatory for credibility

---

## 11. Challenges Faced & Solutions

### 1. Data Leakage
**Problem:**  
Feature rolling windows and target shifting can accidentally leak future data.

**Solution:**  
- Strict chronological splits
- Shift targets backward
- Shift predictions forward during backtesting

---

### 2. Overfitting
**Problem:**  
Tree-based models can overfit noisy financial data.

**Solution:**  
- Conservative hyperparameters
- Focus on out-of-sample Sharpe, not in-sample accuracy

---

### 3. Weak Financial Performance Despite Good ML Metrics
**Problem:**  
Good F1-score but poor Sharpe.

**Solution:**  
- Financial metrics prioritized
- Emphasized drawdown & equity curve shape

---

## 12. Limitations & Risks

- Transaction costs and slippage ignored
- Single-horizon prediction (15 days)
- No position sizing or risk management
- Strategy assumes immediate liquidity
- Regime shifts can break historical patterns

---

## 13. Possible Improvements

- Add volume-based features (OBV, volume ROC)
- Ensemble multiple models
- Dynamic position sizing (volatility targeting)
- Regime detection
- LSTM / Transformer models (advanced)
- Live data ingestion & paper trading pipeline

---

Here you go ✅
Below is **pure `README.md`–ready Markdown**, clean and copy-pasteable.
No commentary, no emojis, no extra text — just documentation.

---


## How to Run This Project (Models, Tickers, Horizons)

Each Jupyter notebook runs the **entire pipeline end-to-end**, including:

- Data download  
- Feature engineering  
- Model training  
- Walk-forward validation  
- Backtesting  
- Performance evaluation  

---

## 📁 Entry Points

The two main execution entry points are:

- `Logistic_regression.ipynb`
- `XGBoost.ipynb`

Each notebook is **self-contained** and can be run independently.

---

## ⚙️ Configurable Parameters

At the **top of each notebook**, key parameters are defined.  
These control the **ticker**, **model behavior**, and **prediction horizon**.

---

###  Ticker Selection

You can change the stock ticker by modifying:

```python
TICKER = "RELIANCE.NS"
````

Examples:

```python
TICKER = "HDFCBANK.NS"
TICKER = "INFY.NS"
TICKER = "AAPL"
TICKER = "MSFT"
```

Any ticker supported by **Yahoo Finance (`yfinance`)** will work.

---

###  Prediction Horizon (Forward Days)

The directional target is defined as:

> **Close[t + HORIZON] > Close[t]**

Change the horizon here:

```python
HORIZON = 15  # number of trading days ahead
```

Examples:

```python
HORIZON = 5    # short-term
HORIZON = 10
HORIZON = 30   # medium-term
```

**Notes:**

* Larger horizons reduce the available sample size
* Very small horizons (1–3 days) increase noise
* The last `HORIZON` rows are automatically dropped to avoid lookahead bias

---

###  Model Choice

There is **no model switch inside a single notebook**.

Instead:

* `Logistic_regression.ipynb` → Logistic Regression pipeline
* `XGBoost.ipynb` → XGBoost pipeline

To compare models:

* Run **both notebooks**
* Results are saved separately and later consolidated into `model_comparison.csv`

---








## 14. Conclusion

This project demonstrates a **production-grade, time-series aware ML pipeline** for financial prediction with:
- Proper validation
- Realistic backtesting
- Clear separation of data, modeling, and evaluation

It emphasizes **robust methodology over optimistic results**, which is critical in quantitative finance.



