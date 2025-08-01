import json

# Load JSON from string (or file)
json_string = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 📈 Bank Nifty ML Notebook (2025 Edition)\n",
    "Modern end-to-end workflow: robust ETL, TA-Lib feature engineering, GARCH volatility, LSTM price forecast, mean–variance optimiser and back-test."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# A ▸ Environment & helper utils\n",
    "%pip -q install yfinance==0.2.38 ta-lib==0.4.28 plotly==5.22.0 arch quantstats backtesting tensorflow==2.16.1\n",
    "\n",
    "import warnings, numpy as np, pandas as pd, ta, plotly.express as px\n",
    "from datetime import date, timedelta\n",
    "import yfinance as yf\n",
    "from arch import arch_model\n",
    "import quantstats as qs\n",
    "from scipy.optimize import minimize\n",
    "from backtesting import Backtest, Strategy\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import LSTM, Dense\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "np.random.seed(42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1  Data Download & Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "BANK_TICKERS = [\n",
    "    'HDFCBANK.NS','AXISBANK.NS','ICICIBANK.NS','KOTAKBANK.NS','SBIN.NS',\n",
    "    'INDUSINDBK.NS','BANKBARODA.NS','PNB.NS','IDFCFIRSTB.NS','AUBANK.NS',\n",
    "    'FEDERALBNK.NS','BANDHANBNK.NS'\n",
    "]\n",
    "\n",
    "end_date   = date.today()\n",
    "start_date = end_date - timedelta(days=3*365)\n",
    "\n",
    "raw = yf.download(BANK_TICKERS, start=start_date, end=end_date,\n",
    "                  auto_adjust=True, progress=False)\n",
    "raw = raw.dropna(how='all')\n",
    "raw = raw.stack(level=1).swaplevel().sort_index()  # tidy multi-index\n",
    "raw.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2  Technical-Indicator Factory (TA-Lib) & Returns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def add_ta(df):\n",
    "    df = df.copy()\n",
    "    df['50_MA']  = ta.trend.sma_indicator(df['Close'], 50)\n",
    "    df['200_MA'] = ta.trend.sma_indicator(df['Close'], 200)\n",
    "    df['RSI']    = ta.momentum.rsi(df['Close'], 14)\n",
    "    bb = ta.volatility.BollingerBands(df['Close'], 20, 2)\n",
    "    df['BB_up'], df['BB_mid'], df['BB_low'] = bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()\n",
    "    return df\n",
    "\n",
    "stock = raw.groupby('Ticker').apply(add_ta).droplevel(0)\n",
    "stock['Return'] = stock.groupby('Ticker')['Close'].pct_change()\n",
    "stock = stock.dropna()\n",
    "stock.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3  Volatility Forecast (GARCH-1,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ticker = 'HDFCBANK.NS'\n",
    "series = 100 * stock.loc[pd.IndexSlice[:, ticker], 'Return']  # % returns\n",
    "garch = arch_model(series, mean='Zero', vol='Garch', p=1, q=1, dist='t')\n",
    "res   = garch.fit(update_freq=0)\n",
    "print(res.summary())\n",
    "\n",
    "forecasts = res.forecast(horizon=5, method='simulation')\n",
    "vol_annualised = np.sqrt(forecasts.variance.iloc[-1] * 252) / 100\n",
    "print(f\"5-day forward vol (ann.): {vol_annualised.iloc[0]:.2%}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4  LSTM Next-Day Price Prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def lstm_forecast(df, lookback=60, epochs=15):\n",
    "    close = df['Close'].values.reshape(-1, 1)\n",
    "    scaler = MinMaxScaler().fit(close)\n",
    "    scaled  = scaler.transform(close)\n",
    "    X, y = [], []\n",
    "    for i in range(lookback, len(scaled)):\n",
    "        X.append(scaled[i-lookback:i, 0])\n",
    "        y.append(scaled[i, 0])\n",
    "    X, y = np.array(X), np.array(y)\n",
    "    X = X.reshape((X.shape[0], X.shape[1], 1))\n",
    "    model = Sequential([\n",
    "        LSTM(64, return_sequences=False, input_shape=(lookback,1)),\n",
    "        Dense(1)\n",
    "    ])\n",
    "    model.compile(optimizer='adam', loss='mse')\n",
    "    model.fit(X, y, epochs=epochs, verbose=0)\n",
    "    last_seq = scaled[-lookback:].reshape(1, lookback, 1)\n",
    "    next_day = scaler.inverse_transform(model.predict(last_seq))[0,0]\n",
    "    return next_day\n",
    "\n",
    "pred = {t: lstm_forecast(g) for t, g in stock.groupby('Ticker')}\n",
    "pd.Series(pred, name='LSTM_next_close').sort_values(ascending=False).head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5  Mean–Variance Optimiser"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "returns = stock['Return'].unstack()\n",
    "mu  = returns.mean()*252\n",
    "cov = returns.cov()*252\n",
    "\n",
    "def mvo(expected_r, cov, risk_aversion=3, w_max=0.3):\n",
    "    n = len(expected_r)\n",
    "    init = np.repeat(1/n, n)\n",
    "    bounds = [(0, w_max)]*n\n",
    "    cons   = ({'type':'eq','fun':lambda w: w.sum()-1})\n",
    "    def util(w): return - (w@expected_r - 0.5*risk_aversion*(w@cov@w))\n",
    "    opt = minimize(util, init, bounds=bounds, constraints=cons)\n",
    "    return opt.x\n",
    "\n",
    "w = mvo(mu.values, cov.values)\n",
    "weights = pd.Series(w, index=mu.index, name='Weight').round(3)\n",
    "weights[weights>0].sort_values(ascending=False).head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6  Back-Testing & Risk Report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "prices = stock['Close'].unstack()\n",
    "def weighted_portfolio(pr, w):\n",
    "    return (pr.pct_change().fillna(0) @ w).add(1).cumprod()\n",
    "\n",
    "curve = weighted_portfolio(prices, w)\n",
    "qs.reports.full(curve)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
'''

# Convert to Python dict
data = json.loads(json_string)

# Now use it as Python
print(data['name'])  # Output: Altaf
print(data['skills'][0])  # Output: Python
