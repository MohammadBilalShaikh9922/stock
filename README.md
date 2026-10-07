# 📈 Stock Market Price Prediction Using Machine Learning

A college-friendly machine learning project that predicts the next trading day's closing price using historical stock market data.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- yfinance
- Plotly
- Streamlit

## ML Algorithm
Random Forest Regression.

## Features
- OHLC prices
- Trading volume
- 5/10/20/50 day moving averages
- Price change
- Volatility
- Lagged closing prices
- Lagged volume
- RSI

## Evaluation
- MAE
- RMSE
- R² Score

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Default ticker: `RELIANCE.NS`

Other examples:
- `TCS.NS`
- `INFY.NS`
- `HDFCBANK.NS`
- `ICICIBANK.NS`
- `SBIN.NS`
- `ITC.NS`

## Deployment
This is a Streamlit application. It can be deployed on Streamlit Community Cloud by connecting a GitHub repository and selecting `app.py` as the main file.

## Disclaimer
This is an educational/research project. Predictions are not guaranteed and should not be treated as financial advice.
