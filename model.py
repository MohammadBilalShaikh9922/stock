import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def download_data(ticker, period="5y"):
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(f"No data found for {ticker}. Please check the stock ticker.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    data = data[required_columns]
    return data


def create_features(data):
    df = data.copy()

    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    df["Price_Change"] = df["Close"].pct_change()
    df["Volatility_5"] = df["Price_Change"].rolling(5).std()

    df["Close_Lag_1"] = df["Close"].shift(1)
    df["Close_Lag_2"] = df["Close"].shift(2)
    df["Close_Lag_3"] = df["Close"].shift(3)
    df["Close_Lag_5"] = df["Close"].shift(5)
    df["Volume_Lag_1"] = df["Volume"].shift(1)

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Target: next trading day's closing price
    df["Target"] = df["Close"].shift(-1)

    return df.dropna().reset_index(drop=True)


def train_model(df):
    features = [
        "Open", "High", "Low", "Close", "Volume",
        "MA_5", "MA_10", "MA_20", "MA_50",
        "Price_Change", "Volatility_5",
        "Close_Lag_1", "Close_Lag_2", "Close_Lag_3",
        "Close_Lag_5", "Volume_Lag_1", "RSI"
    ]

    X = df[features]
    y = df["Target"]

    split_index = int(len(df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return {
        "model": model,
        "features": features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def predict_next_day(model, df, features):
    latest = df.iloc[-1]
    input_data = pd.DataFrame(
        [[latest[f] for f in features]],
        columns=features
    )
    return model.predict(input_data)[0]
