import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from model import download_data, create_features, train_model, predict_next_day

st.set_page_config(
    page_title="Stock Market Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center'>📈 Stock Market Price Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center'>Machine Learning Based Stock Price Prediction using Random Forest</p>",
    unsafe_allow_html=True
)

st.sidebar.header("⚙️ Project Settings")
ticker = st.sidebar.text_input("Stock Ticker", "RELIANCE.NS").upper().strip()
period = st.sidebar.selectbox("Historical Data", ["1y", "2y", "5y", "10y", "max"], index=2)
train_button = st.sidebar.button("🚀 Train Model", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "Example Indian tickers:\n\n"
    "RELIANCE.NS\nTCS.NS\nINFY.NS\nHDFCBANK.NS\n"
    "ICICIBANK.NS\nSBIN.NS\nITC.NS"
)

if train_button:
    with st.spinner("Downloading data and training model..."):
        try:
            raw_data = download_data(ticker, period)
            feature_data = create_features(raw_data)
            results = train_model(feature_data)
            next_prediction = predict_next_day(
                results["model"], feature_data, results["features"]
            )

            st.session_state.data = raw_data
            st.session_state.feature_data = feature_data
            st.session_state.results = results
            st.session_state.next_prediction = next_prediction
            st.session_state.ticker = ticker
            st.session_state.trained = True
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.trained = False

if st.session_state.get("trained", False):
    data = st.session_state.data
    feature_data = st.session_state.feature_data
    results = st.session_state.results
    next_prediction = st.session_state.next_prediction

    latest_close = float(data["Close"].iloc[-1])
    previous_close = float(data["Close"].iloc[-2])
    change = latest_close - previous_close
    change_pct = (change / previous_close) * 100

    st.subheader(f"📊 {st.session_state.ticker} Stock Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close", f"₹{latest_close:,.2f}")
    c2.metric("Daily Change", f"₹{change:,.2f}")
    c3.metric("Daily Change %", f"{change_pct:.2f}%")
    c4.metric("Next Day Prediction", f"₹{next_prediction:,.2f}")

    st.subheader("📈 Historical Stock Price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["Date"], y=data["Close"],
        mode="lines", name="Closing Price"
    ))
    fig.update_layout(
        title=f"{st.session_state.ticker} Closing Price",
        xaxis_title="Date", yaxis_title="Price",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📉 Moving Average Analysis")
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=feature_data["Date"], y=feature_data["Close"],
        name="Closing Price"
    ))
    fig_ma.add_trace(go.Scatter(
        x=feature_data["Date"], y=feature_data["MA_20"],
        name="20-Day Moving Average"
    ))
    fig_ma.add_trace(go.Scatter(
        x=feature_data["Date"], y=feature_data["MA_50"],
        name="50-Day Moving Average"
    ))
    fig_ma.update_layout(xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_ma, use_container_width=True)

    st.subheader("🤖 Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{results['MAE']:.2f}")
    c2.metric("RMSE", f"{results['RMSE']:.2f}")
    c3.metric("R² Score", f"{results['R2']:.4f}")

    st.subheader("🔍 Actual vs Predicted Closing Price")
    comparison = pd.DataFrame({
        "Date": feature_data["Date"].iloc[-len(results["y_test"]):].values,
        "Actual Price": results["y_test"].values,
        "Predicted Price": results["predictions"]
    })

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=comparison["Date"], y=comparison["Actual Price"],
        mode="lines", name="Actual Price"
    ))
    fig_pred.add_trace(go.Scatter(
        x=comparison["Date"], y=comparison["Predicted Price"],
        mode="lines", name="Predicted Price"
    ))
    fig_pred.update_layout(
        xaxis_title="Date", yaxis_title="Closing Price",
        hovermode="x unified"
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    st.subheader("⭐ Feature Importance")
    importance = pd.DataFrame({
        "Feature": results["features"],
        "Importance": results["model"].feature_importances_
    }).sort_values("Importance", ascending=True)

    fig_imp = go.Figure(go.Bar(
        x=importance["Importance"],
        y=importance["Feature"],
        orientation="h"
    ))
    fig_imp.update_layout(xaxis_title="Importance", yaxis_title="Feature")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("📋 Recent Historical Data")
    st.dataframe(data.tail(20), use_container_width=True)

    st.subheader("🔮 Next Trading Day Prediction")
    st.success(f"Predicted Closing Price: ₹{next_prediction:,.2f}")
    st.warning(
        "Educational/research use only. Stock prices are affected by many "
        "factors, and this prediction is not financial advice or a guarantee."
    )
else:
    st.info("Enter a stock ticker in the sidebar and click **Train Model**.")
    st.subheader("📌 About This Project")
    st.write(
        "This project collects historical stock data, creates technical and "
        "lag features, trains a Random Forest regression model, evaluates it "
        "with MAE/RMSE/R², and predicts the next trading day's closing price."
    )
