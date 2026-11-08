import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

st.set_page_config(page_title="Multi-Factor Trading Dashboard", layout="wide")
st.title("⚡ Multi-Factor Trading Dashboard")
st.caption("Educational tool, not financial advice. Free Yahoo data can be ~15 min delayed.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("1. Market Selection")
asset_category = st.sidebar.selectbox("Asset Class", ["Forex Majors", "Global Indices", "Commodities & Crypto"])
if asset_category == "Forex Majors":
    symbol_input = st.sidebar.selectbox("Select Asset", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"])
elif asset_category == "Global Indices":
    symbol_input = st.sidebar.selectbox("Select Index", ["^GSPC", "^IXIC", "^DJI", "^FTSE"])
else:
    symbol_input = st.sidebar.selectbox("Select Asset", ["GC=F", "CL=F", "BTC-USD", "ETH-USD"])

st.sidebar.divider()
st.sidebar.header("2. Risk & Position Sizing")
account_balance = st.sidebar.number_input("Account Balance ($)", min_value=100.0, value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

st.sidebar.divider()
st.sidebar.header("3. Telegram Alerts")
telegram_token = st.sidebar.text_input("Bot Token", type="password")
telegram_chat_id = st.sidebar.text_input("Chat ID")
send_alert_toggle = st.sidebar.checkbox("Enable Automated Alerts", value=False)

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()

# ---------------- INDICATORS ----------------
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, window=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

def calculate_bollinger_bands(data, window=20, num_std=2):
    mean = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = mean + std * num_std
    lower = mean - std * num_std
    width = (upper - lower) / mean
    return upper, lower, width

# ---------------- SCORING ----------------
def evaluate_factors(direction, sweep, rsi, squeeze, trend_agree, session_ok=None):
    out = []
    ok = sweep == direction
    out.append(("Liquidity Sweep", ok,
                f"{direction} sweep of previous day level." if ok else "No sweep matching the bias."))
    if session_ok is not None:
        out.append(("Session", bool(session_ok),
                    "High-liquidity window active." if session_ok else "Low-liquidity window."))
    if pd.isna(rsi):
        rsi_ok = False
    elif direction == "LONG":
        rsi_ok = 40 <= rsi <= 70
    else:
        rsi_ok = 30 <= rsi <= 60
    out.append(("Momentum (RSI)", rsi_ok,
                f"RSI {round(rsi, 1)} supports {direction}." if rsi_ok else f"RSI {None if pd.isna(rsi) else round(rsi, 1)} does not support {direction}."))
    out.append(("Volatility Squeeze", bool(squeeze),
                "Bollinger Bands compressed." if squeeze else "Bandwidth normal."))
    out.append(("Daily Trend Agrees", bool(trend_agree),
                f"Daily trend agrees with {direction} bias." if trend_agree else "Daily trend disagrees with the 1H bias."))
    return out

# ---------------- POSITION SIZING ----------------
CONTRACTS = {
    "EURUSD=X": (100000, "lots"), "GBPUSD=X": (100000, "lots"),
    "USDJPY=X": (100000, "lots"), "AUDUSD=X": (100000, "lots"),
    "GC=F": (100, "contracts (100 oz)"), "CL=F": (1000, "contracts (1000 bbl)"),
}

def size_position(symbol, price, risk_amount, stop_distance):
    if stop_distance <= 0:
        return 0, 1, "units"
    units = risk_amount / stop_distance
    if symbol == "USDJPY=X":  # P&L is in JPY, convert to USD
        units = risk_amount * price / stop_distance
    size, label = CONTRACTS.get(symbol, (1, "units"))
    return units, size, label

# ---------------- TELEGRAM ----------------
def send_telegram_alert(token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
        return r.json().get("ok", False)
    except Exception:
        return False

# ---------------- DATA ----------------
@st.cache_data(ttl=300)
def load_data(symbol):
    t = yf.Ticker(symbol)
    daily = t.history(period="2y", interval="1d").dropna(subset=["Close"])
    hourly = t.history(period="1mo", interval="60m").dropna(subset=["Close"])
    intraday = t.history(period="5d", interval="15m").dropna(subset=["Close"])
    return daily, hourly, intraday

# ---------------- BACKTEST (daily, simplified) ----------------
def run_backtest(df):
    d = df.copy()
    d["RSI"] = calculate_rsi(d["Close"])
    _, _, w = calculate_bollinger_bands(d["Close"])
    d["Squeeze"] = w < w.rolling(20).mean() * 0.85
    d["ATR"] = calculate_atr(d)
    d["SMA20"] = d["Close"].rolling(20).mean()
    d["SMA50"] = d["Close"].rolling(50).mean()
    trades = []
    i = 60
    n = len(d)
    while i < n - 1:
        row, prev = d.iloc[i], d.iloc[i - 1]
        if pd.isna(row["ATR"]) or pd.isna(row["SMA50"]):
            i += 1
            continue
        direction = "LONG" if row["Close"] > row["SMA20"] else "SHORT"
        sweep = None
        if row["Low"] < prev["Low"] and row["Close"] > prev["Low"]:
            sweep = "LONG"
        elif row["High"] > prev["High"] and row["Close"] < prev["High"]:
            sweep = "SHORT"
        agree = row["SMA20"] > row["SMA50"] if direction == "LONG" else row["SMA20"] < row["SMA50"]
        f = evaluate_factors(direction, sweep, row["RSI"], row["Squeeze"], agree, None)
        score = sum(p for _, p, _ in f)
        if score >= len(f) - 1:
            entry, a = row["Close"], row["ATR"]
            sign = 1 if direction == "LONG" else -1
            stop = entry - sign * 1.5 * a
            target = entry + sign * 3.0 * a
            r_result, j = None, i + 1
            for j in range(i + 1, min(i + 11, n)):
                hi, lo = d["High"].iloc[j], d["Low"].iloc[j]
                hit_stop = lo <= stop if sign == 1 else hi >= stop
                hit_target = hi >= target if sign == 1 else lo <= target
                if hit_stop:
                    r_result = -1.0
                    break
                if hit_target:
                    r_result = 2.0
                    break
            if r_result is None:
                r_result = sign * (d["Close"].iloc[j] - entry) / (1.5 * a)
            trades.append({"Date": d.index[i].date(), "Dir": direction,
                           "Entry": round(entry, 4), "R": round(r_result, 2)})
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(trades)

# ---------------- MAIN ----------------
try:
    with st.spinner(f"Loading {symbol_input}..."):
        df_daily, df_hourly, df_intraday = load_data(symbol_input)

    if len(df_daily) < 60 or len(df_hourly) < 20 or df_intraday.empty:
        st.error("Not enough data returned. Try another ticker.")
        st.stop()

    # Key levels
    prev_day = df_daily.iloc[-2]
    pdh, pdl = prev_day["High"], prev_day["Low"]
    price = df_intraday["Close"].iloc[-1]

    # Data freshness
    last_ts = df_intraday.index[-1].to_pydatetime()
    age_min = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60
    if age_min > 30:
        st.warning(f"Last candle is {int(age_min)} min old. Market may be closed, or data is delayed.")
    else:
        st.caption(f"Last candle: {last_ts.strftime('%Y-%m-%d %H:%M %Z')} ({int(age_min)} min ago)")

    # Indicators
    atr = calculate_atr(df_daily).iloc[-1]
    df_intraday["RSI"] = calculate_rsi(df_intraday["Close"])
    rsi = df_intraday["RSI"].iloc[-1]
    upper, lower, bb_width = calculate_bollinger_bands(df_intraday["Close"])
    squeeze = bb_width.iloc[-1] < bb_width.rolling(20).mean().iloc[-1] * 0.85

    # Direction and trend
    hourly_sma = df_hourly["Close"].rolling(20).mean().iloc[-1]
    htf_bullish = df_hourly["Close"].iloc[-1] > hourly_sma
    direction = "LONG" if htf_bullish else "SHORT"
    daily_sma = df_daily["Close"].rolling(20).mean().iloc[-1]
    daily_bullish = df_daily["Close"].iloc[-1] > daily_sma
    trend_agree = daily_bullish == htf_bullish

    # Liquidity sweep (last 8 x 15m candles)
    recent_high = df_intraday["High"].tail(8).max()
    recent_low = df_intraday["Low"].tail(8).min()
    sweep = None
    if recent_low < pdl and price > pdl:
        sweep = "LONG"
    elif recent_high > pdh and price < pdh:
        sweep = "SHORT"

    # Session (UTC)
    hour = datetime.now(timezone.utc).hour
    if 7 <= hour < 12:
        session, session_ok = "London", True
    elif 12 <= hour < 16:
        session, session_ok = "London/NY Overlap", True
    elif 16 <= hour < 21:
        session, session_ok = "New York", True
    else:
        session, session_ok = "Asian / Low Liquidity", False

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", round(price, 4))
    c2.metric("PDH / PDL", f"{round(pdh, 2)} / {round(pdl, 2)}")
    c3.metric("Daily ATR", round(atr, 4))
    c4.metric("RSI (15m)", "N/A" if pd.isna(rsi) else round(rsi, 1))
    c5.metric("1H Bias", "BULLISH 📈" if htf_bullish else "BEARISH 📉")
    st.caption(f"Session: {session}")
    st.divider()

    # Scorecard
    st.subheader("🎯 Confluence Scorecard")
    factors = evaluate_factors(direction, sweep, rsi, squeeze, trend_agree, session_ok)
    score = sum(p for _, p, _ in factors)
    total = len(factors)
    threshold = total - 1
    for name, passed, text in factors:
        (st.success if passed else st.info)(f"{'✅' if passed else '⚪'} **{name}:** {text}")
    st.markdown(f"### Score: {score} / {total} ({direction} bias)")

    # Trade plan
    st.subheader("🧮 Trade Plan & Position Size")
    risk_amount = account_balance * risk_pct / 100.0
    stop_distance = atr * 1.5
    sign = 1 if direction == "LONG" else -1
    stop_price = price - sign * stop_distance
    target_price = price + sign * stop_distance * 2
    units, size, label = size_position(symbol_input, price, risk_amount, stop_distance)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Risk ($)", f"${round(risk_amount, 2)}")
    p2.metric("Stop", round(stop_price, 4))
    p3.metric("Target (2R)", round(target_price, 4))
    p4.metric("Size", f"{round(units / size, 2)} {label}")

    if score >= threshold:
        st.success(f"🟢 {direction} setup: {score}/{total} factors aligned.")
        if send_alert_toggle and telegram_token and telegram_chat_id:
            key = f"{symbol_input}|{direction}"
            last = st.session_state.get("last_alert")
            now = datetime.now(timezone.utc)
            if last is None or last["key"] != key or (now - last["time"]).total_seconds() > 3600:
                msg = (f"🚀 *{direction} SETUP*\n*Asset:* {symbol_input}\n*Price:* {round(price, 4)}\n"
                       f"*Score:* {score}/{total}\n*Stop:* {round(stop_price, 4)}\n"
                       f"*Target:* {round(target_price, 4)}\n*Size:* {round(units / size, 2)} {label}")
                if send_telegram_alert(telegram_token, telegram_chat_id, msg):
                    st.session_state["last_alert"] = {"key": key, "time": now}
                    st.toast("📲 Telegram alert sent")
    else:
        st.warning("🟡 No trade: confluence threshold not reached.")
    st.divider()

    # Chart
    st.subheader("📊 Price Action")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df_intraday.index, open=df_intraday["Open"], high=df_intraday["High"],
                                 low=df_intraday["Low"], close=df_intraday["Close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_intraday.index, y=upper, line=dict(width=1), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_intraday.index, y=lower, line=dict(width=1), name="BB Lower"), row=1, col=1)
    fig.add_hline(y=pdh, line_dash="dot", annotation_text="PDH", row=1, col=1)
    fig.add_hline(y=pdl, line_dash="dot", annotation_text="PDL", row=1, col=1)
    fig.add_trace(go.Scatter(x=df_intraday.index, y=df_intraday["RSI"], name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, showlegend=False,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig)
    st.divider()

    # Backtest
    st.subheader("🧪 Backtest (daily, last 2 years)")
    st.caption("Simplified version of the same logic on daily candles (no session factor, trend = SMA20 vs SMA50). "
               "Stop 1.5 ATR, target 3 ATR, max 10 days. No costs or slippage. Past results do not predict future ones.")
    bt = run_backtest(df_daily)
    if bt.empty:
        st.info("No trades triggered in this period.")
    else:
        wins = (bt["R"] > 0).mean() * 100
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Trades", len(bt))
        b2.metric("Win rate", f"{round(wins, 1)}%")
        b3.metric("Avg R", round(bt["R"].mean(), 2))
        b4.metric("Total R", round(bt["R"].sum(), 1))
        st.line_chart(bt["R"].cumsum().reset_index(drop=True))
        st.dataframe(bt.tail(10), hide_index=True)

except Exception as e:
    st.error(f"Error: {e}")
