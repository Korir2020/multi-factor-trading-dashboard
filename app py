import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone

# Page Configuration
st.set_page_config(page_title="Institutional Trading Intelligence Terminal", layout="wide")

st.title("⚡ Quantitative Multi-Factor Trading Intelligence Terminal")
st.markdown("Hedge-fund grade algorithmic confluence command center processing liquidity sweeps, volatility compression, multi-timeframe trends, and dynamic risk management.")

# --- SIDEBAR COMMAND CENTER ---
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
risk_pct = st.sidebar.slider("Risk Tolerance per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

st.sidebar.divider()
st.sidebar.header("3. Telegram Push Notifications")
telegram_token = st.sidebar.text_input("Bot Token", type="password", placeholder="12345678:ABCDefg...")
telegram_chat_id = st.sidebar.text_input("Chat ID", placeholder="123456789")
send_alert_toggle = st.sidebar.checkbox("Enable Automated Alerts", value=False)

refresh_data = st.sidebar.button("Execute Quantitative Analysis")

# --- QUANTITATIVE FUNCTIONS ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    bb_width = (upper_band - lower_band) / rolling_mean
    return upper_band, lower_band, bb_width

def send_telegram_alert(token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, json=payload)
        return response.json().get("ok", False)
    except Exception:
        return False

# --- DATA ENGINE ---
if refresh_data or symbol_input:
    with st.spinner(f"Ingesting institutional feeds and computing multi-timeframe matrices for {symbol_input}..."):
        try:
            ticker = yf.Ticker(symbol_input)
            df_daily = ticker.history(period="60d", interval="1d")
            df_hourly = ticker.history(period="10d", interval="60m")
            df_intraday = ticker.history(period="5d", interval="15m")
            
            if len(df_daily) < 20 or df_intraday.empty:
                st.error("Insufficient market depth returned. Please select an alternate ticker.")
            else:
                # 1. Key Institutional Levels (PDH / PDL / PDC)
                prev_day = df_daily.iloc[-2]
                pdh = prev_day['High']
                pdl = prev_day['Low']
                pdc = prev_day['Close']
                current_price = df_intraday['Close'].iloc[-1]
                
                # 2. Volatility & ATR
                high_low = df_daily['High'] - df_daily['Low']
                atr = high_low.rolling(window=14).mean().iloc[-1]
                
                # 3. Momentum & Bollinger Squeeze
                df_intraday['RSI'] = calculate_rsi(df_intraday['Close'], window=14)
                current_rsi = df_intraday['RSI'].iloc[-1]
                
                upper, lower, bb_width = calculate_bollinger_bands(df_intraday['Close'])
                current_bb_width = bb_width.iloc[-1]
                avg_bb_width = bb_width.rolling(window=20).mean().iloc[-1]
                is_squeeze = current_bb_width < avg_bb_width * 0.85

                # 4. Multi-Timeframe Trend Filter (1-Hour SMA vs Daily SMA)
                hourly_sma = df_hourly['Close'].rolling(window=20).mean().iloc[-1]
                hourly_price = df_hourly['Close'].iloc[-1]
                htf_bullish = hourly_price > hourly_sma

                # 5. Session & Time Window (UTC)
                now_utc = datetime.now(timezone.utc)
                current_hour = now_utc.hour
                
                if 7 <= current_hour < 12:
                    session = "London Session (Active Liquidity)"
                    session_weight = 1.2
                elif 12 <= current_hour < 16:
                    session = "London / NY Overlap (Peak Volatility)"
                    session_weight = 2.0
                elif 16 <= current_hour < 21:
                    session = "New York Session (Steady Flow)"
                    session_weight = 1.0
                else:
                    session = "Asian Session / Low Liquidity"
                    session_weight = 0.5

                # --- EXECUTIVE METRICS ROW ---
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Current Price", round(current_price, 4))
                col2.metric("PDH / PDL", f"{round(pdh, 2)} / {round(pdl, 2)}")
                col3.metric("ATR Volatility", round(atr, 4))
                col4.metric("14-Period RSI", round(current_rsi, 1) if not np.isnan(current_rsi) else "N/A")
                col5.metric("HTF Bias", "BULLISH 📈" if htf_bullish else "BEARISH 📉")

                st.divider()

                # --- POSITION SIZING & RISK CALCULATOR MODULE ---
                st.subheader("🧮 Dynamic Risk & Position Sizing Engine")
                risk_amount = account_balance * (risk_pct / 100.0)
                # Recommended stop loss distance tied to ATR multiplier (e.g., 1.5 ATR)
                suggested_stop_distance = atr * 1.5
                position_units = risk_amount / suggested_stop_distance if suggested_stop_distance > 0 else 0

                r_col1, r_col2, r_col3 = st.columns(3)
                r_col1.metric("Max Capital at Risk", f"${round(risk_amount, 2)}")
                r_col2.metric("Suggested Stop Distance", round(suggested_stop_distance, 4))
                r_col3.metric("Recommended Position Size", f"{round(position_units, 2)} Units/Lots")

                st.divider()

                # --- 5-FACTOR CONFLUENCE MATRIX ---
                st.subheader("🎯 Algorithmic Confluence Scorecard (5-Factor Model)")
                
                score = 0
                signals = []

                # Factor 1: Liquidity Sweep
                if current_price >= pdh:
                    signals.append(("🚨 LIQUIDITY SWEEP: Price testing/exceeding Previous Day High.", "warning"))
                    score += 1
                elif current_price <= pdl:
                    signals.append(("🚨 LIQUIDITY SWEEP: Price testing/exceeding Previous Day Low.", "warning"))
                    score += 1
                else:
                    signals.append(("⚪ Price trading inside Previous Day range.", "info"))

                # Factor 2: Session Alignment
                if session_weight >= 1.2:
                    signals.append((f"✅ SESSION ALIGNMENT: High-volume window active ({session}).", "success"))
                    score += 1
                else:
                    signals.append((f"⚠️ LOW LIQUIDITY: Current window is {session}.", "error"))

                # Factor 3: Momentum Filter (RSI)
                if not np.isnan(current_rsi):
                    if current_rsi > 70:
                        signals.append((f"⚠️ OVERBOUGHT: RSI at {round(current_rsi, 1)} (>70). Exhaustion risk.", "warning"))
                    elif current_rsi < 30:
                        signals.append((f"⚠️ OVERSOLD: RSI at {round(current_rsi, 1)} (<30). Exhaustion risk.", "warning"))
                    else:
                        signals.append((f"✅ HEALTHY MOMENTUM: RSI balanced at {round(current_rsi, 1)}.", "success"))
                        score += 1

                # Factor 4: Volatility Compression (Bollinger Squeeze)
                if is_squeeze:
                    signals.append(("🔥 VOLATILITY SQUEEZE DETECTED: Bollinger Bands compressed. Expect explosive expansion.", "warning"))
                    score += 1
                else:
                    signals.append(("⚪ Bollinger bandwidth normal.", "info"))

                # Factor 5: Multi-Timeframe Trend Alignment
                if htf_bullish:
                    signals.append(("✅ MACRO ALIGNMENT: 1-Hour trend confirms bullish momentum.", "success"))
                    score += 1
                else:
                    signals.append(("✅ MACRO ALIGNMENT: 1-Hour trend confirms bearish momentum.", "success"))
                    score += 1

                for text, status in signals:
                    if status == "warning": st.warning(text)
                    elif status == "error": st.error(text)
                    elif status == "success": st.success(text)
                    else: st.info(text)

                st.markdown(### f"Total Confluence Rating: **{score} / 5 Factors Aligned**")

                if score >= 4:
                    st.success("🟢 STATUS: Elite multi-factor setup confirmed. Execution parameters locked.")
                    if send_alert_toggle and telegram_token and telegram_chat_id:
                        msg = f"🚀 *ELITE TRADING ALERT* 🚀\n\n*Asset:* {symbol_input}\n*Price:* {round(current_price, 4)}\n*Score:* {score}/5 Factors\n*Position Units:* {round(position_units, 2)}"
                        if send_telegram_alert(telegram_token, telegram_chat_id, msg):
                            st.toast("📲 Telegram push notification dispatched!", icon="⚡")
                else:
                    st.warning("🟡 STATUS: Confluence threshold unreached. Stand by.")

                st.divider()

                # --- VISUAL CHARTING SUITE ---
                st.subheader("📊 Intraday Price Action & Technical Oscillators")
                st.line_chart(df_intraday['Close'])
                st.markdown("**14-Period RSI Oscillator**")
                st.line_chart(df_intraday['RSI'])

        except Exception as e:
            st.error(f"Critical execution error: {e}")
