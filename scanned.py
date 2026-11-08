import os
import sys
import json
import numpy as np
import pandas as pd
import requests
import yfinance as yf

SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
           "^GSPC", "^IXIC", "^DJI", "^FTSE",
           "GC=F", "CL=F", "BTC-USD", "ETH-USD"]
CURRENCIES = {"EURUSD=X": ["EUR", "USD"], "GBPUSD=X": ["GBP", "USD"],
              "USDJPY=X": ["USD", "JPY"], "AUDUSD=X": ["AUD", "USD"]}
CONTRACTS = {"EURUSD=X": (100000, "lots"), "GBPUSD=X": (100000, "lots"),
             "USDJPY=X": (100000, "lots"), "AUDUSD=X": (100000, "lots"),
             "GC=F": (100, "contracts"), "CL=F": (1000, "contracts")}
CRYPTO = {"BTC-USD", "ETH-USD"}

STATE_FILE = "signals.json"
BALANCE = float(os.environ.get("ACCOUNT_BALANCE", "10000"))
RISK_PCT = float(os.environ.get("RISK_PCT", "1"))
MIN_SCORE = 5            # out of 6 factors
COOLDOWN_HOURS = 4
NEWS_WINDOW_MIN = 30
EXPIRE_HOURS = 72


# ---------- indicators ----------
def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + g / l)


def true_range(df):
    pc = df["Close"].shift(1)
    return pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                      (df["Low"] - pc).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df, n=14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bb_width(s, n=20, k=2):
    m = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return (2 * k * sd) / m


# ---------- helpers ----------
def tg(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[no telegram credentials] " + text)
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": text}, timeout=15)
        return bool(r.ok and r.json().get("ok"))
    except Exception as e:
        print("telegram error:", e)
        return False


def load_state():
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
    except Exception:
        s = {}
    s.setdefault("open", [])
    s.setdefault("closed", [])
    s.setdefault("last_alert", {})
    return s


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=1)


def fetch_news():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
        return r.json()
    except Exception as e:
        print("news fetch failed (filter off this run):", e)
        return None


def news_block(symbol, events, now):
    if not events:
        return None
    cur = CURRENCIES.get(symbol, ["USD"])
    for e in events:
        try:
            if e.get("impact") != "High" or e.get("country") not in cur:
                continue
            t = pd.Timestamp(e["date"]).tz_convert("UTC")
            if abs((t - now).total_seconds()) <= NEWS_WINDOW_MIN * 60:
                return e.get("title", "high-impact news")
        except Exception:
            continue
    return None


def position_size(symbol, price, risk_amount, dist):
    if dist <= 0:
        return 0.0, "units"
    units = risk_amount / dist
    if symbol == "USDJPY=X":
        units = risk_amount * price / dist
    size, label = CONTRACTS.get(symbol, (1, "units"))
    return units / size, label


# ---------- analysis ----------
def analyze(symbol, daily, hourly, intraday, now):
    closed = intraday[intraday.index + pd.Timedelta(minutes=15) <= now].dropna(subset=["Close"])
    if len(closed) < 40 or len(daily) < 60 or len(hourly) < 25:
        return None
    # market closed / stale data
    if now - (closed.index[-1] + pd.Timedelta(minutes=15)) > pd.Timedelta(minutes=45):
        return None
    last_date = closed.index[-1].date()
    dd = daily[daily.index.date < last_date].dropna(subset=["Close"])
    if len(dd) < 60:
        return None
    prev = dd.iloc[-1]
    pdh, pdl = prev["High"], prev["Low"]
    price = closed["Close"].iloc[-1]
    atr_d = atr(dd).iloc[-1]
    adx_d = adx(dd).iloc[-1]
    if pd.isna(atr_d) or pd.isna(adx_d):
        return None
    daily_bull = dd["Close"].iloc[-1] > dd["Close"].rolling(50).mean().iloc[-1]

    hh = hourly[hourly.index + pd.Timedelta(minutes=60) <= now].dropna(subset=["Close"])
    if len(hh) < 25:
        return None
    hour_bull = hh["Close"].iloc[-1] > hh["Close"].rolling(20).mean().iloc[-1]
    direction = "LONG" if hour_bull else "SHORT"

    r = rsi(closed["Close"]).iloc[-1]
    w = bb_width(closed["Close"])
    squeeze = bool(w.iloc[-1] < w.rolling(20).mean().iloc[-1] * 0.85)

    last = closed.iloc[-1]
    recent = closed.tail(8)
    if direction == "LONG":
        confirmed = (recent["Low"].min() < pdl and last["Close"] > pdl and last["Close"] > last["Open"])
        extreme = recent["Low"].min()
        rsi_ok = (not pd.isna(r)) and 40 <= r <= 70
    else:
        confirmed = (recent["High"].max() > pdh and last["Close"] < pdh and last["Close"] < last["Open"])
        extreme = recent["High"].max()
        rsi_ok = (not pd.isna(r)) and 30 <= r <= 60

    session_ok = symbol in CRYPTO or 7 <= now.hour < 21
    factors = [
        ("Confirmed sweep", bool(confirmed)),
        ("RSI supports", bool(rsi_ok)),
        ("Volatility squeeze", squeeze),
        ("Daily trend agrees", bool(daily_bull == hour_bull)),
        ("ADX > 20 (trending)", bool(adx_d > 20)),
        ("Active session", bool(session_ok)),
    ]
    score = sum(p for _, p in factors)

    sign = 1 if direction == "LONG" else -1
    stop = extreme - sign * 0.1 * atr_d
    dist = (price - stop) * sign
    target = price + sign * 2 * dist
    if dist <= 0 or dist > 1.5 * atr_d:
        return {"ok": False, "reason": "stop too wide or invalid"}
    return {"ok": bool(confirmed and score >= MIN_SCORE), "symbol": symbol,
            "dir": direction, "price": float(price), "stop": float(stop),
            "target": float(target), "dist": float(dist), "score": int(score),
            "total": len(factors), "factors": factors,
            "time": (closed.index[-1] + pd.Timedelta(minutes=15)).tz_convert("UTC").isoformat()}


def resolve(sig, intraday, now):
    """Return (result, R) if stop/target hit or expired, else None."""
    t0 = pd.Timestamp(sig["time"])
    sign = 1 if sig["dir"] == "LONG" else -1
    c = intraday[intraday.index >= t0]
    for _, row in c.iterrows():
        hit_stop = row["Low"] <= sig["stop"] if sign == 1 else row["High"] >= sig["stop"]
        hit_tgt = row["High"] >= sig["target"] if sign == 1 else row["Low"] <= sig["target"]
        if hit_stop:          # if both in one candle, assume the worse case
            return "LOSS", -1.0
        if hit_tgt:
            return "WIN", 2.0
    if (now - t0) > pd.Timedelta(hours=EXPIRE_HOURS):
        px = intraday["Close"].iloc[-1] if len(intraday) else sig["price"]
        return "EXPIRED", round(sign * (px - sig["price"]) / sig["dist"], 2)
    return None


def record_line(state):
    c = state["closed"]
    w = sum(1 for x in c if x["result"] == "WIN")
    l = sum(1 for x in c if x["result"] == "LOSS")
    tot = round(sum(x["R"] for x in c), 1)
    return f"Record: {w}W {l}L, net {tot:+}R ({len(c)} closed)"


def main():
    if "--test" in sys.argv:
        ok = tg("✅ Scanner test message. Connection works.")
        print("sent" if ok else "FAILED")
        return
    now = pd.Timestamp.now(tz="UTC")
    state = load_state()
    cache = {}

    def get(sym):
        if sym not in cache:
            t = yf.Ticker(sym)
            cache[sym] = (t.history(period="1y", interval="1d"),
                          t.history(period="1mo", interval="60m"),
                          t.history(period="5d", interval="15m"))
        return cache[sym]

    # 1) follow-ups on open signals
    still_open = []
    for sig in state["open"]:
        try:
            res = resolve(sig, get(sig["symbol"])[2], now)
        except Exception as e:
            print("follow-up error", sig["symbol"], e)
            res = None
        if res:
            result, R = res
            sig.update({"result": result, "R": R, "closed_at": now.isoformat()})
            state["closed"].append(sig)
            icon = {"WIN": "🎯", "LOSS": "🛑", "EXPIRED": "⏱"}[result]
            tg(f"{icon} {sig['symbol']} {sig['dir']} {result} ({R:+}R)\n{record_line(state)}")
        else:
            still_open.append(sig)
    state["open"] = still_open

    # 2) scan for new setups
    events = fetch_news()
    for sym in SYMBOLS:
        try:
            d, h, i = get(sym)
            a = analyze(sym, d, h, i, now)
        except Exception as e:
            print("scan error", sym, e)
            continue
        if not a or not a.get("ok"):
            continue
        if any(s["symbol"] == sym for s in state["open"]):
            continue
        key = f"{sym}|{a['dir']}"
        last = state["last_alert"].get(key)
        if last and (now - pd.Timestamp(last)) < pd.Timedelta(hours=COOLDOWN_HOURS):
            continue
        blocked = news_block(sym, events, now)
        if blocked:
            print(f"{sym} skipped, news: {blocked}")
            continue
        risk_amt = BALANCE * RISK_PCT / 100
        size, label = position_size(sym, a["price"], risk_amt, a["dist"])
        passed = ", ".join(n for n, p in a["factors"] if p)
        msg = (f"🚀 {a['dir']} {sym}\nEntry ~ {a['price']:.4f}\nStop: {a['stop']:.4f}\n"
               f"Target (2R): {a['target']:.4f}\nSize: {size:.2f} {label} (risk ${risk_amt:.0f})\n"
               f"Score {a['score']}/{a['total']}: {passed}\nBased on closed 15m candle. Not financial advice.")
        if tg(msg):
            state["last_alert"][key] = now.isoformat()
            state["open"].append({k: a[k] for k in ("symbol", "dir", "price", "stop", "target", "dist", "score", "time")})
    save_state(state)


if __name__ == "__main__":
    main()
