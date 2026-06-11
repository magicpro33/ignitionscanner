"""
IGNITION - Momentum Ignition Scanner
====================================
Detects the earliest minutes of a momentum move by combining:

  IGNITION signals (live, intraday):
    - Relative volume vs 20-day pace (RVOL)
    - Bar-level volume surge (last 3 bars vs session average)
    - Price velocity and acceleration (1-minute bars)
    - VWAP reclaim / cross
    - High-of-day breakout
    - RSI(14) thrust on 5-minute bars
    - MACD bullish cross on 5-minute bars

  FUEL signals (daily, "is this stock primed to move"):
    - Short percent of float (squeeze fuel)
    - Insider net buying, last 90 days
    - Fresh news flow with keyword sentiment
    - Float size (small float = explosive)
    - Distance to 52-week high

Each ticker gets an Ignition Score (0-100). When live confirmation
conditions all fire at once, the ticker is flagged IGNITING and logged
to the alert feed with a timestamp.

Run:  streamlit run ignition_scanner.py
Data: Yahoo Finance via yfinance (free, ~15 min delayed on some feeds).
"""

import time
import math
import re
import gzip
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# Page config and theme
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="IGNITION - Momentum Scanner",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0e14; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: 0.5px; }

.metric-mono, .stDataFrame, code { font-family: 'Space Mono', monospace !important; }

.ignite-banner {
    background: linear-gradient(90deg, #2a1205, #3d1a05);
    border: 1px solid #ff6b1a;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-family: 'Space Mono', monospace;
    color: #ffb380;
    animation: pulse 1.6s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255,107,26,0.45); }
    70% { box-shadow: 0 0 0 12px rgba(255,107,26,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,107,26,0); }
}
.alert-row {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: #d8e0ea;
    padding: 6px 10px;
    border-left: 3px solid #ff6b1a;
    background: #11161f;
    margin-bottom: 4px;
    border-radius: 4px;
}
.fuel-tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 6px;
    background: #16202e;
    border: 1px solid #2b3a4f;
    color: #9fb6d0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Universe presets
# ----------------------------------------------------------------------------
PRESETS = {
    "Nuclear / Uranium / SMR": "SMR,OKLO,NNE,CCJ,UEC,UUUU,EU,URG,LEU,DNN,NXE,BWXT,VST,CEG,TLN",
    "Quantum Computing": "IONQ,QBTS,RGTI,QUBT,ARQQ,LAES",
    "Energy / LNG / Midstream": "LNG,ET,EPD,WMB,KMI,TRGP,AR,EQT,FLNG",
    "AI / Semis": "NVDA,AMD,AVGO,MU,SMCI,VRT,ARM,TSM,MRVL,CRDO",
    "High Short Interest Movers": "GME,AMC,BYND,CVNA,UPST,SOUN,LUNR,ACHR,RKLB",
    "Mega-cap Liquid": "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA",
}

POSITIVE_WORDS = [
    "beat", "beats", "surge", "record", "upgrade", "upgraded", "raises",
    "contract", "award", "awarded", "partnership", "approval", "approved",
    "buyback", "acquisition", "acquire", "breakthrough", "expands", "wins",
    "guidance raised", "outperform", "buy rating", "patent", "milestone",
]
NEGATIVE_WORDS = [
    "miss", "misses", "downgrade", "downgraded", "cuts", "offering",
    "dilution", "lawsuit", "investigation", "recall", "halts", "delay",
    "bankruptcy", "warning", "sell rating", "underperform", "resigns",
]

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def clamp(x, lo=0.0, hi=100.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.0
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.0


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    return (tp * df["Volume"]).cumsum() / cum_vol


def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ----------------------------------------------------------------------------
# Data fetch (cached)
# ----------------------------------------------------------------------------
def alpaca_keys():
    """Read Alpaca keys from Streamlit secrets. Returns (key, secret) or None."""
    try:
        k = st.secrets.get("ALPACA_API_KEY", "")
        s = st.secrets.get("ALPACA_SECRET_KEY", "")
        if k and s:
            return k, s
    except Exception:
        pass
    return None


def _alpaca_bars(ticker: str, timeframe: str, start_iso: str, key: str, secret: str):
    """Fetch bars from Alpaca Market Data v2 (IEX feed works on free plans)."""
    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {
        "timeframe": timeframe,
        "start": start_iso,
        "limit": 10000,
        "adjustment": "raw",
        "feed": "iex",
        "sort": "asc",
    }
    rows = []
    while True:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rows.extend(data.get("bars") or [])
        token = data.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
    df = df.set_index("t").rename(columns={
        "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume",
    })[["Open", "High", "Low", "Close", "Volume"]]
    # Keep regular session only (09:30-16:00 ET) to match yfinance behavior
    df = df.between_time("09:30", "16:00")
    return df if len(df) else None


@st.cache_data(ttl=55, show_spinner=False)
def fetch_intraday(ticker: str):
    """1-minute bars for today plus 5-minute bars for ~5 days.
    Uses Alpaca real-time feed when keys are configured, else Yahoo."""
    keys = alpaca_keys()
    if keys:
        try:
            k, s = keys
            today = datetime.now(timezone.utc) - timedelta(hours=24)
            week = datetime.now(timezone.utc) - timedelta(days=7)
            m1 = _alpaca_bars(ticker, "1Min", today.strftime("%Y-%m-%dT%H:%M:%SZ"), k, s)
            m5 = _alpaca_bars(ticker, "5Min", week.strftime("%Y-%m-%dT%H:%M:%SZ"), k, s)
            if m1 is not None:
                # Trim m1 to the latest session only
                last_day = m1.index[-1].date()
                m1 = m1[m1.index.date == last_day]
                return m1, m5
        except Exception:
            pass  # fall through to Yahoo
    try:
        tk = yf.Ticker(ticker)
        m1 = tk.history(period="1d", interval="1m", prepost=False)
        m5 = tk.history(period="5d", interval="5m", prepost=False)
        m1 = flatten_cols(m1) if m1 is not None else None
        m5 = flatten_cols(m5) if m5 is not None else None
        return m1, m5
    except Exception:
        return None, None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily(ticker: str):
    try:
        d = yf.Ticker(ticker).history(period="3mo", interval="1d")
        return flatten_cols(d)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Nightly screener dump (magicpro33/stock pipeline)
# ----------------------------------------------------------------------------
SCREENER_URL_DEFAULT = (
    "https://raw.githubusercontent.com/magicpro33/stock/main/data/stock_data.json.gz"
)


def _strip_heavy(d: dict) -> dict:
    """object_hook: drop numeric/scalar arrays (OHLC history blobs) as each
    object is parsed, so the 100MB+ dump doesn't blow Streamlit Cloud memory.
    Lists of dicts (record collections) and dict values are preserved; any
    residual dict-valued fields are stripped later when rows are built."""
    out = {}
    for k, v in d.items():
        if isinstance(v, list) and v and not isinstance(v[0], dict):
            continue  # numeric/scalar array, e.g. price history
        out[k] = v
    return out


@st.cache_data(ttl=21600, show_spinner="Loading nightly screener dump...")
def load_screener_dump(url: str) -> pd.DataFrame:
    """Download and parse the nightly stock_data.json.gz into a slim,
    scalar-only DataFrame indexed by ticker."""
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    blob = resp.content
    if url.endswith(".gz") or blob[:2] == b"\x1f\x8b":
        blob = gzip.decompress(blob)
    data = json.loads(blob, object_hook=_strip_heavy)

    records = []
    if isinstance(data, dict):
        # {ticker: {fields}} or {"stocks": [...]} style
        inner = None
        for key in ("stocks", "data", "results", "records"):
            if key in data and isinstance(data.get(key), (list, dict)):
                inner = data[key]
                break
        src = inner if inner is not None else data
        if isinstance(src, dict):
            for tkr, fields in src.items():
                if isinstance(fields, dict):
                    row = {k: v for k, v in fields.items() if not isinstance(v, dict)}
                    row.setdefault("ticker", tkr)
                    records.append(row)
        elif isinstance(src, list):
            records = [
                {k: v for k, v in r.items() if not isinstance(v, dict)}
                for r in src if isinstance(r, dict)
            ]
    elif isinstance(data, list):
        records = [r for r in data if isinstance(r, dict)]

    df = pd.DataFrame(records)
    # Find the ticker column
    tcol = None
    for c in df.columns:
        if str(c).lower() in ("ticker", "symbol", "sym"):
            tcol = c
            break
    if tcol is None:
        raise ValueError("Could not find a ticker/symbol column in the dump.")
    df[tcol] = df[tcol].astype(str).str.upper().str.strip()
    df = df[df[tcol].str.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}")]
    df = df.set_index(tcol)
    return df


def _find_col(cols, *needles):
    """Fuzzy column lookup: first column whose lowercase name contains all
    needles. Lets this survive renames in the nightly pipeline."""
    for c in cols:
        name = str(c).lower().replace(" ", "_")
        if all(n in name for n in needles):
            return c
    return None


def prescore_screener(df: pd.DataFrame) -> pd.Series:
    """Rank the whole dump 0-100 using its own nightly signals. Scale-agnostic:
    numeric signals become percentile ranks, booleans become 0/100."""
    cols = list(df.columns)
    parts = []   # (weight, series 0-100)

    def pct(series):
        s = pd.to_numeric(series, errors="coerce")
        return s.rank(pct=True) * 100.0

    def boolean(series):
        s = series.map(lambda v: 1.0 if v in (True, 1, "1", "true", "True", "Y", "YES", "Yes") else 0.0)
        return s * 100.0

    c = _find_col(cols, "squeeze")
    if c is not None:
        parts.append((0.22, pct(df[c]) if pd.to_numeric(df[c], errors="coerce").notna().any() else boolean(df[c])))

    c = _find_col(cols, "score")
    if c is not None:
        parts.append((0.20, pct(df[c])))

    c = _find_col(cols, "rsi")
    if c is not None:
        r = pd.to_numeric(df[c], errors="coerce")
        # thrust zone 50-70 best; oversold/overbought worth less
        parts.append((0.14, r.map(lambda x: 100.0 if 50 <= x <= 70 else (60.0 if 40 <= x < 50 or 70 < x <= 80 else 20.0) if pd.notna(x) else 0.0)))

    c = _find_col(cols, "golden")
    if c is not None:
        parts.append((0.12, boolean(df[c])))

    c = _find_col(cols, "mfi")
    if c is not None:
        parts.append((0.10, boolean(df[c]) if not pd.to_numeric(df[c], errors="coerce").notna().any() else pct(df[c])))

    c = _find_col(cols, "obv")
    if c is not None:
        parts.append((0.08, pct(df[c])))

    c = _find_col(cols, "volume") or _find_col(cols, "vol")
    if c is not None:
        parts.append((0.08, pct(df[c])))

    c = _find_col(cols, "piotroski")
    if c is not None:
        parts.append((0.06, pct(df[c])))

    if not parts:
        # Nothing recognized: fall back to any numeric columns averaged
        num = df.select_dtypes(include=[np.number])
        return num.rank(pct=True).mean(axis=1).fillna(0) * 100.0

    total_w = sum(w for w, _ in parts)
    out = sum(w * s.fillna(0.0) for w, s in parts) / total_w
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fuel(ticker: str) -> dict:
    """Slow-moving 'is this primed' data: short interest, float,
    insider transactions, news, 52-week levels."""
    out = {
        "short_pct_float": None, "float_shares": None,
        "insider_net_buy_usd": 0.0, "insider_buys": 0, "insider_sells": 0,
        "news_count_48h": 0, "news_sentiment": 0, "latest_headline": "",
        "high_52w": None, "name": ticker,
    }
    try:
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        out["short_pct_float"] = info.get("shortPercentOfFloat")
        out["float_shares"] = info.get("floatShares")
        out["high_52w"] = info.get("fiftyTwoWeekHigh")
        out["name"] = info.get("shortName") or ticker

        # Insider transactions (last ~90 days)
        try:
            ins = tk.insider_transactions
            if ins is not None and len(ins) > 0:
                ins = ins.copy()
                date_col = None
                for c in ["Start Date", "startDate", "Date"]:
                    if c in ins.columns:
                        date_col = c
                        break
                if date_col is not None:
                    ins[date_col] = pd.to_datetime(ins[date_col], errors="coerce")
                    cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
                    ins = ins[ins[date_col] >= cutoff]
                text_col = "Text" if "Text" in ins.columns else None
                val_col = "Value" if "Value" in ins.columns else None
                for _, row in ins.iterrows():
                    txt = str(row.get(text_col, "")).lower() if text_col else ""
                    tr = str(row.get("Transaction", "")).lower()
                    val = row.get(val_col, 0) if val_col else 0
                    try:
                        val = float(val) if pd.notna(val) else 0.0
                    except Exception:
                        val = 0.0
                    is_buy = ("purchase" in txt) or ("buy" in tr) or ("purchase" in tr)
                    is_sell = ("sale" in txt) or ("sell" in tr) or ("sale" in tr)
                    if is_buy:
                        out["insider_buys"] += 1
                        out["insider_net_buy_usd"] += abs(val)
                    elif is_sell:
                        out["insider_sells"] += 1
                        out["insider_net_buy_usd"] -= abs(val)
        except Exception:
            pass

        # News flow, last 48 hours
        try:
            news = tk.news or []
            now = datetime.now(timezone.utc)
            sent = 0
            count = 0
            latest = ""
            for item in news:
                content = item.get("content", item)
                title = content.get("title", "") or ""
                pub = content.get("pubDate") or content.get("providerPublishTime")
                ts = None
                if isinstance(pub, (int, float)):
                    ts = datetime.fromtimestamp(pub, tz=timezone.utc)
                elif isinstance(pub, str):
                    try:
                        ts = pd.to_datetime(pub, utc=True).to_pydatetime()
                    except Exception:
                        ts = None
                if ts is not None and (now - ts) <= timedelta(hours=48):
                    count += 1
                    if not latest:
                        latest = title
                    low = title.lower()
                    sent += sum(1 for w in POSITIVE_WORDS if w in low)
                    sent -= sum(1 for w in NEGATIVE_WORDS if w in low)
            out["news_count_48h"] = count
            out["news_sentiment"] = sent
            out["latest_headline"] = latest
        except Exception:
            pass
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------
# Signal engine
# ----------------------------------------------------------------------------
def compute_signals(ticker: str) -> dict:
    """Returns a dict of raw signals, sub-scores and the composite score."""
    s = {
        "ticker": ticker, "price": None, "chg_pct": None,
        "rvol": 0.0, "surge": 0.0, "velocity": 0.0, "accel": 0.0,
        "above_vwap": False, "vwap_cross": False, "new_hod": False,
        "near_hod": False, "rsi5": None, "macd_cross": False,
        "macd_bull": False, "ignition_score": 0.0, "fuel_score": 0.0,
        "score": 0.0, "igniting": False, "reasons": [], "error": None,
    }

    m1, m5 = fetch_intraday(ticker)
    daily = fetch_daily(ticker)
    fuel = fetch_fuel(ticker)
    s["fuel"] = fuel

    if m1 is None or len(m1) < 6 or daily is None or len(daily) < 21:
        s["error"] = "no data"
        return s

    close = m1["Close"]
    s["price"] = float(close.iloc[-1])
    prev_close = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else float(daily["Close"].iloc[-1])
    s["chg_pct"] = (s["price"] / prev_close - 1.0) * 100.0 if prev_close else None

    # --- RVOL: cumulative volume today vs 20-day average, pace-adjusted ---
    avg_daily_vol = float(daily["Volume"].iloc[-21:-1].mean())
    cum_vol = float(m1["Volume"].sum())
    elapsed = max(len(m1), 1)  # minutes elapsed in session
    expected = avg_daily_vol * min(elapsed / 390.0, 1.0)
    s["rvol"] = cum_vol / expected if expected > 0 else 0.0

    # --- Bar-level surge: last 3 one-minute bars vs session average bar ---
    avg_bar = float(m1["Volume"].iloc[:-3].mean()) if len(m1) > 6 else float(m1["Volume"].mean())
    last3 = float(m1["Volume"].iloc[-3:].mean())
    s["surge"] = last3 / avg_bar if avg_bar > 0 else 0.0

    # --- Velocity and acceleration on 1m closes ---
    if len(close) >= 11:
        vel_now = (close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0
        vel_prev = (close.iloc[-6] / close.iloc[-11] - 1.0) * 100.0
        s["velocity"] = float(vel_now)
        s["accel"] = float(vel_now - vel_prev)
    else:
        s["velocity"] = (close.iloc[-1] / close.iloc[0] - 1.0) * 100.0

    # --- VWAP ---
    vw = vwap(m1)
    s["above_vwap"] = bool(close.iloc[-1] > vw.iloc[-1])
    if len(close) >= 6:
        was_below = bool((close.iloc[-6:-1] < vw.iloc[-6:-1]).any())
        s["vwap_cross"] = s["above_vwap"] and was_below

    # --- High of day ---
    hod = float(m1["High"].max())
    recent_high = float(m1["High"].iloc[-3:].max())
    s["new_hod"] = recent_high >= hod * 0.9999
    s["near_hod"] = s["price"] >= hod * 0.995

    # --- RSI / MACD on 5m bars ---
    if m5 is not None and len(m5) > 30:
        r = rsi(m5["Close"])
        s["rsi5"] = float(r.iloc[-1])
        macd_line, sig_line = macd(m5["Close"])
        s["macd_bull"] = bool(macd_line.iloc[-1] > sig_line.iloc[-1])
        if len(macd_line) >= 4:
            s["macd_cross"] = s["macd_bull"] and bool(
                (macd_line.iloc[-4:-1] <= sig_line.iloc[-4:-1]).any()
            )

    # ------------------------------------------------------------------
    # IGNITION sub-score (0-100)
    # ------------------------------------------------------------------
    rvol_sc = clamp(s["rvol"] / 5.0 * 100.0)
    surge_sc = clamp(s["surge"] / 4.0 * 100.0)
    vel_sc = clamp(s["velocity"] / 1.0 * 100.0) if s["velocity"] > 0 else 0.0
    acc_sc = clamp(s["accel"] / 0.7 * 100.0) if s["accel"] > 0 else 0.0
    if s["vwap_cross"]:
        vwap_sc = 100.0
    elif s["above_vwap"]:
        vwap_sc = 60.0
    else:
        vwap_sc = 0.0
    if s["new_hod"]:
        hod_sc = 100.0
    elif s["near_hod"]:
        hod_sc = 70.0
    else:
        hod_sc = 0.0
    if s["rsi5"] is None:
        rsi_sc = 0.0
    elif 55 <= s["rsi5"] <= 75:
        rsi_sc = 100.0
    elif s["rsi5"] > 75:
        rsi_sc = 70.0  # strong but stretched
    elif s["rsi5"] >= 50:
        rsi_sc = 60.0
    else:
        rsi_sc = clamp(s["rsi5"])
    if s["macd_cross"]:
        macd_sc = 100.0
    elif s["macd_bull"]:
        macd_sc = 60.0
    else:
        macd_sc = 0.0

    s["ignition_score"] = (
        0.25 * rvol_sc + 0.15 * surge_sc + 0.15 * vel_sc + 0.10 * acc_sc
        + 0.10 * vwap_sc + 0.10 * hod_sc + 0.075 * rsi_sc + 0.075 * macd_sc
    )

    # ------------------------------------------------------------------
    # FUEL sub-score (0-100)
    # ------------------------------------------------------------------
    spf = fuel.get("short_pct_float")
    short_sc = clamp((spf or 0) * 100.0 / 20.0 * 100.0) if spf else 0.0

    nb = fuel.get("insider_net_buy_usd", 0.0)
    buys = fuel.get("insider_buys", 0)
    if nb > 1_000_000:
        ins_sc = 100.0
    elif nb > 100_000:
        ins_sc = 75.0
    elif nb > 0 or buys > 0:
        ins_sc = 50.0
    elif nb < -1_000_000:
        ins_sc = 0.0
    else:
        ins_sc = 20.0

    nc = fuel.get("news_count_48h", 0)
    ns = fuel.get("news_sentiment", 0)
    news_sc = clamp(min(nc, 6) / 6.0 * 70.0 + max(ns, 0) * 10.0)

    fl = fuel.get("float_shares")
    if fl is None:
        float_sc = 30.0
    elif fl < 20e6:
        float_sc = 100.0
    elif fl < 50e6:
        float_sc = 80.0
    elif fl < 150e6:
        float_sc = 60.0
    elif fl < 500e6:
        float_sc = 40.0
    else:
        float_sc = 20.0

    h52 = fuel.get("high_52w")
    if h52 and s["price"]:
        dist = (h52 - s["price"]) / h52 * 100.0
        if dist <= 5:
            h52_sc = 100.0
        elif dist <= 15:
            h52_sc = 70.0
        elif dist <= 30:
            h52_sc = 40.0
        else:
            h52_sc = 15.0
    else:
        h52_sc = 30.0

    s["fuel_score"] = (
        0.25 * short_sc + 0.25 * ins_sc + 0.25 * news_sc
        + 0.10 * float_sc + 0.15 * h52_sc
    )

    s["score"] = 0.6 * s["ignition_score"] + 0.4 * s["fuel_score"]

    # ------------------------------------------------------------------
    # IGNITING flag: live confirmation, all conditions at once
    # ------------------------------------------------------------------
    s["igniting"] = (
        s["rvol"] >= 2.0
        and s["surge"] >= 2.0
        and s["velocity"] > 0
        and (s["new_hod"] or s["vwap_cross"])
    )

    # Human-readable reasons
    if s["rvol"] >= 2:
        s["reasons"].append(f"RVOL {s['rvol']:.1f}x")
    if s["surge"] >= 2:
        s["reasons"].append(f"vol surge {s['surge']:.1f}x")
    if s["vwap_cross"]:
        s["reasons"].append("VWAP reclaim")
    if s["new_hod"]:
        s["reasons"].append("new HOD")
    if s["macd_cross"]:
        s["reasons"].append("MACD cross")
    if s["velocity"] > 0.5:
        s["reasons"].append(f"+{s['velocity']:.2f}% / 5min")
    if spf and spf >= 0.15:
        s["reasons"].append(f"short float {spf*100:.0f}%")
    if nb > 0:
        s["reasons"].append("insider buying")
    if nc >= 2:
        s["reasons"].append(f"{nc} headlines 48h")

    return s


# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
st.sidebar.markdown("## IGNITION")
st.sidebar.caption("Momentum ignition scanner")

preset = None
screener_mode = False
source = st.sidebar.radio(
    "Watchlist source",
    ["Preset / Custom", "Nightly Screener Top 10"],
    index=0,
)

if source == "Nightly Screener Top 10":
    screener_mode = True
    screener_url = st.sidebar.text_input("Dump URL", value=SCREENER_URL_DEFAULT)
    pool_size = st.sidebar.slider(
        "Candidate pool (pre-ranked from dump)", 20, 80, 40,
        help="The whole dump is pre-ranked by its own nightly signals; the top "
             "N candidates get the live ignition scan, then the top 10 by "
             "final score are shown.",
    )
    min_price = st.sidebar.number_input("Min price filter", value=1.0, step=0.5)
    tickers = []
    try:
        dump = load_screener_dump(screener_url)
        pre = prescore_screener(dump)
        # Optional price filter if the dump has a price-like column
        pcol = _find_col(dump.columns, "price") or _find_col(dump.columns, "close")
        if pcol is not None:
            prices = pd.to_numeric(dump[pcol], errors="coerce")
            pre = pre[prices >= float(min_price)]
        candidates = pre.sort_values(ascending=False).head(pool_size)
        tickers = list(candidates.index)
        st.sidebar.caption(
            f"Dump loaded: {len(dump)} tickers, {len(dump.columns)} fields. "
            f"Pre-ranked pool: {len(tickers)}."
        )
        with st.sidebar.expander("Detected dump fields"):
            st.write(", ".join(str(c) for c in dump.columns))
        st.session_state["screener_pre"] = candidates
    except Exception as e:
        st.sidebar.error(f"Could not load nightly dump: {e}")
else:
    preset = st.sidebar.selectbox("Watchlist preset", ["Custom"] + list(PRESETS.keys()), index=1)
    default_tickers = PRESETS.get(preset, "SMR,IONQ,UEC,CCJ,NVDA")
    tickers_raw = st.sidebar.text_area("Tickers (comma separated)", value=default_tickers, height=90)
    tickers = [t.strip().upper() for t in re.split(r"[,\s]+", tickers_raw) if t.strip()][:30]

st.sidebar.markdown("---")
alert_threshold = st.sidebar.slider("Alert score threshold", 40, 95, 65)
auto_refresh = st.sidebar.toggle("Auto-refresh", value=True)
refresh_secs = st.sidebar.slider("Refresh every (sec)", 30, 300, 60)
st.sidebar.markdown("---")
if alpaca_keys():
    st.sidebar.success("Data feed: Alpaca (real-time IEX)")
else:
    st.sidebar.info("Data feed: Yahoo (may lag ~15 min). Add Alpaca keys in secrets for real-time.")
st.sidebar.caption(
    "This tool detects momentum early; it does not predict the future. "
    "Not financial advice."
)

if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "alerted" not in st.session_state:
    st.session_state.alerted = set()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
now_str = datetime.now().strftime("%H:%M:%S")
st.markdown(f"# IGNITION <span style='font-size:16px;color:#5b7089;font-family:Space Mono'>last scan {now_str}</span>", unsafe_allow_html=True)
st.caption("Catches the first minutes of a momentum move: volume ignition + price thrust + squeeze/insider/news fuel.")

# ----------------------------------------------------------------------------
# Scan
# ----------------------------------------------------------------------------
results = []
progress = st.progress(0.0, text="Scanning...")
for i, t in enumerate(tickers):
    results.append(compute_signals(t))
    progress.progress((i + 1) / max(len(tickers), 1), text=f"Scanning {t}...")
progress.empty()

ok = [r for r in results if not r.get("error")]
failed = [r["ticker"] for r in results if r.get("error")]
ok.sort(key=lambda r: r["score"], reverse=True)

if screener_mode:
    scanned_n = len(ok)
    ok = ok[:10]
    pre_ranks = st.session_state.get("screener_pre")
    for r in ok:
        if pre_ranks is not None and r["ticker"] in pre_ranks.index:
            r["pre_rank"] = float(pre_ranks[r["ticker"]])
    st.caption(
        f"Nightly Screener mode: pre-ranked the full dump, live-scanned the top "
        f"{scanned_n} candidates, showing the top 10 by ignition score."
    )

# Register new alerts
session_key = datetime.now().strftime("%Y%m%d")
for r in ok:
    key = f"{session_key}:{r['ticker']}"
    if (r["igniting"] or r["score"] >= alert_threshold) and key not in st.session_state.alerted:
        st.session_state.alerted.add(key)
        st.session_state.alerts.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "ticker": r["ticker"],
            "score": round(r["score"]),
            "price": r["price"],
            "why": ", ".join(r["reasons"][:4]) or "score threshold",
            "igniting": r["igniting"],
        })
        st.toast(f"{r['ticker']} score {r['score']:.0f} - {', '.join(r['reasons'][:3])}")

# Igniting banner
igniting_now = [r for r in ok if r["igniting"]]
if igniting_now:
    names = "  |  ".join(
        f"{r['ticker']} {r['score']:.0f} ({', '.join(r['reasons'][:3])})" for r in igniting_now
    )
    st.markdown(f"<div class='ignite-banner'>IGNITING NOW &nbsp; {names}</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Leaderboard table
# ----------------------------------------------------------------------------
if ok:
    rows = []
    for r in ok:
        f = r["fuel"]
        spf = f.get("short_pct_float")
        rows.append({
            "Ticker": r["ticker"],
            "Score": round(r["score"], 1),
            "NightlyRank": round(r["pre_rank"], 0) if r.get("pre_rank") is not None else None,
            "Ignition": round(r["ignition_score"], 1),
            "Fuel": round(r["fuel_score"], 1),
            "Price": round(r["price"], 2) if r["price"] else None,
            "Chg %": round(r["chg_pct"], 2) if r["chg_pct"] is not None else None,
            "RVOL": round(r["rvol"], 2),
            "Surge": round(r["surge"], 2),
            "Vel %/5m": round(r["velocity"], 2),
            "VWAP+": "Y" if r["above_vwap"] else "",
            "HOD": "NEW" if r["new_hod"] else ("near" if r["near_hod"] else ""),
            "RSI5": round(r["rsi5"], 0) if r["rsi5"] is not None else None,
            "Short%Flt": round(spf * 100, 1) if spf else None,
            "InsiderNet$": f.get("insider_net_buy_usd", 0.0),
            "News48h": f.get("news_count_48h", 0),
            "Signals": ", ".join(r["reasons"][:5]),
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
            "Ignition": st.column_config.ProgressColumn("Ignition", min_value=0, max_value=100, format="%.0f"),
            "Fuel": st.column_config.ProgressColumn("Fuel", min_value=0, max_value=100, format="%.0f"),
            "InsiderNet$": st.column_config.NumberColumn("InsiderNet$", format="$%.0f"),
        },
    )
else:
    st.warning("No data returned. Market may be closed, or tickers invalid.")

if failed:
    st.caption(f"No data for: {', '.join(failed)}")

# ----------------------------------------------------------------------------
# Detail chart + alert feed
# ----------------------------------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.markdown("### Chart")
    if ok:
        sel = st.selectbox("Ticker", [r["ticker"] for r in ok], index=0, label_visibility="collapsed")
        m1, _ = fetch_intraday(sel)
        if m1 is not None and len(m1) > 2:
            vw = vwap(m1)
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                vertical_spacing=0.03,
            )
            fig.add_trace(go.Candlestick(
                x=m1.index, open=m1["Open"], high=m1["High"],
                low=m1["Low"], close=m1["Close"], name=sel,
                increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=m1.index, y=vw, name="VWAP",
                line=dict(color="#ff6b1a", width=1.6),
            ), row=1, col=1)
            colors = np.where(m1["Close"] >= m1["Open"], "#1f7a45", "#8a2b2b")
            fig.add_trace(go.Bar(
                x=m1.index, y=m1["Volume"], name="Volume", marker_color=colors,
            ), row=2, col=1)
            fig.update_layout(
                height=460, template="plotly_dark",
                paper_bgcolor="#0a0e14", plot_bgcolor="#0d1320",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_rangeslider_visible=False, showlegend=False,
                font=dict(family="Space Mono"),
            )
            st.plotly_chart(fig, use_container_width=True)

            rsel = next((r for r in ok if r["ticker"] == sel), None)
            if rsel:
                f = rsel["fuel"]
                tags = []
                if f.get("short_pct_float"):
                    tags.append(f"short float {f['short_pct_float']*100:.1f}%")
                if f.get("float_shares"):
                    tags.append(f"float {f['float_shares']/1e6:.0f}M")
                if f.get("insider_buys"):
                    tags.append(f"{f['insider_buys']} insider buys 90d")
                if f.get("news_count_48h"):
                    tags.append(f"{f['news_count_48h']} headlines 48h")
                if tags:
                    st.markdown(" ".join(f"<span class='fuel-tag'>{t}</span>" for t in tags), unsafe_allow_html=True)
                if f.get("latest_headline"):
                    st.caption(f"Latest: {f['latest_headline']}")

with right:
    st.markdown("### Alert feed")
    if st.button("Clear alerts"):
        st.session_state.alerts = []
        st.session_state.alerted = set()
    if st.session_state.alerts:
        for a in st.session_state.alerts[:25]:
            flag = "IGNITING" if a["igniting"] else "ALERT"
            price = f"${a['price']:.2f}" if a.get("price") else ""
            st.markdown(
                f"<div class='alert-row'>{a['time']} &nbsp;<b>{a['ticker']}</b> {price} "
                f"&nbsp;[{flag} {a['score']}]<br>{a['why']}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No alerts yet this session. Alerts fire when a ticker crosses the score threshold or all ignition conditions confirm at once.")

# ----------------------------------------------------------------------------
# Auto refresh
# ----------------------------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()
