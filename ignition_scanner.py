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
.ignite-banner.rev {
    background: linear-gradient(90deg, #170f2a, #221540);
    border-color: #8b5cf6;
    color: #c4b0f5;
    animation: none;
}
.crow.rev { border-color: #8b5cf6; }
.cflag.rev { color: #8b5cf6; border-color: #8b5cf6; }
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
.crow {
    background: #10151f;
    border: 1px solid #1c2533;
    border-radius: 8px;
    padding: 8px 12px 9px 12px;
    margin-bottom: 6px;
}
.ref-table { width: 100%; border-collapse: collapse; font-size: 13.5px; font-family: 'DM Sans', sans-serif; }
.ref-table td { padding: 8px 10px; border-bottom: 1px solid #1c2533; color: #d8e0ea; vertical-align: top; line-height: 1.5; }
.ref-table .grp td { padding-top: 20px; padding-bottom: 6px; font-family: 'Syne', sans-serif; font-weight: 700; font-size: 14px; letter-spacing: 0.6px; border-bottom: 1px solid #2b3a4f; text-transform: uppercase; }
.ref-table .trm { font-family: 'Space Mono', monospace; font-weight: 700; white-space: nowrap; color: #e8eef5; width: 120px; }
.ref-table .lvl { font-family: 'Space Mono', monospace; color: #7e93ab; font-size: 12px; white-space: nowrap; width: 170px; }
.ref-table .mng { color: #9fb6d0; }
.crow.hot { border-color: #ff6b1a; }
.cline {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    font-family: 'Space Mono', monospace;
}
.ctick { font-size: 17px; font-weight: 700; color: #e8eef5; letter-spacing: 0.5px; }
.cflag {
    font-size: 10px; color: #ff6b1a; border: 1px solid #ff6b1a;
    border-radius: 4px; padding: 1px 6px; margin-left: 8px; vertical-align: middle;
}
.cscore { font-size: 20px; font-weight: 700; }
.cbar {
    height: 5px; background: #1c2533; border-radius: 3px;
    margin: 6px 0 5px 0; overflow: hidden;
}
.cfill { height: 100%; border-radius: 3px; }
.csub {
    font-family: 'Space Mono', monospace;
    font-size: 11.5px; color: #7e93ab;
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 4px;
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


# Typical U-shaped intraday volume distribution: fraction of a full day's
# volume that trades in each 30-minute bucket from 9:30 to 16:00 ET.
# Heavy at the open and close, quiet over lunch. Using this instead of a
# linear pace stops RVOL from reading 3-5x on every stock at 9:45 AM.
VOL_CURVE = [0.13, 0.09, 0.075, 0.065, 0.06, 0.055, 0.05,
             0.05, 0.055, 0.06, 0.07, 0.09, 0.15]


def expected_vol_fraction(minutes_elapsed: int) -> float:
    """Cumulative fraction of a typical day's volume expected by N minutes
    into the session, following the U-shaped curve above."""
    m = max(1, min(int(minutes_elapsed), 390))
    full_buckets = m // 30
    frac = sum(VOL_CURVE[:full_buckets])
    if full_buckets < len(VOL_CURVE):
        frac += VOL_CURVE[full_buckets] * (m - full_buckets * 30) / 30.0
    return max(frac, 0.01)


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


def ntfy_config():
    """Read ntfy settings from Streamlit secrets. Returns (server, topic) or None."""
    try:
        topic = st.secrets.get("NTFY_TOPIC", "")
        server = st.secrets.get("NTFY_SERVER", "https://ntfy.sh")
        if topic:
            return server.rstrip("/"), topic
    except Exception:
        pass
    return None


def send_ntfy(title: str, message: str, priority: str = "default", tags: str = "chart_with_upwards_trend"):
    """Push a notification to the phone via ntfy.sh. Fire-and-forget:
    a notification failure must never break the scan loop."""
    cfg = ntfy_config()
    if not cfg:
        return False
    server, topic = cfg
    try:
        requests.post(
            f"{server}/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": tags},
            timeout=5,
        )
        return True
    except Exception:
        return False


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
    # Keep regular session only (09:30-16:00 ET) for intraday bars.
    # Daily bars are stamped pre-market by Alpaca, so skip the filter for them.
    if "Min" in timeframe:
        df = df.between_time("09:30", "16:00")
    return df if len(df) else None


def _bars_ok(df, n):
    return df is not None and len(df) >= n


@st.cache_data(ttl=55, show_spinner=False)
def fetch_intraday(ticker: str):
    """1-minute bars for today plus 5-minute bars for ~5 days.
    Tries Alpaca (real-time) first when keys are configured, but validates the
    result: thin tickers can come back nearly empty on the free IEX feed, so
    anything insufficient falls back to Yahoo, and the better source wins."""
    m1_a = m5_a = None
    if alpaca_keys():
        try:
            k, s = alpaca_keys()
            today = datetime.now(timezone.utc) - timedelta(hours=24)
            week = datetime.now(timezone.utc) - timedelta(days=7)
            m1_a = _alpaca_bars(ticker, "1Min", today.strftime("%Y-%m-%dT%H:%M:%SZ"), k, s)
            m5_a = _alpaca_bars(ticker, "5Min", week.strftime("%Y-%m-%dT%H:%M:%SZ"), k, s)
            if m1_a is not None:
                last_day = m1_a.index[-1].date()
                m1_a = m1_a[m1_a.index.date == last_day]
        except Exception:
            m1_a = m5_a = None
    if _bars_ok(m1_a, 6):
        return m1_a, m5_a

    m1_y = m5_y = None
    try:
        tk = yf.Ticker(ticker)
        m1_y = flatten_cols(tk.history(period="1d", interval="1m", prepost=False))
        m5_y = flatten_cols(tk.history(period="5d", interval="5m", prepost=False))
    except Exception:
        m1_y = m5_y = None
    if _bars_ok(m1_y, 6):
        return m1_y, m5_y

    # Neither source is sufficient: return whichever has the most bars
    a_n = len(m1_a) if m1_a is not None else 0
    y_n = len(m1_y) if m1_y is not None else 0
    return (m1_a, m5_a) if a_n >= y_n else (m1_y, m5_y)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily(ticker: str):
    """~3 months of daily bars for the RVOL baseline and previous close.
    Tries Alpaca first when keys are configured, validates, falls back to
    Yahoo, and keeps whichever source has more history."""
    d_a = None
    if alpaca_keys():
        try:
            k, s = alpaca_keys()
            start = datetime.now(timezone.utc) - timedelta(days=100)
            d_a = _alpaca_bars(ticker, "1Day", start.strftime("%Y-%m-%dT%H:%M:%SZ"), k, s)
        except Exception:
            d_a = None
    if _bars_ok(d_a, 21):
        return d_a
    d_y = None
    try:
        d_y = flatten_cols(yf.Ticker(ticker).history(period="3mo", interval="1d"))
    except Exception:
        d_y = None
    if _bars_ok(d_y, 21):
        return d_y
    a_n = len(d_a) if d_a is not None else 0
    y_n = len(d_y) if d_y is not None else 0
    return d_a if a_n >= y_n else d_y


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
    scalar-only DataFrame indexed by ticker.

    Memory-safe path: the response is streamed, decompressed on the fly, and
    parsed record-by-record with ijson, so the multi-hundred-MB decompressed
    JSON never exists in RAM at once. This matters on Streamlit Cloud, which
    kills the container (-> 'no response from server') around ~1 GB."""
    import decimal
    records = []
    try:
        import ijson
        with requests.get(url, timeout=300, stream=True) as resp:
            resp.raise_for_status()
            stream = resp.raw
            if url.endswith(".gz"):
                stream = gzip.GzipFile(fileobj=resp.raw)
            for tkr, fields in ijson.kvitems(stream, ""):
                if not isinstance(fields, dict):
                    continue
                row = {}
                for k, v in fields.items():
                    if isinstance(v, decimal.Decimal):
                        row[k] = float(v)
                    elif isinstance(v, (str, int, float, bool)) or v is None:
                        row[k] = v
                row.setdefault("ticker", str(tkr))
                records.append(row)
    except Exception:
        records = []

    if not records:
        # Fallback: in-memory parse (handles list-of-records formats too)
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        blob = resp.content
        if url.endswith(".gz") or blob[:2] == b"\x1f\x8b":
            blob = gzip.decompress(blob)
        data = json.loads(blob, object_hook=_strip_heavy)
        del blob
        if isinstance(data, dict):
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
            records = [
                {k: v for k, v in r.items() if not isinstance(v, dict)}
                for r in data if isinstance(r, dict)
            ]

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


@st.cache_data(ttl=86400, show_spinner="Building today's watchlist from nightly dump...")
def screener_watchlist(url: str, pool_size: int, min_price: float, day_key: str):
    """Pin the candidate watchlist ONCE per calendar day from the nightly dump.
    day_key (today's date) is part of the cache key, so this recomputes only
    when the date changes or the settings change. After this, all live data
    comes from the real-time feed - the dump is not touched again."""
    dump = load_screener_dump(url)
    pre = prescore_screener(dump)
    pcol = _find_col(dump.columns, "price") or _find_col(dump.columns, "close")
    if pcol is not None:
        prices = pd.to_numeric(dump[pcol], errors="coerce")
        pre = pre[prices >= float(min_price)]
    candidates = pre.sort_values(ascending=False).head(pool_size)
    meta = {"n_tickers": len(dump), "n_fields": len(dump.columns),
            "fields": [str(c) for c in dump.columns]}
    return list(candidates.index), candidates, meta


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
        "score": 0.0, "igniting": False, "gap_reversal": False,
        "gap_pct": 0.0, "reasons": [], "error": None,
    }

    m1, m5 = fetch_intraday(ticker)
    daily = fetch_daily(ticker)
    fuel = fetch_fuel(ticker)
    s["fuel"] = fuel

    if m1 is None or len(m1) < 6 or daily is None or len(daily) < 5:
        s["error"] = "no data"
        return s

    close = m1["Close"]
    s["price"] = float(close.iloc[-1])
    prev_close = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else float(daily["Close"].iloc[-1])
    s["chg_pct"] = (s["price"] / prev_close - 1.0) * 100.0 if prev_close else None

    # --- RVOL: cumulative volume today vs trailing average, curve-adjusted ---
    # Adaptive window: up to 20 days, fewer if history is short (new listings)
    win = min(20, len(daily) - 1)
    avg_daily_vol = float(daily["Volume"].iloc[-(win + 1):-1].mean())
    cum_vol = float(m1["Volume"].sum())
    elapsed = max(len(m1), 1)  # minutes elapsed in session
    expected = avg_daily_vol * expected_vol_fraction(elapsed)
    s["rvol"] = cum_vol / expected if expected > 0 else 0.0

    # --- Opening gap vs yesterday's close ---
    open_px = float(m1["Open"].iloc[0])
    s["gap_pct"] = (open_px / prev_close - 1.0) * 100.0 if prev_close else 0.0

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
    # Direction-aware flags. The same live footprint (RVOL + surge +
    # velocity + HOD/VWAP trigger) means different things depending on the
    # day's tape:
    #   IGNITING     = footprint fires on a flat/up day -> fresh momentum leg
    #   GAP REVERSAL = footprint fires while the stock is down hard on the
    #                  day or gapped down big -> a bounce attempt inside a
    #                  selloff (tradable, but a different and riskier trade)
    # ------------------------------------------------------------------
    footprint = (
        s["rvol"] >= 2.0
        and s["surge"] >= 2.0
        and s["velocity"] > 0
        and (s["new_hod"] or s["vwap_cross"])
    )
    chg = s["chg_pct"] if s["chg_pct"] is not None else 0.0
    gap = s.get("gap_pct", 0.0) or 0.0
    down_tape = chg <= -4.0 or gap <= -4.0
    s["igniting"] = footprint and not down_tape
    s["gap_reversal"] = footprint and down_tape

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
    if s["gap_reversal"]:
        s["reasons"].insert(0, f"day {chg:+.1f}% - bounce attempt")
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
    help="Preset / Custom: scan a hand-picked list. Nightly Screener Top 10: "
         "your screener's nightly data dump is pre-ranked once per day to pin "
         "a candidate pool; the pool is live-scanned and the 10 highest "
         "scoring stocks are shown.",
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
    min_price = st.sidebar.number_input(
        "Min price filter", value=1.0, step=0.5,
        help="Excludes stocks below this price from the candidate pool. "
             "Filters out sub-$1 names with wide spreads and manipulation risk.",
    )
    tickers = []
    try:
        day_key = datetime.now().strftime("%Y-%m-%d")
        tickers, candidates, meta = screener_watchlist(
            screener_url, pool_size, float(min_price), day_key
        )
        st.sidebar.caption(
            f"Watchlist pinned for {day_key} from dump "
            f"({meta['n_tickers']} tickers, {meta['n_fields']} fields). "
            f"Pool: {len(tickers)}. Live data from real-time feed only."
        )
        with st.sidebar.expander("Detected dump fields"):
            st.write(", ".join(meta["fields"]))
        st.session_state["screener_pre"] = candidates
    except Exception as e:
        st.sidebar.error(f"Could not load nightly dump: {e}")
else:
    preset = st.sidebar.selectbox("Watchlist preset", ["Custom"] + list(PRESETS.keys()), index=1)
    default_tickers = PRESETS.get(preset, "SMR,IONQ,UEC,CCJ,NVDA")
    tickers_raw = st.sidebar.text_area("Tickers (comma separated)", value=default_tickers, height=90)
    tickers = [t.strip().upper() for t in re.split(r"[,\s]+", tickers_raw) if t.strip()][:30]

st.sidebar.markdown("---")
alert_threshold = st.sidebar.slider(
    "Alert score threshold", 40, 95, 65,
    help="A ticker triggers an alert (feed entry + phone push) the first time "
         "its overall Score crosses this line each day. IGNITING alerts fire "
         "regardless of this threshold when all live conditions confirm at once.",
)
view_mode = st.sidebar.radio(
    "Table style", ["Compact (phone)", "Full table"], index=0,
    help="Compact: card rows sized for a phone screen - ticker, color-coded "
         "score, slim bar, key numbers. Full table: the complete sortable "
         "grid with column tooltips (better on desktop).",
)
show_all_cols = st.sidebar.toggle(
    "Show all table columns", value=False,
    help="Off: compact view (Ticker, Score, NightlyRank, Ignition, Fuel). "
         "On: every underlying metric - RVOL, Surge, Velocity, VWAP, HOD, RSI, "
         "short float, insider dollars, news count, and live signals.",
)
paused = st.sidebar.toggle(
    "Pause scanning", value=False, key="paused",
    help="Freezes the scanner: no rescans, no new alerts, and auto-refresh "
         "stops. The last scan's results stay on screen. (Changing the chart "
         "ticker can still fetch that one chart.) Flip off to resume.",
)
auto_refresh = st.sidebar.toggle(
    "Auto-refresh", value=True,
    help="Rescan automatically on the interval below. The page must stay open "
         "(a minimized tab is fine) for scanning and phone alerts to keep running.",
)
refresh_secs = st.sidebar.slider(
    "Refresh every (sec)", 30, 300, 60,
    help="Live bars are cached ~55 seconds, so refreshing faster than 60s "
         "mostly re-reads cache. 60s is the sweet spot.",
)
st.sidebar.markdown("---")
if ntfy_config():
    notify_on = st.sidebar.toggle("Phone notifications (ntfy)", value=True)
    if st.sidebar.button("Send test notification"):
        ok_test = send_ntfy("IGNITION test", "If you can read this, alerts are wired up.", tags="white_check_mark")
        if ok_test:
            st.sidebar.success("Test sent - check your phone.")
        else:
            st.sidebar.error("Send failed - check NTFY_TOPIC in secrets.")
else:
    notify_on = False
    st.sidebar.info("Phone alerts off: add NTFY_TOPIC to secrets to enable ntfy push.")

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
APP_VERSION = "v1.8 - memfix"
last_scan = st.session_state.get("last_scan_time", "--:--:--")
if paused:
    status = f"<span style='color:#f5b942'>PAUSED</span> &nbsp;last scan {last_scan}"
else:
    status = f"last scan {last_scan}"
st.markdown(f"# IGNITION <span style='font-size:16px;color:#5b7089;font-family:Space Mono'>{status} &nbsp;|&nbsp; {APP_VERSION}</span>", unsafe_allow_html=True)
st.caption("Catches the first minutes of a momentum move: volume ignition + price thrust + squeeze/insider/news fuel.")

# ----------------------------------------------------------------------------
# Scan (skipped while paused - last results are reused)
# ----------------------------------------------------------------------------
if paused and "last_results" in st.session_state:
    results = st.session_state["last_results"]
else:
    results = []
    progress = st.progress(0.0, text="Scanning...")
    for i, t in enumerate(tickers):
        results.append(compute_signals(t))
        progress.progress((i + 1) / max(len(tickers), 1), text=f"Scanning {t}...")
    progress.empty()
    st.session_state["last_results"] = results
    st.session_state["last_scan_time"] = datetime.now().strftime("%H:%M:%S")

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

# Register new alerts (skipped while paused - stale results must not re-alert)
session_key = datetime.now().strftime("%Y%m%d")
for r in (ok if not paused else []):
    key = f"{session_key}:{r['ticker']}"
    if (r["igniting"] or r["gap_reversal"] or r["score"] >= alert_threshold) and key not in st.session_state.alerted:
        st.session_state.alerted.add(key)
        kind = "igniting" if r["igniting"] else ("reversal" if r["gap_reversal"] else "alert")
        st.session_state.alerts.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "ticker": r["ticker"],
            "score": round(r["score"]),
            "price": r["price"],
            "why": ", ".join(r["reasons"][:4]) or "score threshold",
            "kind": kind,
        })
        st.toast(f"{r['ticker']} score {r['score']:.0f} - {', '.join(r['reasons'][:3])}")
        if notify_on:
            why = ", ".join(r["reasons"][:4]) or "score threshold"
            price_txt = f" @ ${r['price']:.2f}" if r.get("price") else ""
            if kind == "igniting":
                send_ntfy(
                    f"IGNITING: {r['ticker']}{price_txt}",
                    f"Score {r['score']:.0f} | {why}",
                    priority="urgent", tags="fire",
                )
            elif kind == "reversal":
                send_ntfy(
                    f"GAP REVERSAL: {r['ticker']}{price_txt}",
                    f"Bounce attempt inside a selloff. Score {r['score']:.0f} | {why}",
                    priority="high", tags="warning",
                )
            else:
                send_ntfy(
                    f"{r['ticker']} alert{price_txt}",
                    f"Score {r['score']:.0f} | {why}",
                    priority="default",
                )

# ----------------------------------------------------------------------------
# Reference key (rendered in its own tab)
# ----------------------------------------------------------------------------
REFERENCE_KEY = [
    ("The scores", "#d8e0ea", [
        ("Score", "Overall grade: 60% Ignition + 40% Fuel", "70+ hot, 50+ warm"),
        ("Ignition", "Is money flowing in right now (live, every refresh)", "jumps 30+ pts = act"),
        ("Fuel", "Is the stock primed for a big move (updates hourly)", "high + rising IGN = setup"),
        ("NightlyRank", "Grade from last night's screener dump - yesterday's homework vs today's live Score", "pool picked from top N"),
    ]),
    ("Live ignition signals - 60% of Score", "#ff6b1a", [
        ("RVOL", "Today's volume vs 20-day norm, adjusted for the U-shaped intraday curve", "2x unusual, 5x explosive"),
        ("Surge", "Last 3 minutes' volume vs the session's average bar - the exact minutes buying hits", "2x+ = surge live"),
        ("Vel %/5m", "Price move over the last 5 minutes", "positive + growing"),
        ("Accel", "Is the velocity itself speeding up or fading", "positive = building"),
        ("VWAP+", "Price above the volume-weighted average price; a reclaim from below is a trigger", "Y = buyers in control"),
        ("HOD", "High of day: NEW = fresh breakout, near = coiling within 0.5%", "NEW = trigger"),
        ("RSI5", "RSI(14) on 5-minute bars", "55-75 thrust, 75+ stretched"),
        ("MACD", "Fresh bullish cross on 5-minute bars confirms a trend flip", "cross = confirmation"),
    ]),
    ("Fuel signals - 40% of Score", "#2dd4a7", [
        ("Short%Flt", "Short interest as % of float - forced buyers if price runs", "15%+ = squeeze fuel"),
        ("InsiderNet$", "Insider buys minus sells, last 90 days (SEC Form 4)", "positive = accumulating"),
        ("News48h", "Headlines in the last 48 hours - momentum needs a catalyst", "0 = no sustained move"),
        ("Float", "Tradable shares outstanding - small float means violent moves", "under 50M = explosive"),
        ("52wk dist", "Distance to the 52-week high - momentum lives near highs", "within 5% = best"),
    ]),
    ("Flags and alerts", "#8b5cf6", [
        ("IGNITING", "All four fire at once on a flat/up day: RVOL >=2x, surge >=2x, positive velocity, new HOD or VWAP reclaim", "urgent push, orange"),
        ("GAP REV", "Same footprint but the stock is down 4%+ or gapped down 4%+ - a bounce inside a selloff, riskier trade", "high push, violet"),
        ("ALERT", "Score crossed your sidebar threshold; each ticker alerts once per day", "default push"),
    ]),
]


def render_reference_key():
    """Color-coded glossary of every term in the indicator, as its own tab."""
    rows = ["<table class='ref-table'>"]
    for group, color, terms in REFERENCE_KEY:
        rows.append(
            f"<tr class='grp'><td colspan='3' style='color:{color}'>{group}</td></tr>"
        )
        for term, meaning, level in terms:
            rows.append(
                f"<tr><td class='trm' style='border-left:3px solid {color};"
                f"padding-left:8px'>{term}</td>"
                f"<td class='mng'>{meaning}</td><td class='lvl'>{level}</td></tr>"
            )
    rows.append("</table>")
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.caption(
        "The relationship that matters: Fuel tells you WHICH stocks to watch, "
        "Ignition tells you WHEN. A name at Fuel 70 / Ignition 25 is a loaded "
        "spring doing nothing - yet. When Ignition jumps 30 points in one "
        "refresh, that is the moment this tool exists for. Not financial advice."
    )


# ----------------------------------------------------------------------------
# Tabs: live scanner + reference key
# ----------------------------------------------------------------------------
tab_scan, tab_ref = st.tabs(["Scanner", "Reference key"])

with tab_ref:
    render_reference_key()

# Everything below renders inside the Scanner tab. The tab context is entered
# explicitly so the long display section keeps its flat indentation.
tab_scan.__enter__()

# Igniting / reversal banners
igniting_now = [r for r in ok if r["igniting"]]
reversals_now = [r for r in ok if r["gap_reversal"]]
if igniting_now:
    names = "  |  ".join(
        f"{r['ticker']} {r['score']:.0f} ({', '.join(r['reasons'][:3])})" for r in igniting_now
    )
    st.markdown(f"<div class='ignite-banner'>IGNITING NOW &nbsp; {names}</div>", unsafe_allow_html=True)
if reversals_now:
    names = "  |  ".join(
        f"{r['ticker']} {r['score']:.0f} ({', '.join(r['reasons'][:3])})" for r in reversals_now
    )
    st.markdown(f"<div class='ignite-banner rev'>GAP REVERSAL &nbsp; {names}</div>", unsafe_allow_html=True)

def render_compact(rows_data):
    """Phone-friendly card rows: ticker + big color-coded score, a slim score
    bar, and one line of key numbers. Readable at 380px wide."""
    html = []
    for r in rows_data:
        sc = r["score"]
        color = "#ff6b1a" if sc >= 70 else ("#f5b942" if sc >= 50 else "#8aa0b8")
        hot = " hot" if r["igniting"] else (" rev" if r.get("gap_reversal") else "")
        if r["igniting"]:
            flag = "<span class='cflag'>IGNITING</span>"
        elif r.get("gap_reversal"):
            flag = "<span class='cflag rev'>GAP REV</span>"
        else:
            flag = ""
        pre = r.get("pre_rank")
        pre_txt = f"N {pre:.0f}" if pre is not None else ""
        price_txt = f"${r['price']:.2f}" if r.get("price") else ""
        chg = r.get("chg_pct")
        if chg is not None:
            chg_color = "#22c55e" if chg >= 0 else "#ef4444"
            chg_txt = f"<span style='color:{chg_color}'>{chg:+.1f}%</span>"
        else:
            chg_txt = ""
        rvol_txt = f"RVOL {r['rvol']:.1f}x" if r.get("rvol") else ""
        html.append(
            f"<div class='crow{hot}'>"
            f"<div class='cline'><span><span class='ctick'>{r['ticker']}</span>{flag}</span>"
            f"<span class='cscore' style='color:{color}'>{sc:.0f}</span></div>"
            f"<div class='cbar'><div class='cfill' style='width:{min(sc, 100):.0f}%;background:{color}'></div></div>"
            f"<div class='csub'><span>IGN {r['ignition_score']:.0f}</span>"
            f"<span>FUEL {r['fuel_score']:.0f}</span>"
            + (f"<span>{pre_txt}</span>" if pre_txt else "")
            + (f"<span>{rvol_txt}</span>" if rvol_txt else "")
            + f"<span>{price_txt} {chg_txt}</span></div>"
            f"</div>"
        )
    st.markdown("".join(html), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Leaderboard
# ----------------------------------------------------------------------------
if ok and view_mode == "Compact (phone)":
    render_compact(ok)
elif ok:
    HELP = {
        "Ticker": "Stock symbol. Click a row's ticker in the Chart selector below to inspect it.",
        "Score": "Overall Ignition Score, 0-100. Weighted blend: 60% Ignition (live, is it moving NOW) + 40% Fuel (is it primed to move). Higher = stronger momentum setup.",
        "NightlyRank": "How this stock scored in last night's screener dump pre-ranking (0-100), based on your nightly signals: Short Squeeze, composite score, RSI zone, Golden Cross, MFI, OBV, volume, Piotroski. This is yesterday's homework; Score is today's live grade.",
        "Ignition": "Live momentum sub-score, 0-100, recalculated every refresh. Built from: relative volume vs 20-day pace (25%), last-3-bar volume surge (15%), 5-min price velocity (15%), acceleration (10%), VWAP position/reclaim (10%), high-of-day breakout (10%), RSI thrust (7.5%), MACD cross (7.5%).",
        "Fuel": "Primed-to-move sub-score, 0-100, refreshed hourly. Built from: short % of float (25%), insider net buying 90d (25%), news flow + sentiment 48h (25%), float size (10%), distance to 52-week high (15%). High Fuel + rising Ignition is the setup you want.",
        "Price": "Last traded price from the live feed (Alpaca real-time when keys are set, otherwise Yahoo, which can lag ~15 min).",
        "Chg %": "Percent change vs yesterday's close.",
        "RVOL": "Relative Volume: today's cumulative volume vs the 20-day average, adjusted for the U-shaped intraday volume curve (heavy at open and close, quiet at lunch). 1.0 = normal. 2x+ = unusual money flowing in. 5x+ = explosive. The single most reliable ignition tell.",
        "Surge": "Last 3 one-minute bars' average volume vs this session's average bar. Catches the exact minutes buying pressure hits. 2x+ = surge in progress.",
        "Vel %/5m": "Velocity: percent price move over the last 5 minutes. Positive and growing = thrust.",
        "VWAP+": "Y = price is above VWAP (volume-weighted average price). Above VWAP means buyers in control of the session; a reclaim from below is a classic ignition trigger.",
        "HOD": "High of day status. NEW = just broke to a new session high (breakout trigger). near = within 0.5% of the high, coiling under it.",
        "RSI5": "RSI(14) on 5-minute bars. 55-75 is the momentum thrust zone (strong but not exhausted). Above 75 = stretched, chase risk rises. Below 50 = no momentum yet.",
        "Short%Flt": "Short interest as a percent of float. 15%+ means heavy bets against the stock - squeeze fuel if price starts running and shorts are forced to cover.",
        "InsiderNet$": "Net dollar value of insider buys minus sells over the last 90 days (SEC Form 4 filings). Positive = the people who know the company best are accumulating.",
        "News48h": "Number of news headlines in the last 48 hours. Momentum needs a catalyst; no news usually means no sustained move.",
        "Signals": "Plain-English list of which triggers are currently firing for this ticker.",
    }
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
    if not show_all_cols:
        keep = ["Ticker", "Score", "NightlyRank", "Ignition", "Fuel"]
        df = df[[c for c in keep if c in df.columns]]
        if "NightlyRank" in df.columns and df["NightlyRank"].isna().all():
            df = df.drop(columns=["NightlyRank"])
    col_cfg = {
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f", help=HELP["Score"]),
        "Ignition": st.column_config.ProgressColumn("Ignition", min_value=0, max_value=100, format="%.0f", help=HELP["Ignition"]),
        "Fuel": st.column_config.ProgressColumn("Fuel", min_value=0, max_value=100, format="%.0f", help=HELP["Fuel"]),
        "InsiderNet$": st.column_config.NumberColumn("InsiderNet$", format="$%.0f", help=HELP["InsiderNet$"]),
    }
    for col in df.columns:
        if col not in col_cfg:
            col_cfg[col] = st.column_config.Column(col, help=HELP.get(col, ""))
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=col_cfg,
    )
else:
    st.warning("No data returned. Market may be closed, or tickers invalid.")

if failed:
    st.caption(f"No data for: {', '.join(failed)}")

with st.expander("Metric guide - what everything means"):
    st.markdown("""
**The two halves of the Score**

- **Ignition (60% of Score)** answers *"is money flowing in right now?"* It is rebuilt
  from live bars every refresh. Its loudest inputs are **RVOL** (today's volume vs the
  20-day norm for this time of day) and the **volume Surge** in the last 3 minutes.
  Price **Velocity/Acceleration**, a **VWAP** reclaim, a **new High of Day**, RSI in the
  55-75 thrust zone, and a fresh MACD cross round it out.
- **Fuel (40% of Score)** answers *"is this stock primed to make a big move?"* High
  **short % of float** means forced buyers if price runs. **Insider net buying** means
  informed accumulation. **Fresh news** provides the catalyst. **Small float** makes
  moves violent. **Near 52-week highs** is where momentum lives.

**IGNITING NOW banner / urgent phone alert**

Fires only when ALL of these confirm on the same refresh - the footprint of the first
minutes of a real momentum leg:
- RVOL at least 2x normal pace (pace follows the real U-shaped intraday volume curve,
  so 9:45 AM readings are no longer inflated)
- Last-3-bar volume surge at least 2x the session average
- Positive 5-minute velocity
- A new high of day OR a VWAP reclaim within the last few bars
- AND the stock is NOT down hard on the day (no big gap-down, day change above -4%)

**GAP REVERSAL banner (violet) / high-priority phone alert**

The same live footprint firing while the stock is down 4%+ on the day or gapped down
4%+ at the open - usually a post-earnings flush being bought. This is a bounce attempt
inside a selloff: a real, tradable pattern, but a different and riskier trade than
fresh ignition. Bounces in crushed stocks fail more often than breakouts in strong
ones, which is why it gets its own label instead of the IGNITING banner.

**Alert feed** logs each ticker once per day, the first time it crosses your score
threshold or ignites. Phone pushes mirror the feed when ntfy is configured.

**Chart**: candles are 1-minute bars for the current session; the orange line is VWAP -
price above it means buyers control the session. Volume bars underneath confirm whether
a move has real participation.

**NightlyRank vs Score**: NightlyRank is how the stock graded in last night's screener
dump (yesterday's homework). Score is the live grade. A high NightlyRank with a surging
Ignition number is the combination this tool exists to catch.

*Detects momentum early; does not predict the future. Not financial advice.*
""")

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
            kind = a.get("kind", "alert")
            flag = {"igniting": "IGNITING", "reversal": "GAP REV"}.get(kind, "ALERT")
            border = "#8b5cf6" if kind == "reversal" else "#ff6b1a"
            price = f"${a['price']:.2f}" if a.get("price") else ""
            st.markdown(
                f"<div class='alert-row' style='border-left-color:{border}'>"
                f"{a['time']} &nbsp;<b>{a['ticker']}</b> {price} "
                f"&nbsp;[{flag} {a['score']}]<br>{a['why']}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No alerts yet this session. Alerts fire when a ticker crosses the score threshold or all ignition conditions confirm at once.")

# Close the Scanner tab context entered above
tab_scan.__exit__(None, None, None)

# ----------------------------------------------------------------------------
# Auto refresh
# ----------------------------------------------------------------------------
if auto_refresh and not paused:
    time.sleep(refresh_secs)
    st.rerun()
