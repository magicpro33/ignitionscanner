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
import random
import math
import re
import gzip
import json
import decimal
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

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
/* =====================================================================
   AI UPSCALE BRAND THEME
   Fonts  : Rajdhani (headings) + Plus Jakarta Sans (body) + Space Mono (data)
   Palette:
     --bg-deep   #07111f   deepest background (page)
     --bg-card   #0d1e33   card / row surfaces
     --bg-mid    #122540   borders / dividers
     --bg-hover  #1a3050   hover / selection
     --amber     #f5a623   primary accent (amber)
     --amber-dim #c47d0e   dimmed amber
     --amber-glow rgba(245,166,35,0.35)
     --text-hi   #e8f0fa   high-contrast text
     --text-mid  #8baac8   medium text / labels
     --text-lo   #4a6a8a   low / placeholder text
     --green     #3ddc84   ignition positive
     --red       #e05555   alert / reversal
     --teal      #29b6c8   secondary accent
     --navy-lt   #1e3a5f   light navy highlight
===================================================================== */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Plus+Jakarta+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"]  { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp                       { background: #07111f; }

h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important;
             letter-spacing: 0.8px; color: #ffffff; }

.metric-mono, .stDataFrame, code { font-family: 'Space Mono', monospace !important; }

/* ── Streamlit widget overrides ───────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]  { background: #0d1e33; border-radius: 8px; gap: 2px; }
.stTabs [data-baseweb="tab"]       { color: #b0c8e8; font-family: 'Rajdhani', sans-serif;
                                      font-weight: 600; font-size: 15px; letter-spacing: 0.5px; }
.stTabs [aria-selected="true"]     { color: #f5a623 !important; border-bottom: 2px solid #f5a623; }
.stTabs [data-baseweb="tab-panel"] { background: transparent; }

/* ── IGNITING banner ─────────────────────────────────────────────── */
.ignite-banner {
    background: linear-gradient(90deg, #1a1000, #261800);
    border: 1px solid #f5a623;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-family: 'Space Mono', monospace;
    color: #f5c96a;
    animation: amber-pulse 1.6s infinite;
}
@keyframes amber-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(245,166,35,0.45); }
    70%  { box-shadow: 0 0 0 12px rgba(245,166,35,0); }
    100% { box-shadow: 0 0 0 0 rgba(245,166,35,0); }
}

/* ── GAP REVERSAL banner ─────────────────────────────────────────── */
.ignite-banner.rev {
    background: linear-gradient(90deg, #0e1a2a, #152030);
    border-color: #29b6c8;
    color: #7dd8e4;
    animation: none;
}
.crow.rev  { border-color: #29b6c8; }
.cflag.rev { color: #29b6c8; border-color: #29b6c8; }

/* ── Alert feed rows ─────────────────────────────────────────────── */
.alert-row {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: #ffffff;
    padding: 6px 10px;
    border-left: 3px solid #f5a623;
    background: #0d1e33;
    margin-bottom: 4px;
    border-radius: 4px;
}

/* ── Plain info pills ────────────────────────────────────────────── */
.fuel-tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 6px;
    background: #0d1e33;
    border: 1px solid #1e3a5f;
    color: #b0c8e8;
}
.fuel-tag a       { color: inherit; text-decoration: none; }
.fuel-tag a:hover { text-decoration: underline; opacity: 0.85; }

/* ── Catalyst tag color classes (navy-amber palette) ─────────────── */
.ct-earnings     { background:#0d2215; border-color:#1e6b35; color:#4dd880; }
.ct-fda          { background:#150d22; border-color:#6b35a0; color:#c07ae0; }
.ct-buyout       { background:#211800; border-color:#c47d0e; color:#f5c040; }
.ct-legal        { background:#220d0d; border-color:#a03535; color:#ff4444; }
.ct-partnership  { background:#07111f; border-color:#1e6a8a; color:#29b6c8; }
.ct-squeeze      { background:#1f1200; border-color:#c47d0e; color:#f5a623; }
.ct-breakout     { background:#071a10; border-color:#1a8040; color:#3ddc84; }
.ct-geopolitical { background:#0d1020; border-color:#3a5090; color:#7090d0; }
.ct-rate         { background:#1a1500; border-color:#907020; color:#d0b040; }
.ct-bimodal      { background:#221800; border-color:#f5a623; color:#f5a623; }
.ct-dtc          { background:#0a1828; border-color:#1e4a7a; color:#5090d0; }

/* ── DTC fuel gauge pill ──────────────────────────────────────────── */
.dtc-gauge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    padding: 2px 8px 2px 7px;
    border-radius: 4px;
    margin-right: 6px;
    background: #0a1828;
    border: 1px solid #1e4a7a;
    color: #b0c8e8;
    vertical-align: middle;
}
.dtc-gauge a { color: inherit; text-decoration: none; display: inline-flex; align-items: center; gap: 5px; }
.dtc-gauge a:hover { opacity: 0.85; }
.dtc-bar-track {
    display: inline-block;
    width: 36px; height: 5px;
    background: #122540;
    border-radius: 3px;
    overflow: hidden;
    vertical-align: middle;
}
.dtc-bar-fill {
    display: block;
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
}

/* ── Card icon indicators ─────────────────────────────────────────── */
.cicon { flex-shrink:0; display:flex; align-items:center; justify-content:center; }
.cicon svg { display:block; }
/* ── Stat grid cards (Option 2) ──────────────────────────────────── */
.crow {
    background: #0d1e33;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 11px 13px 12px 13px;
    margin-bottom: 8px;
}
.crow.hot { border-color: #f5a623;
            box-shadow: 0 0 10px rgba(245,166,35,0.15); }
.crow.rev { border-color: #29b6c8; }
.chead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}
.ctick  { font-size: 18px; font-weight: 700; color: #ffffff;
          letter-spacing: 0.3px; font-family: 'Rajdhani', sans-serif; }
.cflag  { font-size: 10px; color: #f5a623; border: 1px solid #f5a623;
          border-radius: 4px; padding: 1px 6px;
          margin-left: 8px; vertical-align: middle;
          font-family: 'Space Mono', monospace; }
.cflag.rev { color: #29b6c8; border-color: #29b6c8; }
.cbar   { height: 6px; background: #122540; border-radius: 3px;
          margin: 0 0 9px 0; overflow: hidden; }
.cfill  { height: 100%; border-radius: 3px; }
.cgrid  { display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
          margin-bottom: 8px; }
.ctile  { background: #07111f; border-radius: 6px; padding: 7px 9px; }
.ctile-lbl { font-family: 'Space Mono', monospace; font-size: 10px;
             letter-spacing: .6px; text-transform: uppercase;
             color: #7a9ab8; margin-bottom: 3px; }
.ctile-val { font-family: 'Space Mono', monospace; font-size: 13px;
             font-weight: 700; color: #ffffff; line-height: 1.2; }
.ctile-sub { font-family: 'Space Mono', monospace; font-size: 11px;
             margin-top: 1px; }
.ref-table { width: 100%; border-collapse: collapse; font-size: 13.5px;
             font-family: 'Plus Jakarta Sans', sans-serif; }
.ref-table td  { padding: 8px 10px; border-bottom: 1px solid #122540;
                 color: #ffffff; vertical-align: top; line-height: 1.5; }
.ref-table .grp td { padding-top: 20px; padding-bottom: 6px;
                     font-family: 'Rajdhani', sans-serif; font-weight: 700;
                     font-size: 14px; letter-spacing: 0.8px;
                     border-bottom: 1px solid #1e3a5f; text-transform: uppercase; }
.ref-table .trm { font-family: 'Space Mono', monospace; font-weight: 700;
                  white-space: nowrap; color: #ffffff; width: 120px; }
.ref-table .lvl { font-family: 'Space Mono', monospace; color: #7a9ab8;
                  font-size: 12px; white-space: nowrap; width: 170px; }
.ref-table .mng { color: #b0c8e8; }

/* ── Card sub-elements ───────────────────────────────────────────── */
.cline  { display: flex; align-items: baseline;
          justify-content: space-between;
          font-family: 'Space Mono', monospace; }
.ctick  { font-size: 17px; font-weight: 700; color: #ffffff;
          letter-spacing: 0.5px; font-family: 'Rajdhani', sans-serif; }
.cflag  { font-size: 10px; color: #f5a623; border: 1px solid #f5a623;
          border-radius: 4px; padding: 1px 6px;
          margin-left: 8px; vertical-align: middle;
          font-family: 'Space Mono', monospace; }
.cscore { font-size: 20px; font-weight: 700;
          font-family: 'Rajdhani', sans-serif; }
.cbar   { height: 5px; background: #122540; border-radius: 3px;
          margin: 6px 0 5px 0; overflow: hidden; }
.cfill  { height: 100%; border-radius: 3px; }
.csub   { font-family: 'Space Mono', monospace; font-size: 11.5px;
          color: #7a9ab8; display: flex; justify-content: space-between;
          flex-wrap: wrap; gap: 4px; }
/* ── Sidebar brand header ────────────────────────────────────────── */
[data-testid="stSidebar"] { background: #07111f; border-right: 1px solid #1e3a5f; }
[data-testid="stSidebar"] .stMarkdown p { color: #b0c8e8; font-size: 13px; }
[data-testid="stSidebar"] label { color: #ffffff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important; font-size: 13px; }
[data-testid="stSidebar"] .stSlider span { color: #b0c8e8 !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label { color: #ffffff !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #ffffff !important; }
[data-testid="stSidebar"] .stTextArea label { color: #ffffff !important; }
[data-testid="stSidebar"] .stNumberInput label { color: #ffffff !important; }
[data-testid="stSidebar"] .stToggle label { color: #ffffff !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {
    border-color: #1e3a5f; margin: 10px 0; }
.sidebar-section {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #7a9ab8;
    margin: 14px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e3a5f;
}
/* ── Streamlit caption color override ───────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] { color: #4a6a8a !important; }
/* ── Progress bar brand color ────────────────────────────────────── */
.stProgress > div > div { background-color: #f5a623 !important; }
/* ── Button brand style ──────────────────────────────────────────── */
.stButton button {
    background: #0d1e33; border: 1px solid #1e3a5f; color: #b0c8e8;
    font-family: 'Plus Jakarta Sans', sans-serif;
    border-radius: 6px; transition: all 0.2s;
}
.stButton button:hover { border-color: #f5a623; color: #f5a623; }
/* ── Success / info / error boxes ────────────────────────────────── */
.stSuccess { background: #071a10 !important; border-color: #3ddc84 !important;
             color: #3ddc84 !important; }
.stInfo    { background: #07111f !important; border-color: #1e3a5f !important;
             color: #b0c8e8 !important; }
.stError   { background: #220d0d !important; border-color: #ff3333 !important;
             color: #ff3333 !important; }
/* ── Selectbox / text area backgrounds ───────────────────────────── */
[data-testid="stSelectbox"] > div,
[data-testid="stTextArea"] textarea {
    background: #0d1e33 !important; border-color: #1e3a5f !important;
    color: #ffffff !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Universe presets
# ----------------------------------------------------------------------------
PRESETS = {
    "Precious Metals / Mining": (
        "HL,AG,FSM,EXK,CDE,FCX,HBM,TECK,RIO,BHP,"
        "SCCO,ALB,GLDG,LAC,TMQ,NISTF,CGAU,CRML,AA,"
        "CENX,XME,COPX,SLVR,SILJ,PSLV,GORO,NVA,USAR,CMP,MP"
    ),
    "Energy / Oil / Gas / Uranium": (
        "CCJ,UROY,EU,LEU,NNE,SMR,OKLO,TXNM,GEV,CEG,"
        "NEE,XE,LNG,XOM,CVX,OXY,EQT,DVN,VLO,SHEL,"
        "BP,DTM,CNQ,AROC,FCG,XLE,XLU,UUUU,UUUG,SMUP"
    ),
    "Defense / Aerospace / Space": (
        "NOC,LMT,RTX,AVAV,ASTS,RKLB,RDW,LASR,SKYT,"
        "MOGA,PKE,LPTH,XAR,ITA,KDEF,ATRO,MRCY,VSAT,"
        "SPCX,BBAI,AIPO,PLTR,SYM,TER,EWY"
    ),
    "Semiconductors / Technology": (
        "AMD,MU,TSM,TSEM,INTC,SWKS,QRVO,SNXX,"
        "TTMI,VICR,SMH,SOXX,XSD,IGV,AMZN,MSFT,"
        "GOOGL,GOOG,NOW,ORCL,DT,TENB,CVLT,EXFY,"
        "INFQ,BOTZ,MAGS,ARKK,FTXR,TCAI"
    ),
    "Quantum / AI / Biotech": (
        "IONQ,QBTS,RGTI,QMCO,BBAI,CMPS,BMEA,ADPT,"
        "TXG,CTKB,BETA,MARA,IBIT,DTCR,ASTS,RDW,"
        "LPTH,SKYT,LASR,SMR,NNE,OKLO,XE,STNG,SEA"
    ),
    "Industrials / Infrastructure": (
        "PWR,MTZ,AGX,POWL,STRL,MWA,MLI,TEX,HLIO,NPO,"
        "GTES,KAI,FELE,CSW,AIT,SXI,MIDD,TRS,SNX,TX,"
        "VICR,XTN,XLI,NEWT,KFRC,OXM,EPC,DBI,GIS,KMB"
    ),
    "Income / Dividends / ETFs": (
        "JEPI,QYLD,SPHD,SRET,DIV,KBWD,PGX,GLAD,AGD,"
        "OTF,FSK,ITUB,OCCI,LTC,SPY,IWM,XLF,XLY,VEU,"
        "VT,VTI,VOO,VUG,VXUS,MAGS,ARKK,BWET,STNG,DTCR,FCG"
    ),
    "Critical Minerals / Materials": (
        "ALB,LAC,TMQ,MP,CRML,USAR,CMP,CGAU,AA,CENX,"
        "SCCO,FCX,TECK,RIO,BHP,XME,COPX,TMC,FLKR,EWY,"
        "TX,NVA,GORO,UUUU,EU,LEU,NNE,CCJ,UROY,CRMX"
    ),
}

# ETFs excluded from All Presets scan — funds don't have ignition signals
# (no insider buying, no earnings catalyst, no squeeze) and slow the scan.
# Individual stocks only for All Presets mode.
ALL_PRESETS_ETF_EXCLUDE = {
    "ARKK","MAGS","SPY","IWM","XLF","XLI","XLU","XLY","XLE","XME",
    "SOXX","SMH","IGV","BOTZ","DTCR","FTXR","XSD","XTN","COPX","SILJ",
    "SLVR","PSLV","JEPI","QYLD","SPHD","SRET","DIV","KBWD","PGX",
    "VTI","VOO","VUG","VEU","VT","VXUS","VTIAX","VTSAX","VTWAX","VFIAX",
    "VIGAX","ITA","XAR","KDEF","SPCX","EWY","FLKR","IBIT","FCG",
    "BWET","SEA","TCAI","AIPO","SMUP","UUUG","CRMX",
}

POSITIVE_WORDS = [
    "beat", "beats", "surge", "record", "upgrade", "upgraded", "raises",
    "contract", "award", "awarded", "partnership", "approval", "approved",
    "buyback", "acquisition", "acquire", "breakthrough", "expands", "wins",
    "guidance raised", "outperform", "buy rating", "patent", "milestone",
    "merger", "takeover", "deal", "agreement", "selected", "chosen",
    "fda approved", "cleared", "accelerated approval", "positive results",
    "positive data", "phase 3", "exceeded", "top-line", "revenue growth",
]
NEGATIVE_WORDS = [
    "miss", "misses", "downgrade", "downgraded", "cuts", "offering",
    "dilution", "lawsuit", "investigation", "recall", "halts", "delay",
    "bankruptcy", "warning", "sell rating", "underperform", "resigns",
    "rejected", "fda rejection", "clinical hold", "adverse", "fraud",
    "subpoena", "default", "going concern", "lowered guidance",
]

# Catalyst keyword buckets — each maps to a catalyst type for tagging
CATALYST_KEYWORDS = {
    "earnings":    ["earnings", "eps", "revenue beat", "quarterly results",
                    "q1", "q2", "q3", "q4", "fiscal", "guidance", "outlook",
                    "profit", "loss", "surprise"],
    "fda":         ["fda", "food and drug", "pdufa", "nda", "bla", "inda",
                    "clinical trial", "phase 1", "phase 2", "phase 3",
                    "approval", "approved", "clearance", "510k", "drug",
                    "biologics", "clinical hold"],
    "legal":       ["lawsuit", "settlement", "verdict", "litigation",
                    "court", "ruling", "judgment", "class action", "sued",
                    "damages", "injunction", "doj", "sec investigation",
                    "subpoena", "antitrust"],
    "buyout":      ["acquisition", "acquire", "merger", "takeover", "buyout",
                    "going private", "lbo", "strategic review", "sale process",
                    "offer to acquire", "bid for", "deal with", "m&a"],
    "partnership": ["partnership", "collaboration", "joint venture", "alliance",
                    "agreement", "contract", "mou", "supply agreement",
                    "licensing deal", "strategic agreement", "selected by"],
    "squeeze":     ["short squeeze", "short interest", "most shorted",
                    "short seller", "short covering", "days to cover"],
    "breakout":    ["52-week high", "all-time high", "breakout", "new high",
                    "technical breakout", "resistance broken", "record high"],
    "geopolitical":["tariff", "sanction", "trade war", "geopolitical",
                    "supply chain", "export ban", "china", "russia", "ukraine",
                    "energy crisis", "oil", "opec", "nato", "war", "conflict",
                    "defense contract", "pentagon"],
    "rate":        ["fed", "federal reserve", "interest rate", "rate hike",
                    "rate cut", "fomc", "powell", "inflation", "cpi", "ppi",
                    "hawkish", "dovish", "treasury yield"],
}

# ── Option 2: minimum keyword hits required before a catalyst tag fires ──
# Prevents a single passing mention from triggering a tag.
# Catalysts with high false-positive risk need more evidence.
CATALYST_MIN_HITS = {
    "earnings":    1,   # very common, 1 hit ok in context
    "fda":         2,   # needs 2 FDA-specific terms (e.g. "fda" + "approval")
    "legal":       2,   # needs 2 legal terms to distinguish from passing refs
    "buyout":      2,   # needs 2 M&A terms (e.g. "acquire" + "merger")
    "partnership": 2,   # "agreement" alone fires too easily
    "squeeze":     1,   # squeeze keywords are specific enough
    "breakout":    1,   # technical terms are specific
    "geopolitical":2,   # "oil" or "china" alone is too broad
    "rate":        2,   # "fed" alone appears in too many general articles
}

# ── Option 1: sector whitelist per catalyst ──────────────────────────────────
# Catalysts only fire for sectors where they are actually meaningful.
# None = fires for ALL sectors (no restriction).
# List = only fires if the stock's sector contains one of these strings.
CATALYST_SECTOR_WHITELIST = {
    "earnings":    None,   # universal — all companies report earnings
    "fda":         [       # only healthcare/pharma/biotech/medical devices
                    "health", "pharma", "biotech", "drug", "life science",
                    "medical", "clinical", "therapeut", "diagnostic",
                    "biolog", "genomic",
                   ],
    "legal":       None,   # any company can face litigation
    "buyout":      None,   # any company can be acquired
    "partnership": None,   # any company can sign deals
    "squeeze":     None,   # short squeeze is universal
    "breakout":    None,   # technical signal, universal
    "geopolitical":[       # sectors directly exposed to macro/trade events
                    "energy", "material", "defense", "industrial",
                    "semiconductor", "technology", "mining", "oil",
                    "chemical", "aerospace", "transport",
                   ],
    "rate":        [       # rate-sensitive sectors only
                    "financial", "bank", "real estate", "reit", "utility",
                    "insurance", "mortgage", "savings", "trust",
                   ],
}

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


# Typical U-shaped intraday volume distribution: fraction of a full day's
# volume that trades in each 30-minute bucket from 9:30 to 16:00 ET.
# Heavy at the open and close, quiet over lunch. Using this instead of a
# linear pace stops RVOL from reading 3-5x on every stock at 9:45 AM.
_VOL_CURVE = [0.13, 0.09, 0.075, 0.065, 0.06, 0.055, 0.05,
              0.05, 0.055, 0.06, 0.07, 0.09, 0.15]
_VOL_CURVE_CUM = []
_acc = 0.0
for _b in _VOL_CURVE:
    _acc += _b
    _VOL_CURVE_CUM.append(_acc)
del _acc, _b


def expected_vol_fraction(minutes_elapsed: int) -> float:
    """Cumulative fraction of a typical day's volume expected by N minutes
    into the session, following the U-shaped curve above."""
    m = max(1, min(int(minutes_elapsed), 390))
    full_buckets = m // 30
    frac = _VOL_CURVE_CUM[full_buckets - 1] if full_buckets > 0 else 0.0
    if full_buckets < len(_VOL_CURVE):
        frac += _VOL_CURVE[full_buckets] * (m - full_buckets * 30) / 30.0
    return max(frac, 0.01)


def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price for intraday bars."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    return (tp * df["Volume"]).cumsum() / cum_vol


# ----------------------------------------------------------------------------
# Data fetch (cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def alpaca_keys():
    """Read Alpaca keys once per server session (cache_resource)."""
    try:
        k = st.secrets.get("ALPACA_API_KEY", "")
        s = st.secrets.get("ALPACA_SECRET_KEY", "")
        if k and s:
            return k, s
    except Exception:
        pass
    return None


@st.cache_resource
def ntfy_config():
    """Read ntfy settings once per server session (cache_resource)."""
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
    except Exception as _ntfy_err:
        st.session_state["ntfy_last_error"] = str(_ntfy_err)
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
    _page_limit = 50  # safety guard: max 50 pages (~500k bars) before bail
    for _ in range(_page_limit):
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
    _ak = alpaca_keys()
    if _ak:
        try:
            k, s = _ak
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
    _ak = alpaca_keys()
    if _ak:
        try:
            k, s = _ak
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


def sector_allows(catalyst: str, sector: str) -> bool:
    """Return True if this catalyst is valid for the stock's sector.
    Option 1: sector whitelist check. None whitelist = universal."""
    whitelist = CATALYST_SECTOR_WHITELIST.get(catalyst)
    if whitelist is None:
        return True  # no restriction
    s = sector.lower()
    return any(w in s for w in whitelist)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fuel(ticker: str) -> dict:
    """Slow-moving 'primed to move' data: short interest, float,
    insider transactions, news, 52-week levels, earnings date,
    sector, institutional ownership, and full catalyst detection."""
    out = {
        # existing
        "short_pct_float": None, "float_shares": None,
        "insider_net_buy_usd": 0.0, "insider_buys": 0, "insider_sells": 0,
        "news_count_48h": 0, "news_sentiment": 0, "latest_headline": "",
        "high_52w": None, "name": ticker,
        "target_mean": None,   # analyst mean price target
        # new catalyst fields
        "earnings_days": None,       # days until next earnings (negative = past)
        "days_to_cover": None,       # short interest / avg daily vol
        "inst_pct": None,            # institutional ownership %
        "sector": "",                # sector string
        "catalyst_tags": [],         # list of detected catalyst types
        "catalyst_score": 0.0,       # 0-100 composite catalyst sub-score
        "bimodal_event": False,      # binary event within ±3 days
        "catalyst_suppressed": [],   # tags detected but filtered out (sector/threshold)
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
        out["target_mean"] = info.get("targetMeanPrice")
        out["sector"] = info.get("sector") or ""

        # Days to cover (short interest / avg daily volume)
        shares_short = info.get("sharesShort")
        avg_vol = info.get("averageVolume") or info.get("averageDailyVolume10Day")
        if shares_short and avg_vol and avg_vol > 0:
            out["days_to_cover"] = round(shares_short / avg_vol, 1)

        # Institutional ownership
        inst = info.get("heldPercentInstitutions")
        if inst is not None:
            out["inst_pct"] = round(float(inst) * 100, 1)

        # Next earnings date
        try:
            cal = tk.calendar
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date") or cal.get("earningsDate")
                if ed is not None:
                    if isinstance(ed, (list, tuple)):
                        ed = ed[0]
                    ed_dt = pd.to_datetime(ed, errors="coerce")
                    if pd.notna(ed_dt):
                        out["earnings_days"] = (ed_dt.date() - datetime.now().date()).days
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                for lbl in ["Earnings Date", "earningsDate"]:
                    if lbl in cal.index:
                        val = cal.loc[lbl].iloc[0]
                        ed_dt = pd.to_datetime(val, errors="coerce")
                        if pd.notna(ed_dt):
                            out["earnings_days"] = (ed_dt.date() - datetime.now().date()).days
                        break
        except Exception:
            pass

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

        # News flow + catalyst detection
        try:
            news = tk.news or []
            now = datetime.now(timezone.utc)
            sent = 0
            count = 0
            latest = ""
            cat_hits: dict[str, int] = {k: 0 for k in CATALYST_KEYWORDS}
            for item in news:
                content = item.get("content", item)
                title = content.get("title", "") or ""
                summary = content.get("summary", "") or ""
                full_text = (title + " " + summary).lower()
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
                    sent += sum(1 for w in POSITIVE_WORDS if w in full_text)
                    sent -= sum(1 for w in NEGATIVE_WORDS if w in full_text)
                # Catalyst scan across all recent news (7 days for catalyst tagging)
                if ts is not None and (now - ts) <= timedelta(days=7):
                    for cat, kws in CATALYST_KEYWORDS.items():
                        cat_hits[cat] += sum(1 for w in kws if w in full_text)
            out["news_count_48h"] = count
            out["news_sentiment"] = sent
            out["latest_headline"] = latest
            # Tag any catalyst with at least 1 keyword hit
            # Apply Option 1 (sector whitelist) + Option 2 (min keyword hits)
            sector = out.get("sector", "")
            out["catalyst_tags"] = [
                c for c, h in cat_hits.items()
                if h >= CATALYST_MIN_HITS.get(c, 1)          # Option 2: threshold
                and sector_allows(c, sector)                   # Option 1: sector
            ]
            # Track which catalysts were suppressed for transparency
            out["catalyst_suppressed"] = [
                c for c, h in cat_hits.items()
                if h > 0 and c not in out["catalyst_tags"]
            ]
        except Exception:
            pass

        # Bimodal event: earnings within ±3 days OR fda/legal catalyst in news
        ed = out["earnings_days"]
        bimodal_news = any(t in out["catalyst_tags"] for t in ("fda", "legal", "buyout"))
        out["bimodal_event"] = (
            (ed is not None and -1 <= ed <= 3) or bimodal_news
        )

        # ------------------------------------------------------------------
        # Catalyst sub-score (0-100)
        # Weights reflect how reliably each catalyst produces a sharp move
        # ------------------------------------------------------------------
        cat_sc = 0.0
        tags = out["catalyst_tags"]

        # Earnings proximity: peak score 2 days before, decays fast after
        if ed is not None:
            if 0 <= ed <= 2:
                cat_sc += 30.0   # imminent earnings = binary event
            elif ed == 3:
                cat_sc += 20.0
            elif 4 <= ed <= 7:
                cat_sc += 12.0
            elif -1 <= ed < 0:
                cat_sc += 15.0   # just reported, still in motion

        if "fda" in tags:
            cat_sc += 25.0       # FDA binary event, often explosive
        if "buyout" in tags:
            cat_sc += 25.0       # M&A premium = instant catalyst
        if "partnership" in tags:
            cat_sc += 15.0
        if "legal" in tags:
            cat_sc += 10.0       # verdict risk, can go either way
        if "squeeze" in tags:
            cat_sc += 10.0
        if "breakout" in tags:
            cat_sc += 8.0
        if "geopolitical" in tags:
            cat_sc += 6.0
        if "rate" in tags:
            cat_sc += 5.0
        if "earnings" in tags and ed is None:
            cat_sc += 8.0       # earnings headlines without a clear date

        # Days-to-cover bonus: >= 5 days = serious squeeze setup
        dtc = out["days_to_cover"]
        if dtc and dtc >= 10:
            cat_sc += 15.0
        elif dtc and dtc >= 5:
            cat_sc += 8.0

        # Bimodal event multiplier: elevates catalyst score across the board
        if out["bimodal_event"]:
            cat_sc = min(cat_sc * 1.25, 100.0)

        out["catalyst_score"] = clamp(cat_sc)

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
        0.18 * short_sc + 0.18 * ins_sc + 0.18 * news_sc
        + 0.08 * float_sc + 0.12 * h52_sc
        + 0.26 * clamp(fuel.get("catalyst_score", 0.0))
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

    # Catalyst tags as readable reasons
    tag_labels = {
        "earnings": f"earnings in {fuel.get('earnings_days')}d" if fuel.get("earnings_days") is not None else "earnings news",
        "fda": "FDA event",
        "legal": "legal catalyst",
        "buyout": "M&A/buyout",
        "partnership": "partnership",
        "squeeze": "squeeze setup",
        "breakout": "breakout news",
        "geopolitical": "geopolitical",
        "rate": "rate catalyst",
    }
    for tag in (fuel.get("catalyst_tags") or []):
        label = tag_labels.get(tag)
        if label:
            s["reasons"].append(label)
    if fuel.get("bimodal_event"):
        s["reasons"].append("BIMODAL EVENT")
    dtc = fuel.get("days_to_cover")
    if dtc and dtc >= 10:
        s["reasons"].append(f"squeeze extreme ({dtc}d)")
    elif dtc and dtc >= 7:
        s["reasons"].append(f"short fuel high ({dtc}d)")
    elif dtc and dtc >= 5:
        s["reasons"].append(f"short fuel mod ({dtc}d)")

    return s


# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
preset = None
screener_mode = False
st.sidebar.markdown(
    "<div style='padding:12px 4px 8px;'>"
    "<div style='font-family:Rajdhani,sans-serif;font-size:20px;font-weight:700;"
    "color:#f5a623;letter-spacing:1px'><a href='https://aiupscalellc.netlify.app/' "
    "target='_blank' rel='noopener' style='color:inherit;text-decoration:none'"
    ">AI UPSCALE</a></div>"
    "</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

st.sidebar.markdown("<div class='sidebar-section'>Watchlist</div>", unsafe_allow_html=True)

# ── Scan ALL presets toggle ──────────────────────────────────────────
all_presets_mode = st.sidebar.toggle(
    "Scan ALL presets",
    value=False,
    key="all_presets_mode",
    help="Combines every ticker across all 8 sector presets into one scan "
         "and shows the top N results by Score. Great for finding the single "
         "best opportunity across your entire watchlist right now.",
)

source = st.sidebar.radio(
    "Watchlist source",
    ["Preset / Custom", "Nightly Screener Top 10"],
    index=0,
    disabled=all_presets_mode,
    help="Preset / Custom: scan a hand-picked sector list. "
         "Nightly Screener Top 10: uses your nightly dump to pre-rank candidates. "
         "Disabled when Scan ALL presets is on.",
)

screener_mode = False
if all_presets_mode:
    # Build deduplicated mega-list across all 8 presets, excluding ETFs
    seen, all_combined = set(), []
    for tkrs in PRESETS.values():
        for t in [x.strip().upper() for x in tkrs.split(",") if x.strip()]:
            if t not in seen and t not in ALL_PRESETS_ETF_EXCLUDE:
                seen.add(t)
                all_combined.append(t)
    tickers = all_combined
    st.sidebar.caption(
        f"ALL PRESETS: {len(tickers)} stocks across "
        f"{len(PRESETS)} sectors (ETFs excluded). "
        f"Top {top_n} shown after scan."
    )
elif source == "Nightly Screener Top 10":
    screener_mode = True
    screener_url = st.sidebar.text_input("Dump URL", value=SCREENER_URL_DEFAULT)
    pool_size = st.sidebar.slider(
        "Candidate pool (pre-ranked from dump)", 20, 80, 40,
        help="The whole dump is pre-ranked by its own nightly signals; the top "
             "N candidates get the live ignition scan.",
    )
    min_price = st.sidebar.number_input(
        "Min price filter", value=1.0, step=0.5,
        help="Excludes stocks below this price from the candidate pool.",
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
            f"Pool: {len(tickers)}."
        )
        with st.sidebar.expander("Detected dump fields"):
            st.write(", ".join(meta["fields"]))
        st.session_state["screener_pre"] = candidates
    except Exception as e:
        st.sidebar.error(f"Could not load nightly dump: {e}")
else:
    preset_names = ["Custom"] + list(PRESETS.keys())
    # Pick a random sector on very first load (not on every rerun)
    if "initial_preset_idx" not in st.session_state:
        st.session_state["initial_preset_idx"] = random.randint(1, len(preset_names) - 1)
    preset = st.sidebar.selectbox(
        "Sector preset",
        preset_names,
        index=st.session_state["initial_preset_idx"],
        help="Select a sector to scan, or choose Custom to enter your own tickers.",
        key="sector_preset",
    )
    # Update stored index so switching presets doesn't snap back on rerun
    st.session_state["initial_preset_idx"] = preset_names.index(preset)
    if preset == "Custom":
        tickers_raw = st.sidebar.text_area(
            "Tickers (comma separated)",
            value="CCJ,SMR,NNE,OKLO,LEU,EU,UUUU,UROY,GEV,CEG",
            height=90,
        )
        tickers = [t.strip().upper() for t in re.split(r"[,\s]+", tickers_raw) if t.strip()][:250]
    else:
        tickers = [t.strip().upper() for t in PRESETS[preset].split(",") if t.strip()]

st.sidebar.markdown("---")
st.sidebar.markdown("<div class='sidebar-section'>Scanner controls</div>", unsafe_allow_html=True)

# ── Top-N results slider ─────────────────────────────────────────────
top_n = st.sidebar.slider(
    "Results to show", 5, 30, 10,
    help="How many stocks to display after the scan, ranked by Score. "
         "Set to 5 to see only the hottest names, 30 for a full board. "
         "Applies to all watchlist modes.",
)

alert_threshold = st.sidebar.slider(
    "Alert score threshold", 40, 95, 65,
    help="A ticker triggers an alert (feed entry + phone push) the first time "
         "its overall Score crosses this line each day. IGNITING alerts fire "
         "regardless of this threshold when all live conditions confirm at once.",
)
view_mode = "Compact (phone)"
show_all_cols = False
st.sidebar.markdown("---")
st.sidebar.markdown("<div class='sidebar-section'>Notifications</div>", unsafe_allow_html=True)
popup_alerts_on = st.sidebar.toggle(
    "Show popup alerts",
    value=False,
    key="popup_alerts_on",
    help="When off (default) the scanner runs silently after each scan. "
         "Turn on to see in-app toast popups for every alert that fires.",
)
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
st.sidebar.markdown("<div class='sidebar-section'>Data feed</div>", unsafe_allow_html=True)
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
APP_VERSION = "v3.2 - icon header"
last_scan = st.session_state.get("last_scan_time", "--:--:--")
st.markdown(
    "<div style='display:flex;align-items:center;gap:14px;margin-bottom:2px'>"
    "<svg width='44' height='44' viewBox='0 0 72 72' fill='none' "
    "xmlns='http://www.w3.org/2000/svg'>"
    "<polyline points='8,36 18,36 24,16 30,56 38,26 44,46 50,36 64,36' "
    "stroke='#f5a623' stroke-width='4' stroke-linecap='round' "
    "stroke-linejoin='round' fill='none'/>"
    "<circle cx='36' cy='36' r='4' fill='#f5a623'/>"
    "</svg>"
    f"<span style='font-family:Space Mono,monospace;font-size:14px;color:#7a9ab8'>"
    f"{APP_VERSION}</span>"
    "</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Scan — runs on first load, watchlist change, or Refresh button press
# ----------------------------------------------------------------------------

# Build a key representing current watchlist so we detect changes
current_watchlist_key = ",".join(sorted(tickers))

# Detect if the watchlist changed since last scan
watchlist_changed = (
    st.session_state.get("last_watchlist_key") != current_watchlist_key
)

# Refresh button (main area, above results)
col_refresh, col_status = st.columns([1, 4])
with col_refresh:
    refresh_clicked = st.button(
        "▶ Refresh scan",
        type="primary",
        help="Run a fresh scan of the current watchlist right now.",
        use_container_width=True,
    )
with col_status:
    scan_count = len(st.session_state.get("last_results", []))
    if scan_count:
        st.markdown(
            f"<div style='padding:8px 0;font-family:Space Mono,monospace;font-size:12px;"
            f"color:#7a9ab8'>{scan_count} tickers scanned</div>",
            unsafe_allow_html=True,
        )

should_scan = refresh_clicked or watchlist_changed or "last_results" not in st.session_state

if should_scan:
    results = []
    progress = st.progress(0.0, text="")
    _scan_label = st.empty()
    for i, t in enumerate(tickers):
        results.append(compute_signals(t))
        progress.progress((i + 1) / max(len(tickers), 1))
        _scan_label.markdown(
            f"<span style='color:#cc0000;font-family:Space Mono,monospace;"
            f"font-size:13px;font-weight:500'>Scanning {t}… ({i+1}/{len(tickers)})</span>",
            unsafe_allow_html=True,
        )
    progress.empty()
    _scan_label.empty()
    st.session_state["last_results"] = results
    st.session_state["last_scan_time"] = datetime.now().strftime("%H:%M:%S")
    st.session_state["last_watchlist_key"] = current_watchlist_key
else:
    results = st.session_state["last_results"]

ok = [r for r in results if not r.get("error")]
failed = [r["ticker"] for r in results if r.get("error")]
ok.sort(key=lambda r: r["score"], reverse=True)

# Apply top-N limit (screener mode, all-presets mode, and normal mode all use it)
if screener_mode:
    scanned_n = len(ok)
    pre_ranks = st.session_state.get("screener_pre")
    if pre_ranks is not None:
        ok = [dict(r) for r in ok[:top_n]]
        for r in ok:
            if r["ticker"] in pre_ranks.index:
                r["pre_rank"] = float(pre_ranks[r["ticker"]])
    else:
        ok = ok[:top_n]
    st.caption(
        f"Nightly Screener mode: pre-ranked the full dump, live-scanned the top "
        f"{scanned_n} candidates, showing the top {top_n} by ignition score."
    )
else:
    ok = ok[:top_n]
    if all_presets_mode:
        st.caption(
            f"All Presets: scanned {len(results)} stocks across "
            f"{len(PRESETS)} sectors (ETFs excluded), showing top {top_n} by Score."
        )

# Register new alerts (only on a fresh scan, not when reusing cached results)
session_key = datetime.now().strftime("%Y%m%d")
for r in (ok if should_scan else []):
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
        if popup_alerts_on:
            st.toast(f"{r['ticker']} {kind.upper()} {r['score']:.0f} — {', '.join(r['reasons'][:3])}")
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
# Stock Analyzer helpers (module-level to avoid re-definition on every rerun)
# ----------------------------------------------------------------------------
def az_pill(label, good):
    bg = "#0d2215" if good is True else ("#220d0d" if good is False else "#1a1500")
    col = "#4dd880" if good is True else ("#ff4444" if good is False else "#d0b040")
    bc  = "#1e6b35" if good is True else ("#a03535" if good is False else "#907020")
    return (f"<span style='display:inline-block;background:{bg};color:{col};"
            f"border:1px solid {bc};border-radius:4px;font-family:Space Mono,monospace;"
            f"font-size:11px;font-weight:500;padding:2px 9px;margin:2px 4px 2px 0'>{label}</span>")


def az_tag(v, hi, lo, fmt="{:.2f}", suffix=""):
    try:
        fv = float(v)
        col = "#4dd880" if fv >= hi else ("#d0b040" if fv >= lo else "#ff4444")
        return f"<span style='font-family:Space Mono,monospace;color:{col}'>{fmt.format(fv)}{suffix}</span>"
    except Exception:
        return "--"


def mrow(label, tooltip, value):
    return (f"<tr style='border-bottom:1px solid #122540'>"
            f"<td style='padding:8px 10px;font-size:13px;color:#b0c8e8;width:40%'>"
            f"<span title='{tooltip}' style='cursor:help;border-bottom:1px dashed #1e3a5f'>{label}</span></td>"
            f"<td style='padding:8px 10px;font-size:13px;font-family:Space Mono,monospace;color:#ffffff'>{value}</td>"
            f"</tr>")


def az_section(title):
    st.markdown(f"<div style='font-family:Rajdhani,sans-serif;font-weight:700;font-size:13px;"
                f"letter-spacing:1px;text-transform:uppercase;color:#f5a623;"
                f"border-bottom:1px solid #1e3a5f;padding-bottom:4px;margin:14px 0 8px'>{title}</div>",
                unsafe_allow_html=True)


def pct_color(v):
    if v is None: return "--"
    col = "#4dd880" if v >= 0 else "#ff4444"
    return f"<span style='font-family:Space Mono,monospace;color:{col}'>{'+' if v >= 0 else ''}{v:.2f}%</span>"


@st.cache_data(ttl=900, show_spinner="Loading analysis…")
def fetch_analyzer(ticker):
    """Fetch full analyst data for the Stock Analyzer section.

    Data priority chain:
      1. Alpaca daily bars for price history (real-time, if keys configured)
      2. yfinance for fundamentals, info fields, and history fallback
      3. Nightly scan dump (stock_data.json.gz) for any fields still missing
    """
    # ── Step 1: Price history — Alpaca first, Yahoo fallback ─────────
    hist = pd.DataFrame()
    _ak = alpaca_keys()
    if _ak:
        try:
            k, s = _ak
            start = (datetime.now(timezone.utc) - timedelta(days=400)).strftime("%Y-%m-%dT%H:%M:%SZ")
            h_alp = _alpaca_bars(ticker, "1Day", start, k, s)
            if h_alp is not None and len(h_alp) >= 50:
                hist = h_alp
        except Exception:
            pass

    # Yahoo fallback for history
    tk = yf.Ticker(ticker)
    if hist.empty:
        try:
            h = tk.history(period="1y", interval="1d")
            if h is not None and not h.empty:
                if isinstance(h.columns, pd.MultiIndex):
                    h.columns = h.columns.get_level_values(0)
                hist = h
        except Exception:
            pass

    # ── Step 2: Fundamentals from yfinance ───────────────────────────
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        pass

    # ── Step 3: Nightly scan dump fallback for missing fields ────────
    # Fields we want that yfinance often omits for smaller/thinner tickers
    SCAN_FIELD_MAP = {
        # scan dump key → info key we'd populate
        "rsi":            "_scan_rsi",
        "short_squeeze":  "_scan_squeeze_score",
        "golden_cross":   "_scan_golden_cross",
        "mfi_sweet_spot": "_scan_mfi",
        "obv":            "_scan_obv",
        "score":          "_scan_composite_score",
        "piotroski":      "_scan_piotroski",
        "price":          "_scan_price",
        "short_pct_float":"shortPercentOfFloat",
        "short_ratio":    "shortRatio",
        "beta":           "beta",
        "pe_ratio":       "trailingPE",
        "forward_pe":     "forwardPE",
        "profit_margin":  "profitMargins",
        "operating_margin":"operatingMargins",
        "roe":            "returnOnEquity",
        "roa":            "returnOnAssets",
        "revenue_growth": "revenueGrowth",
        "earnings_growth":"earningsGrowth",
        "current_ratio":  "currentRatio",
        "market_cap":     "marketCap",
        "float_shares":   "floatShares",
        "target_mean":    "targetMeanPrice",
        "target_low":     "targetLowPrice",
        "target_high":    "targetHighPrice",
    }
    # Fields we still need after yfinance
    missing = [ik for ik in SCAN_FIELD_MAP.values()
               if not ik.startswith("_scan_") and not info.get(ik)]
    if missing:
        try:
            dump = load_screener_dump(SCREENER_URL_DEFAULT)
            sym = ticker.upper()
            if sym in dump.index:
                row = dump.loc[sym]
                for scan_key, info_key in SCAN_FIELD_MAP.items():
                    val = row.get(scan_key) if hasattr(row, "get") else (
                        row[scan_key] if scan_key in row.index else None
                    )
                    if val is not None and (info_key.startswith("_scan_") or not info.get(info_key)):
                        info[info_key] = val
                info["_from_scan_dump"] = True
        except Exception:
            pass

    return info, hist

# ----------------------------------------------------------------------------
# Reference key (rendered in its own tab)
# ----------------------------------------------------------------------------
REFERENCE_KEY = [
    ("Catalyst signals - inside Fuel score", "#f5a623", [
        ("Earnings", "Days until next earnings report. 0-2 days = binary event, highest score. Score decays with distance.", "0-2d = peak score"),
        ("FDA", "Healthcare/Pharma/Biotech only. Requires 2+ FDA-specific keywords (e.g. fda + approval). Sector-gated to prevent false positives on unrelated stocks.", "sector-gated, 2 hits"),
        ("M&A / Buyout", "Any sector. Requires 2+ M&A keywords (e.g. acquire + merger) to avoid single-word false positives.", "2 hits required"),
        ("Partnership", "Any sector. Requires 2+ partnership keywords — single words like agreement are too common alone.", "2 hits required"),
        ("Legal", "Any sector. Requires 2+ legal keywords to confirm a real litigation event.", "2 hits required"),
        ("Squeeze", "Any sector. 1 specific squeeze keyword sufficient — these terms are precise. Backed by DTC gauge.", "1 hit, see DTC"),
        ("Breakout", "Any sector. 1 keyword sufficient — technical terms are unambiguous.", "1 hit"),
        ("Geopolitical", "Energy, Materials, Defense, Industrials, Semis only. Requires 2+ keywords — broad terms like oil or china appear in unrelated news.", "sector-gated, 2 hits"),
        ("Rate", "Financials, Banks, REITs, Utilities, Insurance only. Requires 2+ keywords — fed/inflation appear everywhere.", "sector-gated, 2 hits"),
        ("BIMODAL", "Binary event within 3 days (earnings, FDA, legal) = elevated volatility expected. Score multiplied 1.25x.", "amber badge"),
        ("DTC", "Days to Cover gauge: shares short / avg daily volume. Bar fills red at 10d+ (extreme), amber 7-10d (high), yellow 5-7d (moderate).", "10d+ = extreme"),
        ("Filtered", "Catalysts detected in headlines but blocked by sector whitelist or keyword threshold appear struck-through below the chart. Shows what was caught and why it was filtered.", "transparency"),
    ]),
    ("Price range analysis", "#29b6c8", [
        ("52W Channel", "52-week price channel bar: shows where the current price sits between the year low and high. Red = near 52W low (weak), amber = mid-range, green = near 52W high (strength). Click the position number to understand momentum regime.", "green near high"),
        ("Bollinger Bands", "20-day Bollinger Bands: a volatility channel built from the 20-day moving average ±2 standard deviations. Price near the upper band = overbought/extended. Near the lower band = oversold/potential bounce. Price inside bands = neutral.", "2 std dev band"),
        ("ATR", "Average True Range (14-day): the average daily price range in dollars. Used for stop-loss sizing — place stops 1–1.5x ATR below entry to avoid being shaken out by normal noise. Shows as a % of price for context.", "stop-loss sizing"),
        ("52W High / Low", "Distance to the 52-week high and above the 52-week low. Within 5% of the 52W high = approaching breakout zone. Momentum stocks tend to cluster near their highs.", "5% = breakout zone"),
    ]),
    ("Technical indicators - stock analyzer", "#3ddc84", [
        ("RSI 14d", "Relative Strength Index on daily bars (14-period). Below 30 = oversold. Above 70 = overbought. 45–70 = momentum sweet spot. Used in the Stock Analyzer for the daily trend picture, separate from the RSI5 live signal on 5-minute bars.", "45-70 sweet spot"),
        ("MACD daily", "Moving Average Convergence Divergence on daily bars. MACD line above signal line = buyers in control daily trend. Below = sellers. Used in Stock Analyzer for the multi-week momentum direction.", "line vs signal"),
        ("50-Day MA", "50-day moving average of closing prices. The short-to-medium trend anchor. Price just above the 50MA = ideal low-risk entry zone. Just below = watch for a reclaim. Well below = avoid.", "just above = entry"),
        ("200-Day MA", "200-day moving average. The primary long-term trend line. Golden Cross: 50MA crosses above 200MA = major institutional buy signal. Death Cross: 50MA below 200MA = long-term caution.", "golden vs death"),
        ("Volume ratio", "Today's volume vs 20-day average. Above 1.5x = high participation, confirms price moves. Below 0.5x = low conviction, moves are less reliable.", "1.5x+ = confirms"),
    ]),
    ("Valuation metrics", "#b0c8e8", [
        ("Market Cap", "Share price × shares outstanding. Micro cap <$300M, small <$2B, mid <$10B, large >$10B. Size affects volatility, liquidity, and institutional interest.", "size classification"),
        ("P/E Ratio", "Price-to-Earnings: how much investors pay per $1 of trailing earnings. High P/E = growth expectations priced in. Low P/E = value or declining business. Compare within sector.", "compare in sector"),
        ("Forward P/E", "P/E based on next 12 months estimated earnings. Forward < Trailing = analysts expect earnings growth — good sign. Forward > Trailing = earnings expected to shrink.", "fwd < trail = growth"),
        ("P/B Ratio", "Price-to-Book: price vs balance sheet asset value. Below 1.0 = trading below asset value. Above 3.0 = paying premium for brand, IP, or growth.", "below 1 = cheap"),
        ("P/S Ratio", "Price-to-Sales: useful for pre-profit companies where P/E doesn't apply. Below 2x = reasonable. Above 10x = high growth premium that must be justified.", "below 2x = fair"),
        ("Beta", "Price volatility relative to the S&P 500. 1.0 = moves with market. 1.5 = 50% more volatile. 0.5 = half as volatile. High beta = bigger swings both directions.", "1.5 = high vol"),
        ("Float", "Tradeable shares (excluding insiders and restricted). Smaller float means less supply — any buying pressure creates bigger price moves. Under 20M = explosive.", "under 20M = explosive"),
    ]),
    ("Financial health metrics", "#ffffff", [
        ("Profit Margin", "Net income / revenue. What the company keeps per dollar of sales after ALL expenses. Expanding margin = pricing power. Shrinking = rising costs or competition.", "expanding = strong"),
        ("Operating Margin", "EBIT / revenue. Core business efficiency before interest and taxes. High operating margin with low profit margin = heavy debt load eating into profits.", "core efficiency"),
        ("ROE", "Return on Equity = net income / shareholders equity. Above 15% = strong. Buffett's key metric for durable competitive advantage. Can be inflated by high debt.", "15%+ = strong"),
        ("ROA", "Return on Assets = net income / total assets. Not distorted by debt level. Above 5% = solid. Asset-heavy industries (utilities, pipelines) typically have lower ROA.", "5%+ = solid"),
        ("D/E Ratio", "Debt-to-Equity: total debt / equity. High D/E amplifies both gains and losses. Capital-intensive sectors carry higher D/E by necessity. Rising D/E trend = risk.", "watch the trend"),
        ("Current Ratio", "Current assets / current liabilities. Can the company pay its bills due in the next 12 months? Above 1.5 = comfortable. Below 1.0 = short-term liquidity risk.", "1.5+ = comfortable"),
        ("Revenue Growth", "Year-over-year change in total revenue. Growing top line = expanding business. Consistent double-digit growth = highly attractive to institutional buyers.", "10%+ = strong"),
        ("Earnings Growth", "Year-over-year EPS growth. Growing faster than revenue = increasing efficiency. Shrinking while revenue grows = costs eating into profits.", "10%+ = strong"),
    ]),
    ("The scores", "#ffffff", [
        ("Score", "Overall grade: 60% Ignition + 40% Fuel", "70+ hot, 50+ warm"),
        ("Ignition", "Is money flowing in right now (live, every refresh)", "jumps 30+ pts = act"),
        ("Fuel", "Is the stock primed for a big move (updates hourly)", "high + rising IGN = setup"),
        ("NightlyRank", "Grade from last night's screener dump - yesterday's homework vs today's live Score", "pool picked from top N"),
    ]),
    ("Live ignition signals - 60% of Score", "#f5a623", [
        ("RVOL", "Today's volume vs 20-day norm, adjusted for the U-shaped intraday curve", "2x unusual, 5x explosive"),
        ("Surge", "Last 3 minutes' volume vs the session's average bar - the exact minutes buying hits", "2x+ = surge live"),
        ("Vel %/5m", "Price move over the last 5 minutes", "positive + growing"),
        ("Accel", "Is the velocity itself speeding up or fading", "positive = building"),
        ("VWAP+", "Price above the volume-weighted average price; a reclaim from below is a trigger", "Y = buyers in control"),
        ("HOD", "High of day: NEW = fresh breakout, near = coiling within 0.5%", "NEW = trigger"),
        ("RSI5", "RSI(14) on 5-minute bars", "55-75 thrust, 75+ stretched"),
        ("MACD", "Fresh bullish cross on 5-minute bars confirms a trend flip", "cross = confirmation"),
    ]),
    ("Fuel signals - 40% of Score", "#3ddc84", [
        ("Short%Flt", "Short interest as % of float - forced buyers if price runs", "15%+ = squeeze fuel"),
        ("InsiderNet$", "Insider buys minus sells, last 90 days (SEC Form 4)", "positive = accumulating"),
        ("News48h", "Headlines in the last 48 hours - momentum needs a catalyst", "0 = no sustained move"),
        ("Float", "Tradable shares outstanding - small float means violent moves", "under 50M = explosive"),
        ("52wk dist", "Distance to the 52-week high - momentum lives near highs", "within 5% = best"),
    ]),
    ("Flags and alerts", "#29b6c8", [
        ("IGNITING", "All four fire at once on a flat/up day: RVOL >=2x, surge >=2x, positive velocity, new HOD or VWAP reclaim", "urgent push, orange"),
        ("GAP REV", "Same footprint but the stock is down 4%+ or gapped down 4%+ - a bounce inside a selloff, riskier trade", "high push, teal"),
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

# Banners removed — IGNITING and GAP REV status shown via card icons instead
igniting_now = [r for r in ok if r["igniting"]]
reversals_now = [r for r in ok if r["gap_reversal"]]

# Catalyst tag definitions: label, CSS class, URL builder (lambda ticker -> url)
CATALYST_TAG_META = {
    "earnings":     ("EARNINGS",  "ct-earnings",     lambda t: f"https://finance.yahoo.com/calendar/earnings?symbol={t}"),
    "fda":          ("FDA",       "ct-fda",          lambda t: f"https://www.google.com/search?q={t}+FDA+approval+news&tbm=nws"),
    "buyout":       ("M&A",       "ct-buyout",       lambda t: f"https://www.google.com/search?q={t}+merger+acquisition+buyout&tbm=nws"),
    "legal":        ("LEGAL",     "ct-legal",        lambda t: f"https://www.google.com/search?q={t}+lawsuit+settlement+verdict&tbm=nws"),
    "partnership":  ("PARTNER",   "ct-partnership",  lambda t: f"https://finance.yahoo.com/quote/{t}/news/"),
    "squeeze":      ("SQUEEZE",   "ct-squeeze",      lambda t: f"https://finviz.com/quote.ashx?t={t}"),
    "breakout":     ("BREAKOUT",  "ct-breakout",     lambda t: f"https://finviz.com/quote.ashx?t={t}&ty=c&ta=1&p=d"),
    "geopolitical": ("GEO/MACRO", "ct-geopolitical", lambda t: f"https://www.google.com/search?q={t}+tariff+geopolitical+news&tbm=nws"),
    "rate":         ("FED/RATES", "ct-rate",         lambda t: "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"),
}

BIMODAL_TAG = ("BIMODAL", "ct-bimodal", lambda t: f"https://finance.yahoo.com/calendar/earnings?symbol={t}")
DTC_TAG_CLASS = "ct-dtc"


def catalyst_pill(label: str, css_class: str, url: str) -> str:
    """Return an HTML span pill that is a clickable colored link."""
    return (
        f"<span class='fuel-tag {css_class}'>"
        f"<a href='{url}' target='_blank' rel='noopener'>{label}</a>"
        f"</span>"
    )


def dtc_gauge_pill(ticker: str, dtc: float) -> str:
    """Render DTC as a labeled fuel gauge bar pill.
    Scale: 0d = empty, 15d+ = full. Color shifts low→mid→high."""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    pct = min(dtc / 15.0, 1.0) * 100
    if dtc >= 10:
        color = "#ff3333"   # red  = extreme squeeze fuel
        label = "SQUEEZE"
        intensity = "EXTREME"
    elif dtc >= 7:
        color = "#f5a623"   # amber = high squeeze fuel
        label = "SHORT"
        intensity = "HIGH"
    elif dtc >= 5:
        color = "#d0b040"   # yellow = moderate
        label = "SHORT"
        intensity = "MOD"
    else:
        color = "#5090d0"   # blue = low
        label = "SHORT"
        intensity = "LOW"
    bar = (
        f"<span class='dtc-bar-track'>"
        f"<span class='dtc-bar-fill' style='width:{pct:.0f}%;background:{color}'></span>"
        f"</span>"
    )
    return (
        f"<span class='dtc-gauge' style='border-color:{color};color:{color}'>"
        f"<a href='{url}' target='_blank' rel='noopener'>"
        f"{label} {bar} {intensity} <span style='color:#7a9ab8;font-size:10px'>({dtc}d)</span>"
        f"</a></span>"
    )


def build_catalyst_tags_html(ticker: str, cat_tags: list, bimodal: bool,
                              dtc=None, earnings_days=None) -> str:
    """Build the full row of colored clickable catalyst pills for a ticker."""
    parts = []
    if bimodal:
        label, css, url_fn = BIMODAL_TAG
        parts.append(catalyst_pill(label, css, url_fn(ticker)))
    for tag in cat_tags:
        if tag not in CATALYST_TAG_META:
            continue
        label, css, url_fn = CATALYST_TAG_META[tag]
        if tag == "earnings" and earnings_days is not None:
            label = f"EARN {earnings_days}d" if earnings_days >= 0 else f"EARN -{abs(earnings_days)}d"
        parts.append(catalyst_pill(label, css, url_fn(ticker)))
    if dtc and dtc >= 5:
        parts.append(dtc_gauge_pill(ticker, dtc))
    return "".join(parts)


def _score_icon(sc, igniting, gap_rev):
    """Return an SVG icon that communicates signal strength without a number.

    IGNITING  → animated pulsing flame  (amber)
    GAP REV   → wave/bounce arc         (teal)
    Score 70+ → lightning bolt          (amber, solid)
    Score 50+ → trending arrow up       (amber, outline)
    Score <50 → flat dash               (grey)
    """
    if igniting:
        # Flame — three teardrop paths
        return (
            "<svg width='28' height='28' viewBox='0 0 28 28' fill='none' "
            "xmlns='http://www.w3.org/2000/svg' aria-label='Igniting'>"
            "<style>@keyframes fp{0%,100%{opacity:1}50%{opacity:.55}}"
            ".fp{animation:fp 1.4s ease-in-out infinite}</style>"
            "<path class='fp' d='M14 3C14 3 8 9 8 15a6 6 0 0012 0c0-2.5-1.5-5-2.5-6.5"
            "C17 10 16 11.5 14 12c1-2.5.5-6-0-9z' fill='#f5a623'/>"
            "<path class='fp' style='animation-delay:.2s' d='M14 14c0 0-3 1.5-3 4a3 3 0 006 0"
            "c0-1.5-1-2.8-1.5-3.5C15.5 15.5 15 16.5 14 17c.5-1.2.3-2.5 0-3z' fill='#ff9500'/>"
            "</svg>"
        )
    elif gap_rev:
        # Bounce arc — wave up from bottom
        return (
            "<svg width='28' height='28' viewBox='0 0 28 28' fill='none' "
            "xmlns='http://www.w3.org/2000/svg' aria-label='Gap reversal'>"
            "<path d='M4 20 Q9 8 14 16 Q19 24 24 12' stroke='#29b6c8' "
            "stroke-width='2.5' stroke-linecap='round' fill='none'/>"
            "<path d='M20 10 L24 12 L22 16' stroke='#29b6c8' "
            "stroke-width='2' stroke-linecap='round' stroke-linejoin='round' fill='none'/>"
            "</svg>"
        )
    elif sc >= 70:
        # Lightning bolt — solid amber
        return (
            "<svg width='28' height='28' viewBox='0 0 28 28' fill='none' "
            "xmlns='http://www.w3.org/2000/svg' aria-label='High score'>"
            "<path d='M16 3L7 16h7l-2 9 10-13h-7l2-9z' fill='#f5a623'/>"
            "</svg>"
        )
    elif sc >= 50:
        # Trending up arrow — amber outline
        return (
            "<svg width='28' height='28' viewBox='0 0 28 28' fill='none' "
            "xmlns='http://www.w3.org/2000/svg' aria-label='Rising score'>"
            "<path d='M4 20L11 13l5 4 8-10' stroke='#f5a623' "
            "stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'/>"
            "<path d='M19 7h5v5' stroke='#f5a623' "
            "stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'/>"
            "</svg>"
        )
    else:
        # Flat dash — grey, no momentum
        return (
            "<svg width='28' height='28' viewBox='0 0 28 28' fill='none' "
            "xmlns='http://www.w3.org/2000/svg' aria-label='Low score'>"
            "<path d='M6 14h16' stroke='#4a6a8a' "
            "stroke-width='2.5' stroke-linecap='round'/>"
            "</svg>"
        )


def render_compact(rows_data):
    """Option 2 stat-grid cards: 2×2 metric tiles, icon indicator, no score number."""
    html = []
    for r in rows_data:
        sc   = r["score"]
        ign  = r["igniting"]
        grev = r.get("gap_reversal", False)
        color = "#f5a623" if sc >= 50 else "#4a6a8a"
        hot   = " hot" if ign else (" rev" if grev else "")

        # ── Header flag badge ────────────────────────────────────────
        if ign:
            flag = "<span class='cflag'>IGNITING</span>"
        elif grev:
            flag = "<span class='cflag rev'>GAP REV</span>"
        else:
            flag = ""

        # ── Icon indicator (replaces score number) ───────────────────
        icon_svg = _score_icon(sc, ign, grev)

        # ── Arc gauge (replaces gradient bar + icon) ─────────────────
        # SVG semicircle: arc length of the half-circle path =
        # π × r = π × 40 ≈ 125.7. We use stroke-dasharray=125.7
        # and stroke-dashoffset = 125.7 × (1 - sc/100) to fill.
        arc_len   = 125.7
        fill_len  = arc_len * (sc / 100.0)
        dash_off  = arc_len - fill_len
        # Score color matches the gradient zones
        if sc >= 80:   gc = "#ff2200"
        elif sc >= 65: gc = "#ff6600"
        elif sc >= 50: gc = "#f5a623"
        elif sc >= 35: gc = "#d0b040"
        elif sc >= 20: gc = "#3ddc84"
        else:          gc = "#29b6c8"
        # Gradient id must be unique per card to avoid SVG bleed
        gid = f"ag{abs(hash(r['ticker']))%9999}"
        arc_svg = (
            f"<svg width='80' height='46' viewBox='0 0 90 50' fill='none' "
            f"aria-label='Score {sc:.0f}'>"
            f"<defs><linearGradient id='{gid}' x1='0%' y1='0%' x2='100%' y2='0%'>"
            f"<stop offset='0%' stop-color='#29b6c8'/>"
            f"<stop offset='30%' stop-color='#3ddc84'/>"
            f"<stop offset='60%' stop-color='#f5a623'/>"
            f"<stop offset='100%' stop-color='#ff2200'/>"
            f"</linearGradient></defs>"
            f"<path d='M5 45 A40 40 0 0 1 85 45' stroke='#122540' "
            f"stroke-width='9' stroke-linecap='round' fill='none'/>"
            f"<path d='M5 45 A40 40 0 0 1 85 45' stroke='url(#{gid})' "
            f"stroke-width='9' stroke-linecap='round' fill='none' "
            f"stroke-dasharray='{arc_len:.1f}' stroke-dashoffset='{dash_off:.1f}'/>"
            f"<text x='45' y='43' text-anchor='middle' "
            f"font-family='Space Mono,monospace' font-size='16' font-weight='700' "
            f"fill='{gc}'>{sc:.0f}</text>"
            f"</svg>"
        )

        # ── Price tile ───────────────────────────────────────────────
        price  = r.get("price")
        chg    = r.get("chg_pct")
        if price:
            chg_col = "#3ddc84" if (chg or 0) >= 0 else "#ff3333"
            chg_str = f"<span class='ctile-sub' style='color:{chg_col}'>{chg:+.1f}%</span>" if chg is not None else ""
            price_tile = (
                f"<div class='ctile'>"
                f"<div class='ctile-lbl'>Price</div>"
                f"<div class='ctile-val' style='color:#f5a623'>${price:.2f}</div>"
                f"{chg_str}</div>"
            )
        else:
            price_tile = "<div class='ctile'><div class='ctile-lbl'>Price</div><div class='ctile-val'>--</div></div>"

        # ── Target tile ──────────────────────────────────────────────
        fuel      = r["fuel"]
        tgt       = fuel.get("target_mean")
        if tgt and price and price > 0:
            upside  = (tgt - price) / price * 100
            tgt_col = "#3ddc84" if upside >= 10 else ("#d0b040" if upside >= 0 else "#ff3333")
            sign    = "+" if upside >= 0 else ""
            target_tile = (
                f"<div class='ctile'>"
                f"<div class='ctile-lbl'>Target</div>"
                f"<div class='ctile-val' style='color:{tgt_col}'>${tgt:.2f}</div>"
                f"<div class='ctile-sub' style='color:{tgt_col}'>{sign}{upside:.1f}% upside</div>"
                f"</div>"
            )
        else:
            target_tile = "<div class='ctile'><div class='ctile-lbl'>Target</div><div class='ctile-val' style='color:#4a6a8a'>--</div></div>"

        # ── RVOL tile ────────────────────────────────────────────────
        rvol = r.get("rvol", 0)
        if rvol:
            rvol_col = "#f5a623" if rvol >= 3 else ("#d0b040" if rvol >= 1.5 else "#7a9ab8")
            rvol_lbl = "explosive" if rvol >= 5 else ("high" if rvol >= 2 else ("normal" if rvol >= 0.8 else "low"))
            rvol_tile = (
                f"<div class='ctile'>"
                f"<div class='ctile-lbl'>RVOL</div>"
                f"<div class='ctile-val' style='color:{rvol_col}'>{rvol:.1f}x</div>"
                f"<div class='ctile-sub' style='color:{rvol_col}'>{rvol_lbl}</div>"
                f"</div>"
            )
        else:
            rvol_tile = "<div class='ctile'><div class='ctile-lbl'>RVOL</div><div class='ctile-val' style='color:#4a6a8a'>--</div></div>"

        # ── Catalyst pills ───────────────────────────────────────────
        cat_tags = fuel.get("catalyst_tags") or []
        bimodal  = fuel.get("bimodal_event", False)
        ed       = fuel.get("earnings_days")
        dtc      = fuel.get("days_to_cover")
        tag_html = build_catalyst_tags_html(r["ticker"], cat_tags, bimodal, dtc, ed)

        # ── Assemble card ────────────────────────────────────────────
        parts = [
            f"<div class='crow{hot}'>",
            # Header: ticker + flag left, arc gauge right
            f"<div class='chead'>",
            f"<span><span class='ctick'>{r['ticker']}</span>{flag}</span>",
            f"<div class='cicon' style='margin-top:-6px'>{arc_svg}</div>",
            f"</div>",
            # Stat grid (no bar below header — gauge replaces it)
            f"<div class='cgrid' style='grid-template-columns:1fr 1fr 1fr'>",
            price_tile,
            target_tile,
            rvol_tile,
            f"</div>",
        ]
        if tag_html:
            parts.append(f"<div style='margin-top:2px'>{tag_html}</div>")
        parts.append("</div>")
        html.append("".join(parts))

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

**GAP REVERSAL banner (teal) / high-priority phone alert**

The same live footprint firing while the stock is down 4%+ on the day or gapped down
4%+ at the open - usually a post-earnings flush being bought. This is a bounce attempt
inside a selloff: a real, tradable pattern, but a different and riskier trade than
fresh ignition. Bounces in crushed stocks fail more often than breakouts in strong
ones, which is why it gets its own label instead of the IGNITING banner.

**Alert feed** logs each ticker once per day, the first time it crosses your score
threshold or ignites. Phone pushes mirror the feed when ntfy is configured.

**Chart**: candles are 1-minute bars for the current session; the amber line is VWAP -
price above it means buyers control the session. Volume bars underneath confirm whether
a move has real participation.

**NightlyRank vs Score**: NightlyRank is how the stock graded in last night's screener
dump (yesterday's homework). Score is the live grade. A high NightlyRank with a surging
Ignition number is the combination this tool exists to catch.

*Detects momentum early; does not predict the future. Not financial advice.*
""")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Stock Analyzer  (replaces chart + alert feed)
# ----------------------------------------------------------------------------
st.markdown(
    "<div style='font-family:Rajdhani,sans-serif;font-size:22px;font-weight:700;"
    "color:#ffffff;letter-spacing:.6px;margin:18px 0 4px'>Stock Analyzer</div>",
    unsafe_allow_html=True,
)

if not ok:
    st.info("Run a scan first — then select a stock to analyze.")
else:
    az_col1, az_col2 = st.columns([2, 3])
    with az_col1:
        az_ticker = st.selectbox(
            "Select stock",
            [r["ticker"] for r in ok],
            index=0,
            help="Choose any stock from the current scan results to drill into full analysis.",
            label_visibility="collapsed",
        )

    # ── helper renderers ────────────────────────────────────────────
    # helpers and fetch_analyzer defined at module level

    info, hist = fetch_analyzer(az_ticker)

    # ── Parse core fields ────────────────────────────────────────────
    px    = float(info.get("currentPrice") or info.get("regularMarketPrice") or
                  info.get("previousClose") or 0)
    name  = info.get("shortName") or info.get("longName") or az_ticker
    sec   = info.get("sector") or "Unknown"
    ind   = info.get("industry") or "Unknown"
    mcap  = info.get("marketCap") or 0
    pe    = info.get("trailingPE")
    fwpe  = info.get("forwardPE")
    pb    = info.get("priceToBook")
    ps    = info.get("priceToSalesTrailing12Months")
    beta  = info.get("beta")
    hi52  = info.get("fiftyTwoWeekHigh") or 0
    lo52  = info.get("fiftyTwoWeekLow") or 0
    spf   = info.get("shortPercentOfFloat")
    sratio= info.get("shortRatio")
    am    = info.get("targetMeanPrice")
    al    = info.get("targetLowPrice")
    ahigh = info.get("targetHighPrice")
    nana  = info.get("numberOfAnalystOpinions") or 0
    recky = info.get("recommendationKey") or ""
    rg    = info.get("revenueGrowth")
    eg    = info.get("earningsGrowth")
    pm    = info.get("profitMargins")
    om    = info.get("operatingMargins")
    roe   = info.get("returnOnEquity")
    roa   = info.get("returnOnAssets")
    deq   = info.get("debtToEquity")
    cr    = info.get("currentRatio")
    fl    = info.get("floatShares")
    aus   = ((am - px) / px * 100) if (am and px > 0) else None
    rng52 = ((px - lo52) / (hi52 - lo52) * 100) if (hi52 and lo52 and hi52 != lo52) else None

    # ── Technicals from history ──────────────────────────────────────
    rsi_v = ma50_v = ma200_v = macd_v = macd_s = vol_avg = vol_td = None
    pct1d = pct5d = pct1m = pct3m = atr = None
    bb_upper = bb_mid = bb_lower = None

    if not hist.empty and len(hist) >= 26:
        cl = hist["Close"].dropna()
        vl = hist["Volume"].dropna()
        try:
            dlt = cl.diff(); g = dlt.clip(lower=0).rolling(14).mean()
            ls = (-dlt.clip(upper=0)).rolling(14).mean()
            rs3 = (100 - (100 / (1 + g / ls.replace(0, np.nan)))).dropna()
            rsi_v = float(rs3.iloc[-1]) if not rs3.empty else None
        except Exception: pass
        try:
            e12 = cl.ewm(span=12, adjust=False).mean()
            e26 = cl.ewm(span=26, adjust=False).mean()
            ml = e12 - e26; sl = ml.ewm(span=9, adjust=False).mean()
            macd_v = float(ml.iloc[-1]); macd_s = float(sl.iloc[-1])
        except Exception: pass
        try:
            if len(cl) >= 50:  ma50_v  = float(cl.rolling(50).mean().iloc[-1])
            if len(cl) >= 200: ma200_v = float(cl.rolling(200).mean().iloc[-1])
        except Exception: pass
        try:
            if len(vl) >= 20: vol_avg = float(vl.iloc[-20:].mean()); vol_td = float(vl.iloc[-1])
        except Exception: pass
        try:
            if len(cl) >= 2:  pct1d = (float(cl.iloc[-1]) - float(cl.iloc[-2])) / float(cl.iloc[-2]) * 100
            if len(cl) >= 6:  pct5d = (float(cl.iloc[-1]) - float(cl.iloc[-6])) / float(cl.iloc[-6]) * 100
            if len(cl) >= 22: pct1m = (float(cl.iloc[-1]) - float(cl.iloc[-22])) / float(cl.iloc[-22]) * 100
            if len(cl) >= 66: pct3m = (float(cl.iloc[-1]) - float(cl.iloc[-66])) / float(cl.iloc[-66]) * 100
        except Exception: pass
        try:
            # ATR (14-day Average True Range)
            hi = hist["High"].dropna(); lo = hist["Low"].dropna()
            tr = pd.concat([hi - lo,
                            (hi - cl.shift()).abs(),
                            (lo - cl.shift()).abs()], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
        except Exception: pass
        try:
            # Bollinger Bands (20-day, 2 std)
            if len(cl) >= 20:
                bm = cl.rolling(20).mean()
                bstd = cl.rolling(20).std()
                bb_mid = float(bm.iloc[-1])
                bb_upper = float((bm + 2 * bstd).iloc[-1])
                bb_lower = float((bm - 2 * bstd).iloc[-1])
        except Exception: pass

    # ── Scan result for this ticker ──────────────────────────────────
    scan_r = next((r for r in ok if r["ticker"] == az_ticker), None)
    ign_sc = scan_r["ignition_score"] if scan_r else None
    fuel_sc = scan_r["fuel_score"] if scan_r else None
    rvol = scan_r["rvol"] if scan_r else None
    reasons = scan_r.get("reasons", []) if scan_r else []

    # ── Signal pills ─────────────────────────────────────────────────
    pills_html = ""
    if rsi_v is not None:
        if rsi_v < 30:   pills_html += az_pill("RSI Oversold", True)
        elif rsi_v > 70: pills_html += az_pill("RSI Overbought", False)
        elif 45 < rsi_v < 65: pills_html += az_pill("RSI Sweet Spot", True)
        else:            pills_html += az_pill("RSI Neutral", None)
    if macd_v is not None and macd_s is not None:
        pills_html += az_pill("MACD Bullish" if macd_v > macd_s else "MACD Bearish", macd_v > macd_s)
    if ma50_v and ma200_v:
        pills_html += az_pill("Golden Cross" if ma50_v > ma200_v else "Death Cross", ma50_v > ma200_v)
    if vol_avg and vol_td:
        if vol_td > vol_avg * 1.5: pills_html += az_pill("High Volume", True)
        elif vol_td < vol_avg * 0.5: pills_html += az_pill("Low Volume", None)
    if spf and spf > 0.15: pills_html += az_pill("High Short Interest", None)
    if aus and aus > 15: pills_html += az_pill(f"Analyst Upside {aus:.0f}%", True)
    if scan_r and scan_r.get("igniting"): pills_html += az_pill("IGNITING NOW", True)
    if scan_r and scan_r.get("gap_reversal"): pills_html += az_pill("GAP REVERSAL", False)
    if scan_r and scan_r.get("new_hod"): pills_html += az_pill("New HOD", True)

    # ── Header metrics ───────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Price", f"${px:.2f}" if px else "--")
    m2.metric("Ignition", f"{ign_sc:.0f}" if ign_sc is not None else "--")
    m3.metric("Fuel", f"{fuel_sc:.0f}" if fuel_sc is not None else "--")
    m4.metric("RVOL", f"{rvol:.1f}x" if rvol else "--")
    m5.metric("52W Pos", f"{rng52:.0f}%" if rng52 is not None else "--")
    m6.metric("Target", f"${am:.2f}" if am else "--", delta=f"{aus:.1f}%" if aus else None)

    st.markdown(
        f"<div style='margin:6px 0 4px;font-family:Plus Jakarta Sans,sans-serif'>"
        f"<strong style='color:#ffffff'>{name}</strong>"
        f"  <span style='color:#7a9ab8;font-size:13px'>{sec} / {ind}</span></div>",
        unsafe_allow_html=True,
    )
    if pills_html:
        st.markdown(f"<div style='margin:6px 0 14px'>{pills_html}</div>", unsafe_allow_html=True)

    # Data source badge
    src_parts = []
    _ak_present = alpaca_keys() is not None
    if _ak_present and not hist.empty:
        src_parts.append("<span style='font-family:Space Mono,monospace;font-size:10px;color:#4dd880;border:1px solid #1e6b35;border-radius:3px;padding:1px 7px'>Alpaca history</span>")
    else:
        src_parts.append("<span style='font-family:Space Mono,monospace;font-size:10px;color:#7a9ab8;border:1px solid #1e3a5f;border-radius:3px;padding:1px 7px'>Yahoo history</span>")
    if info.get("_from_scan_dump"):
        src_parts.append("<span style='font-family:Space Mono,monospace;font-size:10px;color:#d0b040;border:1px solid #907020;border-radius:3px;padding:1px 7px'>nightly dump fallback</span>")
    st.markdown("<div style='margin:0 0 10px;display:flex;gap:6px;flex-wrap:wrap'>" + "".join(src_parts) + "</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1e3a5f;margin:12px 0'>", unsafe_allow_html=True)

    # ── Three column layout ──────────────────────────────────────────
    colA, colB, colC = st.columns(3)

    # ── COLUMN A: Price Range Analysis ──────────────────────────────
    with colA:
        az_section("Price Range Analysis")

        # 52-week channel visual
        if hi52 and lo52 and px:
            pct_pos = max(0.0, min(1.0, (px - lo52) / (hi52 - lo52))) if hi52 != lo52 else 0.5
            bar_pct = int(pct_pos * 100)
            # Color: near low=red, mid=amber, near high=green
            bar_col = "#4dd880" if pct_pos > 0.7 else ("#d0b040" if pct_pos > 0.35 else "#ff4444")
            st.markdown(
                f"<div style='background:#0d1e33;border:1px solid #1e3a5f;border-radius:8px;"
                f"padding:14px 16px;margin-bottom:12px'>"
                f"<div style='display:flex;justify-content:space-between;font-family:Space Mono,monospace;"
                f"font-size:11px;color:#7a9ab8;margin-bottom:6px'>"
                f"<span>52W Low ${lo52:.2f}</span><span>52W High ${hi52:.2f}</span></div>"
                f"<div style='background:#122540;border-radius:4px;height:8px;position:relative;overflow:visible'>"
                f"<div style='background:{bar_col};width:{bar_pct}%;height:100%;border-radius:4px'></div>"
                f"<div style='position:absolute;top:-18px;left:calc({bar_pct}% - 1px);"
                f"font-family:Space Mono,monospace;font-size:10px;color:{bar_col}'>"
                f"${px:.2f}</div></div>"
                f"<div style='margin-top:10px;font-family:Space Mono,monospace;font-size:11px;color:#b0c8e8'>"
                f"Position: <strong style='color:{bar_col}'>{rng52:.1f}% of 52W range</strong></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Bollinger Bands channel
        if bb_upper and bb_lower and bb_mid and px:
            bw = bb_upper - bb_lower
            bpos = max(0.0, min(1.0, (px - bb_lower) / bw)) if bw > 0 else 0.5
            bpct = int(bpos * 100)
            bcol = "#ff4444" if bpos > 0.85 else ("#4dd880" if bpos < 0.15 else "#8baac8")
            bb_label = "Near upper band (overbought)" if bpos > 0.85 else ("Near lower band (oversold)" if bpos < 0.15 else "Inside bands (neutral)")
            st.markdown(
                f"<div style='background:#0d1e33;border:1px solid #1e3a5f;border-radius:8px;"
                f"padding:14px 16px;margin-bottom:12px'>"
                f"<div style='display:flex;justify-content:space-between;font-family:Space Mono,monospace;"
                f"font-size:11px;color:#7a9ab8;margin-bottom:6px'>"
                f"<span>BB Lower ${bb_lower:.2f}</span><span>BB Upper ${bb_upper:.2f}</span></div>"
                f"<div style='background:#122540;border-radius:4px;height:8px;position:relative;overflow:visible'>"
                f"<div style='position:absolute;left:50%;width:1px;height:100%;background:#1e3a5f'></div>"
                f"<div style='background:{bcol};width:{bpct}%;height:100%;border-radius:4px'></div>"
                f"<div style='position:absolute;top:-18px;left:calc({bpct}% - 1px);"
                f"font-family:Space Mono,monospace;font-size:10px;color:{bcol}'>${px:.2f}</div></div>"
                f"<div style='margin-top:10px;font-family:Space Mono,monospace;font-size:11px;color:#b0c8e8'>"
                f"Bollinger: <strong style='color:{bcol}'>{bb_label}</strong></div>"
                f"<div style='margin-top:4px;font-family:Space Mono,monospace;font-size:11px;color:#7a9ab8'>"
                f"Mid (20MA): ${bb_mid:.2f} · Width: ${bw:.2f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        az_section("Price Performance")
        perf_rows = [
            mrow("1 Day", "Daily price change vs yesterday's close.", pct_color(pct1d)),
            mrow("5 Day", "Five trading days — roughly one calendar week.", pct_color(pct5d)),
            mrow("1 Month", "~22 trading days. Shows the near-term trend.", pct_color(pct1m)),
            mrow("3 Month", "~66 trading days (one quarter). Used by institutions.", pct_color(pct3m)),
        ]
        st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(perf_rows)}</tbody></table>", unsafe_allow_html=True)

    # ── COLUMN B: Technical Signals + Short/Growth ───────────────────
    with colB:
        az_section("Technical Signals")
        tech_rows = []
        if rsi_v is not None:
            if rsi_v < 30:    ri, rc = "Oversold — potential bounce", "#4dd880"
            elif rsi_v < 45:  ri, rc = "Weak — losing momentum", "#ff4444"
            elif rsi_v < 55:  ri, rc = "Neutral — no clear direction", "#8baac8"
            elif rsi_v < 70:  ri, rc = "Strong — uptrend confirmed", "#4dd880"
            else:             ri, rc = "Overbought — pullback possible", "#ff4444"
            tech_rows.append(mrow("RSI (14d)", "Relative Strength Index 0-100. Below 30 = oversold. Above 70 = overbought. 45-70 = momentum sweet spot.", f"<span style='color:{rc};font-family:Space Mono,monospace'>{rsi_v:.1f}</span> <span style='font-size:11px;color:#7a9ab8'>{ri}</span>"))
        if macd_v is not None:
            mi = "Bullish — momentum building" if macd_v > macd_s else "Bearish — momentum fading"
            mc2 = "#4dd880" if macd_v > macd_s else "#ff4444"
            tech_rows.append(mrow("MACD", "Moving Average Convergence Divergence. MACD above signal line = buyers in control.", f"<span style='color:{mc2};font-family:Space Mono,monospace'>{macd_v:.4f}</span> <span style='font-size:11px;color:#7a9ab8'>{mi}</span>"))
        if ma50_v and px:
            pvs = (px - ma50_v) / ma50_v * 100
            m5i = "Extended — may be overbought" if pvs > 5 else ("Just above — ideal zone" if pvs > 0 else ("Just below — watch reclaim" if pvs > -5 else "Well below — downtrend"))
            m5c = "#4dd880" if 0 < pvs < 5 else ("#d0b040" if pvs > 5 else ("#d0b040" if pvs > -5 else "#ff4444"))
            tech_rows.append(mrow("50-Day MA", "50-Day Moving Average. Price just above = support. Just below = watch for reclaim.", f"${ma50_v:.2f} <span style='color:{m5c};font-size:11px'>({'+' if pvs >= 0 else ''}{pvs:.1f}%)</span>"))
        if ma200_v:
            m2i = "Golden Cross — long-term uptrend" if (ma50_v and ma50_v > ma200_v) else "Death Cross — long-term downtrend"
            m2c = "#4dd880" if (ma50_v and ma50_v > ma200_v) else "#ff4444"
            tech_rows.append(mrow("200-Day MA", "200-Day Moving Average. Golden Cross (50MA above 200MA) = major bull signal.", f"${ma200_v:.2f} <span style='font-size:11px;color:{m2c}'>{m2i}</span>"))
        if vol_avg and vol_td:
            vr = vol_td / vol_avg
            vi = "Very high" if vr > 2 else ("High" if vr > 1.5 else ("Normal" if vr > 0.5 else "Low"))
            vc = "#4dd880" if vr > 1.5 else ("#8baac8" if vr > 0.5 else "#d0b040")
            tech_rows.append(mrow("Volume", "Today's volume vs 20-day average. High volume confirms moves.", f"<span style='color:{vc};font-family:Space Mono,monospace'>{vr:.2f}x avg</span> <span style='font-size:11px;color:#7a9ab8'>{vi}</span>"))
        if ign_sc is not None:
            ic = "#f5a623" if ign_sc >= 70 else ("#c47d0e" if ign_sc >= 50 else "#4a6a8a")
            tech_rows.append(mrow("Ignition Score", "Live momentum score from RVOL, volume surge, velocity, VWAP, HOD, RSI, MACD (0-100).", f"<span style='color:{ic};font-family:Space Mono,monospace;font-size:15px;font-weight:500'>{ign_sc:.0f}</span>"))
        if tech_rows:
            st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(tech_rows)}</tbody></table>", unsafe_allow_html=True)

        az_section("Short Interest & Growth")
        si_rows = []
        if spf:
            si_rows.append(mrow("Short % Float", "% of float sold short. Above 15% = heavy bets against. Above 20% = squeeze setup.", az_tag(spf * 100, 20, 10, "{:.1f}", "%")))
        if sratio:
            si_rows.append(mrow("Days to Cover", "Short interest / avg volume. How many days for shorts to exit. High = squeeze fuel.", az_tag(sratio, 5, 3, "{:.1f}", "d")))
        if rg:
            si_rows.append(mrow("Revenue Growth", "YoY revenue growth. Foundation of long-term stock appreciation.", az_tag(rg * 100, 10, 3, "{:.1f}", "%")))
        if eg:
            si_rows.append(mrow("Earnings Growth", "YoY EPS growth. Earnings growing faster than revenue = increasing efficiency.", az_tag(eg * 100, 10, 3, "{:.1f}", "%")))
        if fuel_sc is not None:
            fc = "#4dd880" if fuel_sc >= 70 else ("#d0b040" if fuel_sc >= 50 else "#4a6a8a")
            si_rows.append(mrow("Fuel Score", "Primed-to-move score: short float, insider buying, news, float size, 52W position (0-100).", f"<span style='color:{fc};font-family:Space Mono,monospace;font-size:15px;font-weight:500'>{fuel_sc:.0f}</span>"))
        if si_rows:
            st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(si_rows)}</tbody></table>", unsafe_allow_html=True)

        az_section("Live Signals")
        if reasons:
            sig_html = "".join(
                f"<span style='display:inline-block;background:#0d1e33;color:#f5a623;"
                f"border:1px solid #1e3a5f;border-radius:4px;font-family:Space Mono,monospace;"
                f"font-size:11px;padding:2px 8px;margin:2px 3px 2px 0'>{r}</span>"
                for r in reasons
            )
            st.markdown(sig_html, unsafe_allow_html=True)
        else:
            st.caption("No active signals this scan.")

    # ── COLUMN C: Valuation + Financial Health + Analyst ────────────
    with colC:
        az_section("Valuation")
        val_rows = []
        if mcap:
            mc_str = f"${mcap/1e9:.2f}B" if mcap >= 1e9 else f"${mcap/1e6:.1f}M"
            val_rows.append(mrow("Market Cap", "Total market value = share price × shares outstanding. Determines if this is micro/small/mid/large cap.", mc_str))
        if pe:
            val_rows.append(mrow("P/E Ratio", "Price-to-Earnings. How much investors pay per dollar of earnings. High P/E = growth expectations. Low P/E = value or declining business.", az_tag(pe, 0, 25, "{:.1f}", "x") if pe < 100 else f"<span style='font-family:Space Mono,monospace;color:#d0b040'>{pe:.1f}x</span>"))
        if fwpe:
            val_rows.append(mrow("Forward P/E", "P/E based on next 12 months estimated earnings. Lower than trailing P/E = analysts expect earnings growth.", f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{fwpe:.1f}x</span>" if fwpe else "--"))
        if pb:
            val_rows.append(mrow("P/B Ratio", "Price-to-Book. Below 1.0 = trading below asset value. Above 3.0 = premium for brand/growth.", az_tag(pb, 0, 3, "{:.2f}", "x")))
        if ps:
            val_rows.append(mrow("P/S Ratio", "Price-to-Sales. Useful for pre-profit companies. Below 2x = reasonable. Above 10x = high growth premium.", f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{ps:.2f}x</span>"))
        if beta:
            val_rows.append(mrow("Beta", "Volatility vs S&P 500. 1.0 = moves with market. 1.5 = 50% more volatile. Under 0.6 = defensive.", f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{beta:.2f}</span>"))
        if fl:
            fl_str = f"{fl/1e9:.2f}B" if fl >= 1e9 else f"{fl/1e6:.0f}M"
            val_rows.append(mrow("Float", "Tradeable shares outstanding. Smaller float = more explosive moves on volume.", f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{fl_str}</span>"))
        if val_rows:
            st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(val_rows)}</tbody></table>", unsafe_allow_html=True)

        az_section("Financial Health")
        hlth_rows = []
        if pm:
            hlth_rows.append(mrow("Profit Margin", "Net income / revenue. How many cents profit per dollar of sales. Expanding = pricing power.", az_tag(pm * 100, 15, 5, "{:.1f}", "%")))
        if om:
            hlth_rows.append(mrow("Operating Margin", "EBIT / revenue. Core business efficiency before interest and taxes.", az_tag(om * 100, 15, 5, "{:.1f}", "%")))
        if roe:
            hlth_rows.append(mrow("Return on Equity", "Net income / shareholders equity. Above 15% = strong. Buffett's key metric.", az_tag(roe * 100, 15, 8, "{:.1f}", "%")))
        if roa:
            hlth_rows.append(mrow("Return on Assets", "Net income / total assets. Not inflated by debt. Above 5% = solid.", az_tag(roa * 100, 8, 3, "{:.1f}", "%")))
        if deq:
            hlth_rows.append(mrow("D/E Ratio", "Total debt / equity. High D/E amplifies gains and losses. Capital-intensive sectors carry more.", f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{deq:.1f}%</span>"))
        if cr:
            hlth_rows.append(mrow("Current Ratio", "Current assets / current liabilities. Above 1.5 = comfortable. Below 1.0 = short-term risk.", az_tag(cr, 1.5, 1.0, "{:.2f}")))
        if hlth_rows:
            st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(hlth_rows)}</tbody></table>", unsafe_allow_html=True)

        if am:
            az_section("Analyst Consensus")
            rcol = "#4dd880" if "buy" in recky.lower() else ("#ff4444" if "sell" in recky.lower() else "#d0b040")
            rdisp = recky.replace("_", " ").title() if recky else "--"
            an_rows = [
                mrow("Recommendation", "Wall Street consensus: Strong Buy / Buy / Hold / Sell. Aggregates all analyst ratings.", f"<span style='color:{rcol};font-weight:500'>{rdisp}</span> <span style='font-size:11px;color:#7a9ab8'>({nana} analysts)</span>"),
                mrow("Mean Target", "Average 12-month analyst price target. Implies expected upside/downside from current price.", f"${am:.2f}" + (f" <span style='font-size:11px;color:{'#4dd880' if aus and aus > 0 else '#ff4444'}'>({'+' if aus and aus >= 0 else ''}{aus:.1f}%)</span>" if aus else "")),
                mrow("Target Range", "Low-to-high analyst target spread. Wide range = high uncertainty. Narrow = strong consensus.", f"${al:.2f} – ${ahigh:.2f}" if (al and ahigh) else "--"),
            ]
            st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(an_rows)}</tbody></table>", unsafe_allow_html=True)

        # Catalyst tags from scan
        f_data = scan_r["fuel"] if scan_r else {}
        cat_tags = f_data.get("catalyst_tags") or []
        bimodal = f_data.get("bimodal_event", False)
        dtc_val = f_data.get("days_to_cover")
        ed = f_data.get("earnings_days")
        if cat_tags or bimodal or (dtc_val and dtc_val >= 5):
            az_section("Catalysts Detected")
            cat_html = build_catalyst_tags_html(az_ticker, cat_tags, bimodal, dtc_val, ed)
            suppressed = f_data.get("catalyst_suppressed") or []
            if suppressed:
                for s in suppressed:
                    cat_html += (
                        f"<span style='font-family:Space Mono,monospace;font-size:10px;"
                        f"color:#7a9ab8;border:1px dashed #1e3a5f;border-radius:4px;"
                        f"padding:1px 6px;margin:2px 3px 2px 0;text-decoration:line-through;"
                        f"display:inline-block'>{s.upper()}</span>"
                    )
            if f_data.get("latest_headline"):
                cat_html += f"<div style='margin-top:8px;font-size:11px;color:#7a9ab8'>{f_data['latest_headline']}</div>"
            st.markdown(cat_html, unsafe_allow_html=True)



# Close the Scanner tab context entered above
tab_scan.__exit__(None, None, None)

# ----------------------------------------------------------------------------
# Auto refresh
# ----------------------------------------------------------------------------
# Auto-refresh removed: scan runs on demand via Refresh button only.
