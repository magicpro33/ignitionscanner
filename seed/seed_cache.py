#!/usr/bin/env python3
"""
IGNITION Preset Cache Seeder
============================
Runs in GitHub Actions on a schedule. Fetches fuel data (fundamentals,
short interest, news, catalysts) for every ticker in every preset list
using yfinance. Does NOT fetch intraday bars — those need a live market
and are meaningless pre-computed. Stores the results as a compressed JSON
file committed back to the repo.

The app loads this file on startup to pre-populate the fuel cache so
switching presets is instant. Intraday signals (RVOL, surge, VWAP etc.)
are always fetched live — they can't be seeded.

Output: data/preset_fuel_cache.json.gz
"""

import gzip
import json
import sys
import time
from datetime import datetime, timezone

import yfinance as yf

# ── Preset definitions (kept in sync with ignition_scanner.py) ───────
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
        "MOG.A,PKE,LPTH,LHX,GD,HII,ATRO,MRCY,VSAT,"
        "SPCX,BBAI,BKSY,PLTR,SYM,TER,HEI"
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
        "T,VZ,MO,PM,BTI,ENB,ET,EPD,GLAD,AGD,"
        "OTF,FSK,ITUB,OCCI,LTC,O,MAIN,ARCC,NLY,AGNC,"
        "BWET,STNG,DTCR,SEA,TX,OXLC,PFLT,CSWC,GAIN,HTGC"
    ),
    "Critical Minerals / Materials": (
        "ALB,LAC,TMQ,MP,CRML,USAR,CMP,CGAU,AA,CENX,"
        "SCCO,FCX,TECK,RIO,BHP,XME,COPX,TMC,FLKR,EWY,"
        "TX,NVA,GORO,UUUU,EU,LEU,NNE,CCJ,UROY,CRMX"
    ),
}

# Fields we extract from yfinance .info — exactly what fetch_fuel uses
INFO_FIELDS = [
    "shortName", "sector", "industry",
    "shortPercentOfFloat", "floatShares", "sharesShort",
    "averageVolume", "averageDailyVolume10Day",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "targetMeanPrice", "targetLowPrice", "targetHighPrice",
    "numberOfAnalystOpinions", "recommendationKey",
    "trailingPE", "forwardPE", "priceToBook",
    "priceToSalesTrailing12Months", "beta",
    "profitMargins", "operatingMargins",
    "returnOnEquity", "returnOnAssets",
    "debtToEquity", "currentRatio",
    "revenueGrowth", "earningsGrowth",
    "earningsSurprise", "earningsSurprisePercent",
    "epsCurrentYear", "epsNextYear", "forwardEps",
    "dividendRate", "dividendYield", "lastDividendValue",
    "exDividendDate", "dividendDate", "lastDividendDate", "payoutRatio",
    "marketCap", "heldPercentInstitutions",
    "currentPrice", "regularMarketPrice", "previousClose",
]

POSITIVE_WORDS = [
    "beat", "beats", "surge", "record", "upgrade", "upgraded", "raises",
    "contract", "award", "awarded", "partnership", "approval", "approved",
    "buyback", "acquisition", "acquire", "breakthrough", "expands", "wins",
    "merger", "takeover", "deal", "agreement", "selected", "chosen",
    "fda approved", "cleared", "exceeded", "top-line", "revenue growth",
]
NEGATIVE_WORDS = [
    "miss", "misses", "downgrade", "downgraded", "cuts", "offering",
    "dilution", "lawsuit", "investigation", "recall", "halts", "delay",
    "bankruptcy", "warning", "rejected", "fda rejection", "adverse",
    "subpoena", "default", "going concern", "lowered guidance",
]

CATALYST_KEYWORDS = {
    "earnings":    ["earnings", "eps", "revenue beat", "quarterly results",
                    "q1", "q2", "q3", "q4", "fiscal", "guidance", "outlook"],
    "fda":         ["fda", "food and drug", "pdufa", "nda", "bla", "approval",
                    "approved", "clearance", "clinical trial", "phase 3"],
    "legal":       ["lawsuit", "settlement", "verdict", "litigation",
                    "court", "ruling", "class action", "doj", "antitrust"],
    "buyout":      ["acquisition", "acquire", "merger", "takeover", "buyout",
                    "going private", "lbo", "strategic review", "m&a"],
    "partnership": ["partnership", "collaboration", "joint venture", "alliance",
                    "agreement", "contract", "mou", "supply agreement"],
    "squeeze":     ["short squeeze", "short interest", "most shorted",
                    "short covering", "days to cover"],
    "breakout":    ["52-week high", "all-time high", "breakout", "new high",
                    "technical breakout", "record high"],
    "geopolitical":["tariff", "sanction", "trade war", "geopolitical",
                    "supply chain", "export ban", "china", "russia",
                    "defense contract", "pentagon"],
    "rate":        ["fed", "federal reserve", "interest rate", "rate hike",
                    "rate cut", "fomc", "powell", "inflation", "cpi"],
    "earn_growth": ["record earnings", "earnings growth", "eps growth",
                    "profit surge", "earnings beat", "record profit",
                    "blowout quarter", "record quarter", "beat estimates",
                    "exceeded expectations", "top-line beat"],
}
CATALYST_MIN_HITS = {
    "earnings": 1, "fda": 2, "legal": 2, "buyout": 2, "partnership": 2,
    "squeeze": 1, "breakout": 1, "geopolitical": 2, "rate": 2,
    "earn_growth": 1,
}
CATALYST_SECTOR_WHITELIST = {
    "earnings": None, "fda": ["health", "pharma", "biotech", "drug",
    "life science", "medical", "clinical", "therapeut", "biolog"],
    "legal": None, "buyout": None, "partnership": None,
    "squeeze": None, "breakout": None,
    "geopolitical": ["energy", "material", "defense", "industrial",
    "semiconductor", "technology", "mining", "oil", "aerospace"],
    "rate": ["financial", "bank", "real estate", "reit", "utility",
    "insurance", "mortgage"],
    "earn_growth": None,
}


def sector_allows(catalyst, sector):
    wl = CATALYST_SECTOR_WHITELIST.get(catalyst)
    if wl is None:
        return True
    return any(w in sector.lower() for w in wl)


def seed_ticker(ticker: str) -> dict:
    """Fetch all seedable (non-intraday) data for one ticker."""
    out = {
        "ticker": ticker,
        "seeded_at": datetime.now(timezone.utc).isoformat(),
        # info fields
        "info": {},
        # computed fuel fields
        "short_pct_float": None, "float_shares": None,
        "days_to_cover": None, "inst_pct": None,
        "high_52w": None, "name": ticker, "sector": "",
        "target_mean": None,
        "insider_net_buy_usd": 0.0, "insider_buys": 0, "insider_sells": 0,
        "news_count_48h": 0, "news_sentiment": 0, "latest_headline": "",
        "earnings_days": None, "catalyst_tags": [], "catalyst_suppressed": [],
        "catalyst_score": 0.0, "bimodal_event": False,
        "earnings_surprise": None, "data_issues": [],
    }
    try:
        tk = yf.Ticker(ticker)

        # ── Core info ────────────────────────────────────────────────
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        out["info"] = {k: info.get(k) for k in INFO_FIELDS if info.get(k) is not None}
        out["name"]             = info.get("shortName") or ticker
        out["sector"]           = info.get("sector") or ""
        out["short_pct_float"]  = info.get("shortPercentOfFloat")
        out["float_shares"]     = info.get("floatShares")
        out["high_52w"]         = info.get("fiftyTwoWeekHigh")
        out["target_mean"]      = info.get("targetMeanPrice")

        shares_short = info.get("sharesShort")
        avg_vol      = info.get("averageVolume") or info.get("averageDailyVolume10Day")
        if shares_short and avg_vol and avg_vol > 0:
            out["days_to_cover"] = round(shares_short / avg_vol, 1)

        inst = info.get("heldPercentInstitutions")
        if inst is not None:
            out["inst_pct"] = round(float(inst) * 100, 1)

        # ── Earnings date ────────────────────────────────────────────
        try:
            from datetime import datetime as dt_cls
            cal = tk.calendar
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date") or cal.get("earningsDate")
                if ed is not None:
                    if isinstance(ed, (list, tuple)):
                        ed = ed[0]
                    import pandas as pd
                    ed_dt = pd.to_datetime(ed, errors="coerce")
                    if pd.notna(ed_dt):
                        out["earnings_days"] = (ed_dt.date() - dt_cls.now().date()).days
        except Exception:
            pass

        # ── Insider transactions ──────────────────────────────────────
        try:
            import pandas as pd
            ins = tk.insider_transactions
            if ins is not None and len(ins) > 0:
                ins = ins.copy()
                date_col = next((c for c in ["Start Date", "startDate", "Date"] if c in ins.columns), None)
                if date_col:
                    ins[date_col] = pd.to_datetime(ins[date_col], errors="coerce")
                    cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
                    ins = ins[ins[date_col] >= cutoff]
                for _, row in ins.iterrows():
                    txt = str(row.get("Text", "")).lower()
                    tr  = str(row.get("Transaction", "")).lower()
                    val = float(row.get("Value", 0) or 0)
                    if "purchase" in txt or "buy" in tr or "purchase" in tr:
                        out["insider_buys"] += 1
                        out["insider_net_buy_usd"] += abs(val)
                    elif "sale" in txt or "sell" in tr or "sale" in tr:
                        out["insider_sells"] += 1
                        out["insider_net_buy_usd"] -= abs(val)
        except Exception:
            pass

        # ── News + catalysts ──────────────────────────────────────────
        try:
            import pandas as pd
            from datetime import timedelta
            news = tk.news or []
            now  = datetime.now(timezone.utc)
            cat_hits = {k: 0 for k in CATALYST_KEYWORDS}
            sent = count = 0
            latest = ""
            for item in news:
                content   = item.get("content", item)
                title     = content.get("title", "") or ""
                summary   = content.get("summary", "") or ""
                full_text = (title + " " + summary).lower()
                pub = content.get("pubDate") or content.get("providerPublishTime")
                ts  = None
                if isinstance(pub, (int, float)):
                    ts = datetime.fromtimestamp(pub, tz=timezone.utc)
                elif isinstance(pub, str):
                    try:
                        ts = pd.to_datetime(pub, utc=True).to_pydatetime()
                    except Exception:
                        ts = None
                if ts and (now - ts) <= timedelta(hours=48):
                    count += 1
                    if not latest:
                        latest = title
                    sent += sum(1 for w in POSITIVE_WORDS if w in full_text)
                    sent -= sum(1 for w in NEGATIVE_WORDS if w in full_text)
                if ts and (now - ts) <= timedelta(days=7):
                    for cat, kws in CATALYST_KEYWORDS.items():
                        cat_hits[cat] += sum(1 for w in kws if w in full_text)
            out["news_count_48h"]  = count
            out["news_sentiment"]  = sent
            out["latest_headline"] = latest
            sector = out["sector"]
            # Earnings growth fundamental check — mirrors app logic exactly:
            # beat estimates by 10%+ OR YoY EPS growth 25%+ adds 2 hits so
            # the tag fires on fundamentals alone.
            try:
                _eg  = float(info.get("earningsGrowth") or 0.0)
                _esp = float(info.get("earningsSurprise") or info.get("earningsSurprisePercent") or 0.0)
                if abs(_esp) > 5:
                    _esp = _esp / 100.0
                if _esp >= 0.10 or _eg >= 0.25:
                    cat_hits["earn_growth"] = cat_hits.get("earn_growth", 0) + 2
                out["earnings_surprise"] = round(_esp * 100, 1) if _esp else None
            except Exception:
                pass

            out["catalyst_tags"] = [
                c for c, h in cat_hits.items()
                if h >= CATALYST_MIN_HITS.get(c, 1) and sector_allows(c, sector)
            ]
            out["catalyst_suppressed"] = [
                c for c, h in cat_hits.items()
                if h > 0 and c not in out["catalyst_tags"]
            ]
        except Exception:
            pass

        # ── Catalyst score ────────────────────────────────────────────
        cat_sc = 0.0
        tags   = out["catalyst_tags"]
        ed     = out["earnings_days"]
        if ed is not None:
            if 0 <= ed <= 2:   cat_sc += 30.0
            elif ed == 3:       cat_sc += 20.0
            elif 4 <= ed <= 7:  cat_sc += 12.0
            elif -1 <= ed < 0:  cat_sc += 15.0
        if "fda"         in tags: cat_sc += 25.0
        if "buyout"      in tags: cat_sc += 25.0
        if "partnership" in tags: cat_sc += 15.0
        if "legal"       in tags: cat_sc += 10.0
        if "squeeze"     in tags: cat_sc += 10.0
        if "breakout"    in tags: cat_sc += 8.0
        if "geopolitical"in tags: cat_sc += 6.0
        if "rate"        in tags: cat_sc += 5.0
        if "earn_growth" in tags: cat_sc += 20.0
        dtc = out["days_to_cover"]
        if dtc and dtc >= 10:    cat_sc += 15.0
        elif dtc and dtc >= 5:   cat_sc += 8.0
        bimodal = (ed is not None and -1 <= ed <= 3) or any(t in tags for t in ("fda","legal","buyout"))
        if bimodal:
            cat_sc = min(cat_sc * 1.25, 100.0)
        out["catalyst_score"]  = round(cat_sc, 2)
        out["bimodal_event"]   = bimodal

    except Exception as e:
        out["seed_error"] = str(e)

    return out


def main():
    import os
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "preset_fuel_cache.json.gz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Deduplicate tickers across all presets
    all_tickers = {}  # ticker → [preset_name, ...]
    for name, tkrs_raw in PRESETS.items():
        for t in [x.strip().upper() for x in tkrs_raw.split(",") if x.strip()]:
            all_tickers.setdefault(t, []).append(name)

    total = len(all_tickers)
    print(f"Seeding {total} unique tickers across {len(PRESETS)} presets...")

    seeded = {}
    errors = []
    for i, (ticker, presets) in enumerate(all_tickers.items(), 1):
        print(f"  [{i:3d}/{total}] {ticker:<10}", end="", flush=True)
        try:
            data = seed_ticker(ticker)
            seeded[ticker] = data
            status = "err" if data.get("seed_error") else "ok"
            print(f" {status}  cats={data.get('catalyst_tags',[])}")
            if data.get("seed_error"):
                errors.append(ticker)
        except Exception as e:
            print(f" FAIL: {e}")
            errors.append(ticker)
            seeded[ticker] = {"ticker": ticker, "seed_error": str(e)}
        # Be polite to yfinance — avoid rate-limit bans
        time.sleep(0.4)

    # Build output structure
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preset_names": list(PRESETS.keys()),
        "presets": {
            name: [t.strip().upper() for t in tkrs.split(",") if t.strip()]
            for name, tkrs in PRESETS.items()
        },
        "fuel": seeded,
        "stats": {
            "total": total,
            "ok": total - len(errors),
            "errors": len(errors),
            "error_tickers": errors,
        },
    }

    # Write compressed
    payload = json.dumps(output, default=str).encode("utf-8")
    with gzip.open(out_path, "wb", compresslevel=6) as f:
        f.write(payload)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nWrote {out_path}  ({size_kb:.1f} KB)")
    print(f"OK: {output['stats']['ok']}/{total}  Errors: {output['stats']['errors']}")
    if errors:
        print(f"Error tickers: {errors}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
