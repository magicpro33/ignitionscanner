# IGNITION - Momentum Ignition Scanner

Detects the earliest minutes of a momentum move and ranks which stocks
are primed to make one.

## Run locally
```
pip install -r requirements.txt
streamlit run ignition_scanner.py
```

## Deploy
Push both files to a GitHub repo and point Streamlit Cloud at
`ignition_scanner.py` (same workflow as your other apps). Single file,
ASCII-safe, no encoding landmines.

## How the score works

**Score = 60% Ignition + 40% Fuel**

### Ignition (live, recalculated every refresh)
| Signal | Weight | What it catches |
|---|---|---|
| RVOL (today vs 20-day pace) | 25% | Unusual money flowing in |
| Volume surge (last 3 bars vs session avg) | 15% | The exact bars where buying hits |
| Velocity (% move last 5 min) | 15% | Price thrust |
| Acceleration (velocity change) | 10% | Move getting faster, not slower |
| VWAP reclaim / above VWAP | 10% | Institutions defending price |
| New high of day | 10% | Breakout trigger |
| RSI(14) on 5m in 55-75 thrust zone | 7.5% | Strength without exhaustion |
| MACD bullish cross on 5m | 7.5% | Trend flip confirmation |

### Fuel (daily, "primed to move")
| Signal | Weight | What it catches |
|---|---|---|
| Short % of float | 25% | Squeeze fuel |
| Insider net buying, 90d | 25% | People with real information accumulating |
| News flow + keyword sentiment, 48h | 25% | Catalyst present |
| Float size | 10% | Small float = explosive moves |
| Distance to 52-week high | 15% | Momentum regimes cluster near highs |

### IGNITING flag (the "it's happening" alert)
Fires only when ALL of these confirm at once:
- RVOL >= 2x
- Last-3-bar volume surge >= 2x
- Positive 5-minute velocity
- New high of day OR VWAP reclaim within the last few bars

That combination is the classic footprint of the first minutes of a real
momentum leg, and it filters out most low-volume head fakes.

## Honest limits (read this once)
- Yahoo data can be delayed up to ~15 minutes on some tickers. For true
  real-time, swap `fetch_intraday` to your Alpaca data feed (you already
  have the API wired up in claudebot) - the signal math stays identical.
- This detects ignition early; nothing predicts it before it exists.
  Anyone selling "exact moment" prediction is selling snake oil.
- Not financial advice. Position sizing and stops are still on you.

## Easy upgrades
- Alpaca websocket feed for sub-second bars (replace yfinance intraday)
- Push alerts to phone via your Twilio setup when IGNITING fires
- Log every alert to CSV and backtest the threshold that would have
  been profitable on your universe
