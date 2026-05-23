#!/usr/bin/env python3
"""
earnings_research.py
Pre-earnings research report generator
Drop into ~/market-analysis and run manually or via cron before earnings week.

Usage:
    python earnings_research.py                        # uses WATCHLIST below
    python earnings_research.py NVDA AMD SOXX NU       # ad-hoc tickers
    python earnings_research.py --no-email             # print to stdout only

Outputs:
    - HTML email via Gmail (same SMTP setup as your existing pipeline)
    - JSON data file in ~/market-analysis/data/earnings_YYYYMMDD.json

Dependencies:
    pip install yfinance anthropic pandas numpy scipy
"""

import sys
import os
import json
import smtplib
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from scipy.stats import norm
import anthropic

# ─── CONFIG ──────────────────────────────────────────────────────────────────

WATCHLIST = ["NVDA", "AMD", "PLTR", "ARM", "NU", "SOXX", "XBI", "SMCI"]

GMAIL_USER   = os.environ.get("GMAIL_USER", "")
GMAIL_PASS   = os.environ.get("GMAIL_APP_PASSWORD", "")
RECIPIENT    = os.environ.get("REPORT_RECIPIENT", GMAIL_USER)
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

DATA_DIR = os.path.expanduser("~/market-analysis/docs")
os.makedirs(DATA_DIR, exist_ok=True)

# ─── DATA COLLECTION ─────────────────────────────────────────────────────────

def get_company_info(ticker: str) -> dict:
    """Company name, description, sector, industry, employees, headquarters, website."""
    t = yf.Ticker(ticker)
    info = t.info or {}

    # Truncate description to ~300 chars for display
    description = info.get("longBusinessSummary", "")
    if len(description) > 320:
        description = description[:317].rsplit(" ", 1)[0] + "…"

    # Exchange-traded funds don't have the same fields
    is_etf = info.get("quoteType", "").upper() in ("ETF", "MUTUALFUND")

    city    = info.get("city", "")
    state   = info.get("state", "")
    country = info.get("country", "")
    hq_parts = [p for p in [city, state, country] if p]
    hq = ", ".join(hq_parts) if hq_parts else None

    employees = info.get("fullTimeEmployees")
    if employees:
        employees = f"{employees:,}"

    fiscal_year_end = info.get("fiscalYearEnd")  # month number
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    fy_end = month_names.get(fiscal_year_end) if fiscal_year_end else None

    return {
        "name":         info.get("longName") or info.get("shortName") or ticker,
        "description":  description,
        "sector":       info.get("sector"),
        "industry":     info.get("industry"),
        "exchange":     info.get("exchange"),
        "quote_type":   info.get("quoteType"),
        "is_etf":       is_etf,
        "headquarters": hq,
        "employees":    employees,
        "website":      info.get("website"),
        "ipo_year":     info.get("ipoExpectedDate") or info.get("firstTradeDateEpochUtc"),
        "fiscal_year_end": fy_end,
        "currency":     info.get("currency", "USD"),
    }


def get_price_data(ticker: str) -> dict:
    """52-week range, trend, recent performance."""
    t = yf.Ticker(ticker)
    hist = t.history(period="1y")
    if hist.empty:
        return {}

    close = hist["Close"]
    current = float(close.iloc[-1])
    high_52w = float(close.max())
    low_52w  = float(close.min())

    # Simple support/resistance: 20/50/200 SMA
    sma20  = float(close.rolling(20).mean().iloc[-1])
    sma50  = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])

    # Recent momentum
    ret_1w  = (current / float(close.iloc[-5])  - 1) * 100 if len(close) >= 5  else None
    ret_1m  = (current / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else None
    ret_3m  = (current / float(close.iloc[-63]) - 1) * 100 if len(close) >= 63 else None

    pct_from_high = (current / high_52w - 1) * 100
    pct_from_low  = (current / low_52w  - 1) * 100

    return {
        "current_price": round(current, 2),
        "52w_high": round(high_52w, 2),
        "52w_low":  round(low_52w, 2),
        "pct_from_52w_high": round(pct_from_high, 1),
        "pct_from_52w_low":  round(pct_from_low, 1),
        "sma_20":  round(sma20, 2),
        "sma_50":  round(sma50, 2),
        "sma_200": round(sma200, 2),
        "above_sma20":  current > sma20,
        "above_sma50":  current > sma50,
        "above_sma200": current > sma200,
        "ret_1w_pct":  round(ret_1w,  1) if ret_1w  is not None else None,
        "ret_1m_pct":  round(ret_1m,  1) if ret_1m  is not None else None,
        "ret_3m_pct":  round(ret_3m,  1) if ret_3m  is not None else None,
    }


def get_earnings_info(ticker: str) -> dict:
    """Next earnings date and analyst estimate data."""
    t = yf.Ticker(ticker)
    info = t.info or {}

    result = {
        "earnings_date": None,
        "eps_estimate": info.get("epsCurrentYear"),
        "eps_forward":  info.get("forwardEps"),
        "revenue_estimate": info.get("revenueEstimate"),   # may be None
        "pe_forward":  info.get("forwardPE"),
        "peg_ratio":   info.get("pegRatio"),
        "revenue_growth_yoy": info.get("revenueGrowth"),
        "earnings_growth_yoy": info.get("earningsGrowth"),
    }

    # Pull next earnings date
    try:
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            now = datetime.now(tz=ed.index.tz)
            upcoming = ed[ed.index > now]
            if not upcoming.empty:
                next_date = upcoming.index[0]
                result["earnings_date"] = next_date.strftime("%Y-%m-%d")
                eps_est = upcoming.iloc[0].get("EPS Estimate")
                if eps_est and not (isinstance(eps_est, float) and np.isnan(eps_est)):
                    result["eps_estimate"] = float(eps_est)
    except Exception:
        pass

    return result


def get_options_signals(ticker: str) -> dict:
    """
    Expected move, IV estimate, put/call ratio.
    Uses nearest expiry AFTER earnings (or nearest expiry if earnings date unknown).
    """
    t = yf.Ticker(ticker)
    info = t.info or {}
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not current_price:
        return {}

    expirations = t.options
    if not expirations:
        return {"note": "No options data available"}

    # Pick best expiry: first one at least 5 days out
    today = datetime.today()
    target_exp = None
    for exp_str in expirations:
        exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
        if (exp_dt - today).days >= 5:
            target_exp = exp_str
            break
    if not target_exp:
        target_exp = expirations[-1]

    try:
        chain = t.option_chain(target_exp)
        calls = chain.calls
        puts  = chain.puts

        # ATM straddle: find strike closest to current price
        atm_strike = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]["strike"].values[0]

        atm_call = calls[calls["strike"] == atm_strike]
        atm_put  = puts [puts ["strike"] == atm_strike]

        call_mid = float((atm_call["bid"].values[0] + atm_call["ask"].values[0]) / 2) if not atm_call.empty else 0
        put_mid  = float((atm_put["bid"].values[0]  + atm_put["ask"].values[0])  / 2) if not atm_put.empty  else 0

        straddle_cost = call_mid + put_mid
        expected_move_pct = (straddle_cost / current_price) * 100

        # Average IV from near-ATM strikes (±5% band)
        low_strike  = current_price * 0.95
        high_strike = current_price * 1.05
        near_calls = calls[(calls["strike"] >= low_strike) & (calls["strike"] <= high_strike)]
        near_puts  = puts [(puts ["strike"] >= low_strike) & (puts ["strike"] <= high_strike)]

        iv_calls = near_calls["impliedVolatility"].dropna()
        iv_puts  = near_puts ["impliedVolatility"].dropna()
        all_iv   = pd.concat([iv_calls, iv_puts])
        avg_iv   = float(all_iv.mean()) * 100 if not all_iv.empty else None

        # Put/call ratio by open interest
        total_call_oi = float(calls["openInterest"].sum())
        total_put_oi  = float(puts["openInterest"].sum())
        pc_ratio = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else None

        # Put/call ratio by volume
        total_call_vol = float(calls["volume"].dropna().sum())
        total_put_vol  = float(puts["volume"].dropna().sum())
        pc_vol_ratio   = round(total_put_vol / total_call_vol, 2) if total_call_vol > 0 else None

        return {
            "expiry_used": target_exp,
            "current_price": round(current_price, 2),
            "atm_strike": atm_strike,
            "straddle_cost": round(straddle_cost, 2),
            "expected_move_pct": round(expected_move_pct, 1),
            "expected_move_up":   round(current_price + straddle_cost, 2),
            "expected_move_down": round(current_price - straddle_cost, 2),
            "avg_iv_pct": round(avg_iv, 1) if avg_iv else None,
            "pc_oi_ratio":  pc_ratio,
            "pc_vol_ratio": pc_vol_ratio,
            "sentiment": (
                "Bearish skew"  if pc_ratio and pc_ratio > 1.2 else
                "Bullish skew"  if pc_ratio and pc_ratio < 0.7 else
                "Neutral"
            ),
        }

    except Exception as e:
        return {"error": str(e)}


def get_fundamental_signals(ticker: str) -> dict:
    """Short interest, analyst ratings, key fundamentals."""
    t = yf.Ticker(ticker)
    info = t.info or {}

    short_ratio    = info.get("shortRatio")        # days to cover
    short_pct_float = info.get("shortPercentOfFloat")
    if short_pct_float:
        short_pct_float = round(float(short_pct_float) * 100, 1)

    # Analyst consensus
    rec = None
    try:
        recs = t.recommendations_summary
        if recs is not None and not recs.empty:
            latest = recs.iloc[0]
            strong_buy  = int(latest.get("strongBuy", 0))
            buy         = int(latest.get("buy", 0))
            hold        = int(latest.get("hold", 0))
            sell        = int(latest.get("sell", 0))
            strong_sell = int(latest.get("strongSell", 0))
            total = strong_buy + buy + hold + sell + strong_sell
            rec = {
                "strong_buy": strong_buy,
                "buy": buy,
                "hold": hold,
                "sell": sell,
                "strong_sell": strong_sell,
                "total_analysts": total,
                "pct_bullish": round((strong_buy + buy) / total * 100, 0) if total else None,
            }
    except Exception:
        pass

    # Price target
    try:
        pt = t.analyst_price_targets
        if pt is not None and not pt.empty and "current" in pt.index:
            price_target = float(pt.loc["current", "mean"]) if "mean" in pt.columns else None
            pt_high = float(pt.loc["current", "high"]) if "high" in pt.columns else None
            pt_low  = float(pt.loc["current", "low"])  if "low"  in pt.columns else None
        else:
            price_target = info.get("targetMeanPrice")
            pt_high = info.get("targetHighPrice")
            pt_low  = info.get("targetLowPrice")
    except Exception:
        price_target = info.get("targetMeanPrice")
        pt_high = info.get("targetHighPrice")
        pt_low  = info.get("targetLowPrice")

    return {
        "short_ratio_days":    short_ratio,
        "short_pct_float":     short_pct_float,
        "market_cap_b":        round(info.get("marketCap", 0) / 1e9, 1) if info.get("marketCap") else None,
        "gross_margins_pct":   round(info.get("grossMargins", 0) * 100, 1) if info.get("grossMargins") else None,
        "operating_margins_pct": round(info.get("operatingMargins", 0) * 100, 1) if info.get("operatingMargins") else None,
        "revenue_growth_yoy_pct": round(info.get("revenueGrowth", 0) * 100, 1) if info.get("revenueGrowth") else None,
        "analyst_consensus": rec,
        "price_target_mean": price_target,
        "price_target_high": pt_high,
        "price_target_low":  pt_low,
    }


def research_ticker(ticker: str) -> dict:
    """Aggregate all signals for a single ticker."""
    print(f"  Pulling data: {ticker}...")
    company = get_company_info(ticker)
    price   = get_price_data(ticker)
    earn    = get_earnings_info(ticker)
    opts    = get_options_signals(ticker)
    fund    = get_fundamental_signals(ticker)

    return {
        "ticker":       ticker,
        "timestamp":    datetime.now().isoformat(),
        "company":      company,
        "price":        price,
        "earnings":     earn,
        "options":      opts,
        "fundamentals": fund,
    }


# ─── CLAUDE SYNTHESIS ────────────────────────────────────────────────────────

def synthesize_with_claude(ticker_data: list[dict]) -> str:
    """Send all ticker data to Claude and get a structured pre-earnings briefing."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    # Build compact summary for each ticker
    summaries = []
    for d in ticker_data:
        co = d.get("company", {})
        summaries.append({
            "ticker":         d["ticker"],
            "company_name":   co.get("name"),
            "sector":         co.get("sector"),
            "industry":       co.get("industry"),
            "is_etf":         co.get("is_etf"),
            "earnings_date":  d["earnings"].get("earnings_date"),
            "price_data":     d["price"],
            "options_signals": d["options"],
            "fundamentals":   d["fundamentals"],
            "earnings_info":  d["earnings"],
        })

    prompt = f"""You are a pre-earnings analyst for an aggressive growth retail investor.
Analyze the following data for {len(summaries)} ticker(s) and produce a structured pre-earnings briefing.

DATA:
{json.dumps(summaries, indent=2)}

For each ticker produce:
1. **Earnings Setup** – date, key metric(s) the market is focused on, expected move %
2. **Options Signals** – IV level (high/moderate/low), P/C ratio interpretation, sentiment skew
3. **Technical Setup** – price vs SMAs, distance from 52w high/low, momentum
4. **Short Interest Risk** – squeeze potential or overhang
5. **Trade Thesis** – directional bias (bullish / bearish / neutral), and ONE specific action:
   - Buy the move (directional), Sell the IV (straddle/strangle), Avoid / wait, or
   - Phased entry setup if volatile name
6. **Key Risk** – what would invalidate the thesis

Close with a 2-sentence "Watch List Priority" ranking the tickers by risk/reward setup quality.

Be direct and specific. No generic disclaimers. Assume the reader has a sophisticated understanding of options, sector ETFs, and tax-location strategy."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# ─── HTML REPORT ─────────────────────────────────────────────────────────────

def build_html_report(ticker_data: list[dict], analysis: str) -> str:
    today = datetime.now().strftime("%A, %B %d, %Y")

    def color_val(val, positive_good=True):
        if val is None: return "<span style='color:#888'>N/A</span>"
        if isinstance(val, bool): return val
        try:
            v = float(val)
            if positive_good:
                c = "#2ecc71" if v > 0 else ("#e74c3c" if v < 0 else "#888")
            else:
                c = "#e74c3c" if v > 0 else ("#2ecc71" if v < 0 else "#888")
            return f"<span style='color:{c}'>{val}</span>"
        except:
            return val

    # Build ticker cards
    cards = ""
    for d in ticker_data:
        t    = d["ticker"]
        c    = d.get("company", {})
        p    = d["price"]
        e    = d["earnings"]
        o    = d["options"]
        f    = d["fundamentals"]

        # Company info banner
        company_name = c.get("name", t)
        sector   = c.get("sector") or ("ETF" if c.get("is_etf") else "—")
        industry = c.get("industry") or "—"
        hq       = c.get("headquarters") or "—"
        employees = c.get("employees") or "—"
        website  = c.get("website") or ""
        fy_end   = c.get("fiscal_year_end") or "—"
        currency = c.get("currency") or "USD"
        description = c.get("description") or ""

        website_link = f'<a href="{website}" style="color:#67e8f9;text-decoration:none">{website.replace("https://","").replace("http://","").rstrip("/")}</a>' if website else "—"

        company_banner = f"""
          <div style="background:#16213e;border-radius:6px;padding:10px 14px;margin-bottom:12px">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">
              <div style="flex:1;min-width:200px">
                <div style="font-size:16px;font-weight:600;color:#e2e8f0">{company_name}</div>
                <div style="margin-top:4px;display:flex;flex-wrap:wrap;gap:6px">
                  {"" if not sector or sector == "—" else f'<span style="background:#1e3a5f;color:#93c5fd;padding:2px 8px;border-radius:3px;font-size:11px">{sector}</span>'}
                  {"" if not industry or industry == "—" else f'<span style="background:#1e3a5f;color:#93c5fd;padding:2px 8px;border-radius:3px;font-size:11px">{industry}</span>'}
                  <span style="background:#1e3a5f;color:#93c5fd;padding:2px 8px;border-radius:3px;font-size:11px">{currency}</span>
                </div>
                {"" if not description else f'<div style="color:#94a3b8;font-size:12px;margin-top:6px;line-height:1.5">{description}</div>'}
              </div>
              <div style="font-size:12px;color:#64748b;text-align:right;white-space:nowrap">
                {"" if hq == "—" else f'<div>🏢 {hq}</div>'}
                {"" if employees == "—" else f'<div>👥 {employees} employees</div>'}
                {"" if fy_end == "—" else f'<div>📅 FY ends {fy_end}</div>'}
                <div>{website_link}</div>
              </div>
            </div>
          </div>"""

        sma_badges = ""
        for sma, label in [("above_sma20","20"), ("above_sma50","50"), ("above_sma200","200")]:
            above = p.get(sma)
            if above is not None:
                bg = "#2ecc71" if above else "#e74c3c"
                sma_badges += f"<span style='background:{bg};color:#fff;padding:2px 6px;border-radius:3px;font-size:11px;margin-right:4px'>SMA{label}</span>"

        sentiment_color = {
            "Bearish skew": "#e74c3c",
            "Bullish skew": "#2ecc71",
            "Neutral":      "#f39c12"
        }.get(o.get("sentiment", ""), "#888")

        cards += f"""
        <div style="background:#1e1e2e;border-radius:8px;padding:16px;margin-bottom:16px;border-left:4px solid #7c3aed">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <h2 style="margin:0;color:#a78bfa;font-size:20px">{t}</h2>
            <div>
              <span style="background:#7c3aed;color:#fff;padding:4px 10px;border-radius:4px;font-size:13px">
                Earnings: {e.get("earnings_date") or "TBD"}
              </span>
            </div>
          </div>
          {company_banner}

          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;font-size:13px">

            <!-- Price / Technical -->
            <div style="background:#2a2a3e;border-radius:6px;padding:10px">
              <div style="color:#888;font-size:11px;text-transform:uppercase;margin-bottom:6px">Price / Technical</div>
              <div style="font-size:18px;font-weight:bold;color:#fff">${p.get('current_price', 'N/A')}</div>
              <div style="margin-top:4px">{sma_badges}</div>
              <table style="width:100%;margin-top:8px;border-collapse:collapse">
                <tr><td style="color:#888">52w High</td><td style="text-align:right">${p.get('52w_high','N/A')} ({color_val(p.get('pct_from_52w_high'), False)}%)</td></tr>
                <tr><td style="color:#888">1W Ret</td><td style="text-align:right">{color_val(p.get('ret_1w_pct'))}%</td></tr>
                <tr><td style="color:#888">1M Ret</td><td style="text-align:right">{color_val(p.get('ret_1m_pct'))}%</td></tr>
                <tr><td style="color:#888">3M Ret</td><td style="text-align:right">{color_val(p.get('ret_3m_pct'))}%</td></tr>
              </table>
            </div>

            <!-- Options Signals -->
            <div style="background:#2a2a3e;border-radius:6px;padding:10px">
              <div style="color:#888;font-size:11px;text-transform:uppercase;margin-bottom:6px">Options Signals</div>
              <div style="font-size:15px;font-weight:bold;color:#fff">
                ±{o.get('expected_move_pct', 'N/A')}% expected move
              </div>
              <div style="color:#aaa;font-size:11px;margin-bottom:6px">
                ${o.get('expected_move_down','—')} ↔ ${o.get('expected_move_up','—')}
              </div>
              <table style="width:100%;border-collapse:collapse">
                <tr><td style="color:#888">IV (ATM avg)</td><td style="text-align:right">{o.get('avg_iv_pct','N/A')}%</td></tr>
                <tr><td style="color:#888">P/C OI Ratio</td><td style="text-align:right">{o.get('pc_oi_ratio','N/A')}</td></tr>
                <tr><td style="color:#888">P/C Vol Ratio</td><td style="text-align:right">{o.get('pc_vol_ratio','N/A')}</td></tr>
                <tr><td style="color:#888">Sentiment</td>
                    <td style="text-align:right;color:{sentiment_color}">{o.get('sentiment','N/A')}</td></tr>
                <tr><td style="color:#888">Expiry</td><td style="text-align:right;font-size:11px">{o.get('expiry_used','N/A')}</td></tr>
              </table>
            </div>

            <!-- Fundamentals -->
            <div style="background:#2a2a3e;border-radius:6px;padding:10px">
              <div style="color:#888;font-size:11px;text-transform:uppercase;margin-bottom:6px">Fundamentals</div>
              <table style="width:100%;border-collapse:collapse">
                <tr><td style="color:#888">Mkt Cap</td><td style="text-align:right">${f.get('market_cap_b','N/A')}B</td></tr>
                <tr><td style="color:#888">Fwd P/E</td><td style="text-align:right">{e.get('pe_forward','N/A')}</td></tr>
                <tr><td style="color:#888">PEG</td><td style="text-align:right">{e.get('peg_ratio','N/A')}</td></tr>
                <tr><td style="color:#888">Rev Growth</td><td style="text-align:right">{color_val(f.get('revenue_growth_yoy_pct'))}%</td></tr>
                <tr><td style="color:#888">Gross Margin</td><td style="text-align:right">{f.get('gross_margins_pct','N/A')}%</td></tr>
                <tr><td style="color:#888">Short Float</td><td style="text-align:right">{f.get('short_pct_float','N/A')}%</td></tr>
                <tr><td style="color:#888">Price Target</td><td style="text-align:right">${f.get('price_target_mean','N/A')}</td></tr>
              </table>
            </div>

          </div>
        </div>"""

    # Convert Claude's markdown-ish text to basic HTML
    analysis_html = (analysis
        .replace("**", "<strong>", 1)
        .replace("**", "</strong>", 1)
    )
    # Simple bold conversion
    import re
    analysis_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', analysis)
    analysis_html = re.sub(r'\n#{1,3} (.+)', r'<h3 style="color:#a78bfa;margin-top:16px">\1</h3>', analysis_html)
    analysis_html = analysis_html.replace("\n\n", "</p><p>").replace("\n", "<br>")
    analysis_html = "<p>" + analysis_html + "</p>"

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: -apple-system, 'Segoe UI', sans-serif; background:#13131f; color:#e2e2f0; margin:0; padding:20px }}
    table td {{ padding: 3px 0; }}
    h3 {{ margin: 12px 0 4px }}
    p {{ line-height: 1.6; margin: 8px 0; }}
  </style>
</head>
<body>
  <div style="max-width:900px;margin:0 auto">
    <div style="border-bottom:1px solid #333;padding-bottom:12px;margin-bottom:20px">
      <h1 style="margin:0;color:#a78bfa">📊 Pre-Earnings Research Brief {" · ".join(d["company"].get("name", d["ticker"]) + " (" + d["ticker"] + ")" for d in ticker_data)}
      <div style="color:#e2e2f0;font-size:15px;margin-top:6px;font-weight:500">{" · ".join(d["company"].get("name", d["ticker"]) + " (" + d["ticker"] + ")" for d in ticker_data)}</div>
      <div style="color:#888;margin-top:4px">{today} · {len(ticker_data)} ticker(s)</div>
    </div>

    {cards}

    <div style="background:#1e1e2e;border-radius:8px;padding:20px;margin-top:8px;border-left:4px solid #06b6d4">
      <h2 style="margin:0 0 12px;color:#67e8f9">🤖 Claude Analysis</h2>
      <div style="font-size:14px;line-height:1.7;color:#cbd5e1">
        {analysis_html}
      </div>
    </div>

    <div style="text-align:center;color:#444;font-size:11px;margin-top:20px;padding-top:12px;border-top:1px solid #222">
      Generated by earnings_research.py · {datetime.now().strftime("%Y-%m-%d %H:%M")} · Data: yfinance
    </div>
  </div>
</body>
</html>"""



# ─── MARKDOWN REPORT ─────────────────────────────────────────────────────────

def na(val, suffix=""):
    """Format a value, returning N/A if None."""
    if val is None:
        return "N/A"
    return f"{val}{suffix}"

def signed(val, suffix=""):
    """Format a numeric value with a +/- sign prefix."""
    if val is None:
        return "N/A"
    try:
        v = float(val)
        prefix = "+" if v > 0 else ""
        return f"{prefix}{val}{suffix}"
    except:
        return str(val)

def sma_status(p):
    parts = []
    for key, label in [("above_sma20","SMA20"), ("above_sma50","SMA50"), ("above_sma200","SMA200")]:
        val = p.get(key)
        if val is not None:
            parts.append(f"{'✅' if val else '❌'} {label}")
    return "  ".join(parts) if parts else "N/A"

def build_md_report(ticker_data: list[dict], analysis: str) -> str:
    today = datetime.now().strftime("%A, %B %d, %Y")
    lines = []

    names_str = " · ".join(
        f'{d["company"].get("name", d["ticker"])} ({d["ticker"]})'
        for d in ticker_data
    )
    lines.append(f"# 📊 Pre-Earnings Research Brief: {names_str}")
    lines.append(f"## {names_str}")
    lines.append(f"*Generated: {today} · {len(ticker_data)} ticker(s) · Data: yfinance*")
    lines.append("")
    lines.append("---")
    lines.append("")

    for d in ticker_data:
        t  = d["ticker"]
        c  = d.get("company", {})
        p  = d["price"]
        e  = d["earnings"]
        o  = d["options"]
        f  = d["fundamentals"]

        # ── Header ──
        company_name = c.get("name", t)
        earnings_date = e.get("earnings_date") or "TBD"
        lines.append(f"## {t} — {company_name}")
        lines.append(f"**Earnings Date:** {earnings_date}")
        lines.append("")

        # ── Company Info ──
        lines.append("### 🏢 Company Info")
        lines.append("")

        info_rows = []
        if c.get("sector"):
            info_rows.append(f"| Sector        | {c['sector']} |")
        if c.get("industry"):
            info_rows.append(f"| Industry      | {c['industry']} |")
        if c.get("headquarters"):
            info_rows.append(f"| Headquarters  | {c['headquarters']} |")
        if c.get("employees"):
            info_rows.append(f"| Employees     | {c['employees']} |")
        if c.get("fiscal_year_end"):
            info_rows.append(f"| FY End        | {c['fiscal_year_end']} |")
        if c.get("exchange"):
            info_rows.append(f"| Exchange      | {c['exchange']} |")
        if c.get("currency"):
            info_rows.append(f"| Currency      | {c['currency']} |")
        if c.get("website"):
            info_rows.append(f"| Website       | {c['website']} |")

        if info_rows:
            lines.append("| Field | Value |")
            lines.append("|-------|-------|")
            lines.extend(info_rows)
            lines.append("")

        if c.get("description"):
            lines.append(f"> {c['description']}")
            lines.append("")

        # ── Price / Technical ──
        lines.append("### 📈 Price & Technical")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Current Price  | ${na(p.get('current_price'))} |")
        lines.append(f"| 52w High       | ${na(p.get('52w_high'))} ({signed(p.get('pct_from_52w_high'))}%) |")
        lines.append(f"| 52w Low        | ${na(p.get('52w_low'))} ({signed(p.get('pct_from_52w_low'))}%) |")
        lines.append(f"| SMA 20/50/200  | ${na(p.get('sma_20'))} / ${na(p.get('sma_50'))} / ${na(p.get('sma_200'))} |")
        lines.append(f"| vs SMAs        | {sma_status(p)} |")
        lines.append(f"| Return 1W      | {signed(p.get('ret_1w_pct'))}% |")
        lines.append(f"| Return 1M      | {signed(p.get('ret_1m_pct'))}% |")
        lines.append(f"| Return 3M      | {signed(p.get('ret_3m_pct'))}% |")
        lines.append("")

        # ── Options Signals ──
        lines.append("### 🎯 Options Signals")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Expected Move  | ±{na(o.get('expected_move_pct'))}% (${na(o.get('expected_move_down'))} ↔ ${na(o.get('expected_move_up'))}) |")
        lines.append(f"| Straddle Cost  | ${na(o.get('straddle_cost'))} |")
        lines.append(f"| IV (ATM avg)   | {na(o.get('avg_iv_pct'))}% |")
        lines.append(f"| P/C OI Ratio   | {na(o.get('pc_oi_ratio'))} |")
        lines.append(f"| P/C Vol Ratio  | {na(o.get('pc_vol_ratio'))} |")
        lines.append(f"| Sentiment      | {na(o.get('sentiment'))} |")
        lines.append(f"| Expiry Used    | {na(o.get('expiry_used'))} |")
        lines.append("")

        # ── Fundamentals ──
        lines.append("### 💰 Fundamentals")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Market Cap     | ${na(f.get('market_cap_b'))}B |")
        lines.append(f"| Fwd P/E        | {na(e.get('pe_forward'))} |")
        lines.append(f"| PEG Ratio      | {na(e.get('peg_ratio'))} |")
        lines.append(f"| EPS Estimate   | {na(e.get('eps_estimate'))} |")
        lines.append(f"| EPS Forward    | {na(e.get('eps_forward'))} |")
        lines.append(f"| Rev Growth YoY | {signed(f.get('revenue_growth_yoy_pct'))}% |")
        lines.append(f"| Gross Margin   | {na(f.get('gross_margins_pct'))}% |")
        lines.append(f"| Op. Margin     | {na(f.get('operating_margins_pct'))}% |")
        lines.append(f"| Short Float    | {na(f.get('short_pct_float'))}% |")
        lines.append(f"| Short Ratio    | {na(f.get('short_ratio_days'))} days |")
        lines.append(f"| Price Target   | ${na(f.get('price_target_mean'))} (low ${na(f.get('price_target_low'))} / high ${na(f.get('price_target_high'))}) |")

        # Analyst consensus
        rec = f.get("analyst_consensus")
        if rec:
            total = rec.get("total_analysts", 0)
            pct_b = rec.get("pct_bullish", "N/A")
            lines.append(f"| Analyst Ratings | {rec.get('strong_buy',0)} str.buy / {rec.get('buy',0)} buy / {rec.get('hold',0)} hold / {rec.get('sell',0)} sell — {pct_b}% bullish ({total} analysts) |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # ── Claude Analysis ──
    lines.append("## 🤖 Claude Analysis")
    lines.append("")
    lines.append(analysis)
    lines.append("")
    lines.append("---")
    lines.append(f"*earnings_research.py · {datetime.now().strftime('%Y-%m-%d %H:%M')} · Data: yfinance*")

    return "\n".join(lines)

# ─── EMAIL SEND ───────────────────────────────────────────────────────────────

def send_email(subject: str, html_body: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = RECIPIENT
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, RECIPIENT, msg.as_string())
    print(f"  ✉️  Email sent to {RECIPIENT}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-earnings research report")
    parser.add_argument("tickers", nargs="*", help="Tickers to research (defaults to WATCHLIST)")
    parser.add_argument("--no-email", action="store_true", help="Print to stdout, skip email")
    parser.add_argument("--no-claude", action="store_true", help="Skip Claude synthesis (faster)")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers] if args.tickers else WATCHLIST
    print(f"\n{'='*55}")
    print(f"  Pre-Earnings Research · {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"{'='*55}\n")

    # Collect data
    all_data = []
    for ticker in tickers:
        try:
            data = research_ticker(ticker)
            all_data.append(data)
        except Exception as e:
            print(f"  ⚠️  {ticker} failed: {e}")

    # Save raw data
    date_str  = datetime.now().strftime("%Y%m%d_%H%M")
    data_path = os.path.join(DATA_DIR, f"earnings_{date_str}.json")
    with open(data_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\n  💾 Raw data saved: {data_path}")

    # Claude synthesis
    analysis = ""
    if not args.no_claude and ANTHROPIC_KEY:
        print("\n  🤖 Running Claude synthesis...")
        try:
            analysis = synthesize_with_claude(all_data)
        except Exception as e:
            analysis = f"Claude synthesis failed: {e}"
            print(f"  ⚠️  {analysis}")
    else:
        analysis = "Claude synthesis skipped (use --no-claude flag or set ANTHROPIC_API_KEY)."

    # Build reports
    html = build_html_report(all_data, analysis)
    md   = build_md_report(all_data, analysis)

    # Always save .md file
    md_path = os.path.join(DATA_DIR, f"earnings_{date_str}.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  📝 Markdown report saved: {md_path}")

    if args.no_email:
        # Print text summary to stdout
        print("\n" + "─"*55)
        for d in all_data:
            o = d["options"]
            e = d["earnings"]
            p = d["price"]
            co = d.get("company", {})
            print(f"\n{d['ticker']} — {co.get('name', d['ticker'])}")
            print(f"  Sector:         {co.get('sector') or 'N/A'}")
            print(f"  Headquarters:   {co.get('headquarters') or 'N/A'}")
            print(f"  Earnings:       {e.get('earnings_date','TBD')}")
            print(f"  Price:          ${p.get('current_price','N/A')}")
            print(f"  Expected Move:  ±{o.get('expected_move_pct','N/A')}%")
            print(f"  IV (avg):       {o.get('avg_iv_pct','N/A')}%")
            print(f"  P/C OI Ratio:   {o.get('pc_oi_ratio','N/A')}")
            print(f"  Sentiment:      {o.get('sentiment','N/A')}")
            print(f"  Short Float:    {d['fundamentals'].get('short_pct_float','N/A')}%")
        print("\n" + "─"*55)
        if analysis:
            print("\nCLAUDE ANALYSIS:\n")
            print(analysis)
    else:
        if not GMAIL_USER or not GMAIL_PASS:
            print("  ⚠️  GMAIL_USER / GMAIL_APP_PASSWORD not set — saving HTML only")
            html_path = os.path.join(DATA_DIR, f"earnings_{date_str}.html")
            with open(html_path, "w") as f:
                f.write(html)
            print(f"  💾 HTML report saved: {html_path}")
        else:
            tickers_str = ", ".join(tickers[:4])
            if len(tickers) > 4:
                tickers_str += f" +{len(tickers)-4} more"
            subject = f"📊 Pre-Earnings Brief: {tickers_str} · {datetime.now().strftime('%b %d')}"
            send_email(subject, html)

    print("\n  ✅ Done.\n")


if __name__ == "__main__":
    main()
