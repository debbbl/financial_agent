"""
tools/market_data.py  — PRODUCTION VERSION
Fetches real news from 4 sources with fallback chain:
  1. Finnhub        (company-specific, best quality)
  2. Alpha Vantage  (finance-focused, includes built-in sentiment)
  3. NewsAPI        (broad coverage, good volume)
  4. DuckDuckGo     (free, no API key, last resort)

Only one source needs to work for the app to function.
Set available API keys in .env — unused sources are skipped gracefully.
"""

import os
import time
import json
import hashlib
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

FINNHUB_API_KEY      = os.getenv("FINNHUB_API_KEY", "")
NEWSAPI_API_KEY      = os.getenv("NEWSAPI_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")

_NEWS_CACHE: dict = {}
CACHE_TTL_SECONDS = 3600


@dataclass
class NewsEvent:
    date: str
    title: str
    source: str
    category: str
    sentiment: str
    sentiment_score: float
    impact: str
    url: str = ""
    summary: str = ""


@dataclass
class StockData:
    ticker: str
    prices: pd.DataFrame
    news: list
    current_price: float
    price_change_pct: float


CATEGORY_KEYWORDS = {
    "earnings": ["earnings","revenue","profit","loss","eps","quarterly","annual","guidance","forecast","beat","miss","dividend","margin","fiscal","results","income","ebitda"],
    "product":  ["launch","product","release","feature","update","iphone","ipad","chip","gpu","software","hardware","app","platform","cloud","innovation","patent","technology","device"],
    "management": ["ceo","cfo","cto","chief","executive","president","director","board","resign","appoint","hire","fired","leadership","insider","buyback","acquisition","merger","layoff"],
    "policy":   ["regulation","antitrust","sec","ftc","doj","government","ban","sanction","tariff","trade","export","china","eu","federal","congress","tax","probe","lawsuit","fine"],
    "competition": ["competitor","competition","rival","market share","beats","surpasses","overtakes","amazon","google","microsoft","apple","meta","nvidia","amd","intel","samsung","byd","openai"],
    "market":   ["market","stock","rally","selloff","surge","drop","analyst","upgrade","downgrade","target","rating","investor","fund","etf","index","fed","interest rate","inflation","economy"],
}

BULLISH_KEYWORDS = ["beat","surge","rally","record","growth","profit","strong","bullish","upgrade","buy","outperform","positive","gain","rises","soars","jumps","exceeds","breakthrough","wins","approved"]
BEARISH_KEYWORDS = ["miss","drop","fall","decline","loss","weak","bearish","downgrade","sell","underperform","negative","cut","slumps","plunges","disappoints","concern","risk","probe","lawsuit","ban","layoff","warning"]


def classify_category(text: str) -> str:
    t = (text or "").lower()
    scores = {cat: sum(1 for kw in kws if kw in t) for cat, kws in CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "market"


def score_sentiment(text: str) -> tuple:
    t = (text or "").lower()
    bull = sum(1 for kw in BULLISH_KEYWORDS if kw in t)
    bear = sum(1 for kw in BEARISH_KEYWORDS if kw in t)
    if bull == 0 and bear == 0:
        return "neutral", 0.0
    total = bull + bear
    raw = (bull - bear) / total
    scaled = raw * min(1.0, total / 3)
    label = "bullish" if scaled > 0.15 else "bearish" if scaled < -0.15 else "neutral"
    return label, round(max(-1.0, min(1.0, scaled)), 3)


def derive_impact(score: float, source: str) -> str:
    credible = any(s in source.lower() for s in ["reuters","bloomberg","wsj","ft","cnbc","sec","finnhub"])
    a = abs(score)
    if a >= 0.6 or (a >= 0.4 and credible): return "high"
    if a >= 0.25: return "medium"
    return "low"


def _cache_key(ticker: str, days: int) -> str:
    return hashlib.md5(f"{ticker}_{days}_{datetime.now().strftime('%Y-%m-%d')}".encode()).hexdigest()


def _is_cached(key: str) -> bool:
    if key not in _NEWS_CACHE: return False
    ts, _ = _NEWS_CACHE[key]
    return (time.time() - ts) < CACHE_TTL_SECONDS


def _normalise(raw_events: list) -> list:
    events, seen = [], set()
    for item in raw_events:
        title = (item.get("title") or "").strip()
        if not title or title in seen: continue
        seen.add(title)
        full = title + " " + (item.get("summary") or "")
        sent_label, sent_score = score_sentiment(full)
        category = item.get("category") or classify_category(full)
        if "sentiment" in item and "sentiment_score" in item:
            sent_label = item["sentiment"]
            sent_score = item["sentiment_score"]
        events.append(NewsEvent(
            date=item["date"], title=title,
            source=item.get("source","Unknown"),
            category=category, sentiment=sent_label,
            sentiment_score=sent_score,
            impact=item.get("impact") or derive_impact(sent_score, item.get("source","")),
            url=item.get("url",""), summary=item.get("summary",""),
        ))
    return events


# ── Source 1: Finnhub ─────────────────────────────────────────────────────────
def fetch_from_finnhub(ticker: str, start_date: str, end_date: str) -> list:
    if not FINNHUB_API_KEY: return []
    try:
        resp = requests.get("https://finnhub.io/api/v1/company-news",
            params={"symbol":ticker,"from":start_date,"to":end_date,"token":FINNHUB_API_KEY}, timeout=10)
        resp.raise_for_status()
        articles = resp.json()
        if not isinstance(articles, list): return []
        results = [{"date": datetime.fromtimestamp(a.get("datetime",0)).strftime("%Y-%m-%d"),
                    "title": a.get("headline",""), "source": a.get("source","Finnhub"),
                    "summary": a.get("summary",""), "url": a.get("url","")} for a in articles]
        print(f"[Finnhub] {len(results)} articles")
        return results
    except Exception as e:
        print(f"[Finnhub] Error: {e}"); return []


# ── Source 2: Alpha Vantage ───────────────────────────────────────────────────
AV_SENT_MAP = {
    "Bearish":("bearish",-0.65),"Somewhat-Bearish":("bearish",-0.35),
    "Neutral":("neutral",0.0),"Somewhat-Bullish":("bullish",0.35),"Bullish":("bullish",0.70),
}
AV_TOPIC_MAP = {"earnings":"earnings","financial_markets":"market","mergers_and_acquisitions":"management",
                "ipo":"management","technology":"product"}

def fetch_from_alphavantage(ticker: str, days: int = 90) -> list:
    if not ALPHAVANTAGE_API_KEY: return []
    try:
        time_from = (datetime.now() - timedelta(days=days)).strftime("%Y%m%dT0000")
        resp = requests.get("https://www.alphavantage.co/query",
            params={"function":"NEWS_SENTIMENT","tickers":ticker,"time_from":time_from,
                    "limit":200,"apikey":ALPHAVANTAGE_API_KEY}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "feed" not in data: return []
        results = []
        for a in data["feed"]:
            try: date_str = datetime.strptime(a.get("time_published","")[:8],"%Y%m%d").strftime("%Y-%m-%d")
            except: continue
            ts = next((t for t in a.get("ticker_sentiment",[]) if t.get("ticker")==ticker), None)
            if ts:
                label = ts.get("ticker_sentiment_label","Neutral")
                sl, ss = AV_SENT_MAP.get(label, ("neutral",0.0))
                ss = float(ts.get("ticker_sentiment_score", ss))
            else:
                sl, ss = score_sentiment(a.get("title","")+a.get("summary",""))
            topics = [t.get("topic","") for t in a.get("topics",[])]
            cat = next((AV_TOPIC_MAP[t] for t in topics if t in AV_TOPIC_MAP), "market")
            results.append({"date":date_str,"title":a.get("title",""),
                "source":a.get("source","Alpha Vantage"),"summary":a.get("summary",""),
                "url":a.get("url",""),"category":cat,"sentiment":sl,
                "sentiment_score":round(max(-1.0,min(1.0,ss)),3)})
        print(f"[AlphaVantage] {len(results)} articles")
        return results
    except Exception as e:
        print(f"[AlphaVantage] Error: {e}"); return []


# ── Source 3: NewsAPI ─────────────────────────────────────────────────────────
def fetch_from_newsapi(ticker: str, company_name: str, days: int = 30) -> list:
    if not NEWSAPI_API_KEY: return []
    try:
        from_date = (datetime.now() - timedelta(days=min(days,29))).strftime("%Y-%m-%d")
        params = {"q":f'"{ticker}" stock price OR "{company_name}" earnings OR "{company_name}" investor',"from":from_date,
                  "sortBy":"publishedAt","language":"en","pageSize":100,"apiKey":NEWSAPI_API_KEY,
                  "sources":"bloomberg,reuters,the-wall-street-journal,cnbc,business-insider,fortune,forbes"}
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        if resp.status_code == 426:
            params.pop("sources")
            resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok": return []
        results = []
        required_words = {ticker.lower(), company_name.lower(), "stock", "shares", "earnings", "revenue", "market", "investor", "trading", "nyse", "nasdaq"}
        
        for a in data.get("articles",[]):
            try: date_str = datetime.strptime(a.get("publishedAt","")[:10],"%Y-%m-%d").strftime("%Y-%m-%d")
            except: continue
            title = a.get("title") or ""
            if len(title) < 10 or "[Removed]" in title: continue
            
            title_lower = title.lower()
            if not any(w in title_lower for w in required_words):
                continue
                
            results.append({"date":date_str,"title":title,
                "source":a.get("source",{}).get("name","NewsAPI"),
                "summary":a.get("description",""),"url":a.get("url","")})
        print(f"[NewsAPI] {len(results)} articles")
        return results
    except Exception as e:
        print(f"[NewsAPI] Error: {e}"); return []


# ── Source 4: DuckDuckGo ──────────────────────────────────────────────────────
def fetch_from_duckduckgo(ticker: str, company_name: str, days: int = 30) -> list:
    try:
        query = quote_plus(f"{ticker} {company_name} stock news")
        resp = requests.get(f"https://duckduckgo.com/news.js?q={query}&o=json&l=us-en&s=0&df=m",
            headers={"User-Agent":"Mozilla/5.0 Chrome/120.0.0.0","Accept":"application/json"}, timeout=10)
        resp.raise_for_status()
        text = resp.text.strip()
        if text.startswith("nrj("): text = text[4:-1]
        articles = json.loads(text).get("results",[])
        cutoff = datetime.now() - timedelta(days=days)
        results = []
        for a in articles:
            try:
                dt = datetime.fromtimestamp(a.get("date",0))
                if dt < cutoff: continue
                title = a.get("title","").strip()
                if title:
                    results.append({"date":dt.strftime("%Y-%m-%d"),"title":title,
                        "source":a.get("source","DuckDuckGo"),"summary":a.get("excerpt",""),
                        "url":a.get("url","")})
            except: continue
        print(f"[DuckDuckGo] {len(results)} articles")
        return results
    except Exception as e:
        print(f"[DuckDuckGo] Error: {e}"); return []


# ── Company name lookup ───────────────────────────────────────────────────────
TICKER_NAMES = {
    "AAPL":"Apple","NVDA":"NVIDIA","TSLA":"Tesla","MSFT":"Microsoft",
    "GOOGL":"Google Alphabet","AMZN":"Amazon","META":"Meta Facebook",
    "AMD":"Advanced Micro Devices","INTC":"Intel","NFLX":"Netflix",
    "JPM":"JPMorgan Chase","V":"Visa","WMT":"Walmart","BABA":"Alibaba",
}

def get_company_name(ticker: str) -> str:
    if ticker in TICKER_NAMES: return TICKER_NAMES[ticker]
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except: return ticker


# ── Main orchestrator ─────────────────────────────────────────────────────────
def fetch_news_all_sources(ticker: str, start_date: str, end_date: str, days: int = 90) -> list:
    key = _cache_key(ticker, days)
    if _is_cached(key):
        print(f"[Cache] Returning cached news for {ticker}")
        return _NEWS_CACHE[key][1]

    company = get_company_name(ticker)
    print(f"\nFetching news for {ticker} ({company})...")
    all_raw = []

    all_raw.extend(fetch_from_finnhub(ticker, start_date, end_date))
    all_raw.extend(fetch_from_alphavantage(ticker, days))
    all_raw.extend(fetch_from_newsapi(ticker, company, min(days, 29)))

    if len(all_raw) < 5:
        all_raw.extend(fetch_from_duckduckgo(ticker, company, days))
    else:
        print(f"[DuckDuckGo] Skipped — already have {len(all_raw)} articles")

    all_raw = [r for r in all_raw if start_date <= r.get("date","9999") <= end_date]
    events = _normalise(all_raw)
    events.sort(key=lambda e: e.date)

    print(f"[Summary] {len(events)} unique events for {ticker} ({start_date} → {end_date})\n")
    _NEWS_CACHE[key] = (time.time(), events)
    return events


# ── Public API ────────────────────────────────────────────────────────────────
def fetch_stock_data(ticker: str, period: str = "3mo", start: str = None, end: str = None) -> StockData:
    stock = yf.Ticker(ticker)
    if start and end:
        hist = stock.history(start=start, end=end)
    else:
        hist = stock.history(period=period)
    if hist.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'")

    hist = hist[["Open","High","Low","Close","Volume"]].copy()
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()

    current_price = float(hist["Close"].iloc[-1])
    prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
    price_change_pct = ((current_price - prev_close) / prev_close) * 100

    start_date = hist.index[0].strftime("%Y-%m-%d")
    end_date = hist.index[-1].strftime("%Y-%m-%d")
    days = (hist.index[-1] - hist.index[0]).days + 1

    news = fetch_news_all_sources(ticker, start_date, end_date, days)
    if not news:
        print(f"\n⚠️  No news found for {ticker}. Check API keys in .env\n")

    return StockData(ticker=ticker, prices=hist, news=news,
                     current_price=current_price, price_change_pct=price_change_pct)


def compute_overall_sentiment(news: list) -> float:
    if not news: return 0.0
    weights = {"high":3,"medium":2,"low":1}
    total_w = sum(weights[n.impact] for n in news)
    weighted_sum = sum(n.sentiment_score * weights[n.impact] for n in news)
    return round(weighted_sum / total_w, 3) if total_w else 0.0


def filter_news(news: list, categories: Optional[list] = None, sentiments: Optional[list] = None) -> list:
    result = news
    if categories is not None: result = [n for n in result if n.category in categories]
    if sentiments is not None: result = [n for n in result if n.sentiment in sentiments]
    return result


def get_range_news(news: list, start_date: str, end_date: str) -> list:
    return [n for n in news if start_date <= n.date <= end_date]