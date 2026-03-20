"""
app.py — Main Streamlit entry point
Financial Agent: Event-Driven Stock Analysis with Agentic AI

Run: streamlit run app.py
"""

import streamlit as st
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

from tools.market_data import (
    fetch_stock_data, filter_news, compute_overall_sentiment,
    get_range_news
)
from agents.financial_agent import FinancialAgent
from ui.chart_builder import (
    build_candlestick_chart, build_sentiment_timeline, build_forecast_gauge
)
from database.db import (
    startup, create_session, update_session, save_news_to_db,
    add_to_watchlist, get_watchlist, get_all_sessions, load_chat_history,
)

load_dotenv()

# ── DB startup (runs once) ────────────────────────────────────────────────────
startup()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f8f9fa;
    border: 0.5px solid #dee2e6;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
  }
  .metric-label { font-size: 12px; color: #6c757d; margin-bottom: 4px; }
  .metric-value { font-size: 22px; font-weight: 600; color: #212529; }
  .metric-sub   { font-size: 12px; font-weight: 500; }
  .bull { color: #3B6D11; } .bear { color: #A32D2D; } .neu  { color: #5F5E5A; }
  .news-card {
    border-left: 3px solid #185FA5;
    padding: 8px 12px;
    margin: 6px 0;
    background: #f8f9fa;
    border-radius: 0 6px 6px 0;
  }
  .cat-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
  }
  .chat-msg-user { background: #e7f3ff; border-radius: 12px; padding: 10px 14px; margin: 6px 0; }
  .chat-msg-ai   { background: #f0fdf4; border-radius: 12px; padding: 10px 14px; margin: 6px 0; }
  .stSelectbox label { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

CAT_BADGE_STYLES = {
    "earnings":    "background:#EAF3DE;color:#3B6D11",
    "product":     "background:#E6F1FB;color:#185FA5",
    "management":  "background:#FAEEDA;color:#854F0B",
    "policy":      "background:#FBEAF0;color:#993556",
    "market":      "background:#EEEDFE;color:#534AB7",
    "competition": "background:#FAECE7;color:#993C1D",
}

POPULAR_TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN"]

# ── Session state init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = create_session()
if "agent" not in st.session_state:
    st.session_state.agent = None
if "stock_data" not in st.session_state:
    st.session_state.stock_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_range" not in st.session_state:
    st.session_state.selected_range = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Financial Agent")
    st.caption("Event-Driven AI Stock Analyst")
    st.divider()

    api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get yours free at console.groq.com",
    )

    st.subheader("Stock Selection")
    ticker = st.selectbox("Ticker", POPULAR_TICKERS, index=0)
    custom = st.text_input("Or enter custom ticker", placeholder="e.g. META")
    if custom.strip():
        ticker = custom.strip().upper()

    period = st.select_slider(
        "Data period",
        options=["1mo", "3mo", "6mo", "1y"],
        value="3mo",
    )

    load_btn = st.button("Load Stock Data", type="primary", use_container_width=True)

    st.divider()
    st.subheader("News Filters")

    col_a, col_b = st.columns(2)
    with col_a:
        f_earnings    = st.checkbox("Earnings", value=True)
        f_product     = st.checkbox("Product", value=True)
        f_management  = st.checkbox("Management", value=True)
    with col_b:
        f_policy      = st.checkbox("Policy", value=True)
        f_market      = st.checkbox("Market", value=True)
        f_competition = st.checkbox("Competition", value=True)

    st.caption("Sentiment")
    col_c, col_d = st.columns(2)
    with col_c:
        f_bullish = st.checkbox("Bullish", value=True)
    with col_d:
        f_bearish = st.checkbox("Bearish", value=True)

    st.divider()
    st.caption("Built for AI Engineer Assessment · Agentic AI System")

    st.divider()
    st.subheader("Session history")
    sessions = get_all_sessions(limit=5)
    for s in sessions:
        label = f"{s.get('ticker','?')} · {s['last_active'][:10]}"
        msgs  = s.get('message_count', 0)
        if st.button(f"{label} ({msgs} msgs)", key=f"sess_{s['session_id'][:8]}", use_container_width=True):
            st.session_state.session_id = s['session_id']
            st.session_state.chat_history = [
                (m['role'], m['content'])
                for m in load_chat_history(s['session_id'])
                if m['role'] in ('user','assistant')
            ]
            st.rerun()


# ── Load stock data ───────────────────────────────────────────────────────────
if load_btn:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        with st.spinner(f"Fetching {ticker} data..."):
            try:
                stock_data = fetch_stock_data(ticker, period)
                agent = FinancialAgent(api_key=api_key, session_id=st.session_state.session_id)
                agent.set_stock_data(stock_data)
                st.session_state.stock_data = stock_data
                st.session_state.agent = agent
                save_news_to_db(ticker, stock_data.news)
                add_to_watchlist(st.session_state.session_id, ticker)
                update_session(st.session_state.session_id, ticker=ticker, period=period)
                st.session_state.chat_history = []
                st.session_state.selected_range = None
                st.success(f"Loaded {ticker} — {len(stock_data.prices)} trading days, {len(stock_data.news)} news events")
            except Exception as e:
                st.error(f"Failed to load data: {e}")


# ── Main dashboard ────────────────────────────────────────────────────────────
if st.session_state.stock_data is None:
    st.info("👈 Select a ticker and click **Load Stock Data** to begin.")
    st.markdown("""
    ### What this agent can do
    - **Event-Driven Charting** — Overlay news events directly on the price chart
    - **AI Range Analysis** — Select a date range and get AI explanation of price moves
    - **Sentiment Forecasting** — 7-day and 30-day bullish/bearish probability
    - **Pattern Matching** — Find historical periods with similar news setups
    - **Natural Language Chat** — Ask anything about the stock in plain English
    """)
    st.stop()

sd = st.session_state.stock_data

# Apply sidebar filters
active_cats = [c for c, v in [
    ("earnings", f_earnings), ("product", f_product), ("management", f_management),
    ("policy", f_policy), ("market", f_market), ("competition", f_competition)
] if v]
active_sents = [s for s, v in [("bullish", f_bullish), ("bearish", f_bearish)] if v]
if f_bullish or f_bearish:
    active_sents += ["neutral"]

filtered_news = filter_news(sd.news, categories=active_cats, sentiments=active_sents)
overall_sentiment = compute_overall_sentiment(sd.news)


# ── Top metrics row ───────────────────────────────────────────────────────────
st.markdown(f"## {sd.ticker} — Event-Driven Analysis")
m1, m2, m3, m4 = st.columns(4)

chg_class = "bull" if sd.price_change_pct >= 0 else "bear"
with m1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Current Price</div>
        <div class="metric-value">${sd.current_price:.2f}</div>
        <div class="metric-sub {chg_class}">{sd.price_change_pct:+.2f}% today</div>
    </div>""", unsafe_allow_html=True)

bull_pct = min(95, max(5, 50 + overall_sentiment * 40))
with m2:
    out_class = "bull" if bull_pct > 55 else "bear" if bull_pct < 45 else "neu"
    out_label = "Bullish" if bull_pct > 55 else "Bearish" if bull_pct < 45 else "Neutral"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">AI Sentiment Forecast</div>
        <div class="metric-value {out_class}">{out_label}</div>
        <div class="metric-sub {out_class}">{bull_pct:.0f}% confidence</div>
    </div>""", unsafe_allow_html=True)

sent_class = "bull" if overall_sentiment > 0 else "bear" if overall_sentiment < 0 else "neu"
with m3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">News Sentiment Score</div>
        <div class="metric-value {sent_class}">{overall_sentiment:+.2f}</div>
        <div class="metric-sub">Scale: -1.0 (bearish) to +1.0 (bullish)</div>
    </div>""", unsafe_allow_html=True)

with m4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Active News Events</div>
        <div class="metric-value">{len(filtered_news)}</div>
        <div class="metric-sub">{len(sd.news)} total · {len(active_cats)} categories shown</div>
    </div>""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Chart & Analysis", "🤖 AI Chat", "📰 News Feed"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Chart
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_chart, col_right = st.columns([3, 1])

    with col_chart:
        st.markdown("**Hover over dots to see news events · Drag to select a range for AI analysis**")
        fig = build_candlestick_chart(
            prices=sd.prices,
            news=filtered_news,
            ticker=sd.ticker,
            selected_range=st.session_state.selected_range,
        )
        chart_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="main_chart")

        # Handle plotly range selection
        if chart_event and hasattr(chart_event, "selection") and chart_event.selection:
            sel = chart_event.selection
            if hasattr(sel, "box") and sel.box:
                box = sel.box[0]
                start = box.get("x")[0][:10] if box.get("x") else None
                end = box.get("x")[1][:10] if box.get("x") else None
                if start and end:
                    st.session_state.selected_range = (start, end)

        # Range selector controls
        st.markdown("**AI Range Analysis**")
        r_col1, r_col2, r_col3 = st.columns([2, 2, 1])
        with r_col1:
            range_start = st.date_input(
                "From",
                value=sd.prices.index[-min(30, len(sd.prices))].date(),
                min_value=sd.prices.index[0].date(),
                max_value=sd.prices.index[-1].date(),
            )
        with r_col2:
            range_end = st.date_input(
                "To",
                value=sd.prices.index[-1].date(),
                min_value=sd.prices.index[0].date(),
                max_value=sd.prices.index[-1].date(),
            )
        with r_col3:
            analyze_btn = st.button("Analyze Range ✨", type="primary", use_container_width=True)

        if analyze_btn:
            start_str = range_start.strftime("%Y-%m-%d")
            end_str   = range_end.strftime("%Y-%m-%d")
            st.session_state.selected_range = (start_str, end_str)
            range_news = get_range_news(sd.news, start_str, end_str)

            with st.spinner("AI is analyzing this price range..."):
                question = (
                    f"Analyze the price movement in {sd.ticker} from {start_str} to {end_str}. "
                    f"Use your tools to examine the price action and explain what news events "
                    f"drove the movement. Be specific and educational."
                )
                answer = st.session_state.agent.chat(question)
                st.session_state.chat_history.append(("Range Analysis", answer))

            st.markdown("#### AI Range Explanation")
            st.info(answer)

        # Sentiment timeline
        st.markdown("**News Sentiment Over Time**")
        fig_sent = build_sentiment_timeline(filtered_news)
        st.plotly_chart(fig_sent, use_container_width=True, key="sent_chart")

    with col_right:
        # Forecast gauge
        fig_gauge = build_forecast_gauge(bull_pct, sd.ticker)
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_chart")

        st.markdown("**Quick AI Queries**")
        quick_questions = [
            f"Why is {sd.ticker} moving today?",
            f"What are the biggest risks for {sd.ticker}?",
            f"Compare bullish vs bearish signals",
            f"Find similar historical patterns",
            f"Summarize earnings news",
        ]
        for q in quick_questions:
            if st.button(q, use_container_width=True, key=f"quick_{q[:20]}"):
                with st.spinner("Analyzing..."):
                    answer = st.session_state.agent.chat(q)
                    st.session_state.chat_history.append((q, answer))
                st.rerun()

        st.markdown("**Event Categories**")
        for cat in active_cats:
            count = sum(1 for n in filtered_news if n.category == cat)
            bull_c = sum(1 for n in filtered_news if n.category == cat and n.sentiment == "bullish")
            bear_c = count - bull_c
            style = CAT_BADGE_STYLES.get(cat, "background:#eee;color:#333")
            st.markdown(
                f'<span class="cat-badge" style="{style}">{cat.title()}</span> '
                f'&nbsp;{count} events &nbsp;'
                f'<span style="color:#3B6D11">▲{bull_c}</span> '
                f'<span style="color:#A32D2D">▼{bear_c}</span>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI Chat
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"### Chat with your AI Financial Agent — {sd.ticker}")
    st.caption(
        "The agent uses tool calls to fetch data before answering. "
        "It follows a ReAct loop: Think → Call tool → Observe result → Respond."
    )

    # Display conversation
    chat_container = st.container()
    with chat_container:
        for role, msg in st.session_state.chat_history:
            if role == "You":
                st.markdown(f'<div class="chat-msg-user"><b>You:</b> {msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-msg-ai"><b>AI Agent:</b> {msg}</div>', unsafe_allow_html=True)

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Ask anything about this stock",
            placeholder=(
                f"e.g. 'Why did {sd.ticker} drop last week?' or "
                f"'What news is driving the current rally?' or "
                f"'Is this a good time to buy?'"
            ),
            height=80,
        )
        col_send, col_reset = st.columns([4, 1])
        with col_send:
            send_btn = st.form_submit_button("Send to Agent ↗", type="primary", use_container_width=True)
        with col_reset:
            reset_btn = st.form_submit_button("Reset Chat", use_container_width=True)

    if send_btn and user_input.strip():
        st.session_state.chat_history.append(("You", user_input))
        with st.spinner("Agent is thinking (using tools)..."):
            answer = st.session_state.agent.chat(user_input)
        st.session_state.chat_history.append(("AI Agent", answer))
        st.rerun()

    if reset_btn:
        st.session_state.chat_history = []
        st.session_state.agent.reset_conversation()
        st.rerun()

    # Show agentic loop explainer
    with st.expander("How the AI agent reasons (ReAct loop)"):
        st.markdown("""
        **Step 1 — Planning:** Claude receives your question + current stock context (price, ticker, news count).

        **Step 2 — Tool selection:** Claude decides which tools to call:
        - `analyze_price_range` — fetch price stats + news for a date window
        - `forecast_trend` — compute bullish/bearish probability
        - `find_similar_periods` — pattern match to historical setups
        - `summarize_news_category` — aggregate by news type

        **Step 3 — Observation:** Tool results are fed back to Claude.

        **Step 4 — Synthesis:** Claude reads the structured data and generates a plain-English explanation.

        This loop can repeat up to 5 times until Claude reaches a final answer.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — News Feed
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"### {sd.ticker} News Feed — {len(filtered_news)} events")

    sort_order = st.radio("Sort by", ["Newest first", "Highest impact first", "Most bearish first"], horizontal=True)

    sorted_news = filtered_news.copy()
    if sort_order == "Newest first":
        sorted_news = sorted(sorted_news, key=lambda n: n.date, reverse=True)
    elif sort_order == "Highest impact first":
        sorted_news = sorted(sorted_news, key=lambda n: abs(n.sentiment_score), reverse=True)
    else:
        sorted_news = sorted(sorted_news, key=lambda n: n.sentiment_score)

    for idx, event in enumerate(sorted_news):
        style = CAT_BADGE_STYLES.get(event.category, "background:#eee;color:#333")
        sent_color = "#3B6D11" if event.sentiment == "bullish" else "#A32D2D" if event.sentiment == "bearish" else "#5F5E5A"
        sent_arrow = "▲" if event.sentiment == "bullish" else "▼" if event.sentiment == "bearish" else "●"

        with st.container():
            st.markdown(f"""
            <div class="news-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                    <div>
                        <span class="cat-badge" style="{style}">{event.category.title()}</span>
                        &nbsp;<span style="color:{sent_color};font-size:13px;font-weight:600">{sent_arrow} {event.sentiment.title()} ({event.sentiment_score:+.2f})</span>
                    </div>
                    <span style="font-size:12px;color:#6c757d">{event.date} · {event.source}</span>
                </div>
                <div style="font-size:14px;color:#212529;font-weight:500">{event.title}</div>
                <div style="font-size:12px;color:#6c757d;margin-top:2px">Impact: {event.impact.title()}</div>
            </div>
            """, unsafe_allow_html=True)

            # Quick AI analysis button per news item
            if st.button(f"AI: explain this event's market impact", key=f"news_{idx}_{event.date}_{event.category}"):
                with st.spinner("Analyzing..."):
                    q = f"Explain the market impact of this news on {sd.ticker}: '{event.title}'. How did it affect price and why?"
                    answer = st.session_state.agent.chat(q)
                    st.session_state.chat_history.append((q, answer))
                st.info(answer)