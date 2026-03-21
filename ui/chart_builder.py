"""
ui/chart_builder.py
Builds interactive Plotly candlestick charts with:
- News event dot overlays (color-coded by sentiment)
- Candlestick hover tooltips showing OHLC + news event count per date
- Click-to-select news dots with details panel support
- Date-highlight vrect for selected news events
- Drag-to-select range highlighting
"""

import plotly.graph_objects as go
import pandas as pd
from collections import Counter
from tools.market_data import NewsEvent


CATEGORY_COLORS = {
    "earnings":    "#639922",   # green
    "product":     "#185FA5",   # blue
    "management":  "#BA7517",   # amber
    "policy":      "#D4537E",   # pink
    "market":      "#7F77DD",   # purple
    "competition": "#D85A30",   # coral
}

SENTIMENT_SYMBOLS = {
    "bullish": "triangle-up",
    "bearish": "triangle-down",
    "neutral": "circle",
}

SENTIMENT_FILL = {
    "bullish": "#639922",
    "bearish": "#E24B4A",
    "neutral": "#888780",
}


def build_candlestick_chart(
    prices: pd.DataFrame,
    news: list[NewsEvent],
    ticker: str,
    selected_range: tuple[str, str] | None = None,
    selected_news_date: str | None = None,
) -> go.Figure:
    """
    Build a Plotly candlestick chart with interactive news overlays.

    Args:
        prices: OHLCV DataFrame indexed by datetime
        news: List of NewsEvent objects to overlay
        ticker: Stock symbol for chart title
        selected_range: Optional (start_date, end_date) tuple to highlight
        selected_news_date: Optional date string to highlight with a vrect

    Returns:
        Plotly Figure object ready for st.plotly_chart()
    """
    fig = go.Figure()

    # ── Build per-date news count lookup ──────────────────────────────────
    news_count_by_date = Counter(n.date for n in news)

    # ── Candlestick base with news-count tooltip ─────────────────────────
    price_dates_str = prices.index.strftime("%Y-%m-%d")
    candle_news_counts = [
        [news_count_by_date.get(d, 0)] for d in price_dates_str
    ]

    fig.add_trace(go.Candlestick(
        x=prices.index,
        open=prices["Open"],
        high=prices["High"],
        low=prices["Low"],
        close=prices["Close"],
        name="Price",
        increasing_line_color="#639922",
        decreasing_line_color="#E24B4A",
        increasing_fillcolor="rgba(99,153,34,0.7)",
        decreasing_fillcolor="rgba(226,75,74,0.7)",
        line=dict(width=1),
        showlegend=False,
        customdata=candle_news_counts,
        hoverinfo="text",
        hovertext=[
            f"{d}  ·  O: ${o:.2f}  H: ${h:.2f}  L: ${l:.2f}  C: ${c:.2f}"
            f"  ·  {news_count_by_date.get(d, 0)} news events"
            for d, o, h, l, c in zip(
                price_dates_str,
                prices["Open"], prices["High"],
                prices["Low"], prices["Close"],
            )
        ],
    ))

    # ── News event dots ───────────────────────────────────────────────────
    # Group news by category so we can toggle them as a legend group
    for category in set(n.category for n in news):
        cat_news = [n for n in news if n.category == category]
        dot_dates, dot_prices, dot_symbols, dot_colors = [], [], [], []
        dot_customdata = []  # [headline, source, sentiment, score, date, url, summary, category, impact]

        for event in cat_news:
            if event.date in price_dates_str.tolist():
                idx = prices.index[price_dates_str == event.date][0]
                candle_high = float(prices.loc[idx, "High"])
                price_at_date = candle_high * 1.003

                dot_dates.append(idx)
                dot_prices.append(price_at_date)
                dot_symbols.append(SENTIMENT_SYMBOLS[event.sentiment])
                dot_customdata.append([
                    event.title,                          # [0] headline
                    event.source,                         # [1] source
                    event.sentiment.title(),              # [2] sentiment label
                    f"{event.sentiment_score:+.2f}",      # [3] sentiment score
                    event.date,                           # [4] date
                    event.url or "",                      # [5] url
                    event.summary or "",                  # [6] summary
                    event.category,                       # [7] category
                    event.impact.title(),                 # [8] impact
                ])

        if dot_dates:
            cat_color = CATEGORY_COLORS.get(category, "#000000")
            fig.add_trace(go.Scatter(
                x=dot_dates,
                y=dot_prices,
                mode="markers",
                marker=dict(
                    symbol=dot_symbols,
                    size=14,  # Increased for easier clicking
                    color=cat_color,
                    line=dict(color="white", width=2),
                ),
                name=category.replace("_", " ").title(),
                legendgroup=category,
                showlegend=True,
                customdata=dot_customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Source: %{customdata[1]}<br>"
                    "Sentiment: %{customdata[2]} (%{customdata[3]})"
                    "<extra></extra>"
                ),
            ))

    # ── Selected news date highlight ─────────────────────────────────────
    if selected_news_date:
        fig.add_vrect(
            x0=selected_news_date, x1=selected_news_date,
            fillcolor="rgba(173,216,230,0.25)",
            line=dict(color="#4A90D9", width=1.5, dash="dash"),
            layer="below",
        )

    # ── Selected range highlight ──────────────────────────────────────────
    if selected_range:
        start, end = selected_range
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(24,95,165,0.10)",
            line=dict(color="#185FA5", width=1, dash="dot"),
            annotation_text="Analysis range",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#185FA5"),
            layer="below",
        )

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"{ticker} — Event-Driven Price Chart",
            font=dict(size=15),
            x=0.01,
        ),
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=False),
            type="date",
            tickformat="%b %d",
            tickmode="auto",
            nticks=10,
            gridcolor="rgba(0,0,0,0.04)",
        ),
        yaxis=dict(
            title="Price (USD)",
            gridcolor="rgba(0,0,0,0.05)",
            tickprefix="$",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        height=480,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        dragmode="select",   # enables box-drag selection
        selectdirection="h", # horizontal only — date range selection
        hovermode="closest",
        clickmode="event+select",
    )

    return fig


def build_sentiment_timeline(news: list[NewsEvent]) -> go.Figure:
    """Bar chart showing daily sentiment score over time."""
    if not news:
        return go.Figure()

    dates = [n.date for n in news]
    scores = [n.sentiment_score for n in news]
    colors = [SENTIMENT_FILL[n.sentiment] for n in news]

    fig = go.Figure(go.Bar(
        x=dates,
        y=scores,
        marker_color=colors,
        text=[f"{s:+.2f}" for s in scores],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Score: %{y:+.2f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.3)")

    fig.update_layout(
        title="News Sentiment Timeline",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1.2, 1.2], gridcolor="rgba(0,0,0,0.05)"),
        height=220,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    return fig


def build_forecast_gauge(bull_prob: float, ticker: str) -> go.Figure:
    """Gauge chart showing bull vs bear probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=bull_prob,
        title={"text": f"{ticker} Bullish Probability", "font": {"size": 14}},
        delta={"reference": 50, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#185FA5"},
            "steps": [
                {"range": [0, 40], "color": "rgba(226,75,74,0.15)"},
                {"range": [40, 60], "color": "rgba(136,135,128,0.1)"},
                {"range": [60, 100], "color": "rgba(99,153,34,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#185FA5", "width": 3},
                "thickness": 0.75,
                "value": bull_prob,
            },
        },
        number={"suffix": "%", "font": {"size": 28}},
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig