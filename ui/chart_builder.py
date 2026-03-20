"""
ui/chart_builder.py
Builds interactive Plotly candlestick charts with:
- News event dot overlays (color-coded by category and sentiment)
- Drag-to-select range highlighting
- Hover tooltips showing news headline + sentiment
"""

import plotly.graph_objects as go
import pandas as pd
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
) -> go.Figure:
    """
    Build a Plotly candlestick chart with interactive news overlays.

    Args:
        prices: OHLCV DataFrame indexed by datetime
        news: List of NewsEvent objects to overlay
        ticker: Stock symbol for chart title
        selected_range: Optional (start_date, end_date) tuple to highlight

    Returns:
        Plotly Figure object ready for st.plotly_chart()
    """
    fig = go.Figure()

    # ── Candlestick base ──────────────────────────────────────────────────
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
    ))

    # ── News event dots ───────────────────────────────────────────────────
    # Group news by category so we can toggle them as a legend group
    for category in set(n.category for n in news):
        cat_news = [n for n in news if n.category == category]
        dot_dates, dot_prices, dot_texts, dot_symbols, dot_colors = [], [], [], [], []

        for event in cat_news:
            if event.date in prices.index.strftime("%Y-%m-%d").tolist():
                idx = prices.index[prices.index.strftime("%Y-%m-%d") == event.date][0]
                price_at_date = float(prices.loc[idx, "High"]) * 1.012  # Sit above candle

                dot_dates.append(idx)
                dot_prices.append(price_at_date)
                dot_texts.append(
                    f"<b>{event.title}</b><br>"
                    f"Source: {event.source}<br>"
                    f"Sentiment: {event.sentiment.title()} ({event.sentiment_score:+.2f})<br>"
                    f"Impact: {event.impact.title()}"
                )
                dot_symbols.append(SENTIMENT_SYMBOLS[event.sentiment])
                dot_colors.append(SENTIMENT_FILL[event.sentiment])

        if dot_dates:
            fig.add_trace(go.Scatter(
                x=dot_dates,
                y=dot_prices,
                mode="markers",
                marker=dict(
                    symbol=dot_symbols,
                    size=10,
                    color=dot_colors,
                    line=dict(color="white", width=1.5),
                ),
                name=category.replace("_", " ").title(),
                legendgroup=category,
                text=dot_texts,
                hoverinfo="text",
                hovertemplate="%{text}<extra></extra>",
            ))

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
            gridcolor="rgba(0,0,0,0.05)",
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
        dragmode="select",  # enables click-drag selection
        hovermode="closest",
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