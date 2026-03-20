# 📈 Financial Agent: Event-Driven Stock Analysis

An intelligent financial analysis dashboard that overlays global news events onto stock price charts. Powered by **Agentic AI (Groq/Claude)**, this tool uses a ReAct reasoning loop to help investors understand the *why* behind price movements.

## 🚀 Features

- **📊 Event-Driven Charting**: Interactive Plotly candlestick charts with news event overlays. Hover over dots to see what happened on specific trading days.
- **🤖 AI Range Analysis**: Select any date range on the chart to get an AI-generated explanation of the price action and key drivers.
- **📰 Smart News Feed**: Highly filterable news feed with:
  - **Quick Date Presets** (1W, 1M, 3M, 6M, YTD, MAX).
  - **Zero-Typing Filters** for Category, Sentiment, and Impact.
  - **Keyword Search** for deep dives into specific topics.
- **🔮 Sentiment Forecasting**: 7-day and 30-day bullish/bearish probability based on aggregate news sentiment and AI analysis.
- **💬 Agentic AI Chat**: A full-featured chat interface where the agent uses tools to:
  - `analyze_price_range`
  - `forecast_trend`
  - `find_similar_periods`
  - `summarize_news_category`

## 📖 How to Use

1. **Load Data**: Enter your Groq API Key and select a ticker (e.g., AAPL, NVDA) in the sidebar. Click **"Load Stock Data"**.
2. **Explore the Chart**:
   - **Hover** over dots on the candlestick chart to view specific news events.
   - **Click** a dot to pin its news details in the side panel.
   - **Drag** a box over an area on the chart to select a date range.
3. **Analyze Moves**: After selecting a range, the side panel will show the top news and an **"Analyze Range ✨"** button. Click it to get an AI breakdown of why the price moved.
4. **News Feed Deep-Dive**: Switch to the **"News Feed"** tab to filter historical news using quick presets and category/impact toggles. Use the search bar to find specific keywords.
5. **AI Chat**: Use the **"AI Chat"** tab to ask complex questions like *"Why did the stock dip in early March?"* or *"Compare recent earnings sentiment with last year."*

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (Python-based interactive UI)
- **AI Engine**: [Groq API](https://groq.com/) (using Llama 3 / Mixtral for high-speed reasoning)
- **Data Source**: [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance API)
- **Database**: 
  - **SQLite**: Session management and chat history storage.
  - **ChromaDB**: (Optional) Vector storage for RAG-based news retrieval.
- **Visualization**: [Plotly](https://plotly.com/) for interactive financial charts.

## 📦 Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/debbbl/financial_agent.git
   cd financial_agent
   ```

2. **Set up a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory:

   ```env
   GROQ_API_KEY=your_api_key_here
   ```

5. **Run the Application**:

   ```bash
   streamlit run app.py
   ```

## 🏗️ Project Structure

- `app.py`: Main entry point and Streamlit UI definition.
- `agents/`: AI logic and agentic tool-calling definitions.
- `tools/`: Market data fetching, news processing, and sentiment analysis tools.
- `ui/`: Custom chart builders and UI component definitions.
- `database/`: DB initialization, session tracking, and persistence logic.
- `tests/`: Unit and integration tests for the financial agent.

## 🤝 Contributing

Feel free to open issues or submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License

[MIT](https://choosealicense.com/licenses/mit/)
