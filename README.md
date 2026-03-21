# 📈 Financial Agent: Event-Driven Stock Analysis

An intelligent financial analysis dashboard built with **Streamlit** that overlays global news events onto interactive stock price charts. Powered by a **Multi-Agent System** (utilizing Groq API and LLMs like Llama 3/4), this platform automates the work of a financial research team to help investors understand the *why* behind price movements.

## 🚀 Key Features

- **🤖 Multi-Agent AI Orchestration**: 
  - **Researcher Agent**: Autonomously calls tools to fetch real-time SEC filings, macroeconomic indicators (FRED), options flow, and news data.
  - **Analyst Agent**: Synthesizes the raw data into a structured investment thesis with bull and bear cases.
  - **Risk Manager Agent**: Acts as a devil's advocate to challenge the thesis and identify blind spots before presenting the final report.
- **📊 Event-Driven Charting**: Interactive Plotly candlestick charts natively embed news events as sentiment-coded dots along the price timeline.
- **⚡ Automated Range Analysis**: Simply drag to highlight a date range on the chart, and the Multi-Agent team will automatically generate a comprehensive report explaining the exact drivers of that period's price action.
- **🧠 Persistent Memory & Pattern Matching**: Utilizes SQLite for session memory and ChromaDB for semantic vector search, allowing the agent to find historical market periods with similar setups.
- **📰 Smart News Feed**: A highly filterable news feed featuring zero-typing toggles for category, sentiment, and impact, plus deep keyword search.

## 📖 Platform Usage Guide

1. **Initialization**: Enter your Groq API Key and select a target stock ticker (e.g., AAPL, NVDA, TSLA) in the left sidebar. Choose your desired historical viewing period and click **"Load Stock Data"**.
2. **Interactive Charting**: 
   - **Hover** over dots on the candlestick chart to instantly read news headlines, sources, and AI-scored sentiment.
   - **Click** a dot to pin that day's news strictly to the side panel. 
   - **Drag** a box over an area on the chart to select a date range. This will instantly trigger the AI team to analyze the movement and stream a **Range Analysis Report** directly to your screen.
3. **Agentic Chat**: Navigate to the **"🤖 AI Chat"** tab to talk directly to your AI portfolio manager. Click on predefined **Quick AI Queries** (like *"Why is the stock moving today?"*), or type your own complex questions. The orchestrator will dynamically adjust its format whether you're asking for a general overview or a specific data point.
4. **News Feed Deep-Dive**: Switch to the **"📰 News Feed"** tab to filter through hundreds of historical news events. Quickly toggle Categories (Earnings, Product, Macro) and Sentiments (Bullish/Bearish) to isolate market catalysts.

## 💻 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/debbbl/financial_agent.git
   cd financial_agent
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate  
   # On Windows: 
   venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   *The dashboard will automatically open in your default web browser at `http://localhost:8501`.*

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **AI Engine**: [Groq API](https://groq.com/) for ultra-low latency LLM inference
- **Data Source**: [yfinance](https://github.com/ranaroussi/yfinance)
- **Database**: SQLite (Relational State) & ChromaDB (Vector Embeddings)
- **Visualization**: [Plotly](https://plotly.com/)
