import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.market_data import score_sentiment, _keyword_sentiment_fallback

def test_sentiment():
    test_cases = [
        "Apple beats estimates but warns on guidance",
        "Tesla stock plunges after weak delivery numbers",
        "NVIDIA hits new record high as AI demand surges",
        "Microsoft to acquire small startup for undisclosed amount",
        "Market remains flat ahead of Fed meeting"
    ]

    print("--- Sentiment Scorer Verification ---\n")

    for text in test_cases:
        print(f"Text: {text}")
        
        # New FinBERT scoring
        label_fin, score_fin = score_sentiment(text)
        print(f"  FinBERT: {label_fin} ({score_fin})")
        
        # Old Keyword scoring
        label_key, score_key = _keyword_sentiment_fallback(text)
        print(f"  Keyword: {label_key} ({score_key})")
        print("-" * 40)

if __name__ == "__main__":
    test_sentiment()
