import os
import json
from transformers import pipeline
from tqdm import tqdm
import time
from datetime import datetime, timedelta
import finnhub

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVE_DIR = os.path.join(BASE_DIR, "data")
INPUT_PATH = os.path.join(SAVE_DIR, "finnhub_stock_news.jsonl")
OUTPUT_PATH = os.path.join(SAVE_DIR, "finnhub_stock_news_with_sentiment.jsonl")

sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment():
    results = []
    with open(INPUT_PATH, "r") as infile:
        for line in tqdm(infile, desc="Analyzing Sentiment"):
            data = json.loads(line)
            text = (data.get("headline", "") + " " + data.get("summary", "")).strip()
            if not text:
                sentiment_result = {"label": "NEUTRAL", "score": 0.0}
            else:
                sentiment_result = sentiment_pipeline(text[:512])[0]

            data["sentiment"] = sentiment_result["label"]
            data["confidence"] = sentiment_result["score"]
            results.append(data)

    with open(OUTPUT_PATH, "w") as outfile:
        for item in results:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")



today = datetime.today()
from_date = (today - timedelta(days=29)).strftime("%Y-%m-%d")
to_date = today.strftime("%Y-%m-%d")



SAVE_DIR = "./data"
os.makedirs(SAVE_DIR, exist_ok=True)

API_KEY = "cvuaqahr01qjg138k2pgcvuaqahr01qjg138k2q0"
client = finnhub.Client(api_key=API_KEY)

COMPANIES = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "NFLX", "INTC", "AMD"]

def get_news(ticker, from_date, to_date):
    try:
        news_items = client.company_news(ticker, _from=from_date, to=to_date)
        print(f"[{ticker}] Fetched {len(news_items)} articles from {from_date} to {to_date}")
        results = []
        for n in news_items:
            published_date = datetime.utcfromtimestamp(n["datetime"]).strftime("%Y-%m-%d")
            results.append({
                "ticker": ticker,
                "date": published_date,
                "headline": n.get("headline", ""),
                "summary": n.get("summary", ""),
                "url": n.get("url", ""),
                "source": n.get("source", "")
            })
        return results
    except Exception as e:
        print(f"[Error] {ticker}: {e}")
        return []

def crawl_finnhub_news():
    today = datetime.today()
    from_date = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    all_news = []
    for ticker in tqdm(COMPANIES):
        news_list = get_news(ticker, from_date, to_date)
        print(f"[{ticker}] {len(news_list)} articles from {from_date} to {to_date}")
        all_news.extend(news_list)
        time.sleep(1.5)

    output_path = os.path.join(SAVE_DIR, "finnhub_stock_news.jsonl")
    with open(output_path, "w") as f:
        for article in all_news:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    crawl_finnhub_news()
    analyze_sentiment()