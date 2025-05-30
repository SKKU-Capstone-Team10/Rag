import json
from apify_client import ApifyClient
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from core.config import settings


# Initialize the ApifyClient with your Apify API token
# Replace '<YOUR_API_TOKEN>' with your token.
client = ApifyClient(settings.APIFY_API_KEY)

# Prepare the Actor input
SYMBOLS = ["MSFT", "NVDA", "AAPL", "AMZN", "GOOG", "META", "TSLA", "AVGO", "NFLX", "AMD"]
LIMIT = 1000
loop = tqdm(SYMBOLS, desc="Processing symbols", unit="symbol")

for symbol in loop:
    input_symbol = ["NASDAQ:" + symbol]

    run_input = {
        "symbols": input_symbol,
        "proxy": {
            "useApifyProxy": True,
            "apifyProxyCountry": "US",
        },
        "resultsLimit": LIMIT,
    }

    # Run the Actor and wait for it to finish
    run = client.actor("mscraper/tradingview-news-scraper").call(run_input=run_input)

    results = list(client.dataset(run["defaultDatasetId"]).iterate_items())

    with open(f"./data/apify/raw/{symbol}.json", "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=2)
