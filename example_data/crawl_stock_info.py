import os
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from tqdm import tqdm
import json


COMPANIES = {
    "AAPL": "apple",
    "MSFT": "microsoft",
    "TSLA": "tesla",
    "NVDA": "nvidia",
    "AMZN": "amazon",
    "GOOGL": "google",
    "META": "meta",
    "NFLX": "netflix",
    "INTC": "intel",
    "AMD": "amd"
}

START_DATE = "2020-01-01"
END_DATE = "2025-04-14"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVE_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(SAVE_DIR, exist_ok=True)



def get_price_history_intraday(ticker, period="7d", interval="1m"):
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    data = data.reset_index()
    data["Datetime"] = data["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return data.to_dict(orient="records")




def get_sec_filings(ticker, count=5):
    cik_lookup_url = f"https://www.sec.gov/files/company_tickers.json"
    cik_db = requests.get(cik_lookup_url).json()
    cik = None
    for item in cik_db.values():
        if item['ticker'].upper() == ticker:
            cik = str(item['cik_str']).zfill(10)
            break
    if cik is None:
        return []

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": "sec-crawler@example.com"}
    r = requests.get(url, headers=headers)
    data = r.json()
    
    filings = []
    for i in range(min(count, len(data["filings"]["recent"]["accessionNumber"]))):
        form_type = data["filings"]["recent"]["form"][i]
        if form_type not in ["10-K", "10-Q", "8-K"]:
            continue
        filings.append({
            "ticker": ticker,
            "form_type": form_type,
            "filed_date": data["filings"]["recent"]["filingDate"][i],
            "accession_num": data["filings"]["recent"]["accessionNumber"][i]
        })
    return filings



def save_all():
    all_data = []
    for ticker, name in tqdm(COMPANIES.items()):
        item = {"ticker": ticker, "company": name}
        
        try:
            item["price_history"] = get_price_history_intraday(ticker)
        except:
            item["price_history"] = []
        try:
            item["sec_filings"] = get_sec_filings(ticker)
        except:
            item["sec_filings"] = []

        all_data.append(item)

    with open(os.path.join(SAVE_DIR, "merged_financial_data.jsonl"), "w") as f:
        for entry in all_data:
            f.write(json.dumps(entry) + "\n")

    print(f"\nsaved {SAVE_DIR}/merged_financial_data.jsonl")





if __name__ == "__main__":
    save_all()
