import os
import dotenv
from openai import OpenAI
from datetime import datetime

def parse_query_with_stock_extraction(input_text, index_looking_at, current_time):
    dotenv.load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API")
    client = OpenAI(api_key=OPENAI_API_KEY)
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


    ## UTC기준 현재 시간
    current_time = f"현재 시간: {datetime.utcnow().year}-{datetime.utcnow().month:02d}-{datetime.utcnow().day:02d} {datetime.utcnow().hour:02d}:{datetime.utcnow().minute:02d}:{datetime.utcnow().second:02d}"
    current_stock_looking_at = COMPANIES[list(COMPANIES.keys())[index_looking_at]]

    ## 질문 입력받기
    """
    대충 질문을 입력받고, 현재 보고 있는 주식과 시간을 기준으로 질문을 분석
    질문에서 언급된 회사들을 추출 및 지금 보고있는 주식도 함께 추출
    구조:
    {
        query_type: "질문 유형", # ["price_change_reason", "current_price", "price_trend", "recent_news", "corporate_event", "financial_metric", "earnings_report", "comparative_analysis", "valuation_check", "investment_advice", "price_forecast", "opinion_analysis"]
        company: "회사 이름",
        mentioned_companies: ["회사1", "회사2", ...],
        is_different_from_current: true/false, # 질문에 있는 회사들이 지금 보고 있는 주식과 다른지
        date_range: ["YYYY-MM-DD", "YYYY-MM-DD"], # 질문에 있는 날짜 범위
        user_query_specific: "자세한 질문" # 사용자가 질문한 내용을 바탕으로 변환된 구체적인 질문
    }

    """
    system_prompt = (
        "You are a query understanding module for a financial RAG system.\n"
        "Given the user's raw question, current focus stock, and current UTC time, "
        "extract all companies mentioned in the question, classify the query type, infer a relevant date range, "
        "and generate a more specific, retrieval-ready query.\n\n"
        "Respond in strict JSON format, but do not add ```json ```:\n"
        "{\n"
        "  \"query_type\": string,\n"
        "  \"company\": string,\n"
        "  \"mentioned_companies\": [list of company names],\n"
        "  \"is_different_from_current\": true/false,\n"
        "  \"date_range\": [\"YYYY-MM-DD\", \"YYYY-MM-DD\"],\n"
        "  \"user_query_specific\": string\n"
        "}\n\n"
        "Guidelines:\n"
        "- The 'company' is the main company being asked about.\n"
        "- 'mentioned_companies' includes all companies found in the question.\n"
        "- Set 'is_different_from_current' = true if the user is asking about a stock other than the current focus.\n"
        "- Classify query_type into one of: "
        "[\"price_change_reason\", \"current_price\", \"price_trend\", \"recent_news\", \"corporate_event\", "
        "\"financial_metric\", \"earnings_report\", \"comparative_analysis\", \"valuation_check\", "
        "\"investment_advice\", \"price_forecast\", \"opinion_analysis\"]\n"
        "- date_range should be inferred based on temporal expressions (e.g., 'this week', 'last 3 days') or default to recent 3 days.\n"
        "- Make 'user_query_specific' a fluent, detailed question suitable for retrieval."
    )

    user_prompt = f"""
User Input: "{input_text}"
Current Focus Stock: "{current_stock_looking_at}"
Current UTC Time: "{current_time}"
"""

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return completion.choices[0].message.content


if __name__ == "__main__":
    index_looking_at = 2 # 지금 현재 보고있는 주식 index
    current_time = f"{datetime.utcnow().year}-{datetime.utcnow().month:02d}-{datetime.utcnow().day:02d} {datetime.utcnow().hour:02d}:{datetime.utcnow().minute:02d}:{datetime.utcnow().second:02d}"
    input_text = input("What do you want to ask?\n")
    result = parse_query_with_stock_extraction(input_text, index_looking_at, current_time)
    print(result)