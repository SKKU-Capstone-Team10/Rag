import os
from pathlib import Path

import ast
import pandas as pd

from fastapi import APIRouter

from schemas.query import QueryRequest, QueryResponse

from model.gpt import chain_title, chain_events
from data.run import MyProcessor

router = APIRouter(prefix="/rag", tags=['RAG'])

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(__file__).resolve().parents[2]
faiss_base_paths = {
    "news":   os.path.join(BASE_DIR, "data/news/index"),
    "prices": os.path.join(BASE_DIR, "data/prices"),
}
runner = MyProcessor("BAAI/bge-large-en", faiss_base_paths)

@router.get('/')
def greet():
    return "RAG api"
    
@router.post('/', response_model=QueryResponse)
async def reply(req: QueryRequest):
    user_query = req.query
    symbol = req.symbol
    is_first_chat = req.is_first_chat

    if not user_query:
        return {"error": "Missing 'query' in request body"}

    answer, refs = runner.process(
        query=user_query,
        db="news",
        symbol=symbol,
        topk=5,
        method="2",
    )
    refs_splitted = []
    for ref in refs:
        splitted = ref.split(' ')
        refs_splitted.append(splitted[-3])
    # Need to create a chat session title
    if is_first_chat == True: 
        title = chain_title.invoke(user_query)
    else:
        title = None
    
    res = QueryResponse(
        title=title,
        reply=answer,
        refs=refs_splitted,
    )
    return res


@router.get('/event/{ticker}')
def generate_event(ticker: str):
    ticker = ticker.upper()

    csv_path = os.path.join(BASE_DIR, "data/news/")
    csv_file = pd.read_csv(os.path.join(csv_path, f"{ticker}.csv"))

    event = chain_events.invoke(
        {'csv': csv_file}
    )

    event_parsed = ast.literal_eval(event)
    return event_parsed