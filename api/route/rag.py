from fastapi import APIRouter

import asyncio

from schemas.query import QueryRequest, QueryResponse

from model.gpt import chain, chain_title, chain_ticker

router = APIRouter(prefix="/rag", tags=['RAG'])

@router.get('/')
def greet():
    return "RAG api"


@router.post('/', response_model=QueryResponse)
async def reply(req: QueryRequest):
    user_query = req.query
    is_first_chat = req.is_first_chat

    if not user_query:
        return {"error": "Missing 'query' in request body"}

    # Create LLM Tasks
    # 1) 항상 실행할 본문·티커 태스크
    reply_task = chain.ainvoke(user_query)
    ticker_task = chain_ticker.ainvoke(user_query)

    # 2) is_first_chat에 따라 제목 태스크만 추가
    if is_first_chat:
        title_task = chain_title.ainvoke(user_query)
        # 세 개의 태스크를 병렬 실행
        title, reply, ticker = await asyncio.gather(
            title_task,
            reply_task,
            ticker_task
        )
    else:
        # 제목 태스크 없이 본문·티커만 병렬 실행
        reply, ticker = await asyncio.gather(
            reply_task,
            ticker_task
        )
        title = None  # 또는 "" 등 원하는 기본값

    if ticker == "False":
        pass
        # skip the attribute for events and datetime
    else:
        pass
        # Fetch the events and datetime
    
    res = QueryResponse(
        title= title,
        reply=reply
    )
    return res