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

    # Call chains in parallel
    reply_task = chain.ainvoke(user_query)
    ticker_task = chain_ticker.ainvoke(user_query)
    if is_first_chat == True: # Need to create a chat session title
        title_task = chain_title.ainvoke(user_query)
        # Execute in parallel
        title, reply, ticker = await asyncio.gather(
            title_task,
            reply_task,
            ticker_task
        )
    else:
        # Execute in parallel
        reply, ticker = await asyncio.gather(
            reply_task,
            ticker_task
        )
        title = None
    
    res = QueryResponse(
        title=title,
        reply=reply,
        ticker=ticker
    )
    return res