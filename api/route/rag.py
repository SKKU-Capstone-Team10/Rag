from fastapi import APIRouter

import asyncio

from schemas.query import QueryRequest, QueryResponse

from model.gpt import chain, chain_title

router = APIRouter(prefix="/rag", tags=['RAG'])

@router.get('/')
def greet():
    return "RAG api"


@router.post('/', response_model=QueryResponse)
async def reply(req: QueryRequest):
    user_query = req.query

    if not user_query:
        return {"error": "Missing 'query' in request body"}

    # Create LLM Tasks
    title_task = chain_title.ainvoke(user_query)
    reply_task = chain.ainvoke(user_query)

    # Execute each tasks
    title, reply = await asyncio.gather(title_task, reply_task)

    res = QueryResponse(
        title= title,
        reply=reply
    )
    return res