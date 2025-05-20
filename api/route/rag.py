from fastapi import APIRouter, Request

import asyncio

from model.gpt import chain, chain_title

router = APIRouter(prefix="/rag", tags=['RAG'])

@router.get('/')
def greet():
    return "RAG api"


@router.post('/')
async def reply(req: Request):
    body = await req.json()
    user_query = body.get("query")

    if not user_query:
        return {"error": "Missing 'query' in request body"}

    # Create LLM Tasks
    title_task = chain_title.ainvoke(user_query)
    reply_task = chain.ainvoke(user_query)

    # Execute each tasks
    title, answer = await asyncio.gather(title_task, reply_task)

    return {
        "title": title,
        "reply": answer
    }