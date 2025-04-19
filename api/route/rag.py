from fastapi import APIRouter, Request

from model.gpt import chain

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

    return {
        "reply": chain.invoke(user_query)
    }