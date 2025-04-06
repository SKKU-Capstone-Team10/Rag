from fastapi import APIRouter

router = APIRouter(prefix="/rag", tags=['RAG'])

@router.get('/')
def greet():
    return "RAG api"

@router.post('/')
def reply():
    return {
        "reply": "Reply from rag server"
    }