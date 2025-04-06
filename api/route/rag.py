from fastapi import APIRouter

router = APIRouter(prefix="/rag", tags=['RAG'])

@router.get('/')
def greet():
    return "RAG api"