from fastapi import APIRouter
from api.route import rag

api_router = APIRouter()
api_router.include_router(rag.router)