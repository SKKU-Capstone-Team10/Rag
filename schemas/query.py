from typing import List
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    symbol: str
    is_first_chat: bool

class QueryResponse(BaseModel):
    title: str
    reply: str
    refs: List