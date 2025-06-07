from typing import List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    symbol: str
    is_first_chat: bool

class QueryResponse(BaseModel):
    title: Optional[str] = None
    reply: str
    refs: List