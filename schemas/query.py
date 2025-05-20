from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    is_first_chat: bool

class QueryResponse(BaseModel):
    title: str
    reply: str
    ticker: str