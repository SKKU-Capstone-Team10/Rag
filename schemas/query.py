from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    title: str
    reply: str