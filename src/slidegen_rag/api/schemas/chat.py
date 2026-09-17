from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    web_search_enabled: bool = False
    generate_provider: str = "gemini"


class CitationOut(BaseModel):
    node_id: int
    text: str
    priority: int


class ChatResponse(BaseModel):
    answer: str
    standalone_query: str
    citations: list[CitationOut]


class SourceExcerptOut(BaseModel):
    node_id: int
    excerpt: str | None = None
