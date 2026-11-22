"""Pydantic request/response schemas for the REST API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to ingest")
    source: Optional[str] = Field(default="inline", description="Source label / filename")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Extra metadata")


class IngestResponse(BaseModel):
    document_id: str
    chunks_added: int
    source: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Override retrieval count")
    include_sources: bool = Field(default=True, description="Include retrieved source chunks")


class SourceChunk(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk] = Field(default_factory=list)
    model: str
    grounded: bool = True


class HealthResponse(BaseModel):
    status: str
    version: str
    provider: str
    documents: int


class ErrorResponse(BaseModel):
    detail: str
