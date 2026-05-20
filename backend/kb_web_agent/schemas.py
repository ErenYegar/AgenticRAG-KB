from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    path: Path
    title: str
    text: str
    line_start: int
    line_end: int


class SourceSnippet(BaseModel):
    source_id: str = Field(serialization_alias="sourceId")
    file_path: Path = Field(serialization_alias="filePath")
    title: str
    line_start: int = Field(serialization_alias="lineStart")
    line_end: int = Field(serialization_alias="lineEnd")
    score: float
    content: str

    model_config = {"populate_by_name": True}


class HealthResponse(BaseModel):
    model: str
    base_url: str = Field(serialization_alias="baseUrl")
    knowledge_base_path: Path = Field(serialization_alias="knowledgeBasePath")
    markdown_file_count: int = Field(serialization_alias="markdownFileCount")
    api_key_configured: bool = Field(serialization_alias="apiKeyConfigured")

    model_config = {"populate_by_name": True}


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = Field(default=None, validation_alias="topK")

    model_config = {"populate_by_name": True}


class SearchResponse(BaseModel):
    sources: list[SourceSnippet]


class ChatRequest(BaseModel):
    message: str
    show_trace: bool = Field(default=True, validation_alias="showTrace")

    model_config = {"populate_by_name": True}


class TraceStep(BaseModel):
    action: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceSnippet]
    trace: list[TraceStep]


# ---------------------------------------------------------------------------
# 第二阶段新增 Schema
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    departments: list[str]


class DocumentMeta(BaseModel):
    doc_id: str = Field(serialization_alias="docId")
    file_name: str = Field(serialization_alias="fileName")
    department: str
    state: str
    progress: float
    chunks_total: int = Field(serialization_alias="chunksTotal")
    error: str = ""

    model_config = {"populate_by_name": True}


class IngestRequest(BaseModel):
    department: str = "default"
    enable_graph: bool = Field(default=False, validation_alias="enableGraph")

    model_config = {"populate_by_name": True}
