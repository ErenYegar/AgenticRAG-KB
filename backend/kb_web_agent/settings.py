from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
DEFAULT_MODEL = "glm-5.1"
DEFAULT_KNOWLEDGE_BASE_PATH = r"D:\workNote"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_MODEL_CACHE_DIR = r"D:\models"  # HuggingFace 模型缓存放 D 盘


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    knowledge_base_path: Path = Path("/mnt/d/workNote")
    max_steps: int = 4
    retrieval_top_k: int = 5
    temperature: float = 0.2
    timeout_seconds: int = 90
    # 第二阶段新增
    chroma_path: Path = Path("chroma_db")
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    model_cache_dir: Path = Path("/mnt/d/models")  # HuggingFace 缓存目录（D 盘）
    jwt_secret: str = ""
    jwt_expire_minutes: int = 480
    enable_vector_store: bool = True
    enable_graph: bool = False  # 图谱功能默认关闭，首次启动较慢


class RawSettings(BaseSettings):
    api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ARK_API_KEY", "LLM_API_KEY", "VOLCENGINE_API_KEY"),
    )
    base_url: str = Field(
        default=DEFAULT_BASE_URL,
        validation_alias=AliasChoices("ARK_BASE_URL", "LLM_BASE_URL"),
    )
    model: str = Field(default=DEFAULT_MODEL, validation_alias=AliasChoices("ARK_MODEL", "LLM_MODEL"))
    knowledge_base_path: str = Field(
        default=DEFAULT_KNOWLEDGE_BASE_PATH,
        validation_alias=AliasChoices("KNOWLEDGE_BASE_PATH", "KB_PATH"),
    )
    max_steps: int = Field(default=4, validation_alias="REACT_MAX_STEPS")
    retrieval_top_k: int = Field(default=5, validation_alias="RETRIEVAL_TOP_K")
    temperature: float = Field(default=0.2, validation_alias="LLM_TEMPERATURE")
    timeout_seconds: int = Field(default=90, validation_alias="LLM_TIMEOUT_SECONDS")
    # 第二阶段新增
    chroma_path: str = Field(default="chroma_db", validation_alias="CHROMA_PATH")
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, validation_alias="EMBEDDING_MODEL")
    model_cache_dir: str = Field(default=DEFAULT_MODEL_CACHE_DIR, validation_alias="MODEL_CACHE_DIR")
    jwt_secret: str = Field(default="", validation_alias="JWT_SECRET")
    jwt_expire_minutes: int = Field(default=480, validation_alias="JWT_EXPIRE_MINUTES")
    enable_vector_store: bool = Field(default=True, validation_alias="ENABLE_VECTOR_STORE")
    enable_graph: bool = Field(default=False, validation_alias="ENABLE_GRAPH")

    model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore")


def normalize_path(raw_path: str | os.PathLike[str]) -> Path:
    raw = str(raw_path).strip().strip('"').strip("'")
    match = re.match(r"^([a-zA-Z]):[\\/]*(.*)$", raw)
    if match and os.name != "nt":
        drive = match.group(1).lower()
        tail = match.group(2).replace("\\", "/").lstrip("/")
        return Path("/mnt") / drive / tail
    return Path(raw.replace("\\", "/")).expanduser()


def load_settings(env_file: Path | None = None) -> Settings:
    raw = RawSettings(_env_file=env_file if env_file and env_file.exists() else ".env")
    return Settings(
        api_key=raw.api_key,
        base_url=normalize_base_url(raw.base_url),
        model=raw.model,
        knowledge_base_path=normalize_path(raw.knowledge_base_path),
        max_steps=raw.max_steps,
        retrieval_top_k=raw.retrieval_top_k,
        temperature=raw.temperature,
        timeout_seconds=raw.timeout_seconds,
        chroma_path=Path(raw.chroma_path),
        embedding_model=raw.embedding_model,
        model_cache_dir=normalize_path(raw.model_cache_dir),
        jwt_secret=raw.jwt_secret,
        jwt_expire_minutes=raw.jwt_expire_minutes,
        enable_vector_store=raw.enable_vector_store,
        enable_graph=raw.enable_graph,
    )


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        return normalized[: -len(suffix)]
    return normalized
