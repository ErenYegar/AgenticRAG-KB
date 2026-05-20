from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from .documents import read_document, split_markdown
from .retriever import KnowledgeRetriever

if TYPE_CHECKING:
    from .graph import KnowledgeGraph, extract_triples_with_llm
    from .vector_store import ChromaVectorStore

logger = logging.getLogger("kb_web_agent.ingestion")


class IngestState(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    EMBEDDING = "embedding"
    GRAPH = "graph"
    DONE = "done"
    ERROR = "error"


@dataclass
class IngestStatus:
    doc_id: str
    file_name: str
    department: str
    state: IngestState = IngestState.PENDING
    progress: float = 0.0  # 0~1
    chunks_total: int = 0
    chunks_done: int = 0
    error: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0


# 全局进度字典（内存中，进程重启丢失）
_ingest_registry: dict[str, IngestStatus] = {}


def get_status(doc_id: str) -> IngestStatus | None:
    return _ingest_registry.get(doc_id)


def list_statuses() -> list[IngestStatus]:
    return list(_ingest_registry.values())


async def ingest_file(
    file_path: Path,
    department: str,
    doc_id: str | None,
    vector_store: "ChromaVectorStore | None",
    bm25_retriever: KnowledgeRetriever | None,
    knowledge_graph: "KnowledgeGraph | None" = None,
    llm_client=None,
    max_chars: int = 1400,
) -> IngestStatus:
    """统一入库 pipeline（在线程池中运行 CPU 密集步骤，避免阻塞事件循环）。

    流程：
    1. 解析文档 → 纯文本
    2. 分块
    3. 向量化并写入 Chroma
    4. 更新 BM25 索引（内存）
    5. LLM 抽取三元组写入图谱（可选）
    """
    doc_id = doc_id or str(uuid.uuid4())
    status = IngestStatus(
        doc_id=doc_id,
        file_name=file_path.name,
        department=department,
    )
    _ingest_registry[doc_id] = status

    try:
        # Step 1: 解析
        status.state = IngestState.PARSING
        logger.info("[Ingest] 开始解析 doc_id=%s file=%s", doc_id, file_path.name)
        text = await asyncio.get_event_loop().run_in_executor(None, read_document, file_path)
        if not text.strip():
            raise ValueError(f"文档内容为空: {file_path.name}")

        # Step 2: 分块
        chunks = await asyncio.get_event_loop().run_in_executor(
            None, split_markdown, text, file_path, max_chars
        )
        status.chunks_total = len(chunks)
        logger.info("[Ingest] 分块完成 doc_id=%s chunks=%d", doc_id, len(chunks))

        # Step 3: 向量化写入 Chroma
        if vector_store is not None:
            status.state = IngestState.EMBEDDING
            n = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: vector_store.add_chunks(chunks, department=department, doc_id=doc_id),
            )
            status.chunks_done = n
            status.progress = 0.7
            logger.info("[Ingest] 向量入库完成 doc_id=%s n=%d", doc_id, n)

        # Step 4: 更新 BM25（重建整个索引，适合中小规模）
        if bm25_retriever is not None:
            await asyncio.get_event_loop().run_in_executor(
                None, _update_bm25, bm25_retriever, chunks
            )
            status.progress = 0.85
            logger.info("[Ingest] BM25 索引已更新 doc_id=%s", doc_id)

        # Step 5: 图谱抽取（可选，每个 chunk 抽取一次，LLM 调用较慢）
        if knowledge_graph is not None and llm_client is not None:
            status.state = IngestState.GRAPH
            from .graph import extract_triples_with_llm

            for chunk in chunks[:10]:  # 限制抽取 chunk 数，避免过多 LLM 调用
                triples = await asyncio.get_event_loop().run_in_executor(
                    None, extract_triples_with_llm, chunk.text, llm_client
                )
                if triples:
                    knowledge_graph.add_triples(
                        triples, doc_id=doc_id, file_path=str(file_path)
                    )
            knowledge_graph.save()
            status.progress = 0.95
            logger.info("[Ingest] 图谱抽取完成 doc_id=%s", doc_id)

        status.state = IngestState.DONE
        status.progress = 1.0
        status.finished_at = time.time()
        logger.info("[Ingest] 入库完成 doc_id=%s", doc_id)

    except Exception as exc:
        status.state = IngestState.ERROR
        status.error = str(exc)
        status.finished_at = time.time()
        logger.error("[Ingest] 入库失败 doc_id=%s err=%s", doc_id, exc)

    return status


def _update_bm25(retriever: KnowledgeRetriever, new_chunks) -> None:
    """将新 chunks 追加到 BM25 索引（原地修改 retriever）。"""
    from .retriever import tokenize
    from rank_bm25 import BM25Okapi

    retriever.chunks.extend(new_chunks)
    retriever.tokenized_chunks.extend(
        tokenize(f"{c.title}\n{c.title}\n{c.path.name}\n{c.text}") for c in new_chunks
    )
    retriever.engine = BM25Okapi(retriever.tokenized_chunks) if retriever.tokenized_chunks else None


async def delete_doc(
    doc_id: str,
    vector_store: "ChromaVectorStore | None",
    knowledge_graph: "KnowledgeGraph | None" = None,
) -> bool:
    """删除文档：从向量库和图谱中移除该 doc_id 的所有数据。"""
    success = True
    if vector_store is not None:
        try:
            n = await asyncio.get_event_loop().run_in_executor(
                None, vector_store.delete_doc, doc_id
            )
            logger.info("[Ingest] 向量删除 doc_id=%s n=%d", doc_id, n)
        except Exception as exc:
            logger.error("[Ingest] 向量删除失败 doc_id=%s err=%s", doc_id, exc)
            success = False

    if knowledge_graph is not None:
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, knowledge_graph.delete_doc, doc_id
            )
            knowledge_graph.save()
        except Exception as exc:
            logger.error("[Ingest] 图谱删除失败 doc_id=%s err=%s", doc_id, exc)
            success = False

    _ingest_registry.pop(doc_id, None)
    return success
