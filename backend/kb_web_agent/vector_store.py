from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .schemas import DocumentChunk, SourceSnippet

if TYPE_CHECKING:
    import chromadb

logger = logging.getLogger("kb_web_agent.vector_store")

_GLOBAL_COLLECTION = "kb_global"


class ChromaVectorStore:
    """基于 Chroma 的向量存储，支持按 department 标签隔离检索。

    - 所有文档写入同一个 collection，通过 metadata["department"] 过滤
    - department="" 表示公共文档，admin 可访问所有，user 只能访问自己的 department 或公共文档
    """

    def __init__(self, persist_directory: str | Path, embedder=None) -> None:
        self.persist_directory = str(persist_directory)
        self._embedder = embedder
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    # ------------------------------------------------------------------
    # 内部初始化（懒加载，避免 import 阻塞）
    # ------------------------------------------------------------------

    def _get_client(self) -> "chromadb.ClientAPI":
        if self._client is None:
            import chromadb

            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info("[VectorStore] Chroma 客户端初始化完成 path=%s", self.persist_directory)
        return self._client

    def _get_collection(self) -> "chromadb.Collection":
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=_GLOBAL_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "[VectorStore] 使用 collection=%s count=%d",
                _GLOBAL_COLLECTION,
                self._collection.count(),
            )
        return self._collection

    def _embedder_instance(self):
        if self._embedder is None:
            from .embedding import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    # ------------------------------------------------------------------
    # 写操作
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        department: str = "",
        doc_id: str = "",
        batch_size: int = 64,
    ) -> int:
        """将文档块向量化并写入 Chroma，返回实际写入条数。"""
        if not chunks:
            return 0
        collection = self._get_collection()
        embedder = self._embedder_instance()

        texts = [chunk.text for chunk in chunks]
        ids = [f"{doc_id}::{chunk.id}" if doc_id else chunk.id for chunk in chunks]
        metadatas = [
            {
                "department": department,
                "doc_id": doc_id,
                "file_path": str(chunk.path),
                "title": chunk.title,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "chunk_id": chunk.id,
            }
            for chunk in chunks
        ]

        total = 0
        t0 = time.perf_counter()
        for start in range(0, len(chunks), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_ids = ids[start : start + batch_size]
            batch_meta = metadatas[start : start + batch_size]
            batch_docs = batch_texts

            embeddings = embedder.encode(batch_texts)
            collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_docs,
                metadatas=batch_meta,
            )
            total += len(batch_texts)

        logger.info(
            "[VectorStore] add_chunks n=%d doc_id=%r elapsed=%.3fs",
            total,
            doc_id,
            time.perf_counter() - t0,
        )
        return total

    def delete_doc(self, doc_id: str) -> int:
        """删除属于某个 doc_id 的所有向量，返回删除条数。"""
        collection = self._get_collection()
        results = collection.get(where={"doc_id": doc_id}, include=[])
        ids = results.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            logger.info("[VectorStore] delete_doc doc_id=%r n=%d", doc_id, len(ids))
        return len(ids)

    # ------------------------------------------------------------------
    # 读操作
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        department_filter: list[str] | None = None,
    ) -> list[SourceSnippet]:
        """语义检索，返回 SourceSnippet 列表（score 为余弦相似度，越高越相关）。

        department_filter:
          - None / [] → 无过滤，查全库（admin 使用）
          - ["dept_a"] → 只返回 dept_a 或公共文档（user 使用）
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        embedder = self._embedder_instance()
        query_vec = embedder.encode_query(query)

        where: dict | None = None
        if department_filter:
            where = {
                "$or": [
                    {"department": {"$in": department_filter}},
                    {"department": ""},
                ]
            }

        t0 = time.perf_counter()
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, max(collection.count(), 1)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        logger.info("[VectorStore] search query=%r elapsed=%.3fs", query[:60], time.perf_counter() - t0)

        snippets: list[SourceSnippet] = []
        if not results["ids"] or not results["ids"][0]:
            return snippets

        for i, doc_id_val in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            cosine_sim = max(0.0, 1.0 - distance)
            text = results["documents"][0][i]

            snippets.append(
                SourceSnippet(
                    source_id=f"S{i + 1}",
                    file_path=Path(meta.get("file_path", "")),
                    title=meta.get("title", ""),
                    line_start=int(meta.get("line_start", 0)),
                    line_end=int(meta.get("line_end", 0)),
                    score=round(cosine_sim, 4),
                    content=text[:800],
                )
            )
        return snippets

    def count(self) -> int:
        """返回 collection 中的向量总数。"""
        return self._get_collection().count()

    def is_ready(self) -> bool:
        """Chroma 是否已有数据（用于决定是否走向量检索分支）。"""
        try:
            return self.count() > 0
        except Exception:
            return False

    def export_metadata(self) -> list[dict]:
        """导出所有文档元数据，用于文档列表接口。"""
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        results = collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in results.get("metadatas", []):
            doc_id = meta.get("doc_id", "")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "file_path": meta.get("file_path", ""),
                    "department": meta.get("department", ""),
                }
        return list(seen.values())
