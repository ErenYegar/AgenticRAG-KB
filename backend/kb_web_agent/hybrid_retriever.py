from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from .retriever import KnowledgeRetriever, tokenize
from .schemas import DocumentChunk, SourceSnippet

if TYPE_CHECKING:
    from .graph import GraphRetriever
    from .vector_store import ChromaVectorStore

logger = logging.getLogger("kb_web_agent.hybrid_retriever")

# 三路融合权重
WEIGHT_VECTOR = 0.6
WEIGHT_BM25 = 0.3
WEIGHT_GRAPH = 0.1


class HybridRetriever:
    """三路混合检索器：向量检索 + BM25 + 知识图谱，按权重融合后返回 Top-K 片段。

    设计原则：
    - 各路检索分别独立运行（线程并行），最后融合排序
    - 若 Chroma 未初始化（is_ready=False），自动降级为纯 BM25
    - 若图谱未初始化，图谱贡献跳过（不影响主流程）
    """

    def __init__(
        self,
        bm25_retriever: KnowledgeRetriever,
        vector_store: "ChromaVectorStore | None" = None,
        graph_retriever: "GraphRetriever | None" = None,
        root: Path | None = None,
    ) -> None:
        self.bm25 = bm25_retriever
        self.vector_store = vector_store
        self.graph = graph_retriever
        self.root = root

    # ------------------------------------------------------------------
    # 公共检索接口（同步，内部并行）
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        department_filter: list[str] | None = None,
    ) -> list[SourceSnippet]:
        """混合检索，返回融合排序后的 Top-K SourceSnippet。"""
        t0 = time.perf_counter()
        use_vector = self.vector_store is not None and self.vector_store.is_ready()

        if use_vector:
            results = self._hybrid_search(query, top_k=top_k * 2, department_filter=department_filter)
        else:
            logger.info("[Hybrid] Chroma 未就绪，降级为纯 BM25")
            results = self.bm25.search(query, top_k=top_k)
            for i, r in enumerate(results):
                r.source_id = f"S{i + 1}"
            elapsed = time.perf_counter() - t0
            logger.info("[Hybrid] BM25-only search elapsed=%.3fs n=%d", elapsed, len(results))
            return results

        elapsed = time.perf_counter() - t0
        logger.info("[Hybrid] hybrid search elapsed=%.3fs n=%d", elapsed, len(results))
        return results[:top_k]

    # ------------------------------------------------------------------
    # 内部：并行三路检索 + 融合
    # ------------------------------------------------------------------

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        department_filter: list[str] | None,
    ) -> list[SourceSnippet]:
        futures_map: dict = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures_map["vector"] = executor.submit(
                self._vector_search, query, top_k, department_filter
            )
            futures_map["bm25"] = executor.submit(self._bm25_search, query, top_k)
            if self.graph is not None:
                futures_map["graph"] = executor.submit(self._graph_search, query)

        vector_results: list[SourceSnippet] = futures_map["vector"].result()
        bm25_results: list[SourceSnippet] = futures_map["bm25"].result()
        graph_hits: set[str] = futures_map["graph"].result() if "graph" in futures_map else set()

        return self._fuse(query, vector_results, bm25_results, graph_hits, top_k)

    def _vector_search(
        self,
        query: str,
        top_k: int,
        department_filter: list[str] | None,
    ) -> list[SourceSnippet]:
        try:
            return self.vector_store.search(query, top_k=top_k, department_filter=department_filter)
        except Exception as exc:
            logger.warning("[Hybrid] 向量检索异常 err=%s", exc)
            return []

    def _bm25_search(self, query: str, top_k: int) -> list[SourceSnippet]:
        try:
            return self.bm25.search(query, top_k=top_k)
        except Exception as exc:
            logger.warning("[Hybrid] BM25 检索异常 err=%s", exc)
            return []

    def _graph_search(self, query: str) -> set[str]:
        """返回与 query 相关的文件路径集合（用于 graph_hit_bonus 计算）。"""
        try:
            return self.graph.search_paths(query)
        except Exception as exc:
            logger.warning("[Hybrid] 图谱检索异常 err=%s", exc)
            return set()

    # ------------------------------------------------------------------
    # 融合排序
    # ------------------------------------------------------------------

    def _fuse(
        self,
        query: str,
        vector_results: list[SourceSnippet],
        bm25_results: list[SourceSnippet],
        graph_hits: set[str],
        top_k: int,
    ) -> list[SourceSnippet]:
        """按 combined_score = cosine×0.6 + bm25_norm×0.3 + graph_hit×0.1 融合。"""

        # 用文件路径+行号作为唯一 key 聚合同一片段
        key_map: dict[str, dict] = {}

        # 向量分数（已归一化到 [0,1]）
        for snip in vector_results:
            k = self._snippet_key(snip)
            entry = key_map.setdefault(k, {"snip": snip, "vec": 0.0, "bm25": 0.0})
            entry["vec"] = snip.score

        # BM25 分数归一化：取最大值做 min-max 归一化
        if bm25_results:
            max_bm25 = max(s.score for s in bm25_results) or 1.0
            for snip in bm25_results:
                k = self._snippet_key(snip)
                entry = key_map.setdefault(k, {"snip": snip, "vec": 0.0, "bm25": 0.0})
                entry["bm25"] = snip.score / max_bm25

        # 图谱命中 bonus
        graph_file_paths = {str(h) for h in graph_hits}

        ranked: list[tuple[str, float]] = []
        for k, entry in key_map.items():
            graph_bonus = WEIGHT_GRAPH if str(entry["snip"].file_path) in graph_file_paths else 0.0
            combined = entry["vec"] * WEIGHT_VECTOR + entry["bm25"] * WEIGHT_BM25 + graph_bonus
            ranked.append((k, combined))

        ranked.sort(key=lambda x: x[1], reverse=True)

        results: list[SourceSnippet] = []
        for pos, (k, combined_score) in enumerate(ranked[:top_k], start=1):
            entry = key_map[k]
            snip = entry["snip"]
            results.append(
                SourceSnippet(
                    source_id=f"S{pos}",
                    file_path=snip.file_path,
                    title=snip.title,
                    line_start=snip.line_start,
                    line_end=snip.line_end,
                    score=round(combined_score, 4),
                    content=snip.content,
                )
            )
        return results

    @staticmethod
    def _snippet_key(snip: SourceSnippet) -> str:
        return f"{snip.file_path}:{snip.line_start}-{snip.line_end}"
