from __future__ import annotations

import logging
from pathlib import Path
import re

import jieba
from rank_bm25 import BM25Okapi

from .schemas import DocumentChunk, SourceSnippet


LATIN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
CJK_SPAN_RE = re.compile(r"[\u3400-\u9fff]+")
jieba.setLogLevel(logging.WARNING)


def tokenize(text: str) -> list[str]:
    tokens = [item.lower() for item in LATIN_RE.findall(text)]
    for span in CJK_SPAN_RE.findall(text):
        tokens.append(span)
        tokens.extend(token for token in jieba.lcut(span) if token.strip())
        tokens.extend(span[index : index + 2] for index in range(len(span) - 1))
    return [token for token in tokens if token.strip()]


class KnowledgeRetriever:
    def __init__(self, chunks: list[DocumentChunk], root: Path | None = None):
        self.chunks = chunks
        self.root = root
        self.tokenized_chunks = [
            tokenize(f"{chunk.title}\n{chunk.title}\n{chunk.path.name}\n{chunk.text}")
            for chunk in chunks
        ]
        self.engine = BM25Okapi(self.tokenized_chunks) if self.tokenized_chunks else None

    def search(self, query: str, top_k: int = 5) -> list[SourceSnippet]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.engine:
            return []
        scores = self.engine.get_scores(query_tokens)
        unique_query_tokens = set(query_tokens)
        ranked: list[tuple[int, float]] = []
        for index, score in enumerate(scores):
            overlap = len(unique_query_tokens & set(self.tokenized_chunks[index])) / len(unique_query_tokens)
            combined = max(float(score), 0.0) + overlap * 0.01
            if combined > 0:
                ranked.append((index, combined))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [
            self._source_for(index, source_number=position, score=score)
            for position, (index, score) in enumerate(ranked[:top_k], start=1)
        ]

    def _source_for(self, index: int, source_number: int, score: float) -> SourceSnippet:
        chunk = self.chunks[index]
        path = chunk.path
        if self.root:
            try:
                path = path.relative_to(self.root)
            except ValueError:
                pass
        elif path.is_absolute():
            path = Path(path.name)
        return SourceSnippet(
            source_id=f"S{source_number}",
            file_path=path,
            title=chunk.title,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            score=round(score, 4),
            content=" ".join(chunk.text.split()),
        )
