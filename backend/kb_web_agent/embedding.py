from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger("kb_web_agent.embedding")


class BGEEmbedder:
    """懒加载 BAAI/bge-m3，首次调用 encode() 时才下载/加载模型，避免阻塞服务启动。

    cache_folder: HuggingFace 模型缓存目录。
                  设置后模型文件下载到该目录，而非默认的 ~/.cache/huggingface（C 盘）。
                  建议设为 D 盘路径，例如 /mnt/d/models（WSL 中）。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_folder: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_folder: str | None = str(cache_folder) if cache_folder else None
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> "SentenceTransformer":
        if self._model is None:
            logger.info(
                "[Embedding] 正在加载模型 %s cache=%s …",
                self.model_name,
                self.cache_folder or "默认(~/.cache)",
            )
            t0 = time.perf_counter()

            # 同时设置环境变量，确保 transformers / huggingface_hub 也使用同一缓存目录
            if self.cache_folder:
                Path(self.cache_folder).mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("HF_HOME", self.cache_folder)
                os.environ.setdefault("TRANSFORMERS_CACHE", self.cache_folder)
                os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", self.cache_folder)

            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_folder,
            )
            logger.info("[Embedding] 模型加载完成 elapsed=%.2fs", time.perf_counter() - t0)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """将文本列表编码为归一化向量列表。"""
        if not texts:
            return []
        t0 = time.perf_counter()
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        logger.info(
            "[Embedding] encode n=%d elapsed=%.3fs",
            len(texts),
            time.perf_counter() - t0,
        )
        return vectors

    def encode_query(self, query: str) -> list[float]:
        """编码单条查询（bge-m3 对 query 有特殊前缀处理）。"""
        return self.encode([f"Represent this sentence for searching relevant passages: {query}"])[0]


_default_embedder: BGEEmbedder | None = None


def get_embedder(
    model_name: str = "BAAI/bge-m3",
    cache_folder: str | Path | None = None,
) -> BGEEmbedder:
    """获取全局单例 Embedder，避免重复加载模型。

    cache_folder: 传入即可将模型缓存到 D 盘，不传则沿用已有单例的设置。
    """
    global _default_embedder
    if _default_embedder is None or _default_embedder.model_name != model_name:
        _default_embedder = BGEEmbedder(model_name, cache_folder=cache_folder)
    return _default_embedder
