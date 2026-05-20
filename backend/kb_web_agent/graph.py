from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger("kb_web_agent.graph")

TRIPLE_EXTRACT_PROMPT = """你是一个知识图谱构建助手。
请从下面的文本中提取实体关系三元组，格式为 JSON 数组：
[{"subject":"...","relation":"...","object":"..."},...]
每个三元组应简洁准确，主语和宾语是名词短语，关系是动词短语。
只输出 JSON 数组，不要其他文字。

文本：
{text}
"""


class KnowledgeGraph:
    """基于 NetworkX 的轻量级知识图谱，支持从文本抽取三元组并持久化。"""

    def __init__(self, persist_path: Path | None = None) -> None:
        self.persist_path = persist_path
        self._graph: nx.DiGraph | None = None
        # doc_id -> set[file_path]：记录每篇文档贡献的节点，用于删除时清理
        self._doc_nodes: dict[str, set[str]] = {}

    @property
    def graph(self) -> "nx.DiGraph":
        if self._graph is None:
            try:
                import networkx as nx
            except ImportError:
                logger.warning("[Graph] networkx 未安装，图谱功能不可用")
                raise
            self._graph = nx.DiGraph()
            if self.persist_path and self.persist_path.exists():
                self._load()
        return self._graph

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self.persist_path.read_text(encoding="utf-8"))
            import networkx as nx

            self._graph = nx.node_link_graph(data["graph"])
            self._doc_nodes = {k: set(v) for k, v in data.get("doc_nodes", {}).items()}
            logger.info("[Graph] 加载图谱 nodes=%d edges=%d", self._graph.number_of_nodes(), self._graph.number_of_edges())
        except Exception as exc:
            logger.warning("[Graph] 加载失败，使用空图谱 err=%s", exc)
            import networkx as nx

            self._graph = nx.DiGraph()

    def save(self) -> None:
        if self.persist_path is None:
            return
        try:
            import networkx as nx

            data = {
                "graph": nx.node_link_data(self._graph),
                "doc_nodes": {k: list(v) for k, v in self._doc_nodes.items()},
            }
            self.persist_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(
                "[Graph] 保存图谱 nodes=%d edges=%d path=%s",
                self._graph.number_of_nodes(),
                self._graph.number_of_edges(),
                self.persist_path,
            )
        except Exception as exc:
            logger.warning("[Graph] 保存失败 err=%s", exc)

    # ------------------------------------------------------------------
    # 写操作
    # ------------------------------------------------------------------

    def add_triples(self, triples: list[dict], doc_id: str = "", file_path: str = "") -> int:
        """将三元组列表写入图谱，返回成功添加数。"""
        g = self.graph
        nodes_added: set[str] = set()
        count = 0
        for triple in triples:
            subj = str(triple.get("subject", "")).strip()
            rel = str(triple.get("relation", "")).strip()
            obj = str(triple.get("object", "")).strip()
            if not (subj and rel and obj):
                continue
            if not g.has_node(subj):
                g.add_node(subj, docs=set())
            if not g.has_node(obj):
                g.add_node(obj, docs=set())
            g.nodes[subj].setdefault("docs", set()).add(file_path)
            g.nodes[obj].setdefault("docs", set()).add(file_path)
            g.add_edge(subj, obj, relation=rel, doc_id=doc_id, file_path=file_path)
            nodes_added.update([subj, obj])
            count += 1

        if doc_id:
            self._doc_nodes.setdefault(doc_id, set()).update(nodes_added)
        return count

    def delete_doc(self, doc_id: str) -> None:
        """删除属于某篇文档的图谱节点（若节点仍有其他文档引用则保留）。"""
        nodes = self._doc_nodes.pop(doc_id, set())
        if not nodes:
            return
        g = self.graph
        to_remove = []
        for node in nodes:
            if not g.has_node(node):
                continue
            # 仅当该节点只被此文档引用时才删除
            other_docs = {
                d for _, _, data in g.in_edges(node, data=True) if data.get("doc_id", "") != doc_id
            } | {
                d for _, _, data in g.out_edges(node, data=True) if data.get("doc_id", "") != doc_id
            }
            if not other_docs:
                to_remove.append(node)
        g.remove_nodes_from(to_remove)
        logger.info("[Graph] delete_doc doc_id=%r removed_nodes=%d", doc_id, len(to_remove))

    # ------------------------------------------------------------------
    # 读操作
    # ------------------------------------------------------------------

    def search_paths(self, query: str) -> set[str]:
        """返回与 query 关键词相关的文件路径集合（用于 hybrid_retriever graph_hit 计算）。"""
        try:
            g = self.graph
        except Exception:
            return set()
        query_lower = query.lower()
        file_paths: set[str] = set()
        for node in g.nodes():
            if query_lower in str(node).lower():
                # 收集与该节点相关的所有边的 file_path
                for _, _, data in g.out_edges(node, data=True):
                    fp = data.get("file_path", "")
                    if fp:
                        file_paths.add(fp)
                for _, _, data in g.in_edges(node, data=True):
                    fp = data.get("file_path", "")
                    if fp:
                        file_paths.add(fp)
        return file_paths

    def neighbors_text(self, entity: str, max_hops: int = 2) -> str:
        """返回实体邻居关系的文本描述，用于在 prompt 中引用图谱上下文。"""
        try:
            g = self.graph
        except Exception:
            return ""
        if not g.has_node(entity):
            return ""
        lines: list[str] = []
        visited: set[str] = {entity}
        frontier = [entity]
        for _ in range(max_hops):
            next_frontier: list[str] = []
            for node in frontier:
                for _, nbr, data in g.out_edges(node, data=True):
                    rel = data.get("relation", "→")
                    lines.append(f"{node} --[{rel}]--> {nbr}")
                    if nbr not in visited:
                        visited.add(nbr)
                        next_frontier.append(nbr)
            frontier = next_frontier
            if not frontier:
                break
        return "\n".join(lines[:50])

    def is_ready(self) -> bool:
        try:
            return self.graph.number_of_nodes() > 0
        except Exception:
            return False


class GraphRetriever:
    """面向 HybridRetriever 的图谱检索适配器。"""

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        self.kg = knowledge_graph

    def search_paths(self, query: str) -> set[str]:
        return self.kg.search_paths(query)


def extract_triples_with_llm(text: str, llm_client) -> list[dict]:
    """调用 LLM 从文本中抽取三元组，返回 list[{subject, relation, object}]。"""
    if len(text) > 2000:
        text = text[:2000]
    prompt = TRIPLE_EXTRACT_PROMPT.format(text=text)
    try:
        raw = llm_client.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        # 尝试从 raw 中提取 JSON 数组
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            triples = json.loads(match.group())
            if isinstance(triples, list):
                return triples
    except Exception as exc:
        logger.warning("[Graph] LLM 三元组抽取失败 err=%s", exc)
    return []
