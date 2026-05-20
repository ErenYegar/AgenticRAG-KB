from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

from .retriever import KnowledgeRetriever
from .schemas import ChatResponse, SourceSnippet, TraceStep
from .settings import Settings

if TYPE_CHECKING:
    from .hybrid_retriever import HybridRetriever

logger = logging.getLogger("kb_web_agent.agent")

SYSTEM_PROMPT = """你是一个本地 Markdown 知识库问答 Agent。
你必须按 ReAct 风格工作：先判断是否需要检索，再调用工具观察证据，最后回答。
每轮只输出一个 JSON 对象。
工具调用格式：{"thought":"为什么检索","action":"search_knowledge_base","action_input":"查询"}
最终回答格式：{"thought":"证据是否足够","action":"final_answer","answer":"带 [S1] 来源标记的中文答案"}
"""


class ReActAgent:
    def __init__(self, settings: Settings, llm_client, retriever: KnowledgeRetriever):
        self.settings = settings
        self.llm_client = llm_client
        self.retriever = retriever

    def answer(self, question: str) -> ChatResponse:
        t_total = time.perf_counter()
        logger.info("[Agent] ===== 开始处理问题 question=%r =====", question)

        # 预搜索：先检索知识库，把结果直接注入首轮 prompt，省去模型"决定搜索"的一次 LLM 调用
        logger.info("[Agent] 预搜索 query=%r top_k=%d", question, self.settings.retrieval_top_k)
        t_ret = time.perf_counter()
        latest_sources = self.retriever.search(question, top_k=self.settings.retrieval_top_k)
        logger.info("[Agent] 预搜索完成 elapsed=%.3fs 命中=%d条", time.perf_counter() - t_ret, len(latest_sources))
        pre_observation = format_sources(latest_sources)

        trace: list[TraceStep] = [
            TraceStep(action="pre_search", content=pre_observation),
        ]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n\n"
                    f"已自动检索知识库，Observation:\n{pre_observation}\n\n"
                    "请根据以上证据直接给出最终答案，按 JSON 格式输出。"
                ),
            },
        ]

        for step in range(self.settings.max_steps):
            logger.info("[Agent] ----- Step %d/%d -----", step + 1, self.settings.max_steps)
            raw = self.llm_client.complete(messages, temperature=self.settings.temperature)
            parsed = parse_json_object(raw)
            if not parsed:
                logger.warning("[Agent] Step %d JSON解析失败，要求模型重试 raw=%r", step + 1, raw[:200])
                trace.append(TraceStep(action="format_error", content=raw))
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "请只输出一个有效 JSON 对象。"})
                continue

            action = str(parsed.get("action", "")).strip()
            thought = str(parsed.get("thought", "")).strip()
            logger.info("[Agent] Step %d action=%r thought=%r", step + 1, action, thought[:100])
            trace.append(TraceStep(action=action, content=thought))

            if action == "search_knowledge_base":
                query = str(parsed.get("action_input", "")).strip() or question
                logger.info("[Agent] 检索知识库 query=%r top_k=%d", query, self.settings.retrieval_top_k)
                t_ret = time.perf_counter()
                latest_sources = self.retriever.search(query, top_k=self.settings.retrieval_top_k)
                logger.info("[Agent] 检索完成 elapsed=%.3fs 命中=%d条", time.perf_counter() - t_ret, len(latest_sources))
                observation = format_sources(latest_sources)
                trace.append(TraceStep(action="observation", content=observation))
                messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
                messages.append({"role": "user", "content": f"Observation:\n{observation}\n继续按 JSON 格式行动。"})
                continue

            if action == "final_answer":
                elapsed = time.perf_counter() - t_total
                logger.info("[Agent] ===== 完成 total_elapsed=%.2fs steps=%d =====", elapsed, step + 1)
                return ChatResponse(
                    answer=str(parsed.get("answer", "")).strip(),
                    sources=latest_sources,
                    trace=trace,
                )

            logger.warning("[Agent] Step %d 未知action=%r", step + 1, action)
            messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
            messages.append({"role": "user", "content": "未知 action。只能使用 search_knowledge_base 或 final_answer。"})

        elapsed = time.perf_counter() - t_total
        logger.warning("[Agent] 达到最大步数 max_steps=%d total_elapsed=%.2fs，触发fallback", self.settings.max_steps, elapsed)
        fallback_sources = self.retriever.search(question, top_k=self.settings.retrieval_top_k)
        trace.append(TraceStep(action="fallback_search", content=format_sources(fallback_sources)))
        return ChatResponse(
            answer="知识库检索已完成，但模型未在限定步骤内给出最终答案。请查看下方来源片段。",
            sources=fallback_sources,
            trace=trace,
        )

    def answer_stream(self, question: str) -> Iterator[str]:
        """SSE 流式生成器，按顺序 yield JSON 行：
        1. {"type":"sources", "sources":[...], "trace":[...]}  — 检索完成立即发送
        2. {"type":"token",   "content":"..."}                 — 每个 token
        3. {"type":"done"}                                     — 结束
        """
        t_total = time.perf_counter()
        logger.info("[Agent/stream] 开始 question=%r", question)

        # 预搜索
        t_ret = time.perf_counter()
        sources = self.retriever.search(question, top_k=self.settings.retrieval_top_k)
        logger.info("[Agent/stream] 预搜索完成 elapsed=%.3fs 命中=%d", time.perf_counter() - t_ret, len(sources))
        pre_observation = format_sources(sources)
        trace = [TraceStep(action="pre_search", content=pre_observation)]

        # 先把来源发给前端，让用户立即看到检索结果
        yield json.dumps(
            {
                "type": "sources",
                "sources": [s.model_dump(by_alias=True, mode="json") for s in sources],
                "trace": [t.model_dump(by_alias=True, mode="json") for t in trace],
            },
            ensure_ascii=False,
        ) + "\n"

        stream_system = (
            "你是一个本地 Markdown 知识库问答助手。"
            "请根据用户提供的知识库片段直接给出简洁准确的中文答案，"
            "用 [S1][S2] 等标记引用来源。不要输出 JSON，不要多余解释，直接回答即可。"
        )
        messages = [
            {"role": "system", "content": stream_system},
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n\n"
                    f"知识库检索结果：\n{pre_observation}\n\n"
                    "请直接给出答案："
                ),
            },
        ]

        # 流式 LLM 调用，边收 token 边转发给前端
        for token in self.llm_client.stream_complete(messages, temperature=self.settings.temperature):
            yield json.dumps({"type": "token", "content": token}, ensure_ascii=False) + "\n"

        elapsed = time.perf_counter() - t_total
        logger.info("[Agent/stream] 完成 total_elapsed=%.2fs", elapsed)
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"


def format_sources(sources: list[SourceSnippet]) -> str:
    if not sources:
        return "未在知识库中找到相关 Markdown 片段。"
    return "\n\n".join(
        [
            f"[{source.source_id}] {source.file_path}:{source.line_start}-{source.line_end}\n"
            f"Title: {source.title}\nContent: {source.content}"
            for source in sources
        ]
    )


def parse_json_object(raw: str) -> dict | None:
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    candidates = [text]
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


# ---------------------------------------------------------------------------
# AgenticRAGAgent — 第二阶段：混合检索 + 查询分析 + 质量评估
# ---------------------------------------------------------------------------

QUERY_ANALYSIS_PROMPT = """请分析以下用户问题的类型，输出一个 JSON 对象：
- simple: 直接可以检索回答的简单问题
- complex: 需要多步推理或多个子问题的复杂问题
- entity_relation: 主要询问实体间关系的问题

只输出 JSON，格式：{{"type":"simple"|"complex"|"entity_relation","sub_queries":["子问题1","子问题2"]}}
sub_queries 仅在 complex 时非空，最多 3 个。

用户问题：{question}"""

QUALITY_EVAL_PROMPT = """请评估以下检索结果是否足以回答用户问题。
只输出 JSON：{{"sufficient":true|false,"reason":"简短理由","refined_query":"如不足则给出更好的查询词，否则为空"}}

用户问题：{question}
检索结果摘要：{summary}"""

STREAM_SYSTEM_PROMPT = (
    "你是一个本地 Markdown 知识库问答助手。"
    "请根据用户提供的知识库片段直接给出简洁准确的中文答案，"
    "用 [S1][S2] 等标记引用来源。不要输出 JSON，不要多余解释，直接回答即可。"
)


class AgenticRAGAgent:
    """第二阶段主 Agent：查询分析 + 混合检索 + 质量评估（最多 2 次细化）+ 流式生成。

    保持 answer_stream() 的 SSE 事件格式与旧版 ReActAgent 完全一致：
    sources / token / done / error
    """

    def __init__(
        self,
        settings: Settings,
        llm_client,
        retriever: "HybridRetriever | KnowledgeRetriever",
        department_filter: list[str] | None = None,
    ) -> None:
        self.settings = settings
        self.llm_client = llm_client
        self.retriever = retriever
        self.department_filter = department_filter  # None = admin 不过滤

    def _search(self, query: str, top_k: int | None = None) -> list[SourceSnippet]:
        k = top_k or self.settings.retrieval_top_k
        if hasattr(self.retriever, "department_filter"):
            return self.retriever.search(query, top_k=k, department_filter=self.department_filter)
        if hasattr(self.retriever, "search"):
            sig = self.retriever.search.__code__.co_varnames
            if "department_filter" in sig:
                return self.retriever.search(query, top_k=k, department_filter=self.department_filter)
            return self.retriever.search(query, top_k=k)
        return []

    def _analyze_query(self, question: str) -> dict:
        """让 LLM 判断问题类型，返回 {type, sub_queries}。"""
        try:
            raw = self.llm_client.complete(
                [{"role": "user", "content": QUERY_ANALYSIS_PROMPT.format(question=question)}],
                temperature=0.0,
            )
            result = parse_json_object(raw)
            if result and "type" in result:
                return result
        except Exception as exc:
            logger.warning("[AgenticRAG] 查询分析失败 err=%s", exc)
        return {"type": "simple", "sub_queries": []}

    def _evaluate_quality(self, question: str, sources: list[SourceSnippet]) -> dict:
        """让 LLM 判断检索结果是否足够，返回 {sufficient, refined_query}。"""
        summary = "\n".join(f"[{s.source_id}] {s.title}: {s.content[:200]}" for s in sources)
        if not summary:
            return {"sufficient": False, "refined_query": question, "reason": "无检索结果"}
        try:
            raw = self.llm_client.complete(
                [
                    {
                        "role": "user",
                        "content": QUALITY_EVAL_PROMPT.format(question=question, summary=summary),
                    }
                ],
                temperature=0.0,
            )
            result = parse_json_object(raw)
            if result and "sufficient" in result:
                return result
        except Exception as exc:
            logger.warning("[AgenticRAG] 质量评估失败 err=%s", exc)
        return {"sufficient": True, "refined_query": "", "reason": "评估异常，直接使用"}

    def answer_stream(self, question: str) -> Iterator[str]:
        """SSE 流式生成器，格式与旧版 ReActAgent.answer_stream 完全一致。"""
        t_total = time.perf_counter()
        logger.info("[AgenticRAG] 开始 question=%r", question)

        # Step 1: 查询分析
        query_info = self._analyze_query(question)
        query_type = query_info.get("type", "simple")
        sub_queries: list[str] = query_info.get("sub_queries", [])
        logger.info("[AgenticRAG] 查询类型=%s sub_queries=%s", query_type, sub_queries)

        trace: list[TraceStep] = [
            TraceStep(action="query_analysis", content=f"type={query_type}"),
        ]

        # Step 2: 初次检索
        all_queries = [question] + (sub_queries if query_type == "complex" else [])
        sources = self._multi_search(all_queries)
        trace.append(TraceStep(action="initial_search", content=f"命中 {len(sources)} 条"))

        # Step 3: 质量评估 + 最多 2 次细化
        for attempt in range(2):
            eval_result = self._evaluate_quality(question, sources)
            if eval_result.get("sufficient", True):
                logger.info("[AgenticRAG] 质量评估通过 attempt=%d", attempt)
                break
            refined = eval_result.get("refined_query", "").strip()
            if not refined or refined == question:
                break
            logger.info("[AgenticRAG] 细化查询 attempt=%d refined=%r", attempt + 1, refined)
            trace.append(TraceStep(action=f"refine_{attempt+1}", content=refined))
            refined_sources = self._search(refined, top_k=self.settings.retrieval_top_k)
            # 合并去重（按 source key），保留分数较高的
            sources = _merge_sources(sources, refined_sources, top_k=self.settings.retrieval_top_k)

        logger.info("[AgenticRAG] 最终来源 n=%d", len(sources))

        # 先发 sources 事件
        yield json.dumps(
            {
                "type": "sources",
                "sources": [s.model_dump(by_alias=True, mode="json") for s in sources],
                "trace": [t.model_dump(by_alias=True, mode="json") for t in trace],
            },
            ensure_ascii=False,
        ) + "\n"

        # Step 4: 流式生成答案
        pre_observation = format_sources(sources)
        messages = [
            {"role": "system", "content": STREAM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n\n"
                    f"知识库检索结果：\n{pre_observation}\n\n"
                    "请直接给出答案："
                ),
            },
        ]

        for token in self.llm_client.stream_complete(messages, temperature=self.settings.temperature):
            yield json.dumps({"type": "token", "content": token}, ensure_ascii=False) + "\n"

        elapsed = time.perf_counter() - t_total
        logger.info("[AgenticRAG] 完成 total_elapsed=%.2fs", elapsed)
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"

    def _multi_search(self, queries: list[str]) -> list[SourceSnippet]:
        """对多个查询分别检索，合并去重后返回。"""
        all_sources: list[SourceSnippet] = []
        for q in queries:
            results = self._search(q)
            all_sources = _merge_sources(all_sources, results, top_k=self.settings.retrieval_top_k * 2)
        return all_sources[: self.settings.retrieval_top_k]


def _merge_sources(
    base: list[SourceSnippet],
    new: list[SourceSnippet],
    top_k: int,
) -> list[SourceSnippet]:
    """合并两个来源列表，相同片段保留分数较高的，按分数重排后截取 top_k。"""
    key_map: dict[str, SourceSnippet] = {}
    for s in base:
        k = f"{s.file_path}:{s.line_start}-{s.line_end}"
        key_map[k] = s
    for s in new:
        k = f"{s.file_path}:{s.line_start}-{s.line_end}"
        if k not in key_map or s.score > key_map[k].score:
            key_map[k] = s
    ranked = sorted(key_map.values(), key=lambda x: x.score, reverse=True)[:top_k]
    for i, snip in enumerate(ranked, start=1):
        snip.source_id = f"S{i}"
    return ranked
