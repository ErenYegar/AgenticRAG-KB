from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .agent import AgenticRAGAgent, ReActAgent
from .auth import User, get_current_user, get_department_filter, require_admin, verify_password, create_token
from .documents import iter_markdown_files, load_document_chunks
from .graph import GraphRetriever, KnowledgeGraph
from .hybrid_retriever import HybridRetriever
from .ingestion import IngestStatus, delete_doc, get_status, ingest_file, list_statuses
from .llm import OpenAICompatibleClient
from .retriever import KnowledgeRetriever
from .schemas import (
    ChatRequest,
    ChatResponse,
    DocumentMeta,
    HealthResponse,
    LoginRequest,
    SearchRequest,
    SearchResponse,
    TokenResponse,
)
from .settings import Settings, load_settings
from .vector_store import ChromaVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kb_web_agent.api")


def create_app(settings: Settings | None = None, llm_client=None) -> FastAPI:
    resolved_settings = settings or load_settings()

    # -----------------------------------------------------------------------
    # 基础设施初始化
    # -----------------------------------------------------------------------
    chunks = load_document_chunks(resolved_settings.knowledge_base_path)
    bm25_retriever = KnowledgeRetriever(chunks, root=resolved_settings.knowledge_base_path)

    client = llm_client or OpenAICompatibleClient(
        api_key=resolved_settings.api_key,
        base_url=resolved_settings.base_url,
        model=resolved_settings.model,
        timeout_seconds=resolved_settings.timeout_seconds,
    )

    # 向量存储（可选）
    vector_store: ChromaVectorStore | None = None
    if resolved_settings.enable_vector_store:
        try:
            from .embedding import get_embedder

            embedder = get_embedder(
                model_name=resolved_settings.embedding_model,
                cache_folder=resolved_settings.model_cache_dir,
            )
            vector_store = ChromaVectorStore(
                persist_directory=resolved_settings.chroma_path,
                embedder=embedder,
            )
            logger.info(
                "[API] ChromaVectorStore 初始化完成 path=%s cache=%s",
                resolved_settings.chroma_path,
                resolved_settings.model_cache_dir,
            )
        except Exception as exc:
            logger.warning("[API] ChromaVectorStore 初始化失败，降级为纯 BM25 err=%s", exc)

    # 知识图谱（可选）
    knowledge_graph: KnowledgeGraph | None = None
    graph_retriever: GraphRetriever | None = None
    if resolved_settings.enable_graph:
        try:
            graph_path = resolved_settings.chroma_path / "graph.json"
            knowledge_graph = KnowledgeGraph(persist_path=graph_path)
            graph_retriever = GraphRetriever(knowledge_graph)
            logger.info("[API] KnowledgeGraph 初始化完成")
        except Exception as exc:
            logger.warning("[API] KnowledgeGraph 初始化失败 err=%s", exc)

    # 混合检索器
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
        graph_retriever=graph_retriever,
        root=resolved_settings.knowledge_base_path,
    )

    # 上传文件临时目录
    upload_dir = Path(resolved_settings.chroma_path) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # FastAPI 应用
    # -----------------------------------------------------------------------
    app = FastAPI(title="AgenticRAG Knowledge Base WebUI", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - t0
        logger.info("[HTTP] %s %s -> %d (%.2fs)", request.method, request.url.path, response.status_code, elapsed)
        return response

    # -----------------------------------------------------------------------
    # 认证接口
    # -----------------------------------------------------------------------

    @app.post("/api/auth/login", response_model=TokenResponse)
    def login(req: LoginRequest) -> TokenResponse:
        user = verify_password(req.username, req.password)
        if not user:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        token = create_token(user, expire_minutes=resolved_settings.jwt_expire_minutes)
        return TokenResponse(
            access_token=token,
            role=user.role,
            departments=user.departments,
        )

    # -----------------------------------------------------------------------
    # 基础接口（无需认证，向后兼容）
    # -----------------------------------------------------------------------

    @app.get("/api/health", response_model=HealthResponse, response_model_by_alias=True)
    def health() -> HealthResponse:
        return HealthResponse(
            model=resolved_settings.model,
            base_url=resolved_settings.base_url,
            knowledge_base_path=resolved_settings.knowledge_base_path,
            markdown_file_count=sum(1 for _ in iter_markdown_files(resolved_settings.knowledge_base_path)),
            api_key_configured=bool(resolved_settings.api_key),
        )

    @app.post("/api/search", response_model=SearchResponse, response_model_by_alias=True)
    def search(request: SearchRequest) -> SearchResponse:
        return SearchResponse(
            sources=bm25_retriever.search(
                request.query,
                top_k=request.top_k or resolved_settings.retrieval_top_k,
            )
        )

    @app.post("/api/chat", response_model=ChatResponse, response_model_by_alias=True)
    def chat(request: ChatRequest) -> ChatResponse:
        logger.info("[API] /api/chat message=%r", request.message)
        agent = ReActAgent(resolved_settings, client, bm25_retriever)
        return agent.answer(request.message)

    @app.post("/api/chat/stream")
    def chat_stream(
        request: ChatRequest,
        dept_filter: list[str] | None = Depends(get_department_filter),
    ) -> StreamingResponse:
        logger.info("[API] /api/chat/stream message=%r dept=%s", request.message, dept_filter)

        # 使用 AgenticRAGAgent（混合检索）
        agent = AgenticRAGAgent(
            settings=resolved_settings,
            llm_client=client,
            retriever=hybrid_retriever,
            department_filter=dept_filter,
        )

        def event_generator():
            try:
                for line in agent.answer_stream(request.message):
                    yield f"data: {line}\n"
            except Exception as exc:
                logger.error("[API] 流式错误 err=%s", exc)
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # -----------------------------------------------------------------------
    # 文档管理接口（需认证）
    # -----------------------------------------------------------------------

    @app.post("/api/docs/upload", response_model=DocumentMeta, response_model_by_alias=True)
    async def upload_doc(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        department: str = Form(default="default"),
        enable_graph_extract: bool = Form(default=False),
        admin: User = Depends(require_admin),
    ) -> DocumentMeta:
        doc_id = str(uuid.uuid4())
        suffix = Path(file.filename or "upload").suffix or ".bin"
        save_path = upload_dir / f"{doc_id}{suffix}"

        content = await file.read()
        save_path.write_bytes(content)
        logger.info("[API] 文件上传 doc_id=%s file=%s size=%d", doc_id, file.filename, len(content))

        llm_for_graph = client if (enable_graph_extract and knowledge_graph is not None) else None

        background_tasks.add_task(
            ingest_file,
            file_path=save_path,
            department=department,
            doc_id=doc_id,
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            knowledge_graph=knowledge_graph,
            llm_client=llm_for_graph,
        )

        return DocumentMeta(
            doc_id=doc_id,
            file_name=file.filename or save_path.name,
            department=department,
            state="pending",
            progress=0.0,
            chunks_total=0,
        )

    @app.get("/api/docs", response_model=list[DocumentMeta], response_model_by_alias=True)
    def list_docs(user: User = Depends(get_current_user)) -> list[DocumentMeta]:
        statuses = list_statuses()
        docs: list[DocumentMeta] = []
        for s in statuses:
            if user.role == "admin" or s.department in (user.departments or []):
                docs.append(
                    DocumentMeta(
                        doc_id=s.doc_id,
                        file_name=s.file_name,
                        department=s.department,
                        state=s.state.value,
                        progress=s.progress,
                        chunks_total=s.chunks_total,
                        error=s.error,
                    )
                )
        # 还可从 vector_store 拉取已持久化的文档元数据
        if vector_store is not None:
            persisted = vector_store.export_metadata()
            existing_ids = {d.doc_id for d in docs}
            for meta in persisted:
                if meta["doc_id"] not in existing_ids:
                    if user.role == "admin" or meta["department"] in (user.departments or []):
                        docs.append(
                            DocumentMeta(
                                doc_id=meta["doc_id"],
                                file_name=Path(meta["file_path"]).name,
                                department=meta["department"],
                                state="done",
                                progress=1.0,
                                chunks_total=0,
                            )
                        )
        return docs

    @app.get("/api/docs/{doc_id}/status", response_model=DocumentMeta, response_model_by_alias=True)
    def doc_status(doc_id: str, user: User = Depends(get_current_user)) -> DocumentMeta:
        s = get_status(doc_id)
        if s is None:
            raise HTTPException(status_code=404, detail="文档不存在或未找到状态")
        if user.role != "admin" and s.department not in (user.departments or []):
            raise HTTPException(status_code=403, detail="无权访问该文档")
        return DocumentMeta(
            doc_id=s.doc_id,
            file_name=s.file_name,
            department=s.department,
            state=s.state.value,
            progress=s.progress,
            chunks_total=s.chunks_total,
            error=s.error,
        )

    @app.delete("/api/docs/{doc_id}")
    async def remove_doc(doc_id: str, admin: User = Depends(require_admin)) -> dict:
        success = await delete_doc(doc_id, vector_store=vector_store, knowledge_graph=knowledge_graph)
        if not success:
            raise HTTPException(status_code=500, detail="部分删除失败，请查看日志")
        return {"deleted": doc_id}

    return app


app = create_app()
