from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient

from kb_web_agent.api import create_app
from kb_web_agent.settings import Settings


# ---------------------------------------------------------------------------
# Fake LLM clients
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """通用 Fake LLM，按 prompt 内容自动识别调用场景，兼容旧版 ReActAgent 和 AgenticRAGAgent。"""

    def complete(self, messages, temperature=0.2, max_tokens=512) -> str:
        user_content = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        # AgenticRAGAgent — 查询分析
        if "sub_queries" in user_content:
            return '{"type":"simple","sub_queries":[]}'
        # AgenticRAGAgent — 质量评估
        if "sufficient" in user_content or "足以回答" in user_content:
            return '{"sufficient":true,"reason":"足够","refined_query":""}'
        # 旧版 ReActAgent — 已有预检索结果，下一步继续
        if "Observation" in user_content and "继续按 JSON" in user_content:
            return '{"thought":"证据足够","action":"final_answer","answer":"测试类路径规范见知识库。[S1]"}'
        # 旧版 ReActAgent — 首次调用（预检索注入）
        if "已自动检索" in user_content:
            return '{"thought":"需要查知识库","action":"search_knowledge_base","action_input":"测试规范"}'
        # 兜底
        return '{"thought":"证据足够","action":"final_answer","answer":"测试答案。[S1]"}'

    def stream_complete(self, messages, temperature=0.2, max_tokens=512) -> Iterator[str]:
        yield "根据知识库，"
        yield "测试类应放在 src/test/java 目录。[S1]"


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------


def make_client(tmp_path: Path) -> TestClient:
    (tmp_path / "rule.md").write_text("# 测试规范\n\n测试类必须放在 src/test/java。", encoding="utf-8")
    settings = Settings(
        api_key="test",
        base_url="https://example.test/api/coding/v3",
        model="glm-5.1",
        knowledge_base_path=tmp_path,
        max_steps=3,
        retrieval_top_k=3,
        enable_vector_store=False,  # 测试环境不需要 Chroma
        enable_graph=False,
    )
    app = create_app(settings=settings, llm_client=FakeLLMClient())
    return TestClient(app)


def _get_token(client: TestClient, username="admin", password="admin123") -> str:
    resp = client.post("/api/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# 健康检查 & 搜索（无需认证，向后兼容）
# ---------------------------------------------------------------------------


def test_health_endpoint_reports_runtime_state(tmp_path):
    client = make_client(tmp_path)

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "glm-5.1"
    assert body["markdownFileCount"] == 1
    assert body["apiKeyConfigured"] is True


def test_search_endpoint_returns_sources(tmp_path):
    client = make_client(tmp_path)

    response = client.post("/api/search", json={"query": "测试类路径", "topK": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["sources"][0]["sourceId"] == "S1"
    assert body["sources"][0]["filePath"] == "rule.md"


def test_chat_endpoint_runs_react_and_returns_answer_with_sources(tmp_path):
    client = make_client(tmp_path)

    response = client.post("/api/chat", json={"message": "测试类应该放哪里？"})

    assert response.status_code == 200
    body = response.json()
    assert "测试类路径规范" in body["answer"]
    assert body["sources"][0]["sourceId"] == "S1"
    assert any(step["action"] == "search_knowledge_base" for step in body["trace"])


# ---------------------------------------------------------------------------
# 认证接口
# ---------------------------------------------------------------------------


def test_login_admin_returns_token(tmp_path):
    client = make_client(tmp_path)

    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})

    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["role"] == "admin"
    assert body["token_type"] == "bearer"


def test_login_wrong_password_returns_401(tmp_path):
    client = make_client(tmp_path)

    resp = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})

    assert resp.status_code == 401


def test_login_user_role(tmp_path):
    client = make_client(tmp_path)

    resp = client.post("/api/auth/login", json={"username": "user", "password": "user123"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "user"
    assert "default" in body["departments"]


# ---------------------------------------------------------------------------
# 流式问答（需要认证）
# ---------------------------------------------------------------------------


def test_chat_stream_without_auth_returns_401(tmp_path):
    client = make_client(tmp_path)

    resp = client.post("/api/chat/stream", json={"message": "测试类应该放哪里？"})

    assert resp.status_code == 401


def test_chat_stream_with_admin_token_returns_sse(tmp_path):
    client = make_client(tmp_path)
    token = _get_token(client)

    resp = client.post(
        "/api/chat/stream",
        json={"message": "测试类应该放哪里？"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    # 解析 SSE 行，收集所有事件类型
    events: list[dict] = []
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            payload = line.removeprefix("data:").strip()
            if payload:
                events.append(json.loads(payload))

    types = [e["type"] for e in events]
    assert "sources" in types, f"缺少 sources 事件，实际: {types}"
    assert "token" in types, f"缺少 token 事件，实际: {types}"
    assert "done" in types, f"缺少 done 事件，实际: {types}"

    # 来源应包含正确的文档片段
    sources_event = next(e for e in events if e["type"] == "sources")
    assert len(sources_event["sources"]) > 0

    # token 拼起来应包含答案
    answer = "".join(e["content"] for e in events if e["type"] == "token")
    assert len(answer) > 0


# ---------------------------------------------------------------------------
# 文档管理接口（需要 admin 认证）
# ---------------------------------------------------------------------------


def test_list_docs_without_auth_returns_401(tmp_path):
    client = make_client(tmp_path)

    resp = client.get("/api/docs")

    assert resp.status_code == 401


def test_list_docs_with_auth_returns_list(tmp_path):
    client = make_client(tmp_path)
    token = _get_token(client)

    resp = client.get("/api/docs", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_upload_doc_without_admin_returns_403(tmp_path):
    client = make_client(tmp_path)
    # user 角色无法上传
    token = _get_token(client, username="user", password="user123")
    file_content = b"# test\nsome content"

    resp = client.post(
        "/api/docs/upload",
        files={"file": ("test.md", file_content, "text/markdown")},
        data={"department": "default"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 403


def test_upload_doc_as_admin_returns_meta(tmp_path):
    client = make_client(tmp_path)
    token = _get_token(client)
    file_content = b"# upload test\nsome content here"

    resp = client.post(
        "/api/docs/upload",
        files={"file": ("upload_test.md", file_content, "text/markdown")},
        data={"department": "default"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert "docId" in body
    assert body["fileName"] == "upload_test.md"
    assert body["department"] == "default"
    assert body["state"] == "pending"
