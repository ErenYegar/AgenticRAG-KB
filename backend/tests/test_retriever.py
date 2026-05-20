from pathlib import Path

from kb_web_agent.documents import load_markdown_chunks
from kb_web_agent.retriever import KnowledgeRetriever, tokenize


def test_tokenize_uses_chinese_words():
    tokens = tokenize("供应链管理系统")

    assert "供应链" in tokens
    assert "管理系统" in tokens


def test_retriever_returns_source_metadata(tmp_path):
    note = tmp_path / "rule.md"
    note.write_text("# 测试规范\n\n测试类必须放在 src/test/java。", encoding="utf-8")
    chunks = load_markdown_chunks(tmp_path)
    retriever = KnowledgeRetriever(chunks)

    results = retriever.search("测试类路径", top_k=1)

    assert results[0].source_id == "S1"
    assert results[0].file_path == Path("rule.md")
    assert "测试类" in results[0].content
