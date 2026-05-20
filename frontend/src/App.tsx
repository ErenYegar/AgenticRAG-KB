import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Bot, Database, FileText, KeyRound, Loader2, LogIn, LogOut, MessageSquare, Send, Trash2, Upload, UploadCloud } from "lucide-react";

import {
  ChatResponse,
  DocumentMeta,
  Health,
  SourceSnippet,
  deleteDoc,
  fetchHealth,
  getRole,
  isLoggedIn,
  listDocs,
  login,
  logout,
  sendQuestionStream,
  uploadDoc,
} from "./api";

type Page = "chat" | "docs" | "login";

type ChatItem =
  | { role: "user"; content: string }
  | { role: "assistant"; content: string; sources: SourceSnippet[]; trace: ChatResponse["trace"] };

// ---------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------

export default function App() {
  const [page, setPage] = useState<Page>(isLoggedIn() ? "chat" : "login");
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState("");

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((err: Error) => setHealthError(err.message));
  }, []);

  function handleLogin() {
    setPage("chat");
  }

  function handleLogout() {
    logout();
    setPage("login");
  }

  if (page === "login") {
    return <LoginPage onSuccess={handleLogin} />;
  }

  return (
    <main className="app-shell">
      <aside className="sidebar" aria-label="运行状态">
        <div className="brand-row">
          <div className="brand-mark">
            <Bot size={24} />
          </div>
          <div>
            <h1>AgenticRAG 知识库问答</h1>
            <p>多格式本地知识库</p>
          </div>
        </div>

        <section className="status-grid">
          <StatusItem icon={<Bot size={18} />} label="模型" value={health?.model.toUpperCase() ?? (healthError || "连接中")} />
          <StatusItem icon={<Database size={18} />} label="知识库" value={health?.knowledgeBasePath ?? (healthError || "读取中")} />
          <StatusItem icon={<FileText size={18} />} label="文档" value={health ? `${health.markdownFileCount} 文件` : "读取中"} />
          <StatusItem icon={<KeyRound size={18} />} label="API Key" value={health?.apiKeyConfigured ? "已配置" : "未配置"} />
        </section>

        <nav className="sidebar-nav">
          <button
            className={`nav-btn ${page === "chat" ? "active" : ""}`}
            onClick={() => setPage("chat")}
          >
            <MessageSquare size={16} /> 问答
          </button>
          {getRole() === "admin" && (
            <button
              className={`nav-btn ${page === "docs" ? "active" : ""}`}
              onClick={() => setPage("docs")}
            >
              <FileText size={16} /> 文档管理
            </button>
          )}
          <button className="nav-btn" onClick={handleLogout}>
            <LogOut size={16} /> 退出登录
          </button>
        </nav>
      </aside>

      {page === "chat" ? <ChatPanel /> : <DocsPanel />}
    </main>
  );
}

// ---------------------------------------------------------------------------
// LoginPage
// ---------------------------------------------------------------------------

function LoginPage({ onSuccess }: { onSuccess: () => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!username || !password) return;
    setLoading(true);
    setError("");
    try {
      await login(username, password);
      onSuccess();
    } catch (err) {
      setError(err instanceof Error ? err.message : "登录失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div style={{ textAlign: "center" }}>
          <div style={{ display: "inline-grid", placeItems: "center", width: 56, height: 56, borderRadius: 12, background: "#18352f", marginBottom: 16 }}>
            <Bot size={28} color="#d9f275" />
          </div>
          <h1>AgenticRAG 知识库问答</h1>
          <p>请登录后使用</p>
        </div>

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="form-field">
            <label htmlFor="username">用户名</label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="admin"
              autoComplete="username"
            />
          </div>
          <div className="form-field">
            <label htmlFor="password">密码</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              autoComplete="current-password"
            />
          </div>
          {error && <div className="error-banner" style={{ margin: 0 }}>{error}</div>}
          <button type="submit" className="btn-primary" disabled={loading || !username || !password}>
            {loading ? <Loader2 size={16} className="spin" /> : <LogIn size={16} />}
            登录
          </button>
        </form>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChatPanel（与原 App 主体相同，提取为独立组件）
// ---------------------------------------------------------------------------

function ChatPanel() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<ChatItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState("");
  const conversationRef = useRef<HTMLDivElement>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading) return;
    setQuestion("");
    setError("");
    setLoading(true);
    setStreaming(false);

    const assistantIndex = messages.length + 1;
    setMessages((items) => [
      ...items,
      { role: "user" as const, content: trimmed },
      { role: "assistant" as const, content: "", sources: [], trace: [] },
    ]);

    let isFirstToken = true;
    let pendingSources: SourceSnippet[] = [];
    let pendingTrace: ChatResponse["trace"] = [];

    try {
      for await (const evt of sendQuestionStream(trimmed)) {
        if (evt.type === "sources") {
          pendingSources = evt.sources;
          pendingTrace = evt.trace;
        } else if (evt.type === "token") {
          if (isFirstToken) {
            isFirstToken = false;
            setLoading(false);
            setStreaming(true);
            setMessages((items) =>
              items.map((item, i) =>
                i === assistantIndex && item.role === "assistant"
                  ? { ...item, content: evt.content, sources: pendingSources, trace: pendingTrace }
                  : item
              )
            );
          } else {
            setMessages((items) =>
              items.map((item, i) =>
                i === assistantIndex && item.role === "assistant"
                  ? { ...item, content: item.content + evt.content }
                  : item
              )
            );
          }
          requestAnimationFrame(() => {
            conversationRef.current?.scrollTo({ top: conversationRef.current.scrollHeight, behavior: "smooth" });
          });
        } else if (evt.type === "error") {
          setError(evt.message);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "请求失败");
    } finally {
      setLoading(false);
      setStreaming(false);
    }
  }

  return (
    <section className="chat-panel" aria-label="问答区">
      <div className="conversation" ref={conversationRef}>
        {messages.length === 0 ? (
          <div className="empty-state">
            <h2>直接问你的知识库</h2>
            <p>Agent 会先检索文档，再用 GLM-5.1 汇总答案，并显示来源片段。</p>
          </div>
        ) : (
          messages.map((item, index) => (
            <article key={`${item.role}-${index}`} className={`message ${item.role}`}>
              <div className="bubble">
                {item.content}
                {item.role === "assistant" && streaming && index === messages.length - 1 ? (
                  <span className="typing-cursor" aria-hidden="true" />
                ) : null}
              </div>
              {item.role === "assistant" && item.sources.length > 0 ? (
                <div className="sources">
                  {item.sources.map((source) => (
                    <SourceCard key={source.sourceId} source={source} />
                  ))}
                </div>
              ) : null}
            </article>
          ))
        )}
        {loading && !streaming ? (
          <div className="loading-row">
            <Loader2 className="spin" size={18} />
            正在检索知识库…
          </div>
        ) : null}
      </div>

      {error ? <div className="error-banner">{error}</div> : null}

      <form className="composer" onSubmit={handleSubmit}>
        <label htmlFor="question">输入问题</label>
        <textarea
          id="question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          rows={2}
          placeholder="例如：测试类应该放在哪里？"
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              e.currentTarget.form?.requestSubmit();
            }
          }}
        />
        <button type="submit" disabled={loading || !question.trim()} aria-label="发送">
          {loading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
          发送
        </button>
      </form>
    </section>
  );
}

// ---------------------------------------------------------------------------
// DocsPanel（文档管理，仅 admin 可见）
// ---------------------------------------------------------------------------

function DocsPanel() {
  const [docs, setDocs] = useState<DocumentMeta[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [department, setDepartment] = useState("default");
  const [uploadError, setUploadError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDocs = useCallback(async () => {
    setLoadingDocs(true);
    try {
      const data = await listDocs();
      setDocs(data);
    } catch {
      // ignore
    } finally {
      setLoadingDocs(false);
    }
  }, []);

  useEffect(() => {
    fetchDocs();
    const interval = setInterval(fetchDocs, 5000);
    return () => clearInterval(interval);
  }, [fetchDocs]);

  async function handleUpload() {
    if (!selectedFile) return;
    setUploading(true);
    setUploadError("");
    try {
      const doc = await uploadDoc(selectedFile, department);
      setDocs((prev) => [doc, ...prev]);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "上传失败");
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(docId: string) {
    if (!confirm("确认删除该文档？此操作不可撤销。")) return;
    try {
      await deleteDoc(docId);
      setDocs((prev) => prev.filter((d) => d.docId !== docId));
    } catch (err) {
      alert(err instanceof Error ? err.message : "删除失败");
    }
  }

  return (
    <div className="docs-page">
      <h2>文档管理</h2>

      {/* 上传区 */}
      <div className="upload-zone">
        <UploadCloud size={40} color="#365248" />
        <p>支持 PDF、Word (.docx)、Excel (.xlsx)、Markdown (.md)、文本 (.txt)</p>

        <div className="upload-controls">
          <label className="upload-label" htmlFor="file-input">
            <Upload size={16} />
            {selectedFile ? selectedFile.name : "选择文件"}
          </label>
          <input
            id="file-input"
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.xlsx,.md,.txt"
            onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
          />
          <select value={department} onChange={(e) => setDepartment(e.target.value)}>
            <option value="default">通用（default）</option>
            <option value="hr">人事（hr）</option>
            <option value="tech">技术（tech）</option>
            <option value="finance">财务（finance）</option>
          </select>
          <button disabled={!selectedFile || uploading} onClick={handleUpload}>
            {uploading ? <Loader2 size={14} className="spin" style={{ display: "inline" }} /> : null}
            {uploading ? " 上传中…" : "开始上传"}
          </button>
        </div>

        {uploadError && <div className="error-banner" style={{ margin: 0, width: "100%" }}>{uploadError}</div>}
      </div>

      {/* 文档列表 */}
      <div className="doc-table">
        <table>
          <thead>
            <tr>
              <th>文件名</th>
              <th>部门</th>
              <th>状态</th>
              <th>进度</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {loadingDocs && docs.length === 0 ? (
              <tr>
                <td colSpan={5} style={{ textAlign: "center", color: "#526158" }}>
                  <Loader2 size={16} className="spin" style={{ display: "inline" }} /> 加载中…
                </td>
              </tr>
            ) : docs.length === 0 ? (
              <tr>
                <td colSpan={5} style={{ textAlign: "center", color: "#526158" }}>暂无文档</td>
              </tr>
            ) : (
              docs.map((doc) => (
                <tr key={doc.docId}>
                  <td style={{ maxWidth: 240, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={doc.fileName}>
                    {doc.fileName}
                  </td>
                  <td>{doc.department}</td>
                  <td>
                    <span className={`state-badge state-${doc.state}`}>{stateLabel(doc.state)}</span>
                    {doc.error && <span style={{ color: "#7a1d1d", fontSize: 12, marginLeft: 6 }}>{doc.error}</span>}
                  </td>
                  <td>
                    <div className="progress-bar-wrap">
                      <div className="progress-bar-fill" style={{ width: `${Math.round(doc.progress * 100)}%` }} />
                    </div>
                  </td>
                  <td>
                    <button className="btn-danger" onClick={() => handleDelete(doc.docId)} title="删除">
                      <Trash2 size={14} style={{ display: "inline" }} />
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 小组件
// ---------------------------------------------------------------------------

function StatusItem({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="status-item">
      <div className="status-icon">{icon}</div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function SourceCard({ source }: { source: SourceSnippet }) {
  return (
    <section className="source-card">
      <div className="source-topline">
        <strong>{source.filePath}:{source.lineStart}-{source.lineEnd}</strong>
        <span>{source.sourceId}</span>
      </div>
      <h3>{source.title}</h3>
      <p>{source.content}</p>
    </section>
  );
}

function stateLabel(state: DocumentMeta["state"]): string {
  const map: Record<DocumentMeta["state"], string> = {
    pending: "等待中",
    parsing: "解析中",
    embedding: "向量化",
    graph: "图谱抽取",
    done: "完成",
    error: "失败",
  };
  return map[state] ?? state;
}
