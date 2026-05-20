// ---------------------------------------------------------------------------
// 类型定义
// ---------------------------------------------------------------------------

export type Health = {
  model: string;
  baseUrl: string;
  knowledgeBasePath: string;
  markdownFileCount: number;
  apiKeyConfigured: boolean;
};

export type SourceSnippet = {
  sourceId: string;
  filePath: string;
  title: string;
  lineStart: number;
  lineEnd: number;
  score: number;
  content: string;
};

export type TraceStep = {
  action: string;
  content: string;
};

export type ChatResponse = {
  answer: string;
  sources: SourceSnippet[];
  trace: TraceStep[];
};

export type StreamEvent =
  | { type: "sources"; sources: SourceSnippet[]; trace: TraceStep[] }
  | { type: "token"; content: string }
  | { type: "done" }
  | { type: "error"; message: string };

export type TokenResponse = {
  access_token: string;
  token_type: string;
  role: "admin" | "user";
  departments: string[];
};

export type DocumentMeta = {
  docId: string;
  fileName: string;
  department: string;
  state: "pending" | "parsing" | "embedding" | "graph" | "done" | "error";
  progress: number;
  chunksTotal: number;
  error?: string;
};

// ---------------------------------------------------------------------------
// 认证 token 管理
// ---------------------------------------------------------------------------

let _token: string | null = sessionStorage.getItem("kb_token");
let _role: "admin" | "user" | null = sessionStorage.getItem("kb_role") as "admin" | "user" | null;

export function setToken(token: string, role: "admin" | "user") {
  _token = token;
  _role = role;
  sessionStorage.setItem("kb_token", token);
  sessionStorage.setItem("kb_role", role);
}

export function clearToken() {
  _token = null;
  _role = null;
  sessionStorage.removeItem("kb_token");
  sessionStorage.removeItem("kb_role");
}

export function getRole(): "admin" | "user" | null {
  return _role;
}

export function isLoggedIn(): boolean {
  return _token !== null;
}

function authHeaders(): Record<string, string> {
  return _token ? { Authorization: `Bearer ${_token}` } : {};
}

// ---------------------------------------------------------------------------
// HTTP 工具函数
// ---------------------------------------------------------------------------

async function requestJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      ...authHeaders(),
      ...(options?.headers ?? {}),
    },
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// 认证接口
// ---------------------------------------------------------------------------

export async function login(username: string, password: string): Promise<TokenResponse> {
  const resp = await requestJson<TokenResponse>("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  setToken(resp.access_token, resp.role);
  return resp;
}

export function logout() {
  clearToken();
}

// ---------------------------------------------------------------------------
// 知识库接口
// ---------------------------------------------------------------------------

export function fetchHealth(): Promise<Health> {
  return requestJson<Health>("/api/health");
}

export function sendQuestion(message: string): Promise<ChatResponse> {
  return requestJson<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
}

export async function* sendQuestionStream(message: string): AsyncGenerator<StreamEvent> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
    },
    body: JSON.stringify({ message }),
  });
  if (!response.ok || !response.body) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.replace(/^data:\s*/, "").trim();
      if (!trimmed) continue;
      try {
        yield JSON.parse(trimmed) as StreamEvent;
      } catch {
        // ignore malformed lines
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 文档管理接口
// ---------------------------------------------------------------------------

export function listDocs(): Promise<DocumentMeta[]> {
  return requestJson<DocumentMeta[]>("/api/docs");
}

export async function uploadDoc(
  file: File,
  department = "default",
  enableGraph = false,
): Promise<DocumentMeta> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("department", department);
  formData.append("enable_graph_extract", String(enableGraph));
  const response = await fetch("/api/docs/upload", {
    method: "POST",
    headers: authHeaders(),
    body: formData,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Upload failed with ${response.status}`);
  }
  return response.json() as Promise<DocumentMeta>;
}

export function deleteDoc(docId: string): Promise<{ deleted: string }> {
  return requestJson<{ deleted: string }>(`/api/docs/${docId}`, {
    method: "DELETE",
  });
}

export function getDocStatus(docId: string): Promise<DocumentMeta> {
  return requestJson<DocumentMeta>(`/api/docs/${docId}/status`);
}
