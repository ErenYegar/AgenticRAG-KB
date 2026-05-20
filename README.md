# AgenticRAG Knowledge Base WebUI

AgenticRAG Knowledge Base WebUI 是一个本地知识库问答系统。项目支持多格式文档入库、混合检索、可选知识图谱增强、流式 LLM 回答、JWT 权限管理，以及基于 React 的文档管理界面。

它适合用于个人或团队私有知识库：文档解析、向量存储和检索在本地完成，答案生成通过兼容 OpenAI Chat Completions 的大模型接口完成。

## 功能特性

- 支持上传和解析 PDF、Word、Excel、Markdown、TXT 文档。
- 使用混合检索流程：Chroma 向量检索、BM25 关键词检索、可选 NetworkX 知识图谱信号。
- 通过 Server-Sent Events（SSE）流式输出答案。
- 提供 Web UI 文档管理能力，管理员可上传和删除文档。
- 支持按部门隔离普通用户检索范围，管理员可检索全库。
- 通过环境变量配置 LLM、Embedding 模型、Chroma 路径、JWT 密钥和运行路径。
- 后端支持 pytest 测试，前端支持 Vitest、Playwright、TypeScript 和 Vite 构建检查。

## 架构说明

### 文档入库流程

```text
上传文件
  -> documents.py 解析支持的文档格式为文本
  -> split_markdown() 按标题和结构切分文本
  -> embedding.py 使用 bge-m3 生成向量
  -> vector_store.py 将向量持久化到 Chroma
  -> retriever.py 刷新 BM25 索引
  -> graph.py 可选：抽取三元组并写入 NetworkX 图谱
```

### 查询流程

```text
用户问题
  -> 携带 JWT Token 请求 POST /api/chat/stream
  -> AgenticRAGAgent
      -> 分析查询类型
      -> 执行混合检索
      -> 评估检索结果质量
      -> 证据不足时细化查询并重试
      -> 调用配置的大模型流式生成答案
  -> React 前端渲染 token 和来源卡片
```

## 目录结构

```text
AgenticRAG-KB/
+-- backend/
|   +-- kb_web_agent/
|   |   +-- api.py              # FastAPI 应用和 HTTP 路由
|   |   +-- agent.py            # ReActAgent 兼容实现和 AgenticRAGAgent
|   |   +-- auth.py             # JWT 认证和角色检查
|   |   +-- documents.py        # 文档解析和分块
|   |   +-- embedding.py        # bge-m3 向量封装
|   |   +-- graph.py            # 可选知识图谱
|   |   +-- hybrid_retriever.py # 向量 + BM25 + 图谱混合检索
|   |   +-- ingestion.py        # 异步入库流程
|   |   +-- llm.py              # OpenAI 兼容 LLM 客户端
|   |   +-- retriever.py        # BM25 检索器
|   |   +-- schemas.py          # Pydantic 数据模型
|   |   +-- settings.py         # 环境配置加载
|   |   +-- vector_store.py     # Chroma 向量存储
|   +-- tests/
|   +-- .env.example
|   +-- pyproject.toml
+-- frontend/
|   +-- src/
|   +-- tests/
|   +-- package.json
|   +-- vite.config.ts
+-- .env.example
+-- .gitattributes
+-- .gitignore
+-- LICENSE
+-- README.md
```

## 环境要求

| 组件 | 要求 / 说明 |
| --- | --- |
| Python | 3.12+ |
| Node.js | 18+ |
| 后端 | FastAPI、Chroma、sentence-transformers、BM25、NetworkX |
| 前端 | React 19、Vite、TypeScript |
| LLM API | 任意兼容 OpenAI Chat Completions 的接口 |
| Embedding | 默认使用 `BAAI/bge-m3` |

后端配置支持 Windows 风格路径，并会在 WSL/Linux 环境中自动转换为可用路径。

## 快速开始

克隆仓库后，分别安装后端和前端依赖。

```bash
git clone <your-repo-url>
cd AgenticRAG-KB
```

### 后端

```bash
cd backend
python -m venv .venv

# Linux/macOS/WSL
source .venv/bin/activate

# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip wheel
pip install -e ".[test]"
cp .env.example .env
```

编辑 `backend/.env`，填写 API Key、知识库路径、模型缓存路径和 JWT 密钥。

启动后端服务：

```bash
uvicorn kb_web_agent.api:app --host 127.0.0.1 --port 8000 --reload
```

### 前端

```bash
cd frontend
npm install
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`，使用下方开发账号登录。

## 配置说明

运行后端前，复制配置模板：

```bash
cp backend/.env.example backend/.env
```

主要环境变量如下：

| 变量 | 是否必填 | 说明 |
| --- | --- | --- |
| `ARK_API_KEY` | 是 | OpenAI 兼容 LLM 接口的 API Key。 |
| `ARK_BASE_URL` | 是 | LLM API Base URL，默认是火山引擎 Ark coding endpoint。 |
| `ARK_MODEL` | 是 | 聊天模型名称，默认 `glm-5.1`。 |
| `KNOWLEDGE_BASE_PATH` | 是 | 后端启动时自动加载 Markdown/TXT 文件的知识库目录。 |
| `REACT_MAX_STEPS` | 否 | 兼容 ReAct Agent 的最大推理步数。 |
| `RETRIEVAL_TOP_K` | 否 | 每次检索返回的片段数量。 |
| `CHROMA_PATH` | 否 | Chroma 持久化目录，默认相对 `backend/`。 |
| `EMBEDDING_MODEL` | 否 | Embedding 模型名称，默认 `BAAI/bge-m3`。 |
| `MODEL_CACHE_DIR` | 否 | Hugging Face 模型缓存目录。 |
| `ENABLE_VECTOR_STORE` | 否 | 设为 `false` 时退化为仅 BM25 检索。 |
| `ENABLE_GRAPH` | 否 | 设为 `true` 后入库时启用 LLM 三元组抽取。 |
| `JWT_SECRET` | 是 | JWT 签名密钥，请使用强随机值。 |
| `JWT_EXPIRE_MINUTES` | 否 | Token 有效期，单位分钟。 |
| `ADMIN_PASSWORD` | 否 | 覆盖开发环境 admin 默认密码。 |
| `USER_PASSWORD` | 否 | 覆盖开发环境 user 默认密码。 |

不要提交 `.env`、API Key、JWT 密钥、上传文件、Chroma 数据、模型缓存、虚拟环境或构建产物。

## 账号与权限

开发账号定义在 `backend/kb_web_agent/auth.py`。

| 用户名 | 默认密码 | 角色 | 权限 |
| --- | --- | --- | --- |
| `admin` | `admin123` | admin | 问答、查看所有部门、上传文档、删除文档。 |
| `user` | `user123` | user | 问答，并只能查看该用户部门范围内的文档。 |

生产环境建议替换内存用户表，使用持久化用户数据和密码哈希。

## API

FastAPI 交互式文档地址：`http://127.0.0.1:8000/docs`。

### 公开接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/health` | 健康检查和配置摘要。 |
| `POST` | `/api/search` | 本地 BM25 片段检索。 |
| `POST` | `/api/chat` | 同步问答兼容接口。 |

### 需要认证的接口

请求头需要包含 `Authorization: Bearer <token>`。

| 方法 | 路径 | 角色 | 说明 |
| --- | --- | --- | --- |
| `POST` | `/api/auth/login` | public | 登录并返回 JWT。 |
| `POST` | `/api/chat/stream` | user/admin | 主流式问答接口。 |
| `GET` | `/api/docs` | user/admin | 查看可见文档列表。 |
| `POST` | `/api/docs/upload` | admin | 上传文档并启动入库。 |
| `GET` | `/api/docs/{doc_id}/status` | user/admin | 查询入库进度。 |
| `DELETE` | `/api/docs/{doc_id}` | admin | 删除文档。 |

登录请求示例：

```json
{ "username": "admin", "password": "admin123" }
```

流式问答请求示例：

```json
{ "message": "API 测试应该放在哪个目录？" }
```

SSE 返回示例：

```text
data: {"type":"sources","sources":[...],"trace":[...]}
data: {"type":"token","content":"根据"}
data: {"type":"done"}
```

## 开发与测试

### 后端测试

```bash
cd backend
pytest -v
```

### 前端测试与构建

```bash
cd frontend
npm test -- --reporter verbose
npm run build
```

运行 Playwright 测试前，需要先安装浏览器：

```bash
cd frontend
npm run install:browsers
npm run test:e2e
```

## 运行注意事项

- 首次调用 Embedding 时可能会下载默认的 `BAAI/bge-m3` 模型，请为 `MODEL_CACHE_DIR` 准备足够磁盘空间。
- `ENABLE_GRAPH=true` 会在入库时增加 LLM 调用，可能显著增加耗时和费用。
- `CHROMA_PATH` 保存运行时向量数据；如果不想重新入库，请定期备份。
- Chroma 目录下的上传文件和运行数据属于本地数据，不应提交到仓库。
- 当前 CORS 配置偏向本地开发，公开部署前应限制允许来源。
- 内存 BM25 索引会在启动时从配置的知识库文件重建；生产环境可按需补充上传文档的索引重建策略。

## 发布或部署前安全检查

- 确认 `.env` 文件已被忽略，并且没有提交任何真实密钥。
- 如果密钥曾出现在共享文件或终端日志中，请立即轮换。
- 设置强随机 `JWT_SECRET`。
- 通过环境变量修改默认密码，或替换演示用认证层。
- 部署前检查 CORS、认证逻辑、上传限制和文档删除策略。

## License

本项目使用 Apache License 2.0 许可证，详见 `LICENSE`。
