# AgenticRAG Knowledge Base WebUI

AgenticRAG Knowledge Base WebUI is a local knowledge-base question-answering application. It combines multi-format document ingestion, hybrid retrieval, optional knowledge-graph enrichment, streaming LLM answers, JWT authentication, and a React document-management interface.

The project is designed for private or team knowledge bases that need local document parsing and retrieval while calling an OpenAI-compatible chat-completion API for reasoning and answer generation.

## Features

- Upload and parse PDF, Word, Excel, Markdown, and TXT files.
- Retrieve with a hybrid pipeline: Chroma vector search, BM25 keyword search, and optional NetworkX knowledge-graph signals.
- Stream answers through Server-Sent Events (SSE).
- Manage documents from the Web UI with admin-only upload and delete actions.
- Isolate results by department for normal users while allowing admins to search the full knowledge base.
- Configure LLM, embedding model, Chroma path, JWT secret, and runtime paths through environment variables.
- Run backend tests with pytest and frontend checks with Vitest, Playwright, TypeScript, and Vite.

## Architecture

### Ingestion Flow

```text
Uploaded file
  -> documents.py parses supported formats into text
  -> split_markdown() chunks content by headings
  -> embedding.py creates bge-m3 embeddings
  -> vector_store.py persists vectors in Chroma
  -> retriever.py refreshes the BM25 index
  -> graph.py optionally extracts triples into a NetworkX graph
```

### Query Flow

```text
User question
  -> POST /api/chat/stream with a JWT token
  -> AgenticRAGAgent
      -> analyze query type
      -> run hybrid retrieval
      -> evaluate retrieval quality
      -> refine and retry when evidence is insufficient
      -> stream the final answer from the configured LLM
  -> React UI renders tokens and source cards
```

## Repository Layout

```text
AgenticRAG-KB/
+-- backend/
|   +-- kb_web_agent/
|   |   +-- api.py              # FastAPI app and HTTP routes
|   |   +-- agent.py            # ReActAgent compatibility plus AgenticRAGAgent
|   |   +-- auth.py             # JWT auth and role checks
|   |   +-- documents.py        # Document parsing and chunking
|   |   +-- embedding.py        # bge-m3 embedding wrapper
|   |   +-- graph.py            # Optional knowledge graph
|   |   +-- hybrid_retriever.py # Vector + BM25 + graph retrieval
|   |   +-- ingestion.py        # Async ingestion pipeline
|   |   +-- llm.py              # OpenAI-compatible LLM client
|   |   +-- retriever.py        # BM25 retriever
|   |   +-- schemas.py          # Pydantic schemas
|   |   +-- settings.py         # Environment configuration
|   |   +-- vector_store.py     # Chroma vector store
|   +-- tests/
|   +-- .env.example
|   +-- pyproject.toml
+-- frontend/
|   +-- src/
|   +-- tests/
|   +-- package.json
|   +-- vite.config.ts
+-- .env.example
+-- .gitignore
+-- README.md
```

## Requirements

| Component | Version / Notes |
| --- | --- |
| Python | 3.12+ |
| Node.js | 18+ |
| Backend | FastAPI, Chroma, sentence-transformers, BM25, NetworkX |
| Frontend | React 19, Vite, TypeScript |
| LLM API | Any OpenAI-compatible chat-completion endpoint |
| Embeddings | `BAAI/bge-m3` by default |

The backend supports Windows-style paths in configuration and normalizes them for WSL/Linux when needed.

## Quick Start

Clone the repository and install each side separately.

```bash
git clone <your-repo-url>
cd AgenticRAG-KB
```

### Backend

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

Edit `backend/.env` and fill in your API key, knowledge-base path, model cache path, and JWT secret.

Run the API server:

```bash
uvicorn kb_web_agent.api:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173` and log in with one of the development accounts listed below.

## Configuration

Copy one of the templates before running the backend:

```bash
cp backend/.env.example backend/.env
```

Key environment variables:

| Variable | Required | Description |
| --- | --- | --- |
| `ARK_API_KEY` | Yes | API key for the OpenAI-compatible LLM endpoint. |
| `ARK_BASE_URL` | Yes | Base URL for the LLM API. Defaults to Volcengine Ark coding endpoint. |
| `ARK_MODEL` | Yes | Chat model name. Defaults to `glm-5.1`. |
| `KNOWLEDGE_BASE_PATH` | Yes | Folder containing Markdown/TXT files loaded on startup. |
| `REACT_MAX_STEPS` | No | Maximum reasoning steps for the compatibility agent. |
| `RETRIEVAL_TOP_K` | No | Number of retrieved snippets per search. |
| `CHROMA_PATH` | No | Chroma persistence directory, relative to `backend/` by default. |
| `EMBEDDING_MODEL` | No | Embedding model name. Defaults to `BAAI/bge-m3`. |
| `MODEL_CACHE_DIR` | No | Hugging Face model cache directory. |
| `ENABLE_VECTOR_STORE` | No | Set to `false` to fall back to BM25-only retrieval. |
| `ENABLE_GRAPH` | No | Set to `true` to enable LLM triple extraction during ingestion. |
| `JWT_SECRET` | Yes | Secret used to sign JWT tokens. Use a strong random value. |
| `JWT_EXPIRE_MINUTES` | No | Token lifetime in minutes. |
| `ADMIN_PASSWORD` | No | Overrides the development admin password. |
| `USER_PASSWORD` | No | Overrides the development user password. |

Never commit `.env`, API keys, JWT secrets, uploaded files, Chroma data, model caches, virtual environments, or build output.

## Accounts And Permissions

Development accounts are defined in `backend/kb_web_agent/auth.py`.

| Username | Default password | Role | Access |
| --- | --- | --- | --- |
| `admin` | `admin123` | admin | Ask questions, view all departments, upload files, delete files. |
| `user` | `user123` | user | Ask questions and view documents allowed by the user's departments. |

For production use, replace the in-memory user table with persistent users and hashed passwords.

## API

FastAPI exposes interactive API documentation at `http://127.0.0.1:8000/docs`.

### Public Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Health and configuration summary. |
| `POST` | `/api/search` | BM25 search for local snippets. |
| `POST` | `/api/chat` | Synchronous compatibility chat endpoint. |

### Authenticated Endpoints

Use `Authorization: Bearer <token>`.

| Method | Path | Role | Description |
| --- | --- | --- | --- |
| `POST` | `/api/auth/login` | public | Login and receive a JWT. |
| `POST` | `/api/chat/stream` | user/admin | Main streaming chat endpoint. |
| `GET` | `/api/docs` | user/admin | List visible documents. |
| `POST` | `/api/docs/upload` | admin | Upload a document and start ingestion. |
| `GET` | `/api/docs/{doc_id}/status` | user/admin | Check ingestion progress. |
| `DELETE` | `/api/docs/{doc_id}` | admin | Delete a document. |

Example login request:

```json
{ "username": "admin", "password": "admin123" }
```

Example streaming chat request:

```json
{ "message": "Where should API tests live?" }
```

Example SSE payloads:

```text
data: {"type":"sources","sources":[...],"trace":[...]}
data: {"type":"token","content":"Based"}
data: {"type":"done"}
```

## Development

### Backend Tests

```bash
cd backend
pytest -v
```

### Frontend Tests And Build

```bash
cd frontend
npm test -- --reporter verbose
npm run build
```

For Playwright tests, install browsers first:

```bash
cd frontend
npm run install:browsers
npm run test:e2e
```

## Operational Notes

- The first embedding call can download the default `BAAI/bge-m3` model. Configure `MODEL_CACHE_DIR` to a location with enough space.
- `ENABLE_GRAPH=true` adds LLM calls during ingestion and can increase cost and latency.
- `CHROMA_PATH` stores runtime vector data. Back it up if you do not want to re-ingest documents.
- Uploaded files under the Chroma/runtime directory are local data and should not be committed.
- CORS is permissive for local development. Restrict it before deploying publicly.
- The in-memory BM25 index is rebuilt from configured knowledge-base files on startup. Uploaded documents persist in Chroma, but production deployments may need an explicit index rebuild strategy.

## Security Checklist Before Publishing Or Deploying

- Confirm `.env` files are ignored and contain no committed secrets.
- Rotate any API key that may have appeared in a shared file or terminal log.
- Set a strong `JWT_SECRET`.
- Change default passwords through environment variables or replace the demo auth layer.
- Review CORS, authentication, upload limits, and document deletion policies for your deployment.

## License

No license file is included yet. Add a license before inviting external reuse or contributions.
