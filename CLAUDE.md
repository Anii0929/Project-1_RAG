# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install/sync dependencies
uv sync

# Add new dependency
uv add <package-name>
```

### Environment Setup
- Create `.env` file with `ANTHROPIC_API_KEY=your_key_here`
- Requires Python 3.13+ and uv package manager

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) chatbot system** built with FastAPI backend and vanilla JS frontend that answers questions about course materials.

### Core Components Architecture

**RAG System Pipeline** (`backend/rag_system.py`):
- **DocumentProcessor**: Chunks course documents into searchable segments
- **VectorStore**: ChromaDB-based semantic search using sentence transformers
- **AIGenerator**: Anthropic Claude integration with tool-based search
- **SessionManager**: Conversation history management
- **ToolManager**: Search tool orchestration for Claude

**Data Flow**:
1. Documents → DocumentProcessor → CourseChunks 
2. CourseChunks → VectorStore (ChromaDB embeddings)
3. User Query → AIGenerator → ToolManager → VectorStore search
4. Search Results → Claude → Response with sources

### Key Technical Patterns

**Tool-Based Search Architecture**: Claude uses search tools rather than direct vector retrieval - the `CourseSearchTool` is registered with `ToolManager` and Claude calls it via function calling.

**Dual Collection Design**: 
- `course_metadata` collection: Course-level information
- `course_content` collection: Chunked content for detailed search

**Session Management**: Conversation history limited to `MAX_HISTORY=2` exchanges to manage context length.

**Embedding Strategy**: Uses `all-MiniLM-L6-v2` with 800-character chunks and 100-character overlap for optimal retrieval.

### File Structure
```
backend/           # FastAPI application
├── app.py         # Main FastAPI app with API endpoints
├── rag_system.py  # Core RAG orchestrator
├── config.py      # Configuration with environment variables
├── ai_generator.py    # Claude API integration
├── vector_store.py    # ChromaDB vector operations
├── document_processor.py # Text chunking and processing
├── search_tools.py    # Tool system for Claude function calling
├── session_manager.py # Conversation history
└── models.py      # Data models (Course, CourseChunk, etc.)

frontend/          # Static web interface
├── index.html     # Main UI
├── script.js      # Frontend logic
└── style.css      # Styling

docs/             # Course material storage
└── course*.txt   # Course content files
```

### Configuration
Key settings in `config.py`:
- `CHUNK_SIZE=800` and `CHUNK_OVERLAP=100` for document processing
- `MAX_RESULTS=5` for search results
- `ANTHROPIC_MODEL="claude-sonnet-4-20250514"`
- `EMBEDDING_MODEL="all-MiniLM-L6-v2"`

### API Endpoints
- `POST /api/query` - Main chat endpoint with session support
- `GET /api/courses` - Course analytics and statistics
- Static files served from `/` (frontend)

### Development Notes
- ChromaDB data stored in `./chroma_db/` (auto-created)
- CORS enabled for development
- Documents auto-loaded from `../docs` on startup
- Session-based conversation tracking
- No-cache headers for development static files