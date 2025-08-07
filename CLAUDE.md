# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation & Setup
```bash
# Install dependencies using uv package manager
uv sync

# Create .env file with required API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Running the Application
```bash
# Quick start with provided script
chmod +x run.sh
./run.sh

# Manual start (preferred for development)
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Development Workflow
- **Frontend**: Static files served from `frontend/` directory (HTML/CSS/JS)
- **Backend**: FastAPI server with auto-reload during development
- **Database**: ChromaDB persisted locally in `backend/chroma_db/`

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials with the following architecture:

### Core Components
- **`RAGSystem`** (`rag_system.py`): Main orchestrator that coordinates all components
- **`VectorStore`** (`vector_store.py`): ChromaDB-based vector storage with two collections:
  - `course_catalog`: Course metadata (titles, instructors, links)
  - `course_content`: Chunked course content for semantic search
- **`AIGenerator`** (`ai_generator.py`): Anthropic Claude API integration with tool support
- **`DocumentProcessor`** (`document_processor.py`): Processes course documents into chunks
- **`SessionManager`** (`session_manager.py`): Manages conversation history per session

### Key Design Patterns
- **Tool-based AI Search**: Uses Claude's function calling to search course content
- **Two-tier Vector Storage**: Separate collections for metadata vs content search
- **Session-aware Conversations**: Maintains conversation history per user session
- **Chunked Content Processing**: Documents split into 800-character chunks with 100-char overlap

### Configuration
All settings centralized in `config.py` with defaults:
- Chunk size: 800 characters (overlap: 100)
- Max search results: 5
- Embedding model: `all-MiniLM-L6-v2`
- Claude model: `claude-sonnet-4-20250514`

### API Endpoints
- `POST /api/query`: Process user queries with RAG
- `GET /api/courses`: Get course analytics
- `/`: Serves frontend static files

### Document Processing Flow
1. Documents from `docs/` folder are processed on startup
2. Each document becomes a `Course` with multiple `Lesson` objects
3. Content is chunked into `CourseChunk` objects for vector search
4. Both metadata and content chunks are stored separately for optimal retrieval

## Dependencies

The project uses **uv** as the package manager with core dependencies:
- `chromadb==1.0.15`: Vector database
- `anthropic==0.58.2`: Claude API integration
- `sentence-transformers==5.0.0`: Embedding generation
- `fastapi==0.116.1`: Web framework
- `uvicorn==0.35.0`: ASGI server

## File Structure Notes
- `docs/`: Course materials (TXT files processed on startup)
- `backend/chroma_db/`: Persisted vector database
- `frontend/`: Static web interface files
- `main.py`: Entry point (unused in favor of FastAPI app)