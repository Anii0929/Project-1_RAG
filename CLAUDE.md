# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application

```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Development Setup

```bash
# Install dependencies (requires Python 3.13+)
uv sync

# Create .env file with required API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

## Architecture

This is a RAG (Retrieval-Augmented Generation) system for course materials with three main layers:

### Core Components

- **RAGSystem** ([backend/rag_system.py](backend/rag_system.py)): Main orchestrator that coordinates all components
- **VectorStore** ([backend/vector_store.py](backend/vector_store.py)): ChromaDB integration with dual collections - `course_catalog` for metadata and `course_content` for chunks
- **AIGenerator** ([backend/ai_generator.py](backend/ai_generator.py)): Claude integration using tool-calling for structured search
- **DocumentProcessor** ([backend/document_processor.py](backend/document_processor.py)): Chunks documents (800 chars, 100 overlap)

### Key Design Decisions

1. **Tool-Based Search**: Uses Anthropic's tool-calling feature instead of direct RAG retrieval. The AI decides when and how to search.

2. **Dual Embedding Strategy**:

   - Course titles are embedded separately for semantic course name resolution
   - Course content is embedded with metadata for filtered search

3. **Embedding Creation**: Happens automatically on startup from [/docs](docs) folder and when calling `add_course_*` methods. Uses `all-MiniLM-L6-v2` model via SentenceTransformers.

4. **Session Management**: Maintains conversation history (last 2 exchanges) for context-aware responses.

## API Endpoints

- `POST /api/query` - Main RAG query with session support
- `GET /api/courses` - Course statistics and metadata
- Static files served from [/frontend](frontend)

## Configuration

Required environment variable:

- `ANTHROPIC_API_KEY` - Claude API access

Key settings in [backend/config.py](backend/config.py):

- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
- `CHUNK_SIZE`: 800 characters
- `MAX_RESULTS`: 5 search results
- `CHROMA_PATH`: ./chroma_db (persistent storage)

## Data Flow

1. Documents in [/docs](docs) are loaded on startup
2. Text is chunked and embedded into ChromaDB
3. User queries trigger tool-based search via Claude
4. Search results provide context for answer generation
5. Responses include source citations

## Important Notes

- ChromaDB data persists in [backend/chroma_db/](backend/chroma_db/)
- Duplicate courses are automatically skipped during loading
- Frontend uses vanilla JavaScript (no framework)
- No formal testing framework - this is a learning/demo project
