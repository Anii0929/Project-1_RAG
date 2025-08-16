# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Environment Setup
```bash
# Install dependencies
uv sync

# Environment variables required in .env:
ANTHROPIC_API_KEY=your_key_here
```

### Development Server
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Uses uvicorn with auto-reload for development

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) system for course materials with a clear separation between frontend, API, and processing layers.

### Core RAG Flow
1. **Document Processing**: Course materials in `docs/` are parsed into structured lessons and chunked for vector storage
2. **Query Processing**: User queries trigger semantic search through ChromaDB, then Claude synthesizes responses
3. **Session Management**: Conversation history is maintained per session for context-aware responses

### Key Components

**RAG System (`rag_system.py`)**: Main orchestrator that coordinates all components. Handles the complete query lifecycle from user input to response generation.

**Document Processor (`document_processor.py`)**: Parses course documents with expected format:
```
Course Title: [title]
Course Link: [url]  
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [lesson_url]
[content...]
```

**Vector Store (`vector_store.py`)**: ChromaDB integration with sentence transformers for semantic search. Stores both course metadata and content chunks with configurable overlap.

**AI Generator (`ai_generator.py`)**: Anthropic Claude integration with tool calling. Uses a specialized system prompt for educational content and decides when to search vs. use general knowledge.

**Session Manager (`session_manager.py`)**: Maintains conversation history with configurable message limits. Creates unique session IDs for context preservation.

### Configuration System
All settings centralized in `config.py` with environment variable support:
- Chunk size/overlap for document processing
- Embedding model selection  
- Search result limits
- Conversation history depth
- Claude model selection

### Data Models
Pydantic models in `models.py` define the core entities:
- `Course`: Container with lessons and metadata
- `Lesson`: Individual lesson with optional links
- `CourseChunk`: Vector-searchable content pieces with course/lesson context

### Tool Integration
The system uses a tool management pattern where Claude can call search tools via the `search_tools.py` module. Tools are registered with the AI generator and can be invoked based on query analysis.

### Frontend Integration
Static files served from `frontend/` with a chat interface that maintains session state and displays responses with source citations. Uses relative API paths for deployment flexibility.

## File Structure Context

- `backend/app.py`: FastAPI application with CORS configuration and static file serving
- `docs/`: Course materials automatically loaded on startup
- `chroma_db/`: Persistent vector database storage
- Frontend files use cache-busting for development
- No test framework currently configured

## Development Notes

- Documents are automatically processed and indexed on server startup
- The system expects course documents to follow the structured format for proper parsing
- Session state is maintained in memory (not persistent across restarts)
- Vector embeddings use sentence-transformers with the all-MiniLM-L6-v2 model
- Claude model configured for claude-sonnet-4-20250514 with educational prompt optimization
- always use uv to run the server do not use pip directly
- make sure to use uv to all dependency
- use uv to run Python files