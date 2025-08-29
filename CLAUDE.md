# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install dependencies
uv sync

# Set up environment variables (required)
# Create .env file with: ANTHROPIC_API_KEY=your_api_key_here
```

### Running the Application
```bash
# Quick start with script
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Development Tools
```bash
# Run from backend directory
cd backend

# Start development server with hot reload
uv run uvicorn app:app --reload --port 8000

# Access points:
# - Web Interface: http://localhost:8000
# - API Documentation: http://localhost:8000/docs
```

## Architecture Overview

This is a full-stack RAG (Retrieval-Augmented Generation) chatbot system for querying course materials. The system follows a modular architecture with clear separation between components:

### Core Components

**RAGSystem** (`backend/rag_system.py`): Main orchestrator that coordinates all components
- Integrates document processing, vector storage, AI generation, and session management
- Provides high-level methods for adding documents and processing queries
- Manages the tool system for enhanced search capabilities

**VectorStore** (`backend/vector_store.py`): ChromaDB-based semantic search
- Uses sentence-transformers for embeddings (all-MiniLM-L6-v2)
- Stores both course metadata and content chunks in separate collections
- Provides semantic search with configurable result limits

**DocumentProcessor** (`backend/document_processor.py`): Text processing and structuring
- Extracts structured course information (title, lessons, instructor)
- Performs intelligent text chunking with sentence-based boundaries and overlap
- Parses course documents into Course and CourseChunk models

**AIGenerator** (`backend/ai_generator.py`): Anthropic Claude integration
- Handles prompt construction and response generation
- Manages conversation context and source attribution
- Uses Claude Sonnet 4 for high-quality responses

**SessionManager** (`backend/session_manager.py`): Conversation state management
- Maintains conversation history with configurable message limits
- Provides session isolation for multiple users
- Manages message storage and retrieval

### Data Models (`backend/models.py`)
- **Course**: Represents complete course with lessons and metadata
- **Lesson**: Individual lesson with title and optional link
- **CourseChunk**: Text chunks for vector storage with course/lesson context

### Frontend Architecture
- **Static files**: `frontend/` directory with HTML, CSS, and JavaScript
- **Real-time chat**: WebSocket-like communication via REST API
- **Responsive UI**: Sidebar with course stats and suggested questions
- **Markdown rendering**: Uses marked.js for rich text display

### API Structure (`backend/app.py`)
- **POST /api/query**: Main query endpoint with session management
- **GET /api/courses**: Course statistics and metadata
- **FastAPI**: Modern async web framework with automatic OpenAPI docs
- **CORS enabled**: Supports cross-origin requests

## Configuration System

All settings centralized in `backend/config.py`:
- **ANTHROPIC_API_KEY**: Required for Claude API access
- **ANTHROPIC_MODEL**: "claude-sonnet-4-20250514"
- **EMBEDDING_MODEL**: "all-MiniLM-L6-v2"
- **CHUNK_SIZE**: 800 characters for text chunks
- **CHUNK_OVERLAP**: 100 characters overlap between chunks
- **MAX_RESULTS**: 5 semantic search results
- **MAX_HISTORY**: 2 conversation messages to remember
- **CHROMA_PATH**: "./chroma_db" for vector storage

## Document Processing

The system processes course documents from the `docs/` directory on startup:
- Expects plain text files with course content
- Automatically extracts course structure and metadata
- Creates semantic embeddings for all content
- Supports dynamic addition of new course materials

## Key Integration Points

1. **FastAPI + Frontend**: Static file serving with API endpoints
2. **ChromaDB + SentenceTransformers**: Vector storage and similarity search
3. **Anthropic Claude**: AI response generation with context
4. **Session Management**: Conversation state across multiple interactions
5. **Tool System**: Extensible search and query enhancement capabilities

## Development Notes

- Uses `uv` as the Python package manager
- Requires Python 3.13 or higher
- ChromaDB creates persistent storage in `./chroma_db`
- Environment variables loaded from `.env` file
- Frontend uses relative API paths for flexible deployment
- always use uv for package management and not pip
- use uv to run python files