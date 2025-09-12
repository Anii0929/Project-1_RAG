# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
- **Start the application**: `./run.sh` or manually `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Install dependencies**: `uv sync`
- **Setup environment**: Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`

### Application URLs
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Architecture

This is a RAG (Retrieval-Augmented Generation) system built with FastAPI backend and simple HTML/JS frontend.

### Core Components
- **RAGSystem** (`backend/rag_system.py`): Main orchestrator that coordinates all components
- **VectorStore** (`backend/vector_store.py`): ChromaDB integration for semantic search
- **AIGenerator** (`backend/ai_generator.py`): Anthropic Claude integration with tool support
- **DocumentProcessor** (`backend/document_processor.py`): Processes PDF/DOCX/TXT files into chunks
- **SessionManager** (`backend/session_manager.py`): Handles conversation history
- **ToolManager** (`backend/search_tools.py`): Provides AI tools for course search

### Data Models
- **Course**: Represents a course with title, instructor, and lessons
- **Lesson**: Individual lesson with number, title, and optional link
- **CourseChunk**: Text chunks for vector storage with metadata

### Key Patterns
- The system uses tool-based AI interaction where Claude can call search functions
- Documents are chunked for vector storage and semantic search
- Conversation history is maintained per session
- ChromaDB handles vector embeddings using sentence-transformers
- Frontend is served statically from the FastAPI backend

### Document Processing Flow
1. Documents placed in `docs/` folder are auto-loaded on startup
2. `DocumentProcessor` extracts course structure and creates chunks
3. `VectorStore` stores both course metadata and content chunks
4. `RAGSystem` coordinates queries through AI tools and search

### Dependencies
- Python 3.13+ with uv package manager
- ChromaDB for vector storage
- Anthropic API for Claude integration
- FastAPI for web framework
- sentence-transformers for embeddings