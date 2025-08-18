# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system for course materials that combines:
- **FastAPI backend** (`backend/`) with RAG functionality
- **HTML/CSS/JS frontend** (`frontend/`) for web interface  
- **ChromaDB vector database** for semantic search
- **Anthropic Claude** for AI-powered responses
- **Course documents** in `docs/` folder (txt files)

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Activate virtual environment  
source .venv/bin/activate

# Set up environment variables (create .env file)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Access Points
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture

### Core Components
- **RAGSystem** (`backend/rag_system.py`) - Main orchestrator that coordinates all components
- **VectorStore** (`backend/vector_store.py`) - ChromaDB interface for semantic search
- **AIGenerator** (`backend/ai_generator.py`) - Anthropic Claude integration with tool support
- **DocumentProcessor** (`backend/document_processor.py`) - Processes course documents into chunks
- **SessionManager** (`backend/session_manager.py`) - Manages conversation context
- **ToolManager** (`backend/search_tools.py`) - Tool-based search functionality

### Data Flow
1. Course documents in `docs/` are processed into chunks on startup
2. User queries go through the RAG system which uses tools to search the vector store
3. Claude generates responses using retrieved context and conversation history
4. Frontend displays responses with source information

### Key Patterns
- **Tool-based architecture**: Claude uses search tools rather than direct vector queries
- **Session management**: Conversation context is maintained per session
- **Modular design**: Each component has a single responsibility
- **Configuration centralization**: All settings in `backend/config.py`

### Database Structure
- **ChromaDB collections**: Separate collections for course metadata and content chunks
- **Embedding model**: `all-MiniLM-L6-v2` via SentenceTransformers
- **Chunk strategy**: 800 characters with 100 character overlap

## Configuration

### Environment Variables
- `ANTHROPIC_API_KEY` - Required for Claude API access
- All other settings configured in `backend/config.py`

### Key Settings (`backend/config.py`)
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters  
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation exchanges

### Document Processing
- Supported formats: PDF, DOCX, TXT
- Course documents should be placed in `docs/` folder
- Documents are automatically loaded on application startup
- Duplicate detection prevents re-processing existing courses
- use uv to run python files