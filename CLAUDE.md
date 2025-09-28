# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Course Materials RAG (Retrieval-Augmented Generation) system that enables users to query course materials and receive intelligent, context-aware responses. The system uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.

## Development Commands

### Starting the Application

**Quick Start (Recommended):**
```bash
./run.sh
```

**Manual Start:**
```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Environment Setup

1. Install dependencies:
```bash
uv sync
```

2. Set up environment variables by creating `.env` file:
```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Application Access

- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Architecture Overview

### Project Structure

- **`backend/`** - FastAPI application with RAG system components
- **`frontend/`** - Static HTML/CSS/JS web interface
- **`docs/`** - Course materials (text files) that get processed and indexed
- **`main.py`** - Simple entry point (not used in web app)

### Core Components

**RAG System (`backend/rag_system.py`):**
- Main orchestrator that coordinates all components
- Handles document processing, vector storage, AI generation, and session management

**Document Processor (`backend/document_processor.py`):**
- Processes course documents from the `docs/` folder
- Extracts course metadata (title, lessons, links)
- Chunks text content for vector storage

**Vector Store (`backend/vector_store.py`):**
- ChromaDB-based vector storage for semantic search
- Stores both course metadata and content chunks
- Uses sentence-transformers for embeddings

**AI Generator (`backend/ai_generator.py`):**
- Anthropic Claude integration for generating responses
- Uses retrieved context to provide accurate answers

**Session Manager (`backend/session_manager.py`):**
- Manages conversation history and context
- Maintains session state for multi-turn conversations

**Search Tools (`backend/search_tools.py`):**
- Provides structured search capabilities
- Includes CourseSearchTool for semantic search

### Key Configuration (`backend/config.py`)

- **Embedding Model:** `all-MiniLM-L6-v2`
- **AI Model:** `claude-sonnet-4-20250514`
- **Chunk Size:** 800 characters with 100 character overlap
- **Database:** ChromaDB stored in `./chroma_db`

### Data Flow

1. Course documents in `docs/` are processed on startup
2. Text is chunked and embedded into ChromaDB
3. User queries are processed through semantic search
4. Relevant context is retrieved and sent to Claude
5. AI-generated responses are returned with source attribution

### API Endpoints

- `POST /api/query` - Submit queries and get AI responses
- `GET /api/courses` - Get course statistics and analytics

## Notes

- No testing framework is currently configured
- No linting/formatting tools are set up in pyproject.toml
- The system automatically loads documents from `docs/` on startup
- Frontend is served as static files through FastAPI
- CORS is enabled for all origins (development configuration)