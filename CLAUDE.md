# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Course Materials RAG (Retrieval-Augmented Generation) system - a full-stack web application that enables semantic search and AI-powered question answering over course documents. The system uses ChromaDB for vector storage, Anthropic's Claude for generation, and serves a web interface for user interaction.

## Essential Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000

# Development with different port (if 8000 is occupied)
cd backend && uv run uvicorn app:app --reload --port 8001
```

### Setup Commands
```bash
# Install dependencies
uv sync

# Create environment file (required)
cp .env.example .env
# Then edit .env with your ANTHROPIC_API_KEY
```

## Architecture

### Core Components

The system follows a modular RAG architecture with clear separation of concerns:

**RAGSystem (backend/rag_system.py)** - Main orchestrator that coordinates all components:
- DocumentProcessor: Chunks course documents into searchable segments
- VectorStore: Manages ChromaDB for semantic search using sentence transformers
- AIGenerator: Handles Anthropic Claude API calls for response generation
- SessionManager: Maintains conversation context and history
- ToolManager: Coordinates search tools for enhanced retrieval

**Data Models (backend/models.py)**:
- `Course`: Contains title, instructor, lessons, and course metadata
- `Lesson`: Individual lessons with titles and links
- `CourseChunk`: Text segments with course/lesson context for vector storage

**Frontend Architecture**:
- Vanilla HTML/CSS/JS with pink-themed UI (frontend/style.css)
- JavaScript handles API communication and UI updates (frontend/script.js)
- FastAPI serves both static files and API endpoints

### Key Configuration (backend/config.py)

- Uses Claude Sonnet 4 model (`claude-sonnet-4-20250514`)
- Embedding model: `all-MiniLM-L6-v2`
- Document chunks: 800 characters with 100 character overlap
- Conversation history: 2 messages retained
- ChromaDB path: `./chroma_db`

### API Endpoints

- `POST /api/query`: Submit questions, returns AI response with sources
- `GET /api/courses`: Retrieve course statistics and titles
- `/`: Static file serving for frontend

### Document Processing Flow

1. Course documents (`.txt` files) placed in `docs/` folder
2. DocumentProcessor extracts course metadata and lessons from structured text
3. Content is chunked and stored in ChromaDB with course/lesson context
4. User queries trigger semantic search + Claude generation with retrieved context
5. Session manager maintains conversation history for follow-up questions

### Development Notes

- Uses `uv` for Python dependency management
- FastAPI with auto-reload for backend development
- Frontend styling uses CSS custom properties for theming
- ChromaDB persists to local filesystem for data retention
- Requires ANTHROPIC_API_KEY environment variable