# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Running the Application
```bash
# Quick start using provided script
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install dependencies
uv sync

# Run Python modules with uv
uv run [command]
```

### Environment Setup
- Create `.env` file in root with `ANTHROPIC_API_KEY=your_key_here`
- Requires Python 3.13+ and uv package manager

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) chatbot system** with the following structure:

### Core Components
- **RAGSystem** (`backend/rag_system.py`): Main orchestrator that coordinates all components
- **DocumentProcessor** (`backend/document_processor.py`): Converts documents into Course objects and chunks
- **VectorStore** (`backend/vector_store.py`): ChromaDB integration for semantic search
- **AIGenerator** (`backend/ai_generator.py`): Anthropic Claude API integration with tool support
- **SessionManager** (`backend/session_manager.py`): Conversation history management
- **ToolManager** (`backend/search_tools.py`): Tool-based search system for AI agent

### Data Flow
1. Documents in `docs/` folder are processed into Course objects with lessons
2. Content is chunked and stored in ChromaDB with embeddings
3. User queries trigger tool-based searches through Claude AI
4. AI generates responses using retrieved context and maintains conversation history

### Key Models
- **Course**: Contains title, description, lessons, and metadata
- **Lesson**: Individual course sections with content
- **CourseChunk**: Text chunks for vector search with course/lesson context

### API Structure
- FastAPI backend serves both API endpoints (`/api/query`, `/api/courses`) and static frontend
- Frontend is a simple HTML/JS interface in `frontend/` directory
- No authentication or user management - single session model

### Tool-Based Architecture
The system uses Claude's tool calling capabilities rather than traditional RAG retrieval, allowing the AI to search the knowledge base dynamically during response generation.

## Application URLs
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`