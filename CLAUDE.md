# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Course Materials RAG System - A Retrieval-Augmented Generation system that enables intelligent querying of course materials using vector search and Claude AI. Built with FastAPI backend and vanilla JavaScript frontend.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Dependencies Management
```bash
# Install dependencies
uv sync

# Add new dependency
uv add <package-name>
```

### Environment Setup
Create `.env` file in root directory:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Architecture Overview

The system follows a modular RAG architecture centered around the `RAGSystem` class in `backend/rag_system.py`:

### Core Components

1. **RAGSystem** (`rag_system.py`) - Main orchestrator that coordinates all components
2. **VectorStore** (`vector_store.py`) - ChromaDB integration for semantic search using sentence-transformers
3. **AIGenerator** (`ai_generator.py`) - Claude API integration with tool-calling capabilities
4. **DocumentProcessor** (`document_processor.py`) - Processes course documents into structured chunks
5. **SessionManager** (`session_manager.py`) - Manages conversation history and sessions
6. **ToolManager & CourseSearchTool** (`search_tools.py`) - Implements tool-based search for AI

### Data Models (`models.py`)
- `Course` - Represents complete courses with lessons
- `CourseChunk` - Text chunks for vector storage
- `Lesson` - Individual lessons within courses

### Key Architectural Patterns

- **Tool-based RAG**: Uses Claude's tool-calling to perform searches instead of direct context injection
- **Session Management**: Maintains conversation history for context-aware responses
- **Modular Design**: Each component has clear responsibilities and interfaces
- **Vector Search**: ChromaDB with sentence-transformers for semantic similarity

### API Structure (`app.py`)
- `/api/query` - Main RAG query endpoint
- `/api/courses` - Course analytics and statistics
- Static file serving for frontend at root path

### Configuration (`config.py`)
- Centralized settings using dataclass pattern
- Environment variable integration via python-dotenv
- Configurable chunking, embedding, and API parameters

## File Processing Flow

1. Documents placed in `docs/` folder are auto-loaded on startup
2. `DocumentProcessor` chunks documents and extracts course metadata
3. `VectorStore` creates embeddings and stores in ChromaDB
4. AI queries use tool-calling to search relevant content
5. Responses synthesized from search results with conversation context

## Testing and Development

The application uses uv for dependency management. No specific test framework is configured - check for test files or add testing setup as needed.

## Key Dependencies

- **FastAPI**: Web framework and API
- **ChromaDB**: Vector database for embeddings
- **Anthropic**: Claude AI integration
- **sentence-transformers**: Text embeddings
- **python-dotenv**: Environment variable management