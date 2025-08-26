# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- **Quick start**: `./run.sh` (creates docs directory, starts server on port 8000)
- **Manual start**: `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Install dependencies**: `uv sync`

### Environment Setup
- Create `.env` file with: `ANTHROPIC_API_KEY=your_key_here`
- Application loads documents from `/docs` folder on startup
- ChromaDB storage location: `./backend/chroma_db`

### Access Points
- Web interface: http://localhost:8000
- API documentation: http://localhost:8000/docs

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for querying course materials with AI-powered responses.

### Core Architecture Pattern
The system uses a **tool-based RAG architecture** where Claude doesn't receive pre-retrieved context. Instead, Claude actively decides when to search and calls search tools during response generation.

### Key Components & Data Flow

**Request Processing Flow:**
1. Frontend sends query to `/api/query` endpoint
2. `RAGSystem` orchestrates the response using conversation history
3. `AIGenerator` (Claude API) processes query with available tools
4. Claude calls `CourseSearchTool` when it needs information
5. Search tool performs vector similarity search via `VectorStore` (ChromaDB)
6. Retrieved chunks provide context for AI response generation
7. Response returned with answer, sources, and session_id

**Component Responsibilities:**
- `app.py` - FastAPI server, CORS, static files, API endpoints
- `rag_system.py` - Main orchestrator connecting all components
- `ai_generator.py` - Claude API integration with tool calling
- `search_tools.py` - Tool-based search functionality using vector similarity
- `vector_store.py` - ChromaDB integration for semantic search
- `document_processor.py` - Text chunking and course structure extraction
- `session_manager.py` - Conversation history management
- `models.py` - Data models (Course, Lesson, CourseChunk)

### Data Models
- **Course**: Contains title, instructor, lessons list, course_link
- **Lesson**: Contains lesson_number, title, lesson_link
- **CourseChunk**: Text chunks for vector storage with course/lesson metadata

### Vector Search Implementation
- Uses ChromaDB with sentence-transformers embeddings (`all-MiniLM-L6-v2`)
- Documents chunked into ~800 character pieces with 100 character overlap
- Semantic similarity search returns top 5 most relevant chunks
- Search tool tracks sources for response attribution

### Configuration
All settings in `backend/config.py`:
- Chunk size: 800 characters
- Chunk overlap: 100 characters  
- Max search results: 5
- Max conversation history: 2 exchanges
- Default model: `claude-sonnet-4-20250514`

### Document Processing
- Supports PDF, DOCX, TXT files from `/docs` folder
- Extracts course structure with lessons and metadata
- Creates vector embeddings for all text chunks
- Avoids reprocessing existing courses (checks by title)

### Session Management
- Creates unique session IDs for conversation tracking
- Maintains conversation history (2 exchanges max)
- Thread-safe session storage for concurrent users