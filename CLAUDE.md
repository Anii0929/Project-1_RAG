# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

**Development:**
```bash
# Start the application
./run.sh

# Manual start (from backend directory)
cd backend && uv run uvicorn app:app --reload --port 8000

# Install dependencies
uv sync

# Make run script executable (if needed)
chmod +x run.sh
```

**Environment Setup:**
Create `.env` file in project root:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Access Points:**
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

**Testing & Development:**
- Uses pytest framework with tests in `backend/tests/`
- Uses Python 3.13+ with uv package manager
- All dependencies managed via `pyproject.toml`

## Code Quality & Development Workflow

**Essential Commands:**
```bash
# Install all dependencies (including dev tools)
uv sync --group dev

# Format code (black + isort)
./scripts/format.sh

# Run linting checks
./scripts/lint.sh

# Comprehensive quality check (format, lint, test)
./scripts/check-quality.sh
```

**Code Quality Tools:**
- **Black**: Automatic code formatting with 88-character line length
- **isort**: Import sorting compatible with Black's style
- **flake8**: Linting for code quality and style consistency
- **pytest**: Unit testing framework

**Development Workflow:**
1. **Before making changes**: Run `./scripts/check-quality.sh` to ensure clean baseline
2. **During development**: Use `./scripts/format.sh` to auto-format code
3. **Before committing**: Run `./scripts/check-quality.sh` to catch issues
4. **Code style**: All code automatically formatted to Black standards (88 chars)

**Quality Standards:**
- All Python code formatted with Black (line length: 88)
- Imports organized with isort (Black-compatible profile)
- Code passes flake8 linting (E203, W503 ignored for Black compatibility)
- Comprehensive test coverage in `backend/tests/`
- No unused imports or variables in production code

**Configuration Files:**
- `pyproject.toml`: Black, isort, and project configuration
- `scripts/`: Development automation scripts for quality checks

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) system** that answers questions about course materials using ChromaDB vector storage and Anthropic's Claude AI.

### Core Data Flow
1. **Document Processing**: Course files in `docs/` are processed into structured Course/Lesson objects
2. **Vector Storage**: Content is chunked and stored in ChromaDB with dual collections (`course_catalog` + `course_content`)
3. **Query Processing**: User queries trigger semantic search through tool-based AI function calling
4. **Response Generation**: Claude synthesizes search results into contextual answers

### Component Architecture

**Backend Components** (`backend/`):
- `app.py` - FastAPI server with `/api/query` and `/api/courses` endpoints
- `rag_system.py` - Main orchestrator coordinating all components
- `ai_generator.py` - Claude API integration with tool execution support
- `search_tools.py` - Tool-based search system using function calling
- `vector_store.py` - ChromaDB interface with dual collection management
- `document_processor.py` - Parses course documents into structured data
- `session_manager.py` - Conversation history management
- `models.py` - Pydantic models (Course, Lesson, CourseChunk)
- `config.py` - Configuration management with environment variables

**Frontend** (`frontend/`):
- `index.html` - Chat interface with course statistics sidebar
- `script.js` - API communication and UI updates
- `style.css` - Styling for chat interface

### Data Models and Processing

**Document Format Expected:**
```
Course Title: [title]
Course Link: [url] 
Course Instructor: [instructor]

Lesson 1: [lesson title]
Lesson Link: [url]
[lesson content]

Lesson 2: [lesson title]
[lesson content]
```

**Vector Storage Strategy:**
- `course_catalog` collection: Course metadata for semantic name matching
- `course_content` collection: Text chunks with course/lesson metadata for content search
- Chunks include contextual headers: "Course [title] Lesson [n] content: [text]"

**Tool-Based Search:**
- AI decides when to use search tools based on query type
- `CourseSearchTool` supports filtering by course name and lesson number
- Course name resolution uses semantic matching in catalog collection
- Search results are formatted with course/lesson headers and tracked as sources

### Key Configuration

**Chunking Settings** (`config.py`):
- `CHUNK_SIZE: 800` - Character limit per chunk
- `CHUNK_OVERLAP: 100` - Overlap between chunks
- `MAX_RESULTS: 5` - Search results returned
- `MAX_HISTORY: 2` - Conversation exchanges remembered

**AI Model:**
- Uses `claude-sonnet-4-20250514` with system prompt optimized for educational content
- Temperature: 0 for consistent responses
- Max tokens: 800 per response

### Session Management
- Sessions auto-created if not provided
- Conversation history limited to prevent context overflow
- History included in AI prompts for contextual responses

### Startup Process
- Application auto-loads documents from `docs/` folder on startup
- Existing courses are detected to avoid reprocessing
- ChromaDB collections are persisted in `./chroma_db` directory
- Use `uv` to manage all dependencies (not pip)

### Development Notes
- FastAPI server runs on port 8000 with auto-reload enabled
- Frontend served as static files from `frontend/` directory
- ChromaDB storage persisted locally in `./chroma_db`
- Course documents expected in `docs/` folder at startup
- Environment variables loaded from `.env` file in project root