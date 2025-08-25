# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Environment Setup
```bash
# Install dependencies (including dev tools)
uv sync --group dev

# Create .env file from example
cp .env.example .env

# Choose AI provider in .env file:
# AI_PROVIDER=ollama (recommended - free, local)
# AI_PROVIDER=huggingface (free, cloud)
# AI_PROVIDER=search_only (no AI generation)
# AI_PROVIDER=anthropic (requires API key + credits)
```

### Code Quality Tools
```bash
# Format code automatically
./scripts/format.sh

# Run all quality checks
./scripts/quality.sh

# Individual tool usage:
uv run black .                    # Format code
uv run isort .                    # Sort imports
uv run flake8 .                   # Check code style
uv run mypy backend/ main.py      # Type checking
```

### AI Provider Setup

**Option 1: Ollama (Recommended - Free & Local)**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2

# Install Python dependency
uv add ollama

# Set in .env: AI_PROVIDER=ollama
```

**Option 2: Hugging Face (Free & Cloud)**
```bash
# Install dependencies
uv add transformers torch

# Set in .env: AI_PROVIDER=huggingface
```

**Option 3: Search Only (No AI costs)**
```bash
# No additional setup needed
# Set in .env: AI_PROVIDER=search_only
```

**Option 4: Anthropic (Requires credits)**
```bash
# Set in .env: 
# AI_PROVIDER=anthropic
# ANTHROPIC_API_KEY=your_api_key_here
```

### Access Points
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) chatbot** for querying course materials using semantic search and Claude AI.

### Core Architecture Pattern
The system follows a **tool-based RAG pattern** where Claude autonomously decides when to search course content:

1. **Query Processing**: User queries flow through FastAPI → RAGSystem → AIGenerator
2. **Tool Decision**: Claude evaluates if a search is needed using the `search_course_content` tool
3. **Vector Search**: ChromaDB performs semantic similarity search on course chunks
4. **Response Generation**: Claude synthesizes search results into conversational responses

### Key Components Interaction

**RAGSystem** (`rag_system.py`) - Central orchestrator that:
- Coordinates document processing pipeline during startup
- Manages query flow between components
- Handles session state and conversation history

**Document Processing Pipeline**:
- `DocumentProcessor` parses structured course files (Title → Instructor → Lessons)
- Text chunked at 800 chars with 100-char overlap, preserving sentence boundaries
- Each chunk enhanced with course/lesson context for better search relevance
- `VectorStore` embeds chunks using sentence-transformers and stores in ChromaDB

**Tool-Based Search**:
- `CourseSearchTool` registered with `ToolManager` for Claude's use
- Claude autonomously decides when to invoke `search_course_content`
- Search supports course name filtering and lesson number filtering
- Results formatted with course/lesson context headers

**AI Integration**:
- `AIGenerator` manages Claude API calls with tool definitions
- System prompt optimized for educational content with search instructions
- Conversation history maintained via `SessionManager` (max 2 exchanges)
- Tool execution handled through separate API call cycle

### Data Models
- `Course`: Container for course metadata + lessons list
- `Lesson`: Individual lesson with number, title, optional link
- `CourseChunk`: Text chunk with course/lesson context for vector search

### Expected Document Format
Course documents in `/docs/` should follow this structure:
```
Course Title: [title]
Course Link: [url]  
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [lesson_url]
[lesson content...]

Lesson 1: Next Topic
[more content...]
```

### Configuration
All settings centralized in `config.py`:
- **AI Provider**: Configurable via `AI_PROVIDER` environment variable
  - `ollama`: Local LLM (recommended, free)
  - `huggingface`: Free transformers models
  - `search_only`: Search without AI generation
  - `anthropic`: Claude API (requires credits)
- **Embedding model**: `all-MiniLM-L6-v2`
- **Chunk size**: 800 characters, 100 overlap
- **Max search results**: 5, conversation history: 2

### Frontend Integration
Simple HTML/JS frontend (`frontend/`) communicates via:
- `POST /api/query` for chat interactions
- `GET /api/courses` for course statistics
- Session management maintained across conversation turns
- Sources displayed to user with course/lesson attribution

### Startup Behavior
On application startup:
1. Documents in `/docs/` automatically processed and indexed
2. ChromaDB collections created/loaded from `./chroma_db/`
3. Existing courses detected to avoid reprocessing
4. FastAPI serves both API endpoints and static frontend files

### Key Design Decisions
- **Flexible AI providers**: Support for multiple AI backends (local/cloud/free)
- **Tool-based search**: AI decides when to search rather than always searching (when supported)
- **Context preservation**: Chunks include course/lesson metadata for better retrieval
- **Session continuity**: Conversation history maintained for follow-up questions  
- **Source attribution**: UI shows which course materials were referenced with clickable links
- **Deduplication**: Course titles used as unique identifiers to prevent reprocessing
- **Cost optimization**: Default to free/local options to avoid API costs

### Troubleshooting
- **No response or errors**: Check AI provider setup and dependencies
- **Ollama not working**: Ensure Ollama is installed and model is pulled
- **Search-only mode**: Returns formatted search results without AI generation
- **Slow responses**: HuggingFace models may be slower; consider Ollama for better performance

## Development Best Practices
- Always use uv to run the server, do not use pip directly
- Make sure to use uv to manage all dependencies
- Run `./scripts/format.sh` before committing changes to ensure consistent formatting
- Use `./scripts/quality.sh` to check code quality and catch issues early
- Code is automatically formatted with Black (88 character line length)
- Import sorting is handled by isort (configured for Black compatibility)
- Type hints are checked with mypy for better code reliability
- Code style is enforced by flake8 with sensible defaults