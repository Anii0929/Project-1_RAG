# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) chatbot system that enables semantic search and Q&A over course materials. The system uses ChromaDB for vector storage, Anthropic's Claude API with tool calling, and FastAPI for the backend.

## Development Commands

### Setup
```bash
# Install Python dependencies
uv sync

# Create .env file with Anthropic API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Running the Application
```bash
# Run using the provided script
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

The application runs on `http://localhost:8000` with API docs at `http://localhost:8000/docs`.

## Architecture

### Document Processing Pipeline
The system follows this flow for ingesting course materials:

1. **DocumentProcessor** (`document_processor.py`): Parses course documents with expected format:
   - Line 1: `Course Title: [title]`
   - Line 2: `Course Link: [url]`
   - Line 3: `Course Instructor: [instructor]`
   - Subsequent lines: Lesson markers (`Lesson N: [title]`) followed by content
   - Chunks text using sentence-based splitting with configurable overlap (config: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`)

2. **VectorStore** (`vector_store.py`): Manages two ChromaDB collections:
   - `course_catalog`: Stores course metadata (titles, instructors, lesson structure) for semantic course name resolution
   - `course_content`: Stores text chunks with metadata (course_title, lesson_number, chunk_index)

3. **Search Resolution**: When searching:
   - Course names are fuzzy-matched via vector search on the catalog
   - Content search supports optional filters by course title and lesson number

### RAG Query Flow
When a user submits a query:

1. **RAGSystem** (`rag_system.py`) orchestrates the components
2. **AIGenerator** (`ai_generator.py`) calls Claude with tool definitions
3. **ToolManager** (`search_tools.py`) provides the `search_course_content` tool
4. Claude decides whether to use the search tool based on the query
5. **CourseSearchTool** executes searches via VectorStore
6. Tool results are sent back to Claude for final response generation
7. **SessionManager** (`session_manager.py`) maintains conversation history (max 2 exchanges by default)

### Key Design Patterns
- **Tool-based search**: The AI decides when to search rather than always retrieving context
- **Dual collection strategy**: Separates course metadata from content for efficient name resolution
- **Chunk contextualization**: First chunk of each lesson gets prefixed with "Lesson N content:", last lesson chunks include course title for better retrieval

## Configuration

All settings are in `backend/config.py`:
- `ANTHROPIC_MODEL`: Currently `claude-sonnet-4-20250514`
- `EMBEDDING_MODEL`: `all-MiniLM-L6-v2` (SentenceTransformer)
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters
- `MAX_RESULTS`: 5 search results per query
- `MAX_HISTORY`: 2 conversation exchanges retained
- `CHROMA_PATH`: `./chroma_db` (relative to backend directory)

## Data Models

Core models in `backend/models.py`:
- **Course**: `title` (unique ID), `course_link`, `instructor`, `lessons[]`
- **Lesson**: `lesson_number`, `title`, `lesson_link`
- **CourseChunk**: `content`, `course_title`, `lesson_number`, `chunk_index`

## Document Format

Course documents in `docs/` must follow this structure:
```
Course Title: [Title]
Course Link: [URL]
Course Instructor: [Name]

Lesson 0: [Lesson Title]
Lesson Link: [URL]
[Lesson content...]

Lesson 1: [Lesson Title]
Lesson Link: [URL]
[Lesson content...]
```

## API Endpoints

- `POST /api/query`: Submit a query with optional `session_id`
- `GET /api/courses`: Get statistics on loaded courses
- On startup, documents from `../docs` are automatically loaded into ChromaDB
- always use uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies
- add and commit these changes