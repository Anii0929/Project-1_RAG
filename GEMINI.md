# GEMINI.md - Course Materials RAG System

## Project Overview

This is a full-stack Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses. The application allows users to query course content and receive intelligent, context-aware answers.

### Key Technologies

- **Backend**: Python, FastAPI
- **Frontend**: HTML, CSS, JavaScript
- **AI**: Anthropic's Claude API
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers
- **Package Management**: uv

### Architecture

The system follows a client-server architecture:

1. **Frontend**: A web interface that allows users to interact with the chatbot
2. **Backend**: A FastAPI server that handles API requests and orchestrates the RAG pipeline
3. **Vector Store**: ChromaDB for storing and retrieving course content embeddings
4. **AI Generator**: Anthropic's Claude API for generating natural language responses
5. **Document Processor**: Processes course documents into structured chunks for vector storage

## Building and Running

### Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- An Anthropic API key

### Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**:
   ```bash
   uv sync
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory with your Anthropic API key:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

### Running the Application

#### Quick Start (Recommended)

```bash
chmod +x run.sh
./run.sh
```

#### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Accessing the Application

Once running, the application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Development Conventions

### Code Structure

- **backend/**: Contains all Python backend code
  - `app.py`: Main FastAPI application
  - `rag_system.py`: Core RAG system orchestrator
  - `document_processor.py`: Handles document parsing and chunking
  - `vector_store.py`: Manages ChromaDB interactions
  - `ai_generator.py`: Interfaces with Anthropic's Claude API
  - `search_tools.py`: Implements tool-based search functionality
  - `session_manager.py`: Manages conversation history
  - `config.py`: Application configuration
  - `models.py`: Data models using Pydantic

- **frontend/**: Contains all frontend code
  - `index.html`: Main HTML file
  - `script.js`: Client-side JavaScript logic
  - `style.css`: Styling

- **docs/**: Directory for course documents (loaded automatically on startup)

### Adding Course Documents

To add new course materials to the system:
1. Place documents in the `docs/` directory
2. Supported formats: `.pdf`, `.docx`, `.txt`
3. Documents are automatically processed when the server starts
4. Document format expected:
   - Line 1: Course Title: [title]
   - Line 2: Course Link: [url]
   - Line 3: Course Instructor: [instructor]
   - Following lines: Lesson markers and content (Lesson 1: Introduction, etc.)

### API Endpoints

- `POST /api/query`: Process a query about course materials
- `GET /api/courses`: Get course statistics and titles

### Development Workflow

1. Make changes to the code
2. The server automatically reloads with `--reload` flag
3. Test changes through the web interface or API documentation
4. Check the console for any error messages

## Testing

Currently, the project doesn't have explicit unit tests. Testing is done through:
1. Manual testing via the web interface
2. API testing through the auto-generated FastAPI docs at `http://localhost:8000/docs`