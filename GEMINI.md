## Project Overview

This project is a full-stack Retrieval-Augmented Generation (RAG) application. It provides a web-based chat interface for users to ask questions about a collection of course materials. The backend is built with Python using FastAPI, and the frontend is a simple HTML, CSS, and JavaScript application.

### Key Features

- **RAG System**: The core of the application is a RAG system that uses a vector store (ChromaDB) to find relevant course material and an AI model (Anthropic's Claude) to generate answers.
- **Web Interface**: A user-friendly chat interface allows users to interact with the RAG system.
- **Course Management**: The application can ingest course materials from text files, process them into chunks, and store them in the vector store.
- **Semantic Search**: The system uses sentence transformers to create embeddings for semantic search, allowing users to find relevant content even if their query doesn't exactly match the text.
- **Tool-based Search**: The AI model uses a "search" tool to query the vector store, which allows for more complex and targeted searches.
- **Conversation History**: The application maintains a conversation history for each user session, allowing the AI to have context for follow-up questions.

## Project Structure

The project is divided into two main parts: a `backend` and a `frontend`.

### Backend

The backend is a Python application built with FastAPI. It's responsible for the following:

- **`app.py`**: The main FastAPI application. It defines the API endpoints, handles CORS, and serves the frontend.
- **`rag_system.py`**: The main orchestrator for the RAG system. It integrates the document processor, vector store, and AI generator.
- **`document_processor.py`**: This module is responsible for reading course documents, splitting them into chunks, and extracting metadata.
- **`vector_store.py`**: This module manages the ChromaDB vector store. It handles adding, and searching for course content.
- **`ai_generator.py`**: This module interacts with Anthropic's Claude API to generate AI responses.
- **`search_tools.py`**: This module defines the "search" tool that the AI model uses to query the vector store.
- **`session_manager.py`**: This module manages user sessions and conversation history.
- **`config.py`**: This file contains the configuration for the application, such as API keys and model names.
- **`models.py`**: This file defines the Pydantic models used in the application.

### Frontend

The frontend is a simple HTML, CSS, and JavaScript application.

- **`index.html`**: The main HTML file for the application.
- **`style.css`**: The CSS file for styling the application.
- **`script.js`**: The JavaScript file that handles user interactions, sends requests to the backend, and displays the results.

## How it Works

1.  **Initialization**: When the application starts, it loads the course materials from the `docs` directory, processes them, and stores them in the ChromaDB vector store.
2.  **User Query**: A user enters a query in the chat interface.
3.  **API Request**: The frontend sends the query to the backend's `/api/query` endpoint.
4.  **RAG Process**:
    - The `RAGSystem` receives the query.
    - The `AIGenerator` sends the query to the Claude API, along with the definition of the `search_course_content` tool.
    - Claude determines that it needs to use the search tool and returns a "tool use" response.
    - The `ToolManager` executes the `search_course_content` tool, which queries the `VectorStore`.
    - The `VectorStore` performs a semantic search to find the most relevant course chunks.
    - The search results are returned to the `AIGenerator`.
    - The `AIGenerator` sends the search results back to Claude, which then generates a final answer.
5.  **Response**: The backend sends the answer and the sources back to the frontend.
6.  **Display**: The frontend displays the answer and the sources in the chat interface.

## How to Run

1.  Install the dependencies: `uv sync`
2.  Set up the environment variables in a `.env` file.
3.  Run the application: `./run.sh` or `cd backend && uv run uvicorn app:app --reload --port 8000`
4.  Open a browser to `http://localhost:8000`.
