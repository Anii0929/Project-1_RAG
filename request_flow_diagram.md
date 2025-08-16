# RAG System Request Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Frontend<br/>(script.js)
    participant FastAPI as FastAPI<br/>(app.py)
    participant RAG as RAG System<br/>(rag_system.py)
    participant Session as Session Manager<br/>(session_manager.py)
    participant AI as AI Generator<br/>(ai_generator.py)
    participant Tools as Tool Manager<br/>(search_tools.py)
    participant Vector as Vector Store<br/>(vector_store.py)
    participant ChromaDB as ChromaDB<br/>(chroma_db/)
    participant Claude as Anthropic Claude<br/>(API)

    User->>Frontend: 1. Types query & clicks send
    Note over Frontend: chatInput.value = "What is RAG?"
    
    Frontend->>Frontend: 2. Add user message to UI
    Frontend->>Frontend: 3. Show loading indicator
    
    Frontend->>FastAPI: 4. POST /api/query<br/>{query: "What is RAG?", session_id: null}
    
    FastAPI->>RAG: 5. rag_system.query(query, session_id)
    
    RAG->>Session: 6. create_session() or get_history(session_id)
    Session-->>RAG: 7. session_id & conversation history
    
    RAG->>AI: 8. generate_response(query, history, tools, tool_manager)
    
    AI->>Claude: 9. messages.create() with system prompt & tools
    Note over Claude: Claude decides: "This needs a search"
    
    Claude->>AI: 10. Tool use request: search("RAG explanation")
    
    AI->>Tools: 11. Execute tool: CourseSearchTool.search()
    
    Tools->>Vector: 12. search_content("RAG explanation")
    
    Vector->>ChromaDB: 13. query() with embeddings
    Note over ChromaDB: Semantic search through<br/>course chunks
    
    ChromaDB-->>Vector: 14. Relevant course chunks + metadata
    
    Vector-->>Tools: 15. SearchResults with documents & sources
    
    Tools-->>AI: 16. Tool response with search results
    
    AI->>Claude: 17. Tool results back to Claude
    Note over Claude: Claude synthesizes<br/>search results into answer
    
    Claude-->>AI: 18. Generated natural language response
    
    AI-->>RAG: 19. Final answer string
    
    RAG->>Session: 20. add_exchange(session_id, query, answer)
    
    RAG-->>FastAPI: 21. (answer, sources) tuple
    
    FastAPI-->>Frontend: 22. QueryResponse JSON<br/>{answer: "RAG is...", sources: [...], session_id: "abc123"}
    
    Frontend->>Frontend: 23. Remove loading indicator
    Frontend->>Frontend: 24. addMessage() with markdown rendering
    Frontend->>Frontend: 25. Display sources in collapsible section
    
    Frontend-->>User: 26. Show formatted response with sources
```

## Component Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[index.html<br/>Chat Interface]
        JS[script.js<br/>Event Handlers & API Calls]
        CSS[style.css<br/>Styling]
    end
    
    subgraph "API Layer"
        FastAPI[app.py<br/>REST Endpoints<br/>/api/query, /api/courses]
    end
    
    subgraph "RAG Core"
        RAGSys[rag_system.py<br/>Main Orchestrator]
        Session[session_manager.py<br/>Conversation History]
        AI[ai_generator.py<br/>Claude Integration]
        Tools[search_tools.py<br/>Tool Management]
    end
    
    subgraph "Data Processing"
        DocProc[document_processor.py<br/>Text Chunking & Parsing]
        Vector[vector_store.py<br/>Embedding & Search]
        Models[models.py<br/>Data Structures]
    end
    
    subgraph "Storage"
        ChromaDB[(ChromaDB<br/>Vector Database)]
        Docs[docs/<br/>Course Materials]
    end
    
    subgraph "External APIs"
        Claude[Anthropic Claude<br/>AI Generation]
    end
    
    UI --> JS
    JS --> FastAPI
    FastAPI --> RAGSys
    RAGSys --> Session
    RAGSys --> AI
    AI --> Tools
    AI --> Claude
    Tools --> Vector
    Vector --> ChromaDB
    DocProc --> Vector
    DocProc --> Docs
    Vector --> Models
    Session --> Models
    
    classDef frontend fill:#e1f5fe
    classDef api fill:#f3e5f5
    classDef core fill:#e8f5e8
    classDef data fill:#fff3e0
    classDef storage fill:#fce4ec
    classDef external fill:#f1f8e9
    
    class UI,JS,CSS frontend
    class FastAPI api
    class RAGSys,Session,AI,Tools core
    class DocProc,Vector,Models data
    class ChromaDB,Docs storage
    class Claude external
```

## Data Flow Summary

1. **User Input** → Frontend captures and validates
2. **HTTP Request** → JSON payload to FastAPI endpoint  
3. **Session Management** → Create/retrieve conversation context
4. **AI Processing** → Claude analyzes query and decides on tool usage
5. **Vector Search** → Semantic search through course chunks
6. **Response Generation** → Claude synthesizes search results
7. **Response Delivery** → JSON back to frontend with sources
8. **UI Update** → Markdown rendering and source display