# RAG System Class Relationships and Workflows

## Main Classes and Their Relationships

```
┌─────────────────┐
│   FastAPI App   │
│   (app.py)      │
└─────────┬───────┘
          │
          v
┌─────────────────┐
│   RAGSystem     │ ◄─── Main Orchestrator
│ (rag_system.py) │
└─────────┬───────┘
          │
          │ Initializes and coordinates:
          │
    ┌─────┴─────┬─────────┬──────────┬───────────┬─────────────┐
    │           │         │          │           │             │
    v           v         v          v           v             v
┌───────────┐ ┌──────────┐ ┌────────┐ ┌─────────┐ ┌───────────┐ ┌──────────┐
│Document   │ │Vector    │ │AI      │ │Session  │ │Tool       │ │Course    │
│Processor  │ │Store     │ │Generator│ │Manager  │ │Manager    │ │SearchTool│
└───────────┘ └──────────┘ └────────┘ └─────────┘ └───────────┘ └──────────┘
```

## Workflow 1: Document Ingestion (Startup)

```
User starts app → FastAPI startup event
                      │
                      v
┌─────────────────────────────────────────────────────────────┐
│                   RAGSystem                                 │
│  add_course_folder() / add_course_document()               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   v
┌─────────────────────────────────────────────────────────────┐
│              DocumentProcessor                              │
│  • process_course_document()                               │
│  • Reads files, extracts content, creates Course object    │
│  • Chunks text into CourseChunk objects                    │
└──────────────────┬──────────────────────────────────────────┘
                   │ Returns: Course + List[CourseChunk]
                   v
┌─────────────────────────────────────────────────────────────┐
│                  VectorStore                               │
│  • add_course_metadata(course)                            │
│  • add_course_content(course_chunks)                      │
│  • Creates embeddings and stores in ChromaDB              │
└─────────────────────────────────────────────────────────────┘
                   │
                   v
              ChromaDB Files
           (./chroma_db/ directory)
```

## Workflow 2: Query Processing (User Chat)

```
User sends query → FastAPI /api/query endpoint
                      │
                      v
┌─────────────────────────────────────────────────────────────┐
│                   RAGSystem                                 │
│  query(query, session_id)                                  │
└──────┬──────────────────────────────────┬───────────────────┘
       │                                  │
       v                                  v
┌─────────────────┐              ┌─────────────────┐
│ SessionManager  │              │   AIGenerator   │
│ • get_history() │              │ • generate_     │
│ • add_exchange()│              │   response()    │
└─────────────────┘              └─────────┬───────┘
                                          │
                                          v
                                 ┌─────────────────┐
                                 │  ToolManager    │
                                 │ • Provides tool │
                                 │   definitions   │
                                 └─────────┬───────┘
                                          │
                                          v
                          Claude API calls CourseSearchTool
                                          │
                                          v
                                 ┌─────────────────┐
                                 │CourseSearchTool │
                                 │• search_courses │
                                 └─────────┬───────┘
                                          │
                                          v
                                 ┌─────────────────┐
                                 │  VectorStore    │
                                 │• search_content │
                                 │• Uses ChromaDB  │
                                 │  for similarity │
                                 └─────────┬───────┘
                                          │
                                          v
                              Results back to Claude
                                          │
                                          v
                                Response to User
```

## Key Data Models Flow

```
File (.txt/.pdf/.docx)
          │
          v (DocumentProcessor)
     Course Object
          │
          ├─── CourseChunk Objects
          │              │
          v              v (VectorStore)
   course_metadata   course_content
    collection       collection
          │              │
          └──── ChromaDB ┘
                  │
                  v (Search)
            SearchResults
                  │
                  v (AIGenerator)
            Final Response
```

## Dependencies and Relationships

- **RAGSystem**: Central orchestrator, depends on all other components
- **VectorStore**: Depends on Course and CourseChunk models, uses ChromaDB
- **DocumentProcessor**: Creates Course and CourseChunk objects
- **AIGenerator**: Uses ToolManager, makes Anthropic API calls
- **ToolManager**: Manages CourseSearchTool, interfaces with VectorStore
- **SessionManager**: Independent conversation tracking
- **Config**: Injected into RAGSystem, provides settings to all components

## External Dependencies

- **ChromaDB**: Persistent vector database (file-based)
- **Anthropic API**: Claude model for response generation  
- **SentenceTransformers**: Embedding model (all-MiniLM-L6-v2)
- **FastAPI**: Web framework and API endpoints