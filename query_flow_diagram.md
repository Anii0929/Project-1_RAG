# RAG Chatbot Query Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                FRONTEND                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   User Input    │───▶│   script.js     │───▶│  Loading UI     │        │
│  │                 │    │  sendMessage()  │    │                 │        │
│  │ "What is MCP?"  │    │                 │    │     ●●●         │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│                                   │                                        │
│                                   ▼                                        │
│                          POST /api/query                                   │
│                          {                                                 │
│                            "query": "What is MCP?",                       │
│                            "session_id": "session_1"                      │
│                          }                                                 │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND API                                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   FastAPI       │───▶│   QueryRequest  │───▶│   app.py        │        │
│  │   /api/query    │    │   Validation    │    │   endpoint      │        │
│  │                 │    │                 │    │                 │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│                                                          │                 │
│                                                          ▼                 │
│                                                 rag_system.query()         │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG ORCHESTRATION                                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   RAGSystem     │───▶│ Session Manager │───▶│   AI Generator  │        │
│  │  rag_system.py  │    │ Get History     │    │  ai_generator.py│        │
│  │                 │    │                 │    │                 │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│                                                          │                 │
│                                                          ▼                 │
│                                          Build prompt + tools              │
│                                          "Answer this question..."         │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLAUDE AI                                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   Claude API    │───▶│   Tool Choice   │───▶│  Tool Execution │        │
│  │  anthropic.py   │    │   Decision      │    │                 │        │
│  │                 │    │ "Need to search"│    │ search_course_  │        │
│  └─────────────────┘    └─────────────────┘    │     content     │        │
│                                                 └─────────────────┘        │
│                                                          │                 │
│                                                          ▼                 │
│                                               CourseSearchTool.execute()   │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VECTOR SEARCH                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  VectorStore    │───▶│   ChromaDB      │───▶│ Search Results  │        │
│  │  vector_store.py│    │   Embedding     │    │                 │        │
│  │                 │    │   Similarity    │    │ Documents +     │        │
│  └─────────────────┘    └─────────────────┘    │ Metadata        │        │
│                                                 └─────────────────┘        │
│                                                          │                 │
│                                                          ▼                 │
│                                              Format with course context    │
│                                              "[Course Title - Lesson 1]    │
│                                               MCP stands for..."            │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RESPONSE GENERATION                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   Tool Results  │───▶│   Claude API    │───▶│ Final Response  │        │
│  │   to Claude     │    │   Final Call    │    │                 │        │
│  │                 │    │                 │    │ "MCP (Model     │        │
│  └─────────────────┘    └─────────────────┘    │  Context        │        │
│                                                 │  Protocol)..."  │        │
│                                                 └─────────────────┘        │
│                                                          │                 │
│                                                          ▼                 │
│                                              Update conversation history   │
│                                              Extract sources               │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RESPONSE ASSEMBLY                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   QueryResponse │───▶│   JSON Format   │───▶│   HTTP 200      │        │
│  │   Model         │    │                 │    │   Response      │        │
│  │                 │    │ {               │    │                 │        │
│  └─────────────────┘    │  "answer": "...",│   └─────────────────┘        │
│                         │  "sources": [], │                               │
│                         │  "session_id"   │                               │
│                         │ }               │                               │
│                         └─────────────────┘                               │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FRONTEND DISPLAY                                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  Remove Loading │───▶│   Display       │───▶│   Update UI     │        │
│  │   Animation     │    │   Response      │    │                 │        │
│  │                 │    │                 │    │ ┌─────────────┐ │        │
│  └─────────────────┘    │ "MCP (Model     │    │ │   Sources   │ │        │
│                         │  Context        │    │ │ ▼ Course1   │ │        │
│                         │  Protocol)..."  │    │ │   Lesson 2  │ │        │
│                         │                 │    │ └─────────────┘ │        │
│                         └─────────────────┘    └─────────────────┘        │
│                                                          │                 │
│                                                          ▼                 │
│                                              Re-enable input for next      │
│                                              query                         │
└─────────────────────────────────────────────────────────────────────────────┘

Key Data Flow:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. User Query → HTTP Request → FastAPI Validation
2. RAG System → Session History → AI Generator  
3. Claude decides to use search tool → Vector Store Search
4. ChromaDB returns relevant chunks → Tool formats results
5. Claude generates final response → Session updated
6. JSON response → Frontend display with sources

Components Involved:
────────────────────
• Frontend: script.js, HTML/CSS
• API Layer: app.py (FastAPI)
• Orchestration: rag_system.py
• AI: ai_generator.py (Claude API)
• Tools: search_tools.py
• Storage: vector_store.py (ChromaDB)
• Session: session_manager.py
```