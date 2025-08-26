# RAG System Request Flow Diagram

```
┌─────────────────┐
│   Frontend      │
│  (index.html)   │
└─────────────────┘
         │
         │ 1. User types question
         │    POST /api/query
         │    {query: "...", session_id: "..."}
         ▼
┌─────────────────┐
│   FastAPI       │
│   (app.py)      │
│  /api/query     │
└─────────────────┘
         │
         │ 2. Validate request
         │    Create/get session
         ▼
┌─────────────────┐
│   RAG System    │
│ (rag_system.py) │
│   .query()      │
└─────────────────┘
         │
         │ 3. Get conversation history
         │    Construct prompt
         ▼
┌─────────────────┐
│  AI Generator   │
│(ai_generator.py)│
│ Claude API      │
└─────────────────┘
         │
         │ 4. Claude decides it needs
         │    to search for information
         ▼
┌─────────────────┐
│  Search Tools   │
│(search_tools.py)│
│CourseSearchTool │
└─────────────────┘
         │
         │ 5. Perform semantic search
         │    similarity_search(query)
         ▼
┌─────────────────┐
│  Vector Store   │
│(vector_store.py)│
│   ChromaDB      │
└─────────────────┘
         │
         │ 6. Vector similarity search:
         │    query → embedding
         │    find similar chunks
         ▼
┌─────────────────┐
│  Course Chunks  │
│   (embedded)    │
│ [chunk1, chunk2,│
│  chunk3, ...]   │
└─────────────────┘
         │
         │ 7. Return relevant chunks
         │    with source info
         ▼
┌─────────────────┐
│  Search Tools   │
│ (CourseSearch   │
│     Tool)       │
└─────────────────┘
         │
         │ 8. Return search results
         │    to Claude
         ▼
┌─────────────────┐
│  AI Generator   │
│ (Claude API)    │
│                 │
└─────────────────┘
         │
         │ 9. Generate response using
         │    retrieved context
         ▼
┌─────────────────┐
│   RAG System    │
│ - Update history│
│ - Get sources   │
│ - Return result │
└─────────────────┘
         │
         │ 10. Return QueryResponse
         │     {answer, sources, session_id}
         ▼
┌─────────────────┐
│   FastAPI       │
│   Response      │
│                 │
└─────────────────┘
         │
         │ 11. JSON response
         ▼
┌─────────────────┐
│   Frontend      │
│ - Display answer│
│ - Show sources  │
│ - Update chat   │
└─────────────────┘
```

## Vector Search Detail

```
User Query: "What is RAG?"
     │
     ▼
┌─────────────────────────────────────┐
│         Vector Search Process       │
│                                     │
│ 1. Query → Embedding Vector         │
│    "What is RAG?" → [0.1, 0.3, ...] │
│                                     │
│ 2. ChromaDB Similarity Search       │
│    Compare with stored embeddings   │
│                                     │
│ 3. Find Most Similar Chunks         │
│    - Course1_chunk_5: 0.85 similar  │
│    - Course3_chunk_12: 0.78 similar │
│    - Course2_chunk_3: 0.72 similar  │
│                                     │
│ 4. Return Top Matches with Sources  │
│    [{text: "RAG is...", source: ""},│
│     {text: "Retrieval...", source:""}]│
└─────────────────────────────────────┘
```

## Key Components

- **Tool-Based Architecture**: Claude actively calls search tools when needed
- **Semantic Search**: Vector embeddings enable meaning-based retrieval
- **Session Management**: Conversation history maintained across requests  
- **Source Tracking**: Search tool tracks which documents were used
- **Async Processing**: FastAPI handles requests asynchronously