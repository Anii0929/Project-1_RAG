# User Query Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 FRONTEND                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  1. User types query in chat input                                             │
│  2. JavaScript captures Enter/click event                                      │
│  3. Disables input, shows loading animation                                    │
│  4. POST /api/query { query, session_id }                                     │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI BACKEND                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  5. app.py receives POST request                                              │
│  6. Creates session_id if not provided                                        │
│  7. Calls rag_system.query(query, session_id)                                │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               RAG SYSTEM                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  8. rag_system.py orchestrates the request                                    │
│  9. Retrieves conversation history from session_manager                       │
│ 10. Formats query as prompt for AI                                           │
│ 11. Calls ai_generator.generate_response()                                   │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              AI GENERATOR                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 12. ai_generator.py calls Anthropic Claude API                               │
│ 13. Sends query + system prompt + tools + history                            │
│ 14. Claude analyzes query and decides if search needed                       │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼ (if search needed)
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TOOL EXECUTION                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 15. Claude calls search_course_content tool                                  │
│ 16. tool_manager routes to CourseSearchTool                                  │
│ 17. CourseSearchTool.execute() called with parameters                        │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              VECTOR SEARCH                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 18. vector_store.search() executed                                           │
│ 19. If course_name provided, resolves via semantic search in course_catalog  │
│ 20. Searches course_content collection with ChromaDB                         │
│ 21. Uses Sentence Transformers embeddings for similarity                     │
│ 22. Returns top-k results with metadata                                      │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RESPONSE SYNTHESIS                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 23. CourseSearchTool formats results with course/lesson context              │
│ 24. Tool result sent back to Claude as tool_result message                   │
│ 25. Claude synthesizes final response using search context                   │
│ 26. AI generates natural language answer                                     │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            RESPONSE FLOW BACK                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 27. rag_system gets sources from tool_manager                                │
│ 28. Updates session conversation history                                     │
│ 29. Returns (response, sources) tuple                                        │
│ 30. FastAPI packages as QueryResponse JSON                                   │
│ 31. HTTP response sent to frontend                                           │
└─────────────────────┬───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND DISPLAY                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 32. JavaScript receives response                                             │
│ 33. Removes loading animation                                                │
│ 34. Renders AI response with markdown parsing                                │
│ 35. Shows collapsible sources section                                        │
│ 36. Re-enables input for next query                                          │
└─────────────────────────────────────────────────────────────────────────────────┘

KEY COMPONENTS:
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   ChromaDB      │  │ Sentence Trans- │  │ Session Manager │  │ Document Proc-  │
│ (Vector Store)  │  │ formers (Embed) │  │ (History)       │  │ essor (Chunks)  │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘

DATA FLOW:
User Query → API → RAG System → AI + Tools → Vector Search → Response Synthesis → Display