# RAG Chatbot Diagnostic Report

## Problem Statement
The RAG chatbot returns "query failed" for content-related questions.

## Root Cause Analysis

After comprehensive testing, the issue is **NOT** caused by:
- ❌ Empty database (found 4 courses with content)
- ❌ Vector search functionality (works perfectly)  
- ❌ Tool registration (both tools properly registered)
- ❌ AI provider setup (Ollama working correctly)
- ❌ Basic system architecture (all components functional)

## Actual Issues Identified

### 1. 🔴 **PRIMARY ISSUE: Tool Parameter Validation Error**

**Problem**: The `get_course_outline` tool requires a `course_name` parameter, but the AI sometimes calls it without providing this parameter when users ask general questions like "What courses are available?"

**Evidence**:
```
Response preview: Error executing tool: CourseOutlineTool.execute() missing 1 required positional argument: 'course_name'
```

**Impact**: Causes "query failed" responses for queries about available courses.

### 2. 🟡 **SECONDARY ISSUE: ChromaDB Metadata Handling**

**Problem**: ChromaDB v1.0.15 cannot handle `None` values in metadata fields.

**Evidence**:
```
TypeError: 'NoneType' object cannot be converted to 'PyString'
```

**Impact**: Prevents adding new course metadata when optional fields contain `None`.

### 3. 🟡 **MINOR ISSUE: System Prompt Ambiguity**

**Problem**: The system prompt doesn't clearly handle general course listing queries.

**Current prompt issues**:
- Says to use `get_course_outline` for "course overview" but doesn't specify that it needs a course name
- No guidance for "list all courses" type queries

## Test Results Summary

| Component | Status | Details |
|-----------|--------|---------|
| VectorStore initialization | ✅ PASS | Works correctly |
| VectorStore empty search | ✅ PASS | Handles empty DB correctly |
| VectorStore with data | ✅ PASS | Found 4 courses, search returns 5 results |
| CourseSearchTool direct execution | ✅ PASS | Works perfectly |
| ToolManager execution | ✅ PASS | Works perfectly |
| Ollama availability | ✅ PASS | gemma3:4b model available and working |
| End-to-end queries | 🟡 PARTIAL | 3/4 test queries successful |
| AI provider integration | ✅ PASS | Ollama generates coherent responses |

## Proposed Fixes

### Fix 1: Add Course Catalog Listing Tool 🔧

**Create a new tool for listing all available courses:**

```python
class CourseListTool(Tool):
    """Tool for listing all available courses"""
    
    def get_tool_definition(self) -> Dict[str, Any]:
        return {
            "name": "list_all_courses", 
            "description": "List ALL available courses in the system. Use this when user asks 'what courses are available', 'show me all courses', or similar general course listing queries.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    
    def execute(self) -> str:
        """List all available courses with basic info."""
        course_titles = self.store.get_existing_course_titles()
        if not course_titles:
            return "No courses are currently available in the system."
            
        courses_info = []
        for title in course_titles:
            courses_info.append(f"• {title}")
            
        return f"Available courses ({len(course_titles)}):\n\n" + "\n".join(courses_info)
```

### Fix 2: Improve Tool Parameter Validation 🛡️

**Add better error handling in CourseOutlineTool:**

```python
def execute(self, course_name: str = None) -> str:
    """Execute with better parameter validation."""
    if not course_name:
        # Fallback: list available courses
        titles = self.store.get_existing_course_titles()
        if not titles:
            return "No courses available. Please specify a course name to get its outline."
        return f"Please specify which course outline you want. Available courses: {', '.join(titles)}"
    
    # Continue with existing logic...
```

### Fix 3: Update System Prompt 📝

**Improve AI tool selection guidance:**

```python
SYSTEM_PROMPT = """You are a course materials assistant. You MUST use the provided tools to answer questions about course content and structure.

CRITICAL TOOL USAGE RULES:
- For general questions about available courses → use list_all_courses tool
- For specific course outline/structure (when course name is mentioned) → use get_course_outline tool  
- For questions about course content/concepts → use search_course_content tool
- NEVER answer course-related questions using your own knowledge - ALWAYS use tools first
- Use exactly ONE tool per query

TOOL SELECTION GUIDE:
- list_all_courses: "what courses are available", "show me all courses", "list courses"
- get_course_outline: "outline of MCP course", "structure of [specific course]", "lessons in [course name]"
- search_course_content: "what is MCP", "explain concepts", "how does X work"
"""
```

### Fix 4: ChromaDB Metadata Sanitization 🧹

**Add metadata sanitization in VectorStore:**

```python
def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values from metadata to prevent ChromaDB errors."""
    return {k: v for k, v in metadata.items() if v is not None}

def add_course_metadata(self, course: Course):
    """Add course metadata with sanitized values."""
    metadata = {
        "title": course.title,
        "instructor": course.instructor,
        "course_link": course.course_link,
        "lessons_json": json.dumps(lessons_metadata),
        "lesson_count": len(course.lessons)
    }
    
    # Sanitize to remove None values
    metadata = self._sanitize_metadata(metadata)
    
    self.course_catalog.add(
        documents=[course.title],
        metadatas=[metadata],
        ids=[course.title]
    )
```

## Implementation Priority

1. **🔴 HIGH PRIORITY**: Fix 1 (Add CourseListTool) - Solves the main "query failed" issue
2. **🔴 HIGH PRIORITY**: Fix 3 (Update system prompt) - Prevents AI from making invalid tool calls  
3. **🟡 MEDIUM PRIORITY**: Fix 2 (Parameter validation) - Adds defensive error handling
4. **🟡 LOW PRIORITY**: Fix 4 (Metadata sanitization) - Fixes edge case with new data

## Verification Steps

After implementing fixes:

1. Test general queries: "What courses are available?"
2. Test specific queries: "Outline of MCP course"  
3. Test content queries: "What is MCP protocol?"
4. Verify all tools work with existing data
5. Test adding new course data

## Expected Outcome

With these fixes:
- ✅ General course listing queries will work
- ✅ Specific course outline queries will work
- ✅ Content search queries will continue working
- ✅ System will be more robust against edge cases
- ✅ No more "query failed" errors for supported query types

## Current System Status

**CONCLUSION**: The RAG system is fundamentally sound. The database has content, search works, AI provider works. The issue is a specific tool parameter validation problem that affects certain types of queries. With the proposed fixes, the system should work reliably for all supported query types.