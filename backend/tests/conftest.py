import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, Mock

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from config import Config
from vector_store import VectorStore
from document_processor import DocumentProcessor
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from session_manager import SessionManager
from models import Course, Lesson, CourseChunk

@pytest.fixture
def temp_chroma_path():
    """Create temporary ChromaDB path for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_config(temp_chroma_path):
    """Create test configuration with temporary paths"""
    config = Config()
    config.CHROMA_PATH = temp_chroma_path
    config.MAX_RESULTS = 5  # Set to reasonable value for testing
    config.ANTHROPIC_API_KEY = "test-key"
    return config

@pytest.fixture
def vector_store(test_config):
    """Create vector store instance for testing"""
    return VectorStore(
        chroma_path=test_config.CHROMA_PATH,
        embedding_model=test_config.EMBEDDING_MODEL,
        max_results=test_config.MAX_RESULTS
    )

@pytest.fixture
def sample_course():
    """Create sample course data for testing"""
    lessons = [
        Lesson(lesson_number=1, title="Introduction", lesson_link="https://example.com/lesson1"),
        Lesson(lesson_number=2, title="Overview", lesson_link="https://example.com/lesson2"),
        Lesson(lesson_number=3, title="Advanced Topics", lesson_link="https://example.com/lesson3")
    ]
    return Course(
        title="Test Course",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=lessons
    )

@pytest.fixture
def sample_chunks(sample_course):
    """Create sample course chunks for testing"""
    chunks = []
    for i in range(5):
        chunk = CourseChunk(
            content=f"This is test content for chunk {i+1}. It contains information about testing.",
            course_title=sample_course.title,
            lesson_number=1 if i < 2 else 2,
            chunk_index=i
        )
        chunks.append(chunk)
    return chunks

@pytest.fixture
def populated_vector_store(vector_store, sample_course, sample_chunks):
    """Vector store with sample data loaded"""
    vector_store.add_course_metadata(sample_course)
    vector_store.add_course_content(sample_chunks)
    return vector_store

@pytest.fixture
def course_search_tool(populated_vector_store):
    """CourseSearchTool with populated data"""
    return CourseSearchTool(populated_vector_store)

@pytest.fixture
def tool_manager(course_search_tool):
    """ToolManager with registered tools"""
    tm = ToolManager()
    tm.register_tool(course_search_tool)
    return tm

@pytest.fixture
def ai_generator(test_config):
    """AI generator instance for testing"""
    return AIGenerator(
        api_key=test_config.ANTHROPIC_API_KEY,
        model=test_config.ANTHROPIC_MODEL
    )

@pytest.fixture
def rag_system(test_config):
    """RAG system instance for testing"""
    return RAGSystem(test_config)

@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API testing"""
    mock_rag = Mock()
    mock_rag.session_manager = Mock()
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    mock_rag.session_manager.clear_session.return_value = None
    mock_rag.query.return_value = (
        "Test answer from the RAG system",
        [{"text": "Test source content", "link": "https://example.com/test"}]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course 1", "Test Course 2"]
    }
    return mock_rag

@pytest.fixture
def test_app(mock_rag_system):
    """Create test FastAPI app without static file mounting issues"""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from typing import List, Optional
    
    app = FastAPI(title="Course Materials RAG System Test")
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Inject mock RAG system
    app.state.rag_system = mock_rag_system
    
    @app.post("/api/session/new")
    async def create_new_session(prev_session_id: Optional[str] = None):
        try:
            if prev_session_id:
                app.state.rag_system.session_manager.clear_session(prev_session_id)
            session_id = app.state.rag_system.session_manager.create_session()
            return JSONResponse(content={"session_id": session_id})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = app.state.rag_system.session_manager.create_session()

            answer, sources = app.state.rag_system.query(request.query, session_id)

            source_objects = []
            for source in sources:
                if isinstance(source, dict) and "text" in source:
                    source_objects.append(Source(text=source["text"], link=source.get("link")))
                else:
                    source_objects.append(Source(text=str(source), link=None))

            return QueryResponse(
                answer=answer,
                sources=source_objects,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = app.state.rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System Test API"}
    
    return app

@pytest.fixture
def test_client(test_app):
    """Create test client for API testing"""
    return TestClient(test_app)