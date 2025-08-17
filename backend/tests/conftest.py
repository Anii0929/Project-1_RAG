import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Add backend directory to path for imports
backend_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_path)

from config import config
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class MockConfig:
    """Mock configuration for testing"""
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    CHROMA_PATH = "./test_chroma"
    EMBEDDING_MODEL = "test-model"
    MAX_RESULTS = 5
    ANTHROPIC_API_KEY = "fake_key"
    ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
    MAX_HISTORY = 2


@pytest.fixture
def mock_config():
    """Provide mock configuration for tests"""
    return MockConfig()


@pytest.fixture
def sample_course():
    """Provide sample course data for testing"""
    return Course(
        title="Python Programming",
        instructor="John Doe",
        course_link="https://python-course.com",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Introduction to Python",
                lesson_link="https://python-course.com/lesson1",
                content="Python is a high-level programming language."
            ),
            Lesson(
                lesson_number=2,
                title="Variables and Data Types",
                lesson_link="https://python-course.com/lesson2",
                content="Python has various data types like int, str, list."
            )
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Provide sample course chunks for testing"""
    return [
        CourseChunk(
            content="Python is a high-level programming language used for web development.",
            course_title="Python Programming",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Python supports multiple programming paradigms including object-oriented.",
            course_title="Python Programming", 
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Variables in Python are dynamically typed and don't need explicit declaration.",
            course_title="Python Programming",
            lesson_number=2,
            chunk_index=0
        )
    ]


@pytest.fixture
def mock_rag_system(mock_config):
    """Provide a mocked RAG system for testing"""
    with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
         patch('rag_system.VectorStore') as mock_vector_store, \
         patch('rag_system.AIGenerator') as mock_ai_gen, \
         patch('rag_system.SessionManager') as mock_session_mgr, \
         patch('rag_system.ToolManager') as mock_tool_mgr, \
         patch('rag_system.CourseSearchTool') as mock_search_tool, \
         patch('rag_system.CourseOutlineTool') as mock_outline_tool:
        
        rag_system = RAGSystem(mock_config)
        
        # Configure mock behaviors
        mock_ai_gen.return_value.generate_response.return_value = "Mock AI response"
        mock_tool_mgr.return_value.get_last_sources.return_value = []
        mock_tool_mgr.return_value.get_tool_definitions.return_value = []
        mock_session_mgr.return_value.create_session.return_value = "test_session_123"
        mock_session_mgr.return_value.get_conversation_history.return_value = None
        mock_vector_store.return_value.get_course_count.return_value = 0
        mock_vector_store.return_value.get_existing_course_titles.return_value = []
        
        return rag_system


@pytest.fixture
def test_app():
    """Create a FastAPI test app without static file mounting issues"""
    from pydantic import BaseModel
    from typing import List, Optional, Union
    from fastapi import HTTPException
    
    # Create test app instance
    app = FastAPI(title="Course Materials RAG System Test", root_path="")
    
    # Add middleware (same as main app)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models (copy from main app)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class SourceLink(BaseModel):
        title: str
        course_link: Optional[str] = None
        lesson_link: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, SourceLink]]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    class ClearSessionRequest(BaseModel):
        session_id: str
    
    class ClearSessionResponse(BaseModel):
        success: bool
        message: str
    
    # Mock RAG system for testing
    mock_rag_system = Mock()
    
    # API Endpoints (copy from main app but with mock rag_system)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/clear-session", response_model=ClearSessionResponse)
    async def clear_session(request: ClearSessionRequest):
        try:
            mock_rag_system.session_manager.clear_session(request.session_id)
            return ClearSessionResponse(
                success=True,
                message=f"Session {request.session_id} cleared successfully"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def read_index():
        return {"message": "RAG System API", "status": "running"}
    
    # Store mock rag system reference for access in tests
    app.state.mock_rag_system = mock_rag_system
    
    return app


@pytest.fixture
def client(test_app):
    """Provide a test client for the FastAPI app"""
    return TestClient(test_app)


@pytest.fixture
def mock_query_response():
    """Provide mock query response data"""
    return {
        "answer": "Python is a high-level programming language.",
        "sources": [
            {
                "title": "Python Programming - Lesson 1",
                "course_link": "https://python-course.com",
                "lesson_link": "https://python-course.com/lesson1"
            }
        ],
        "session_id": "test_session_123"
    }


@pytest.fixture
def mock_course_analytics():
    """Provide mock course analytics data"""
    return {
        "total_courses": 3,
        "course_titles": [
            "Python Programming",
            "Web Development",
            "Data Science"
        ]
    }


@pytest.fixture(autouse=True)
def setup_environment():
    """Setup test environment variables"""
    os.environ["ANTHROPIC_API_KEY"] = "test_key"
    yield
    # Cleanup after test
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]


@pytest.fixture
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()