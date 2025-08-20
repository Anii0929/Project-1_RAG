import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import tempfile
import json

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from config import Config
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[Dict[str, Optional[str]]]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]


@pytest.fixture
def mock_config():
    """Create mock configuration for testing"""
    config = Mock(spec=Config)
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "./test_chroma"
    config.EMBEDDING_MODEL = "test-model"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
    config.MAX_HISTORY = 2
    return config


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for API testing"""
    mock_rag = Mock(spec=RAGSystem)
    
    # Mock session manager
    mock_session_manager = Mock()
    mock_session_manager.create_session.return_value = "test_session_123"
    mock_rag.session_manager = mock_session_manager
    
    # Default query response
    mock_rag.query.return_value = (
        "This is a test response about Python programming.",
        [{"text": "Python Basics - Lesson 1", "link": "https://example.com/lesson1"}]
    )
    
    # Default analytics response
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Python Basics", "Web Development", "Data Science"]
    }
    
    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """Create FastAPI test app without static file mounting"""
    app = FastAPI(title="Test Course Materials RAG System", root_path="")
    
    # Add middleware
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
    
    # Add API endpoints (without static file mounting)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        from fastapi import HTTPException
        try:
            # Create session if not provided
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            # Process query using RAG system
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
        """Get course analytics and statistics"""
        from fastapi import HTTPException
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        """Root endpoint for testing"""
        return {"message": "Test RAG System API"}
    
    return app


@pytest.fixture
def client(test_app):
    """Create test client for API testing"""
    return TestClient(test_app)


@pytest.fixture
def sample_course():
    """Create sample course data for testing"""
    lessons = [
        Lesson(
            lesson_number=1,
            lesson_title="Introduction to Python",
            lesson_link="https://example.com/lesson1",
            content="Python is a high-level programming language..."
        ),
        Lesson(
            lesson_number=2,
            lesson_title="Variables and Data Types",
            lesson_link="https://example.com/lesson2",
            content="Variables in Python are containers for storing data..."
        )
    ]
    
    course = Course(
        title="Python Programming Basics",
        course_link="https://example.com/python-course",
        instructor="Jane Doe",
        lessons=lessons
    )
    
    return course


@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    chunks = []
    for lesson in sample_course.lessons:
        chunk = CourseChunk(
            content=f"Course {sample_course.title} Lesson {lesson.lesson_number} content: {lesson.content[:100]}...",
            course_title=sample_course.title,
            lesson_number=lesson.lesson_number,
            lesson_title=lesson.lesson_title,
            lesson_link=lesson.lesson_link,
            instructor=sample_course.instructor,
            course_link=sample_course.course_link
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing"""
    return SearchResults(
        documents=[
            "Python is a high-level programming language that's great for beginners.",
            "Variables in Python are created when you assign a value to them.",
        ],
        metadata=[
            {
                "course_title": "Python Programming Basics",
                "lesson_number": 1,
                "lesson_title": "Introduction to Python",
                "lesson_link": "https://example.com/lesson1"
            },
            {
                "course_title": "Python Programming Basics", 
                "lesson_number": 2,
                "lesson_title": "Variables and Data Types",
                "lesson_link": "https://example.com/lesson2"
            }
        ],
        distances=[0.1, 0.15],
        error=None
    )


@pytest.fixture
def mock_dependencies():
    """Mock all RAG system dependencies for comprehensive testing"""
    mocks = {}
    
    with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
         patch('rag_system.VectorStore') as mock_vector_store, \
         patch('rag_system.AIGenerator') as mock_ai_gen, \
         patch('rag_system.SessionManager') as mock_session_mgr:
        
        mocks['document_processor'] = mock_doc_proc
        mocks['vector_store'] = mock_vector_store  
        mocks['ai_generator'] = mock_ai_gen
        mocks['session_manager'] = mock_session_mgr
        
        yield mocks


@pytest.fixture
def temp_docs_dir():
    """Create temporary directory with sample course documents"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample course file
        course_content = """Course Title: Python Programming Basics
Course Link: https://example.com/python-course
Course Instructor: Jane Doe

Lesson 1: Introduction to Python
Lesson Link: https://example.com/lesson1
Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.

Lesson 2: Variables and Data Types
Lesson Link: https://example.com/lesson2
Variables in Python are containers for storing data values. Unlike other programming languages, Python has no command for declaring a variable. A variable is created the moment you first assign a value to it.
"""
        
        course_file = os.path.join(temp_dir, "python_basics.txt")
        with open(course_file, 'w') as f:
            f.write(course_content)
        
        yield temp_dir


@pytest.fixture(autouse=True)
def prevent_real_api_calls():
    """Prevent real API calls during testing"""
    with patch('anthropic.Anthropic'), \
         patch('chromadb.PersistentClient'), \
         patch('sentence_transformers.SentenceTransformer'):
        yield


@pytest.fixture
def mock_session():
    """Create mock session data for testing"""
    return {
        "session_id": "test_session_123",
        "conversation_history": "User: Hello\nAssistant: Hi! How can I help you with course materials today?",
        "created_at": "2024-01-01T10:00:00Z"
    }


@pytest.fixture  
def api_test_data():
    """Common test data for API tests"""
    return {
        "valid_query": {
            "query": "What is Python programming?",
            "session_id": "test_session_123"
        },
        "query_without_session": {
            "query": "Explain variables in Python"
        },
        "expected_response": {
            "answer": "Python is a high-level programming language...",
            "sources": [
                {"text": "Python Basics - Lesson 1", "link": "https://example.com/lesson1"}
            ],
            "session_id": "test_session_123"
        },
        "expected_stats": {
            "total_courses": 3,
            "course_titles": ["Python Basics", "Web Development", "Data Science"]
        }
    }


@pytest.fixture
def error_scenarios():
    """Error scenarios for testing exception handling"""
    return {
        "rag_system_error": Exception("RAG system processing failed"),
        "invalid_query": {"query": ""},
        "malformed_request": {"invalid_field": "test"},
        "session_error": Exception("Session management failed"),
        "search_error": Exception("Search service unavailable")
    }