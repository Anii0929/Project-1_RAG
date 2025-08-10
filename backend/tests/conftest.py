import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
from config import Config
from vector_store import VectorStore
from document_processor import DocumentProcessor
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
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