import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from config import Config

@pytest.fixture
def mock_config():
    config = Config()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config

@pytest.fixture
def sample_courses():
    courses = [
        Course(
            title="Introduction to MCP",
            course_link="https://example.com/mcp",
            instructor="Dr. Smith",
            lessons=[
                Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/mcp/lesson1"),
                Lesson(lesson_number=2, title="Core Concepts", lesson_link="https://example.com/mcp/lesson2"),
                Lesson(lesson_number=3, title="Advanced Topics", lesson_link="https://example.com/mcp/lesson3"),
            ]
        ),
        Course(
            title="Advanced AI Systems",
            course_link="https://example.com/ai",
            instructor="Prof. Johnson",
            lessons=[
                Lesson(lesson_number=1, title="Neural Networks", lesson_link="https://example.com/ai/lesson1"),
                Lesson(lesson_number=2, title="Deep Learning", lesson_link="https://example.com/ai/lesson2"),
            ]
        )
    ]
    return courses

@pytest.fixture
def sample_course_chunks():
    chunks = [
        CourseChunk(
            content="MCP is a powerful framework for building modern applications.",
            course_title="Introduction to MCP",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="The core concepts of MCP include components, services, and pipelines.",
            course_title="Introduction to MCP",
            lesson_number=2,
            chunk_index=0
        ),
        CourseChunk(
            content="Neural networks are the foundation of deep learning systems.",
            course_title="Advanced AI Systems",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Deep learning enables complex pattern recognition tasks.",
            course_title="Advanced AI Systems",
            lesson_number=2,
            chunk_index=0
        ),
    ]
    return chunks

@pytest.fixture
def sample_search_results():
    return SearchResults(
        documents=[
            "MCP is a powerful framework for building modern applications.",
            "The core concepts of MCP include components, services, and pipelines."
        ],
        metadata=[
            {"course_title": "Introduction to MCP", "lesson_number": 1},
            {"course_title": "Introduction to MCP", "lesson_number": 2}
        ],
        distances=[0.1, 0.2]
    )

@pytest.fixture
def empty_search_results():
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )

@pytest.fixture
def mock_vector_store():
    mock_store = Mock()
    
    mock_store.search.return_value = SearchResults(
        documents=["Sample content about MCP framework"],
        metadata=[{"course_title": "Introduction to MCP", "lesson_number": 1}],
        distances=[0.1]
    )
    
    mock_store.get_lesson_link.return_value = "https://example.com/mcp/lesson1"
    
    mock_store.get_course_outline.return_value = {
        "course_title": "Introduction to MCP",
        "instructor": "Dr. Smith",
        "course_link": "https://example.com/mcp",
        "lesson_count": 3,
        "lessons": [
            {"lesson_number": 1, "lesson_title": "Getting Started"},
            {"lesson_number": 2, "lesson_title": "Core Concepts"},
            {"lesson_number": 3, "lesson_title": "Advanced Topics"}
        ]
    }
    
    mock_store.get_course_count.return_value = 2
    mock_store.get_existing_course_titles.return_value = ["Introduction to MCP", "Advanced AI Systems"]
    mock_store.add_course_metadata = Mock()
    mock_store.add_course_content = Mock()
    
    return mock_store

@pytest.fixture
def mock_anthropic_client():
    mock_client = Mock()
    
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock(text="This is a test response from Claude.")]
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client

@pytest.fixture
def mock_anthropic_client_with_tool_use():
    mock_client = Mock()
    
    initial_response = Mock()
    initial_response.stop_reason = "tool_use"
    
    tool_use_block = Mock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "search_course_content"
    tool_use_block.id = "tool_123"
    tool_use_block.input = {"query": "MCP framework", "course_name": "Introduction to MCP"}
    
    initial_response.content = [tool_use_block]
    
    final_response = Mock()
    final_response.stop_reason = "end_turn"
    final_response.content = [Mock(text="Based on the search, MCP is a powerful framework.")]
    
    mock_client.messages.create.side_effect = [initial_response, final_response]
    
    return mock_client

@pytest.fixture
def mock_document_processor():
    mock_processor = Mock()
    
    course = Course(
        title="Test Course",
        lessons=[Lesson(lesson_number=1, title="Test Lesson")]
    )
    
    chunks = [
        CourseChunk(
            content="Test content",
            course_title="Test Course",
            lesson_number=1,
            chunk_index=0
        )
    ]
    
    mock_processor.process_course_document.return_value = (course, chunks)
    
    return mock_processor

@pytest.fixture
def mock_session_manager():
    mock_manager = Mock()
    mock_manager.get_conversation_history.return_value = "Previous conversation context"
    mock_manager.add_exchange = Mock()
    return mock_manager

@pytest.fixture
def mock_tool_manager():
    mock_manager = Mock()
    
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"},
                    "lesson_number": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    ]
    
    mock_manager.execute_tool.return_value = "Search results: MCP is a framework"
    mock_manager.get_last_sources.return_value = [
        {"text": "Introduction to MCP - Lesson 1", "url": "https://example.com/mcp/lesson1"}
    ]
    mock_manager.reset_sources = Mock()
    
    return mock_manager

@pytest.fixture
def mock_chroma_collection():
    mock_collection = Mock()
    
    mock_collection.query.return_value = {
        'documents': [["Sample document content"]],
        'metadatas': [[{"course_title": "Test Course", "lesson_number": 1}]],
        'distances': [[0.1]]
    }
    
    mock_collection.add = Mock()
    mock_collection.get = Mock(return_value={'ids': []})
    mock_collection.delete = Mock()
    
    return mock_collection

@pytest.fixture
def mock_chroma_client(mock_chroma_collection):
    mock_client = Mock()
    mock_client.get_or_create_collection.return_value = mock_chroma_collection
    return mock_client