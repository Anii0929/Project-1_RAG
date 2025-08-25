"""
Pytest configuration and shared fixtures for RAG system testing.
"""

import json
import os
import shutil

# Add backend to path for imports
import sys
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults


@pytest.fixture
def sample_course():
    """Create a sample course with lessons for testing."""
    lessons = [
        Lesson(
            lesson_number=0,
            title="Introduction to MCP",
            lesson_link="https://example.com/lesson0",
        ),
        Lesson(
            lesson_number=1,
            title="Building Your First MCP App",
            lesson_link="https://example.com/lesson1",
        ),
        Lesson(lesson_number=2, title="Advanced MCP Features", lesson_link=None),
    ]

    return Course(
        title="MCP: Build Rich-Context AI Apps with Anthropic",
        instructor="Anthropic Team",
        course_link="https://example.com/mcp-course",
        lessons=lessons,
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing."""
    return [
        CourseChunk(
            content="MCP (Model Context Protocol) is a revolutionary approach to building AI applications that can seamlessly integrate with external data sources and tools.",
            course_title="MCP: Build Rich-Context AI Apps with Anthropic",
            lesson_number=0,
            chunk_index=0,
        ),
        CourseChunk(
            content="In this lesson, we'll explore the fundamental concepts of MCP and how it enables AI agents to access real-time information from various APIs and databases.",
            course_title="MCP: Build Rich-Context AI Apps with Anthropic",
            lesson_number=0,
            chunk_index=1,
        ),
        CourseChunk(
            content="Building your first MCP application requires understanding the client-server architecture and how to implement proper data connectors.",
            course_title="MCP: Build Rich-Context AI Apps with Anthropic",
            lesson_number=1,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore with predictable responses."""
    mock_store = Mock()

    # Mock successful search results
    successful_results = SearchResults(
        documents=[
            "MCP enables AI apps to access external data",
            "Building MCP requires understanding protocols",
        ],
        metadata=[
            {
                "course_title": "MCP: Build Rich-Context AI Apps with Anthropic",
                "lesson_number": 0,
            },
            {
                "course_title": "MCP: Build Rich-Context AI Apps with Anthropic",
                "lesson_number": 1,
            },
        ],
        distances=[0.1, 0.2],
    )

    # Mock empty results
    empty_results = SearchResults(documents=[], metadata=[], distances=[])

    # Mock error results
    error_results = SearchResults.empty("Search error: Database connection failed")

    mock_store.search.return_value = successful_results
    mock_store._resolve_course_name.return_value = (
        "MCP: Build Rich-Context AI Apps with Anthropic"
    )
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

    # Store different result types for specific test scenarios
    mock_store._successful_results = successful_results
    mock_store._empty_results = empty_results
    mock_store._error_results = error_results

    return mock_store


@pytest.fixture
def mock_chroma_collection():
    """Create a mock ChromaDB collection."""
    mock_collection = Mock()

    # Mock query results
    mock_collection.query.return_value = {
        "documents": [["Sample document 1", "Sample document 2"]],
        "metadatas": [
            [
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 2},
            ]
        ],
        "distances": [[0.1, 0.2]],
    }

    # Mock get results for course catalog
    mock_collection.get.return_value = {
        "ids": ["Test Course"],
        "metadatas": [
            {
                "title": "Test Course",
                "instructor": "Test Instructor",
                "course_link": "https://example.com/course",
                "lessons_json": json.dumps(
                    [
                        {
                            "lesson_number": 1,
                            "lesson_title": "Lesson 1",
                            "lesson_link": "https://example.com/lesson1",
                        }
                    ]
                ),
            }
        ],
    }

    return mock_collection


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing AI generator."""
    mock_client = Mock()

    # Mock successful response without tools
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock()]
    mock_response.content[0].text = "This is a test response from Claude."

    # Mock tool use response
    mock_tool_response = Mock()
    mock_tool_response.stop_reason = "tool_use"
    mock_tool_content = Mock()
    mock_tool_content.type = "tool_use"
    mock_tool_content.name = "search_course_content"
    mock_tool_content.id = "tool_123"
    mock_tool_content.input = {"query": "test query"}
    mock_tool_response.content = [mock_tool_content]

    mock_client.messages.create.return_value = mock_response

    # Store both response types for different test scenarios
    mock_client._normal_response = mock_response
    mock_client._tool_response = mock_tool_response

    return mock_client


@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB instance for integration tests."""
    temp_dir = tempfile.mkdtemp()

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_tool_manager():
    """Create a mock ToolManager with tools registered."""
    from search_tools import ToolManager

    mock_manager = Mock(spec=ToolManager)
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course content",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    mock_manager.execute_tool.return_value = "Mock search results"
    mock_manager.get_last_sources.return_value = []
    mock_manager.reset_sources.return_value = None

    return mock_manager


# Test utilities


def create_test_search_results(
    documents: List[str] = None, metadata: List[Dict] = None, error: str = None
) -> SearchResults:
    """Helper function to create SearchResults for testing."""
    if error:
        return SearchResults.empty(error)

    documents = documents or ["Test document"]
    metadata = metadata or [{"course_title": "Test Course", "lesson_number": 1}]
    distances = [0.1] * len(documents)

    return SearchResults(documents=documents, metadata=metadata, distances=distances)


def assert_search_results_equal(result1: SearchResults, result2: SearchResults):
    """Helper function to assert SearchResults are equal."""
    assert result1.documents == result2.documents
    assert result1.metadata == result2.metadata
    assert result1.distances == result2.distances
    assert result1.error == result2.error
