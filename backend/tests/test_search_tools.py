import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults, VectorStore


class TestCourseSearchTool:
    """Test suite for CourseSearchTool"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing"""
        mock_store = Mock(spec=VectorStore)
        return mock_store

    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create a CourseSearchTool instance with mock vector store"""
        return CourseSearchTool(mock_vector_store)

    def test_tool_definition(self, search_tool):
        """Test that tool definition is properly structured"""
        definition = search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]

    def test_execute_successful_search(self, search_tool, mock_vector_store):
        """Test successful search execution"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["Sample course content about Python"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        result = search_tool.execute("Python basics")

        assert isinstance(result, str)
        assert "Python Basics" in result
        assert "Lesson 1" in result
        assert "Sample course content" in result
        mock_vector_store.search.assert_called_once_with(
            query="Python basics", course_name=None, lesson_number=None
        )

    def test_execute_with_course_filter(self, search_tool, mock_vector_store):
        """Test search execution with course name filter"""
        mock_results = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 2}],
            distances=[0.2],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = None

        result = search_tool.execute("variables", course_name="Python", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="variables", course_name="Python", lesson_number=2
        )
        assert "Advanced Python" in result

    def test_execute_with_error(self, search_tool, mock_vector_store):
        """Test handling of search errors"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Search error: Database connection failed",
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute("test query")

        assert result == "Search error: Database connection failed"

    def test_execute_empty_results(self, search_tool, mock_vector_store):
        """Test handling of empty search results"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute("nonexistent content")

        assert "No relevant content found" in result

    def test_execute_empty_results_with_filters(self, search_tool, mock_vector_store):
        """Test empty results with course and lesson filters"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute("test", course_name="Test Course", lesson_number=1)

        assert "No relevant content found in course 'Test Course' in lesson 1" in result

    def test_format_results_multiple_sources(self, search_tool, mock_vector_store):
        """Test formatting of multiple search results"""
        mock_results = SearchResults(
            documents=["First document content", "Second document content"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson1",
            "https://example.com/lesson2",
        ]

        result = search_tool.execute("test query")

        assert "[Course A - Lesson 1]" in result
        assert "[Course B - Lesson 2]" in result
        assert "First document content" in result
        assert "Second document content" in result

        # Check that sources are tracked
        assert len(search_tool.last_sources) == 2
        assert search_tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert search_tool.last_sources[0]["link"] == "https://example.com/lesson1"

    def test_source_tracking_reset(self, search_tool, mock_vector_store):
        """Test that sources are properly reset between searches"""
        # First search
        mock_results_1 = SearchResults(
            documents=["Content 1"],
            metadata=[{"course_title": "Course 1", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results_1
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        search_tool.execute("query 1")
        assert len(search_tool.last_sources) == 1

        # Second search with different results
        mock_results_2 = SearchResults(
            documents=["Content 2", "Content 3"],
            metadata=[
                {"course_title": "Course 2", "lesson_number": 1},
                {"course_title": "Course 2", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results_2

        search_tool.execute("query 2")

        # Sources should be updated, not appended
        assert len(search_tool.last_sources) == 2
        assert search_tool.last_sources[0]["text"] == "Course 2 - Lesson 1"


class TestCourseOutlineTool:
    """Test suite for CourseOutlineTool"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing"""
        mock_store = Mock(spec=VectorStore)
        return mock_store

    @pytest.fixture
    def outline_tool(self, mock_vector_store):
        """Create a CourseOutlineTool instance with mock vector store"""
        return CourseOutlineTool(mock_vector_store)

    def test_tool_definition(self, outline_tool):
        """Test that tool definition is properly structured"""
        definition = outline_tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["course_name"]

    def test_execute_successful_outline(self, outline_tool, mock_vector_store):
        """Test successful course outline retrieval"""
        # Mock course name resolution
        mock_vector_store._resolve_course_name.return_value = "Python Programming"

        # Mock course catalog data
        mock_course_catalog = Mock()
        mock_vector_store.course_catalog = mock_course_catalog
        mock_course_catalog.get.return_value = {
            "metadatas": [
                {
                    "title": "Python Programming",
                    "instructor": "John Doe",
                    "course_link": "https://example.com/python",
                    "lessons_json": '[{"lesson_number": 1, "lesson_title": "Introduction"}, {"lesson_number": 2, "lesson_title": "Variables"}]',
                }
            ]
        }

        result = outline_tool.execute("Python")

        assert "Course: Python Programming" in result
        assert "Instructor: John Doe" in result
        assert "Course Link: https://example.com/python" in result
        assert "1. Introduction" in result
        assert "2. Variables" in result

    def test_execute_course_not_found(self, outline_tool, mock_vector_store):
        """Test handling when course is not found"""
        mock_vector_store._resolve_course_name.return_value = None

        result = outline_tool.execute("Nonexistent Course")

        assert "No course found matching 'Nonexistent Course'" in result

    def test_execute_metadata_not_found(self, outline_tool, mock_vector_store):
        """Test handling when course metadata is not found"""
        mock_vector_store._resolve_course_name.return_value = "Found Course"

        mock_course_catalog = Mock()
        mock_vector_store.course_catalog = mock_course_catalog
        mock_course_catalog.get.return_value = None

        result = outline_tool.execute("Found Course")

        assert "Course metadata not found for 'Found Course'" in result


class TestToolManager:
    """Test suite for ToolManager"""

    @pytest.fixture
    def tool_manager(self):
        """Create a ToolManager instance"""
        return ToolManager()

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool for testing"""
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "A test tool",
        }
        mock_tool.execute.return_value = "Test result"
        return mock_tool

    def test_register_tool(self, tool_manager, mock_tool):
        """Test tool registration"""
        tool_manager.register_tool(mock_tool)

        assert "test_tool" in tool_manager.tools
        assert tool_manager.tools["test_tool"] is mock_tool

    def test_get_tool_definitions(self, tool_manager, mock_tool):
        """Test getting all tool definitions"""
        tool_manager.register_tool(mock_tool)

        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "test_tool"

    def test_execute_tool(self, tool_manager, mock_tool):
        """Test tool execution"""
        tool_manager.register_tool(mock_tool)

        result = tool_manager.execute_tool("test_tool", param1="value1")

        assert result == "Test result"
        mock_tool.execute.assert_called_once_with(param1="value1")

    def test_execute_nonexistent_tool(self, tool_manager):
        """Test execution of non-existent tool"""
        result = tool_manager.execute_tool("nonexistent_tool")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_source_tracking(self, tool_manager):
        """Test source tracking functionality"""
        # Create mock tool with last_sources
        mock_tool_with_sources = Mock()
        mock_tool_with_sources.last_sources = [{"text": "Source 1"}]
        mock_tool_with_sources.get_tool_definition.return_value = {
            "name": "source_tool"
        }

        tool_manager.register_tool(mock_tool_with_sources)

        sources = tool_manager.get_last_sources()
        assert sources == [{"text": "Source 1"}]

        # Test reset
        tool_manager.reset_sources()
        assert mock_tool_with_sources.last_sources == []


# Integration test to verify tools work with real-like data
class TestIntegration:
    """Integration tests with more realistic scenarios"""

    def test_search_tool_with_realistic_data(self):
        """Test search tool with realistic vector store responses"""
        # This test would need to be run against actual data
        # For now, we'll create a comprehensive mock scenario

        mock_store = Mock()
        search_tool = CourseSearchTool(mock_store)

        # Simulate a realistic search result
        realistic_results = SearchResults(
            documents=[
                "Python variables are containers for storing data values. Unlike other programming languages, Python has no command for declaring a variable.",
                "In Python, you can assign values to variables using the assignment operator (=). The variable name goes on the left side of the operator.",
            ],
            metadata=[
                {"course_title": "Python Programming Fundamentals", "lesson_number": 1},
                {"course_title": "Python Programming Fundamentals", "lesson_number": 2},
            ],
            distances=[0.15, 0.18],
            error=None,
        )

        mock_store.search.return_value = realistic_results
        mock_store.get_lesson_link.side_effect = [
            "https://learn.example.com/python/lesson1",
            "https://learn.example.com/python/lesson2",
        ]

        result = search_tool.execute("How do I create variables in Python?")

        # Verify the result contains expected content
        assert "Python Programming Fundamentals" in result
        assert "variables are containers" in result
        assert "assignment operator" in result
        assert "[Python Programming Fundamentals - Lesson 1]" in result
        assert "[Python Programming Fundamentals - Lesson 2]" in result

        # Verify sources are properly tracked
        assert len(search_tool.last_sources) == 2
        assert (
            search_tool.last_sources[0]["link"]
            == "https://learn.example.com/python/lesson1"
        )


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
