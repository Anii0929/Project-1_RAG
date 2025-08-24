"""
Tests for CourseSearchTool functionality.

This module tests the CourseSearchTool's execute method and result formatting
with various scenarios including successful searches, empty results, and error conditions.
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool functionality."""
    
    @pytest.mark.unit
    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly formatted."""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
    
    @pytest.mark.unit
    def test_execute_successful_search(self, mock_vector_store):
        """Test successful search execution with results."""
        tool = CourseSearchTool(mock_vector_store)
        
        # Configure mock to return successful results
        mock_vector_store.search.return_value = mock_vector_store._successful_results
        
        result = tool.execute("test query")
        
        assert "MCP enables AI apps" in result
        assert "[MCP: Build Rich-Context AI Apps with Anthropic - Lesson 0]" in result
        assert len(tool.last_sources) == 2
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )
    
    @pytest.mark.unit
    def test_execute_with_course_name(self, mock_vector_store):
        """Test search execution with course name filter."""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_vector_store._successful_results
        
        result = tool.execute("test query", course_name="MCP")
        
        assert "MCP enables AI apps" in result
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="MCP",
            lesson_number=None
        )
    
    @pytest.mark.unit
    def test_execute_with_lesson_number(self, mock_vector_store):
        """Test search execution with lesson number filter."""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_vector_store._successful_results
        
        result = tool.execute("test query", lesson_number=1)
        
        assert "MCP enables AI apps" in result
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=1
        )
    
    @pytest.mark.unit
    def test_execute_with_all_parameters(self, mock_vector_store):
        """Test search execution with all parameters."""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_vector_store._successful_results
        
        result = tool.execute("test query", course_name="MCP", lesson_number=1)
        
        assert "MCP enables AI apps" in result
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="MCP",
            lesson_number=1
        )
    
    @pytest.mark.unit
    def test_execute_empty_results(self, mock_vector_store):
        """Test handling of empty search results."""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_vector_store._empty_results
        
        result = tool.execute("nonexistent query")
        
        assert result == "No relevant content found."
        assert len(tool.last_sources) == 0
    
    @pytest.mark.unit
    def test_execute_empty_results_with_course_filter(self, mock_vector_store):
        """Test empty results with course name filter includes filter info."""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_vector_store._empty_results
        
        result = tool.execute("test query", course_name="NonexistentCourse")
        
        assert "No relevant content found in course 'NonexistentCourse'." == result
    
    @pytest.mark.unit
    def test_execute_empty_results_with_lesson_filter(self, mock_vector_store):
        """Test empty results with lesson number filter includes filter info."""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_vector_store._empty_results
        
        result = tool.execute("test query", lesson_number=99)
        
        assert "No relevant content found in lesson 99." == result
    
    @pytest.mark.unit
    def test_execute_empty_results_with_both_filters(self, mock_vector_store):
        """Test empty results with both filters includes both in message."""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_vector_store._empty_results
        
        result = tool.execute("test query", course_name="TestCourse", lesson_number=5)
        
        assert "No relevant content found in course 'TestCourse' in lesson 5." == result
    
    @pytest.mark.unit
    def test_execute_error_handling(self, mock_vector_store):
        """Test handling of search errors."""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_vector_store._error_results
        
        result = tool.execute("test query")
        
        assert result == "Search error: Database connection failed"
        assert len(tool.last_sources) == 0
    
    @pytest.mark.unit
    def test_format_results_with_lesson_links(self, mock_vector_store):
        """Test result formatting includes lesson links when available."""
        tool = CourseSearchTool(mock_vector_store)
        
        # Mock get_lesson_link to return a link
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        search_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        
        formatted = tool._format_results(search_results)
        
        assert "[Test Course - Lesson 1]" in formatted
        assert "Test content" in formatted
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course - Lesson 1"
        assert tool.last_sources[0]["link"] == "https://example.com/lesson1"
    
    @pytest.mark.unit
    def test_format_results_without_lesson_number(self, mock_vector_store):
        """Test result formatting when lesson number is not available."""
        tool = CourseSearchTool(mock_vector_store)
        
        search_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course"}],  # No lesson_number
            distances=[0.1]
        )
        
        formatted = tool._format_results(search_results)
        
        assert "[Test Course]" in formatted
        assert "Test content" in formatted
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course"
        assert tool.last_sources[0]["link"] is None
    
    @pytest.mark.unit 
    def test_format_results_unknown_course(self, mock_vector_store):
        """Test result formatting handles unknown course title."""
        tool = CourseSearchTool(mock_vector_store)
        
        search_results = SearchResults(
            documents=["Test content"],
            metadata=[{}],  # Empty metadata
            distances=[0.1]
        )
        
        formatted = tool._format_results(search_results)
        
        assert "[unknown]" in formatted
        assert "Test content" in formatted


class TestCourseSearchToolIntegration:
    """Integration tests for CourseSearchTool with ToolManager."""
    
    @pytest.mark.integration
    def test_tool_registration(self, mock_vector_store):
        """Test that CourseSearchTool can be registered with ToolManager."""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        
        tool_manager.register_tool(search_tool)
        
        definitions = tool_manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
    
    @pytest.mark.integration
    def test_tool_execution_via_manager(self, mock_vector_store):
        """Test tool execution through ToolManager."""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        mock_vector_store.search.return_value = mock_vector_store._successful_results
        
        result = tool_manager.execute_tool(
            "search_course_content",
            query="test query",
            course_name="MCP"
        )
        
        assert "MCP enables AI apps" in result
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="MCP",
            lesson_number=None
        )
    
    @pytest.mark.integration
    def test_get_last_sources_via_manager(self, mock_vector_store):
        """Test source retrieval through ToolManager."""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        mock_vector_store.search.return_value = mock_vector_store._successful_results
        
        # Execute search
        tool_manager.execute_tool("search_course_content", query="test")
        
        # Get sources
        sources = tool_manager.get_last_sources()
        assert len(sources) == 2
        assert sources[0]["text"] == "MCP: Build Rich-Context AI Apps with Anthropic - Lesson 0"
    
    @pytest.mark.integration
    def test_reset_sources_via_manager(self, mock_vector_store):
        """Test source reset through ToolManager."""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        mock_vector_store.search.return_value = mock_vector_store._successful_results
        
        # Execute search and get sources
        tool_manager.execute_tool("search_course_content", query="test")
        sources_before = tool_manager.get_last_sources()
        assert len(sources_before) > 0
        
        # Reset sources
        tool_manager.reset_sources()
        sources_after = tool_manager.get_last_sources()
        assert len(sources_after) == 0


class TestCourseSearchToolEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.unit
    def test_execute_with_none_query(self, mock_vector_store):
        """Test behavior with None query (should not crash)."""
        tool = CourseSearchTool(mock_vector_store)
        
        # This might raise an exception or handle gracefully
        # depending on implementation
        try:
            result = tool.execute(None)
            assert isinstance(result, str)  # Should return some error message
        except (TypeError, AttributeError):
            # Expected if not handling None gracefully
            pass
    
    @pytest.mark.unit
    def test_execute_with_empty_query(self, mock_vector_store):
        """Test behavior with empty query string."""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_vector_store._empty_results
        
        result = tool.execute("")
        
        assert isinstance(result, str)
        mock_vector_store.search.assert_called_once_with(
            query="",
            course_name=None,
            lesson_number=None
        )
    
    @pytest.mark.unit
    def test_execute_with_negative_lesson_number(self, mock_vector_store):
        """Test behavior with negative lesson number."""
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = mock_vector_store._empty_results
        
        result = tool.execute("test", lesson_number=-1)
        
        assert isinstance(result, str)
        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name=None,
            lesson_number=-1
        )
    
    @pytest.mark.unit
    def test_format_results_with_mismatched_arrays(self, mock_vector_store):
        """Test format results handles mismatched document/metadata arrays."""
        tool = CourseSearchTool(mock_vector_store)
        
        # More documents than metadata
        search_results = SearchResults(
            documents=["Doc 1", "Doc 2"],
            metadata=[{"course_title": "Course 1"}],  # Only one metadata item
            distances=[0.1, 0.2]
        )
        
        # Should not crash, might skip extra documents or handle gracefully
        formatted = tool._format_results(search_results)
        assert isinstance(formatted, str)