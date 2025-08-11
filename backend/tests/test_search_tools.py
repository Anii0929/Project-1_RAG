from unittest.mock import Mock, patch

import pytest
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool execute method"""

    def test_execute_successful_search(self, course_search_tool):
        """Test successful search execution"""
        result = course_search_tool.execute("testing")

        # Should not be empty and should contain formatted results
        assert result != ""
        assert isinstance(result, str)
        assert "Test Course" in result

    def test_execute_with_course_filter(self, course_search_tool):
        """Test search with course name filter"""
        result = course_search_tool.execute("testing", course_name="Test Course")

        assert result != ""
        assert "Test Course" in result

    def test_execute_with_lesson_filter(self, course_search_tool):
        """Test search with lesson number filter"""
        result = course_search_tool.execute("testing", lesson_number=1)

        assert result != ""
        assert "Lesson 1" in result

    def test_execute_with_both_filters(self, course_search_tool):
        """Test search with both course and lesson filters"""
        result = course_search_tool.execute(
            "testing", course_name="Test Course", lesson_number=1
        )

        assert result != ""
        assert "Test Course" in result
        assert "Lesson 1" in result

    def test_execute_no_results_found(self, course_search_tool):
        """Test when search returns no results"""
        result = course_search_tool.execute("nonexistent_topic")

        # Should return a "not found" message
        assert "No relevant content found" in result

    def test_execute_course_not_found(self, course_search_tool):
        """Test when specified course doesn't exist"""
        result = course_search_tool.execute("testing", course_name="Nonexistent Course")

        # Should return course not found message
        assert "No course found matching" in result

    def test_execute_stores_sources(self, course_search_tool):
        """Test that execute stores sources for later retrieval"""
        # Clear any existing sources
        course_search_tool.last_sources = []

        result = course_search_tool.execute("testing")

        # Should have stored sources
        assert len(course_search_tool.last_sources) > 0

        # Sources should have proper structure
        for source in course_search_tool.last_sources:
            assert isinstance(source, dict)
            assert "text" in source
            assert "link" in source

    def test_execute_with_zero_max_results(self, vector_store):
        """Test behavior when max_results is 0 (current config issue)"""
        # Create tool with vector store that has 0 max_results
        vector_store.max_results = 0
        tool = CourseSearchTool(vector_store)

        result = tool.execute("testing")

        # With 0 max_results, should return no results
        assert "No relevant content found" in result or result == ""

    def test_format_results_with_links(self, course_search_tool):
        """Test that _format_results properly formats with lesson links"""
        # Create mock search results
        mock_results = SearchResults(
            documents=["Test content 1", "Test content 2"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
        )

        formatted = course_search_tool._format_results(mock_results)

        assert "[Test Course - Lesson 1]" in formatted
        assert "[Test Course - Lesson 2]" in formatted
        assert "Test content 1" in formatted
        assert "Test content 2" in formatted

    def test_get_tool_definition(self, course_search_tool):
        """Test tool definition structure"""
        definition = course_search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert "properties" in definition["input_schema"]
        assert "query" in definition["input_schema"]["properties"]
        assert "required" in definition["input_schema"]
        assert "query" in definition["input_schema"]["required"]


class TestToolManager:
    """Test suite for ToolManager"""

    def test_register_tool(self, course_search_tool):
        """Test tool registration"""
        tm = ToolManager()
        tm.register_tool(course_search_tool)

        assert "search_course_content" in tm.tools

    def test_get_tool_definitions(self, tool_manager):
        """Test getting all tool definitions"""
        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) > 0
        assert isinstance(definitions, list)

        # Should contain search tool definition
        search_def = next(
            (d for d in definitions if d["name"] == "search_course_content"), None
        )
        assert search_def is not None

    def test_execute_tool(self, tool_manager):
        """Test executing tool through manager"""
        result = tool_manager.execute_tool("search_course_content", query="testing")

        assert isinstance(result, str)
        assert result != ""

    def test_execute_nonexistent_tool(self, tool_manager):
        """Test executing tool that doesn't exist"""
        result = tool_manager.execute_tool("nonexistent_tool", query="testing")

        assert "not found" in result

    def test_get_last_sources(self, tool_manager):
        """Test retrieving sources from last search"""
        # Execute a search to generate sources
        tool_manager.execute_tool("search_course_content", query="testing")

        sources = tool_manager.get_last_sources()

        assert isinstance(sources, list)
        # Should have sources if search was successful
        if sources:  # Only test structure if sources exist
            for source in sources:
                assert isinstance(source, dict)
                assert "text" in source

    def test_reset_sources(self, tool_manager):
        """Test resetting sources"""
        # Execute a search to generate sources
        tool_manager.execute_tool("search_course_content", query="testing")

        # Reset sources
        tool_manager.reset_sources()

        sources = tool_manager.get_last_sources()
        assert len(sources) == 0


class TestSearchToolsIntegration:
    """Integration tests for search tools"""

    def test_end_to_end_search_flow(self, tool_manager):
        """Test complete search flow from tool manager to results"""
        # Execute search
        result = tool_manager.execute_tool(
            "search_course_content", query="testing", course_name="Test Course"
        )

        # Verify result
        assert isinstance(result, str)
        assert result != ""

        # Verify sources were captured
        sources = tool_manager.get_last_sources()
        assert isinstance(sources, list)

        # If sources exist, verify structure
        if sources:
            for source in sources:
                assert "text" in source
                assert "link" in source
