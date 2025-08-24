"""
Tests for CourseListTool functionality.

This module tests the new CourseListTool that addresses the "query failed" issue
for general course listing queries.
"""

import pytest
from unittest.mock import Mock
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from search_tools import CourseListTool, ToolManager


class TestCourseListTool:
    """Test suite for CourseListTool functionality."""
    
    @pytest.mark.unit
    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly formatted."""
        tool = CourseListTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "list_all_courses"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == []
        assert len(definition["input_schema"]["properties"]) == 0
    
    @pytest.mark.unit
    def test_execute_empty_courses(self, mock_vector_store):
        """Test execution with no courses available."""
        tool = CourseListTool(mock_vector_store)
        
        # Mock empty course list
        mock_vector_store.get_existing_course_titles.return_value = []
        
        result = tool.execute()
        
        assert result == "No courses are currently available in the system."
        assert len(tool.last_sources) == 0
    
    @pytest.mark.unit
    def test_execute_with_courses(self, mock_vector_store):
        """Test execution with available courses."""
        tool = CourseListTool(mock_vector_store)
        
        # Mock course data
        mock_vector_store.get_existing_course_titles.return_value = [
            "MCP Course", "Chroma Course"
        ]
        mock_vector_store.get_all_courses_metadata.return_value = [
            {
                'title': 'MCP Course',
                'instructor': 'Anthropic',
                'lesson_count': 5,
                'course_link': 'https://example.com/mcp'
            },
            {
                'title': 'Chroma Course',
                'instructor': 'ChromaDB Team',
                'lesson_count': 3,
                'course_link': None
            }
        ]
        
        result = tool.execute()
        
        assert "Available courses (2):" in result
        assert "**MCP Course**" in result
        assert "by Anthropic" in result
        assert "(5 lessons)" in result
        assert "[Course Link]" in result
        assert "**Chroma Course**" in result
        assert "by ChromaDB Team" in result
        assert "(3 lessons)" in result
        
        # Check sources
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Course Catalog (2 courses)"
        assert tool.last_sources[0]["link"] is None
    
    @pytest.mark.unit
    def test_execute_with_minimal_metadata(self, mock_vector_store):
        """Test execution with courses that have minimal metadata."""
        tool = CourseListTool(mock_vector_store)
        
        mock_vector_store.get_existing_course_titles.return_value = ["Test Course"]
        mock_vector_store.get_all_courses_metadata.return_value = [
            {
                'title': 'Test Course'
                # Missing instructor, lesson_count, course_link
            }
        ]
        
        result = tool.execute()
        
        assert "Available courses (1):" in result
        assert "**Test Course**" in result
        # Should not show instructor, lesson count, or link if not available
        assert "by Unknown Instructor" not in result
        assert "(0 lessons)" not in result
        assert "[Course Link]" not in result
    
    @pytest.mark.unit
    def test_execute_error_handling(self, mock_vector_store):
        """Test execution handles errors gracefully."""
        tool = CourseListTool(mock_vector_store)
        
        # Mock exception in course titles retrieval
        mock_vector_store.get_existing_course_titles.side_effect = Exception("Database error")
        
        result = tool.execute()
        
        assert "Error retrieving course list: Database error" in result
    
    @pytest.mark.integration
    def test_tool_registration_with_manager(self, mock_vector_store):
        """Test that CourseListTool can be registered with ToolManager."""
        tool_manager = ToolManager()
        list_tool = CourseListTool(mock_vector_store)
        
        tool_manager.register_tool(list_tool)
        
        definitions = tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in definitions]
        assert "list_all_courses" in tool_names
    
    @pytest.mark.integration
    def test_tool_execution_via_manager(self, mock_vector_store):
        """Test tool execution through ToolManager."""
        tool_manager = ToolManager()
        list_tool = CourseListTool(mock_vector_store)
        tool_manager.register_tool(list_tool)
        
        # Mock data
        mock_vector_store.get_existing_course_titles.return_value = ["Test Course"]
        mock_vector_store.get_all_courses_metadata.return_value = [
            {'title': 'Test Course', 'instructor': 'Test Instructor', 'lesson_count': 1}
        ]
        
        result = tool_manager.execute_tool("list_all_courses")
        
        assert "Available courses (1):" in result
        assert "**Test Course**" in result
    
    @pytest.mark.integration
    def test_sources_retrieval_via_manager(self, mock_vector_store):
        """Test source retrieval through ToolManager."""
        tool_manager = ToolManager()
        list_tool = CourseListTool(mock_vector_store)
        tool_manager.register_tool(list_tool)
        
        mock_vector_store.get_existing_course_titles.return_value = ["Course A", "Course B"]
        mock_vector_store.get_all_courses_metadata.return_value = [
            {'title': 'Course A'},
            {'title': 'Course B'}
        ]
        
        # Execute tool
        tool_manager.execute_tool("list_all_courses")
        
        # Get sources
        sources = tool_manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["text"] == "Course Catalog (2 courses)"