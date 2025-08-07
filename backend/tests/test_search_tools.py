import pytest
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults

class TestCourseSearchTool:
    
    def test_get_tool_definition(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["properties"]["query"]["type"] == "string"
        assert definition["input_schema"]["properties"]["course_name"]["type"] == "string"
        assert definition["input_schema"]["properties"]["lesson_number"]["type"] == "integer"
        assert definition["input_schema"]["required"] == ["query"]
    
    def test_execute_with_query_only(self, mock_vector_store, sample_search_results):
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/mcp/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="MCP framework")
        
        mock_vector_store.search.assert_called_once_with(
            query="MCP framework",
            course_name=None,
            lesson_number=None
        )
        
        assert "[Introduction to MCP - Lesson 1]" in result
        assert "MCP is a powerful framework" in result
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Introduction to MCP - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com/mcp/lesson1"
    
    def test_execute_with_course_filter(self, mock_vector_store, sample_search_results):
        mock_vector_store.search.return_value = sample_search_results
        mock_vector_store.get_lesson_link.return_value = None
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="components", course_name="Introduction to MCP")
        
        mock_vector_store.search.assert_called_once_with(
            query="components",
            course_name="Introduction to MCP",
            lesson_number=None
        )
        
        assert "[Introduction to MCP - Lesson 1]" in result
        assert "[Introduction to MCP - Lesson 2]" in result
        assert tool.last_sources[0]["url"] is None
    
    def test_execute_with_lesson_filter(self, mock_vector_store):
        filtered_results = SearchResults(
            documents=["Content from lesson 2"],
            metadata=[{"course_title": "Introduction to MCP", "lesson_number": 2}],
            distances=[0.15]
        )
        mock_vector_store.search.return_value = filtered_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/mcp/lesson2"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="concepts", lesson_number=2)
        
        mock_vector_store.search.assert_called_once_with(
            query="concepts",
            course_name=None,
            lesson_number=2
        )
        
        assert "[Introduction to MCP - Lesson 2]" in result
        assert "Content from lesson 2" in result
    
    def test_execute_with_both_filters(self, mock_vector_store):
        specific_results = SearchResults(
            documents=["Specific content from MCP lesson 3"],
            metadata=[{"course_title": "Introduction to MCP", "lesson_number": 3}],
            distances=[0.05]
        )
        mock_vector_store.search.return_value = specific_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(
            query="advanced topics",
            course_name="Introduction to MCP",
            lesson_number=3
        )
        
        mock_vector_store.search.assert_called_once_with(
            query="advanced topics",
            course_name="Introduction to MCP",
            lesson_number=3
        )
        
        assert "[Introduction to MCP - Lesson 3]" in result
        assert "Specific content from MCP lesson 3" in result
    
    def test_execute_with_error(self, mock_vector_store):
        error_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Course not found"
        )
        mock_vector_store.search.return_value = error_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", course_name="NonExistent")
        
        assert result == "Course not found"
        assert tool.last_sources == []
    
    def test_execute_with_empty_results(self, mock_vector_store, empty_search_results):
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="nonexistent content")
        
        assert result == "No relevant content found."
        assert tool.last_sources == []
    
    def test_execute_empty_results_with_filters(self, mock_vector_store, empty_search_results):
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(
            query="nonexistent",
            course_name="Test Course",
            lesson_number=5
        )
        
        assert result == "No relevant content found in course 'Test Course' in lesson 5."
    
    def test_format_results_multiple_documents(self, mock_vector_store):
        results = SearchResults(
            documents=[
                "First document content",
                "Second document content",
                "Third document content"
            ],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": None},
                {"course_title": "Course A", "lesson_number": 2}
            ],
            distances=[0.1, 0.2, 0.3]
        )
        
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/a/1",
            None,
            "https://example.com/a/2"
        ]
        
        tool = CourseSearchTool(mock_vector_store)
        formatted = tool._format_results(results)
        
        assert "[Course A - Lesson 1]" in formatted
        assert "[Course B]" in formatted
        assert "[Course A - Lesson 2]" in formatted
        assert "First document content" in formatted
        assert "Second document content" in formatted
        assert "Third document content" in formatted
        
        assert len(tool.last_sources) == 3
        assert tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com/a/1"
        assert tool.last_sources[1]["text"] == "Course B"
        assert tool.last_sources[1]["url"] is None


class TestCourseOutlineTool:
    
    def test_get_tool_definition(self, mock_vector_store):
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["properties"]["course_title"]["type"] == "string"
        assert definition["input_schema"]["required"] == ["course_title"]
    
    def test_execute_successful(self, mock_vector_store):
        outline_data = {
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
        mock_vector_store.get_course_outline.return_value = outline_data
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_title="Introduction to MCP")
        
        mock_vector_store.get_course_outline.assert_called_once_with("Introduction to MCP")
        
        assert "Course: Introduction to MCP" in result
        assert "Instructor: Dr. Smith" in result
        assert "Course Link: https://example.com/mcp" in result
        assert "Total Lessons: 3" in result
        assert "1. Getting Started" in result
        assert "2. Core Concepts" in result
        assert "3. Advanced Topics" in result
    
    def test_execute_course_not_found(self, mock_vector_store):
        mock_vector_store.get_course_outline.return_value = None
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_title="Nonexistent Course")
        
        assert result == "No course found matching 'Nonexistent Course'"
    
    def test_format_outline_minimal_data(self, mock_vector_store):
        minimal_outline = {
            "course_title": "Basic Course",
            "lessons": []
        }
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool._format_outline(minimal_outline)
        
        assert "Course: Basic Course" in result
        assert "Total Lessons: 0" in result
        assert "Instructor:" not in result
        assert "Course Link:" not in result
    
    def test_format_outline_with_missing_lesson_details(self, mock_vector_store):
        outline = {
            "course_title": "Test Course",
            "lessons": [
                {"lesson_number": 1},
                {"lesson_title": "Unnamed Lesson"},
                {"lesson_number": 3, "lesson_title": "Complete Lesson"}
            ]
        }
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool._format_outline(outline)
        
        assert "1. Untitled" in result
        assert "N/A. Unnamed Lesson" in result
        assert "3. Complete Lesson" in result


class TestToolManager:
    
    def test_register_tool(self, mock_vector_store):
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(tool)
        
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool
    
    def test_register_tool_without_name(self, mock_vector_store):
        manager = ToolManager()
        
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"description": "No name"}
        
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)
    
    def test_get_tool_definitions(self, mock_vector_store):
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        definitions = manager.get_tool_definitions()
        
        assert len(definitions) == 2
        assert any(d["name"] == "search_course_content" for d in definitions)
        assert any(d["name"] == "get_course_outline" for d in definitions)
    
    def test_execute_tool_successful(self, mock_vector_store, sample_search_results):
        mock_vector_store.search.return_value = sample_search_results
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        result = manager.execute_tool("search_course_content", query="test query")
        
        assert "Introduction to MCP" in result
    
    def test_execute_tool_not_found(self):
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", param="value")
        
        assert result == "Tool 'nonexistent_tool' not found"
    
    def test_get_last_sources(self, mock_vector_store):
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        search_tool.last_sources = [
            {"text": "Source 1", "url": "https://example.com/1"}
        ]
        manager.register_tool(search_tool)
        
        sources = manager.get_last_sources()
        assert sources == [{"text": "Source 1", "url": "https://example.com/1"}]
    
    def test_get_last_sources_empty(self, mock_vector_store):
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        
        sources = manager.get_last_sources()
        assert sources == []
    
    def test_reset_sources(self, mock_vector_store):
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        search_tool.last_sources = [{"text": "Source", "url": "url"}]
        manager.register_tool(search_tool)
        
        manager.reset_sources()
        
        assert search_tool.last_sources == []