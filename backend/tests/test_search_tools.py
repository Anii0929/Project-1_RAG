import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool(unittest.TestCase):
    """Test suite for CourseSearchTool.execute method"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.tool = CourseSearchTool(self.mock_vector_store)
    
    def test_successful_search_with_results(self):
        """Test successful search that returns valid results"""
        # Arrange
        mock_results = SearchResults(
            documents=["This is course content about Python", "Another content chunk"],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 1},
                {"course_title": "Python Basics", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Act
        result = self.tool.execute(query="Python programming")
        
        # Assert
        self.assertIsInstance(result, str)
        self.assertIn("Python Basics", result)
        self.assertIn("This is course content about Python", result)
        self.assertEqual(len(self.tool.last_sources), 2)
        self.mock_vector_store.search.assert_called_once_with(
            query="Python programming",
            course_name=None,
            lesson_number=None
        )
    
    def test_search_with_course_filter(self):
        """Test search with course name filter"""
        # Arrange
        mock_results = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 3}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Act
        result = self.tool.execute(
            query="decorators",
            course_name="Advanced Python"
        )
        
        # Assert
        self.assertIn("Advanced Python", result)
        self.assertIn("Filtered content", result)
        self.mock_vector_store.search.assert_called_once_with(
            query="decorators",
            course_name="Advanced Python",
            lesson_number=None
        )
    
    def test_search_with_lesson_filter(self):
        """Test search with lesson number filter"""
        # Arrange
        mock_results = SearchResults(
            documents=["Lesson specific content"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 5}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Act
        result = self.tool.execute(
            query="functions",
            lesson_number=5
        )
        
        # Assert
        self.assertIn("Lesson 5", result)
        self.mock_vector_store.search.assert_called_once_with(
            query="functions",
            course_name=None,
            lesson_number=5
        )
    
    def test_search_with_error_result(self):
        """Test handling of error in search results"""
        # Arrange
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Act
        result = self.tool.execute(query="test query")
        
        # Assert
        self.assertEqual(result, "Database connection failed")
        self.assertEqual(self.tool.last_sources, [])
    
    def test_search_with_empty_results(self):
        """Test handling of empty search results"""
        # Arrange
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Act
        result = self.tool.execute(query="nonexistent content")
        
        # Assert
        self.assertEqual(result, "No relevant content found.")
        self.assertEqual(self.tool.last_sources, [])
    
    def test_empty_results_with_filters(self):
        """Test empty results message includes filter information"""
        # Arrange
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Act
        result = self.tool.execute(
            query="test",
            course_name="Test Course",
            lesson_number=3
        )
        
        # Assert
        self.assertEqual(
            result,
            "No relevant content found in course 'Test Course' in lesson 3."
        )
    
    def test_format_results_with_missing_metadata(self):
        """Test that formatting handles missing metadata gracefully"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content without metadata"],
            metadata=[{}],  # Empty metadata
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Act
        result = self.tool.execute(query="test")
        
        # Assert
        self.assertIn("unknown", result)  # Should use 'unknown' for missing course_title
        self.assertIn("Content without metadata", result)
    
    def test_vector_store_exception_handling(self):
        """Test handling of exceptions from vector store"""
        # Arrange
        self.mock_vector_store.search.side_effect = Exception("Unexpected error")
        
        # Act & Assert
        with self.assertRaises(Exception) as context:
            self.tool.execute(query="test")
        self.assertIn("Unexpected error", str(context.exception))
    
    def test_tool_definition(self):
        """Test that tool definition is correctly structured"""
        # Act
        definition = self.tool.get_tool_definition()
        
        # Assert
        self.assertEqual(definition["name"], "search_course_content")
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)
        self.assertEqual(definition["input_schema"]["type"], "object")
        self.assertIn("query", definition["input_schema"]["properties"])
        self.assertEqual(definition["input_schema"]["required"], ["query"])


class TestCourseOutlineTool(unittest.TestCase):
    """Test suite for CourseOutlineTool"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.tool = CourseOutlineTool(self.mock_vector_store)
    
    def test_successful_outline_retrieval(self):
        """Test successful retrieval of course outline"""
        # Arrange
        self.mock_vector_store._resolve_course_name.return_value = "Python Complete Course"
        self.mock_vector_store.course_catalog.get.return_value = {
            'metadatas': [{
                'title': 'Python Complete Course',
                'course_link': 'https://example.com/python',
                'lessons_json': '[{"lesson_number": 1, "lesson_title": "Introduction"}, {"lesson_number": 2, "lesson_title": "Variables"}]'
            }]
        }
        
        # Act
        result = self.tool.execute(course_title="Python")
        
        # Assert
        self.assertIn("Python Complete Course", result)
        self.assertIn("https://example.com/python", result)
        self.assertIn("Lesson 1: Introduction", result)
        self.assertIn("Lesson 2: Variables", result)
        self.assertEqual(self.tool.last_sources, ["Python Complete Course"])
    
    def test_course_not_found(self):
        """Test handling when course is not found"""
        # Arrange
        self.mock_vector_store._resolve_course_name.return_value = None
        
        # Act
        result = self.tool.execute(course_title="Nonexistent Course")
        
        # Assert
        self.assertEqual(result, "No course found matching 'Nonexistent Course'")
        self.assertEqual(self.tool.last_sources, [])
    
    def test_outline_retrieval_exception(self):
        """Test exception handling in outline retrieval"""
        # Arrange
        self.mock_vector_store._resolve_course_name.return_value = "Python Course"
        self.mock_vector_store.course_catalog.get.side_effect = Exception("Database error")
        
        # Act
        result = self.tool.execute(course_title="Python")
        
        # Assert
        self.assertIn("Error retrieving course outline", result)
        self.assertIn("Database error", result)


class TestToolManager(unittest.TestCase):
    """Test suite for ToolManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = ToolManager()
        self.mock_tool = Mock()
        self.mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "Test tool"
        }
    
    def test_register_tool(self):
        """Test tool registration"""
        # Act
        self.manager.register_tool(self.mock_tool)
        
        # Assert
        self.assertIn("test_tool", self.manager.tools)
        self.assertEqual(self.manager.tools["test_tool"], self.mock_tool)
    
    def test_register_tool_without_name(self):
        """Test that registering tool without name raises error"""
        # Arrange
        self.mock_tool.get_tool_definition.return_value = {"description": "No name"}
        
        # Act & Assert
        with self.assertRaises(ValueError) as context:
            self.manager.register_tool(self.mock_tool)
        self.assertIn("Tool must have a 'name'", str(context.exception))
    
    def test_execute_tool(self):
        """Test tool execution"""
        # Arrange
        self.mock_tool.execute.return_value = "Tool result"
        self.manager.register_tool(self.mock_tool)
        
        # Act
        result = self.manager.execute_tool("test_tool", query="test")
        
        # Assert
        self.assertEqual(result, "Tool result")
        self.mock_tool.execute.assert_called_once_with(query="test")
    
    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist"""
        # Act
        result = self.manager.execute_tool("nonexistent_tool")
        
        # Assert
        self.assertEqual(result, "Tool 'nonexistent_tool' not found")
    
    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        # Arrange
        self.manager.register_tool(self.mock_tool)
        
        # Act
        definitions = self.manager.get_tool_definitions()
        
        # Assert
        self.assertEqual(len(definitions), 1)
        self.assertEqual(definitions[0]["name"], "test_tool")
    
    def test_get_last_sources(self):
        """Test retrieving last sources from tools"""
        # Arrange
        mock_search_tool = Mock()
        mock_search_tool.last_sources = ["Source 1", "Source 2"]
        mock_search_tool.get_tool_definition.return_value = {"name": "search"}
        self.manager.register_tool(mock_search_tool)
        
        # Act
        sources = self.manager.get_last_sources()
        
        # Assert
        self.assertEqual(sources, ["Source 1", "Source 2"])
    
    def test_reset_sources(self):
        """Test resetting sources for all tools"""
        # Arrange
        mock_search_tool = Mock()
        mock_search_tool.last_sources = ["Source 1"]
        mock_search_tool.get_tool_definition.return_value = {"name": "search"}
        self.manager.register_tool(mock_search_tool)
        
        # Act
        self.manager.reset_sources()
        
        # Assert
        self.assertEqual(mock_search_tool.last_sources, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)