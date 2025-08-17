import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend directory to path for imports
backend_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_path)

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool(unittest.TestCase):
    """Test cases for CourseSearchTool.execute method"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_execute_basic_query_success(self):
        """Test basic query execution with successful results"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["This is course content about Python basics"],
            metadata=[{
                'course_title': 'Python Programming',
                'lesson_number': 1
            }],
            distances=[0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_course_link.return_value = "https://course.com"
        self.mock_vector_store.get_lesson_link.return_value = "https://lesson.com"
        
        # Execute the search
        result = self.search_tool.execute("Python basics")
        
        # Verify the call was made correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="Python basics",
            course_name=None,
            lesson_number=None
        )
        
        # Verify the result format
        self.assertIn("Python Programming", result)
        self.assertIn("Lesson 1", result)
        self.assertIn("This is course content about Python basics", result)
        
        # Verify sources were tracked
        self.assertEqual(len(self.search_tool.last_sources), 1)
        self.assertEqual(self.search_tool.last_sources[0]['title'], "Python Programming - Lesson 1")
    
    def test_execute_with_course_filter(self):
        """Test query execution with course name filter"""
        mock_results = SearchResults(
            documents=["Advanced Python concepts"],
            metadata=[{
                'course_title': 'Advanced Python',
                'lesson_number': 2
            }],
            distances=[0.15],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_course_link.return_value = "https://advanced.com"
        self.mock_vector_store.get_lesson_link.return_value = "https://advanced-lesson.com"
        
        result = self.search_tool.execute("concepts", course_name="Advanced Python")
        
        # Verify the call included course filter
        self.mock_vector_store.search.assert_called_once_with(
            query="concepts",
            course_name="Advanced Python",
            lesson_number=None
        )
        
        self.assertIn("Advanced Python", result)
        self.assertIn("Lesson 2", result)
    
    def test_execute_with_lesson_filter(self):
        """Test query execution with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{
                'course_title': 'Web Development',
                'lesson_number': 3
            }],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_course_link.return_value = "https://web.com"
        self.mock_vector_store.get_lesson_link.return_value = "https://web-lesson3.com"
        
        result = self.search_tool.execute("content", lesson_number=3)
        
        # Verify the call included lesson filter
        self.mock_vector_store.search.assert_called_once_with(
            query="content",
            course_name=None,
            lesson_number=3
        )
        
        self.assertIn("Web Development", result)
        self.assertIn("Lesson 3", result)
    
    def test_execute_with_both_filters(self):
        """Test query execution with both course and lesson filters"""
        mock_results = SearchResults(
            documents=["Specific lesson content"],
            metadata=[{
                'course_title': 'Data Science',
                'lesson_number': 5
            }],
            distances=[0.05],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_course_link.return_value = "https://datascience.com"
        self.mock_vector_store.get_lesson_link.return_value = "https://datascience-lesson5.com"
        
        result = self.search_tool.execute("content", course_name="Data Science", lesson_number=5)
        
        # Verify the call included both filters
        self.mock_vector_store.search.assert_called_once_with(
            query="content",
            course_name="Data Science",
            lesson_number=5
        )
        
        self.assertIn("Data Science", result)
        self.assertIn("Lesson 5", result)
    
    def test_execute_with_search_error(self):
        """Test query execution when vector store returns an error"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query")
        
        # Should return the error message
        self.assertEqual(result, "Database connection failed")
    
    def test_execute_with_empty_results(self):
        """Test query execution when no results are found"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("nonexistent topic")
        
        # Should return no results message
        self.assertEqual(result, "No relevant content found.")
    
    def test_execute_empty_results_with_filters(self):
        """Test empty results message includes filter information"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("topic", course_name="Missing Course", lesson_number=99)
        
        # Should include filter info in the message
        self.assertIn("Missing Course", result)
        self.assertIn("lesson 99", result)
        self.assertIn("No relevant content found", result)
    
    def test_execute_multiple_results(self):
        """Test query execution with multiple search results"""
        mock_results = SearchResults(
            documents=[
                "First result about algorithms",
                "Second result about data structures"
            ],
            metadata=[
                {'course_title': 'Computer Science', 'lesson_number': 1},
                {'course_title': 'Computer Science', 'lesson_number': 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_course_link.return_value = "https://cs.com"
        self.mock_vector_store.get_lesson_link.side_effect = lambda course, lesson: f"https://cs.com/lesson{lesson}"
        
        result = self.search_tool.execute("algorithms")
        
        # Should contain both results
        self.assertIn("First result about algorithms", result)
        self.assertIn("Second result about data structures", result)
        self.assertIn("Lesson 1", result)
        self.assertIn("Lesson 2", result)
        
        # Should track multiple sources
        self.assertEqual(len(self.search_tool.last_sources), 2)
    
    def test_execute_missing_metadata_fields(self):
        """Test query execution with incomplete metadata"""
        mock_results = SearchResults(
            documents=["Content with missing metadata"],
            metadata=[{
                'course_title': 'Incomplete Course'
                # Missing lesson_number
            }],
            distances=[0.3],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_course_link.return_value = "https://incomplete.com"
        self.mock_vector_store.get_lesson_link.return_value = None
        
        result = self.search_tool.execute("test")
        
        # Should handle missing lesson number gracefully
        self.assertIn("Incomplete Course", result)
        self.assertNotIn("Lesson", result)  # No lesson number should be shown
        
        # Source should not include lesson info
        self.assertEqual(self.search_tool.last_sources[0]['title'], "Incomplete Course")
    
    def test_execute_no_links_available(self):
        """Test query execution when links are not available"""
        mock_results = SearchResults(
            documents=["Content without links"],
            metadata=[{
                'course_title': 'No Links Course',
                'lesson_number': 1
            }],
            distances=[0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_course_link.return_value = None
        self.mock_vector_store.get_lesson_link.return_value = None
        
        result = self.search_tool.execute("test")
        
        # Should still work without links
        self.assertIn("No Links Course", result)
        self.assertIn("Lesson 1", result)
        
        # Sources should have None for links
        source = self.search_tool.last_sources[0]
        self.assertIsNone(source['course_link'])
        self.assertIsNone(source['lesson_link'])


if __name__ == '__main__':
    unittest.main()