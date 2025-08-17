import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend directory to path for imports
backend_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_path)

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class MockConfig:
    """Mock configuration for testing"""
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    CHROMA_PATH = "./test_chroma"
    EMBEDDING_MODEL = "test-model"
    MAX_RESULTS = 5
    ANTHROPIC_API_KEY = "fake_key"
    ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
    MAX_HISTORY = 2


class TestRAGSystem(unittest.TestCase):
    """Test cases for RAG system content-query handling"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.config = MockConfig()
        
        # Create mocks for all dependencies
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager') as mock_session_mgr, \
             patch('rag_system.ToolManager') as mock_tool_mgr, \
             patch('rag_system.CourseSearchTool') as mock_search_tool, \
             patch('rag_system.CourseOutlineTool') as mock_outline_tool:
            
            # Initialize RAG system with mocked dependencies
            self.rag_system = RAGSystem(self.config)
            
            # Store mock references
            self.mock_doc_processor = self.rag_system.document_processor
            self.mock_vector_store = self.rag_system.vector_store
            self.mock_ai_generator = self.rag_system.ai_generator
            self.mock_session_manager = self.rag_system.session_manager
            self.mock_tool_manager = self.rag_system.tool_manager
    
    def test_query_basic_content_question(self):
        """Test basic content query handling"""
        # Mock AI generator response
        self.mock_ai_generator.generate_response.return_value = "Python is a programming language"
        
        # Mock tool manager sources
        mock_sources = [
            {
                "title": "Python Course - Lesson 1",
                "course_link": "https://python.com",
                "lesson_link": "https://python.com/lesson1"
            }
        ]
        self.mock_tool_manager.get_last_sources.return_value = mock_sources
        
        # Execute query
        response, sources = self.rag_system.query("What is Python?")
        
        # Verify response
        self.assertEqual(response, "Python is a programming language")
        self.assertEqual(sources, mock_sources)
        
        # Verify AI generator was called correctly
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        self.assertIn("What is Python?", call_args['query'])
        self.assertIsNotNone(call_args['tools'])
        self.assertEqual(call_args['tool_manager'], self.mock_tool_manager)
        
        # Verify sources were retrieved and reset
        self.mock_tool_manager.get_last_sources.assert_called_once()
        self.mock_tool_manager.reset_sources.assert_called_once()
    
    def test_query_with_session_id(self):
        """Test query handling with session context"""
        # Mock session manager
        mock_history = "Previous conversation about programming"
        self.mock_session_manager.get_conversation_history.return_value = mock_history
        
        # Mock AI generator response
        self.mock_ai_generator.generate_response.return_value = "Continuing our discussion..."
        self.mock_tool_manager.get_last_sources.return_value = []
        
        # Execute query with session
        session_id = "test_session_123"
        response, sources = self.rag_system.query("Continue", session_id=session_id)
        
        # Verify session history was retrieved
        self.mock_session_manager.get_conversation_history.assert_called_once_with(session_id)
        
        # Verify AI generator received history
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        self.assertEqual(call_args['conversation_history'], mock_history)
        
        # Verify session was updated with exchange
        self.mock_session_manager.add_exchange.assert_called_once_with(
            session_id, 
            "Continue", 
            "Continuing our discussion..."
        )
    
    def test_query_without_session_id(self):
        """Test query handling without session context"""
        # Mock AI generator response
        self.mock_ai_generator.generate_response.return_value = "Standalone response"
        self.mock_tool_manager.get_last_sources.return_value = []
        
        # Execute query without session
        response, sources = self.rag_system.query("Standalone question")
        
        # Verify no session manager calls were made
        self.mock_session_manager.get_conversation_history.assert_not_called()
        self.mock_session_manager.add_exchange.assert_not_called()
        
        # Verify AI generator called without history
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        self.assertIsNone(call_args['conversation_history'])
    
    def test_query_with_tool_sources(self):
        """Test query handling when tools return sources"""
        # Mock AI response
        self.mock_ai_generator.generate_response.return_value = "Here's what I found about algorithms"
        
        # Mock multiple sources from tools
        mock_sources = [
            {
                "title": "Algorithm Course - Lesson 1",
                "course_link": "https://algo.com",
                "lesson_link": "https://algo.com/lesson1"
            },
            {
                "title": "Algorithm Course - Lesson 2", 
                "course_link": "https://algo.com",
                "lesson_link": "https://algo.com/lesson2"
            }
        ]
        self.mock_tool_manager.get_last_sources.return_value = mock_sources
        
        # Execute query
        response, sources = self.rag_system.query("Explain algorithms")
        
        # Verify sources are returned
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["title"], "Algorithm Course - Lesson 1")
        self.assertEqual(sources[1]["title"], "Algorithm Course - Lesson 2")
    
    def test_query_prompt_formatting(self):
        """Test that query prompt is formatted correctly"""
        # Mock AI response
        self.mock_ai_generator.generate_response.return_value = "Response"
        self.mock_tool_manager.get_last_sources.return_value = []
        
        # Execute query
        user_query = "How do I learn Python?"
        self.rag_system.query(user_query)
        
        # Verify prompt formatting
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        expected_prompt = f"Answer this question about course materials: {user_query}"
        self.assertEqual(call_args['query'], expected_prompt)
    
    def test_query_tool_definitions_passed(self):
        """Test that tool definitions are passed to AI generator"""
        # Mock tool definitions
        mock_tool_defs = [
            {"name": "search_course_content", "description": "Search courses"},
            {"name": "get_course_outline", "description": "Get outlines"}
        ]
        self.mock_tool_manager.get_tool_definitions.return_value = mock_tool_defs
        
        # Mock AI response
        self.mock_ai_generator.generate_response.return_value = "Response"
        self.mock_tool_manager.get_last_sources.return_value = []
        
        # Execute query
        self.rag_system.query("Test query")
        
        # Verify tool definitions were retrieved and passed
        self.mock_tool_manager.get_tool_definitions.assert_called_once()
        
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        self.assertEqual(call_args['tools'], mock_tool_defs)
    
    def test_query_error_handling(self):
        """Test query handling when AI generator raises exception"""
        # Mock AI generator to raise exception
        self.mock_ai_generator.generate_response.side_effect = Exception("API Error")
        
        # Execute query and expect exception to propagate
        with self.assertRaises(Exception) as context:
            self.rag_system.query("Test query")
        
        self.assertIn("API Error", str(context.exception))
    
    def test_query_empty_sources_handling(self):
        """Test query handling when no sources are returned"""
        # Mock AI response
        self.mock_ai_generator.generate_response.return_value = "General knowledge response"
        
        # Mock empty sources
        self.mock_tool_manager.get_last_sources.return_value = []
        
        # Execute query
        response, sources = self.rag_system.query("General question")
        
        # Verify empty sources are handled correctly
        self.assertEqual(response, "General knowledge response")
        self.assertEqual(sources, [])
    
    def test_add_course_document_integration(self):
        """Test adding course document integrates with query functionality"""
        # Mock document processing
        mock_course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="https://test.com",
            lessons=[Lesson(lesson_number=1, title="Intro", lesson_link="https://test.com/lesson1")]
        )
        mock_chunks = [
            CourseChunk(content="Test content", course_title="Test Course", lesson_number=1, chunk_index=0)
        ]
        self.mock_doc_processor.process_course_document.return_value = (mock_course, mock_chunks)
        
        # Add document
        course, chunk_count = self.rag_system.add_course_document("test_path.txt")
        
        # Verify document was processed and added to vector store
        self.mock_doc_processor.process_course_document.assert_called_once_with("test_path.txt")
        self.mock_vector_store.add_course_metadata.assert_called_once_with(mock_course)
        self.mock_vector_store.add_course_content.assert_called_once_with(mock_chunks)
        
        # Verify return values
        self.assertEqual(course, mock_course)
        self.assertEqual(chunk_count, 1)
    
    def test_get_course_analytics_integration(self):
        """Test course analytics retrieval"""
        # Mock vector store analytics
        self.mock_vector_store.get_course_count.return_value = 5
        self.mock_vector_store.get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3", "Course 4", "Course 5"
        ]
        
        # Get analytics
        analytics = self.rag_system.get_course_analytics()
        
        # Verify analytics structure
        expected_analytics = {
            "total_courses": 5,
            "course_titles": ["Course 1", "Course 2", "Course 3", "Course 4", "Course 5"]
        }
        self.assertEqual(analytics, expected_analytics)
    
    def test_complex_query_workflow(self):
        """Test complex query workflow with multiple interactions"""
        session_id = "complex_session"
        
        # Mock session history
        initial_history = None
        updated_history = "User: What is Python?\nAI: Python is a programming language"
        
        # First query
        self.mock_session_manager.get_conversation_history.return_value = initial_history
        self.mock_ai_generator.generate_response.return_value = "Python is a programming language"
        self.mock_tool_manager.get_last_sources.return_value = [{"title": "Python Course"}]
        
        response1, sources1 = self.rag_system.query("What is Python?", session_id)
        
        # Second query with updated history
        self.mock_session_manager.get_conversation_history.return_value = updated_history
        self.mock_ai_generator.generate_response.return_value = "Python is used for web development, data science..."
        self.mock_tool_manager.get_last_sources.return_value = [{"title": "Python Applications"}]
        
        response2, sources2 = self.rag_system.query("What is it used for?", session_id)
        
        # Verify both queries were processed correctly
        self.assertEqual(response1, "Python is a programming language")
        self.assertEqual(response2, "Python is used for web development, data science...")
        
        # Verify session was updated twice
        self.assertEqual(self.mock_session_manager.add_exchange.call_count, 2)


if __name__ == '__main__':
    unittest.main()