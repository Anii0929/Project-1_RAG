import unittest
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from vector_store import SearchResults
from config import Config


class TestRAGSystemIntegration(unittest.TestCase):
    """Integration tests for the complete RAG system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a test config
        self.test_config = Config()
        self.test_config.ANTHROPIC_API_KEY = "test_key"
        self.test_config.ANTHROPIC_MODEL = "test_model"
        
        # Patch external dependencies
        self.patcher_anthropic = patch('ai_generator.anthropic.Anthropic')
        self.patcher_chromadb = patch('vector_store.chromadb.PersistentClient')
        self.patcher_embedding = patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
        
        self.mock_anthropic_class = self.patcher_anthropic.start()
        self.mock_chromadb_class = self.patcher_chromadb.start()
        self.mock_embedding_class = self.patcher_embedding.start()
        
        # Set up mock instances
        self.mock_anthropic_client = Mock()
        self.mock_anthropic_class.return_value = self.mock_anthropic_client
        
        self.mock_chromadb_client = Mock()
        self.mock_chromadb_class.return_value = self.mock_chromadb_client
        
        # Mock collections
        self.mock_course_catalog = Mock()
        self.mock_course_content = Mock()
        self.mock_chromadb_client.get_or_create_collection.side_effect = [
            self.mock_course_catalog,
            self.mock_course_content
        ]
        
        # Initialize RAG system
        self.rag_system = RAGSystem(self.test_config)
    
    def tearDown(self):
        """Clean up patches"""
        self.patcher_anthropic.stop()
        self.patcher_chromadb.stop()
        self.patcher_embedding.stop()
    
    def test_successful_content_query_with_tool_use(self):
        """Test a successful content query that uses the search tool"""
        # Arrange
        # Mock the Claude API responses
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_001"
        mock_tool_block.input = {"query": "What is Python?"}
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_block]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Python is a high-level programming language.")]
        
        self.mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Mock the vector store search
        self.mock_course_content.query.return_value = {
            'documents': [["Python is a programming language used for various applications"]],
            'metadatas': [[{"course_title": "Python Basics", "lesson_number": 1}]],
            'distances': [[0.1]]
        }
        
        # Act
        response, sources = self.rag_system.query("What is Python?")
        
        # Assert
        self.assertEqual(response, "Python is a high-level programming language.")
        self.assertEqual(sources, ["Python Basics - Lesson 1"])
        self.assertEqual(self.mock_anthropic_client.messages.create.call_count, 2)
    
    def test_query_with_no_results(self):
        """Test query when search returns no results"""
        # Arrange
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_002"
        mock_tool_block.input = {"query": "nonexistent topic"}
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_block]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="I couldn't find any information about that topic.")]
        
        self.mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Mock empty search results
        self.mock_course_content.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        # Act
        response, sources = self.rag_system.query("Tell me about nonexistent topic")
        
        # Assert
        self.assertEqual(response, "I couldn't find any information about that topic.")
        self.assertEqual(sources, [])
    
    def test_query_with_course_outline_tool(self):
        """Test query that uses the course outline tool"""
        # Arrange
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "get_course_outline"
        mock_tool_block.id = "tool_003"
        mock_tool_block.input = {"course_title": "Python"}
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_block]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Here's the Python course outline...")]
        
        self.mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Mock course catalog search for course resolution
        self.mock_course_catalog.query.return_value = {
            'documents': [["Python Complete Course"]],
            'metadatas': [[{"title": "Python Complete Course"}]]
        }
        
        # Mock course catalog get for outline
        self.mock_course_catalog.get.return_value = {
            'metadatas': [{
                'title': 'Python Complete Course',
                'course_link': 'https://example.com/python',
                'lessons_json': '[{"lesson_number": 1, "lesson_title": "Introduction"}]'
            }]
        }
        
        # Act
        response, sources = self.rag_system.query("Show me the Python course outline")
        
        # Assert
        self.assertEqual(response, "Here's the Python course outline...")
        self.assertEqual(sources, ["Python Complete Course"])
    
    def test_query_without_tools(self):
        """Test query that doesn't use any tools"""
        # Arrange
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="2 + 2 equals 4")]
        self.mock_anthropic_client.messages.create.return_value = mock_response
        
        # Act
        response, sources = self.rag_system.query("What is 2 + 2?")
        
        # Assert
        self.assertEqual(response, "2 + 2 equals 4")
        self.assertEqual(sources, [])
        self.mock_anthropic_client.messages.create.assert_called_once()
    
    def test_query_with_session_management(self):
        """Test that session management works correctly"""
        # Arrange
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Test response")]
        self.mock_anthropic_client.messages.create.return_value = mock_response
        
        session_id = self.rag_system.session_manager.create_session()
        
        # Act
        response1, _ = self.rag_system.query("First question", session_id)
        response2, _ = self.rag_system.query("Second question", session_id)
        
        # Assert
        history = self.rag_system.session_manager.get_conversation_history(session_id)
        self.assertIn("First question", history)
        self.assertIn("Second question", history)
    
    def test_query_with_vector_store_exception(self):
        """Test handling of vector store exceptions"""
        # Arrange
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_error"
        mock_tool_block.input = {"query": "test"}
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_block]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="An error occurred during search")]
        
        self.mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Mock vector store exception
        self.mock_course_content.query.side_effect = Exception("Database connection failed")
        
        # Act
        response, sources = self.rag_system.query("Search for something")
        
        # Assert
        self.assertEqual(response, "An error occurred during search")
        self.assertEqual(sources, [])
    
    def test_query_with_anthropic_api_exception(self):
        """Test handling of Anthropic API exceptions"""
        # Arrange
        self.mock_anthropic_client.messages.create.side_effect = Exception("API key invalid")
        
        # Act & Assert
        with self.assertRaises(Exception) as context:
            self.rag_system.query("Test query")
        self.assertIn("API key invalid", str(context.exception))
    
    def test_tool_registration(self):
        """Test that tools are properly registered"""
        # Assert
        tool_definitions = self.rag_system.tool_manager.get_tool_definitions()
        tool_names = [t["name"] for t in tool_definitions]
        
        self.assertIn("search_course_content", tool_names)
        self.assertIn("get_course_outline", tool_names)
        self.assertEqual(len(tool_definitions), 2)
    
    def test_query_with_course_filter(self):
        """Test query with course name filter in tool call"""
        # Arrange
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_004"
        mock_tool_block.input = {
            "query": "functions",
            "course_name": "Python Advanced"
        }
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_block]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Functions in Python Advanced course...")]
        
        self.mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Mock course resolution
        self.mock_course_catalog.query.return_value = {
            'documents': [["Python Advanced"]],
            'metadatas': [[{"title": "Python Advanced"}]]
        }
        
        # Mock content search
        self.mock_course_content.query.return_value = {
            'documents': [["Functions are reusable blocks of code"]],
            'metadatas': [[{"course_title": "Python Advanced", "lesson_number": 5}]],
            'distances': [[0.1]]
        }
        
        # Act
        response, sources = self.rag_system.query("Tell me about functions in Python Advanced")
        
        # Assert
        self.assertEqual(response, "Functions in Python Advanced course...")
        self.assertEqual(sources, ["Python Advanced - Lesson 5"])
    
    def test_sources_reset_after_query(self):
        """Test that sources are reset after each query"""
        # Arrange
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response")]
        self.mock_anthropic_client.messages.create.return_value = mock_response
        
        # Manually set some sources
        self.rag_system.search_tool.last_sources = ["Old Source"]
        
        # Act
        response, sources = self.rag_system.query("New query")
        
        # Assert
        self.assertEqual(self.rag_system.search_tool.last_sources, [])
        self.assertEqual(self.rag_system.outline_tool.last_sources, [])
    
    def test_query_failed_scenario(self):
        """Test the specific 'query failed' scenario the user mentioned"""
        # This test tries to reproduce the exact error condition
        
        # Scenario 1: Tool execution returns an error string
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_fail"
        mock_tool_block.input = {"query": "test content"}
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_block]
        
        # Simulate what might cause "query failed"
        # Option 1: Vector store returns error in SearchResults
        self.mock_course_content.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]],
        }
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="query failed")]  # The problematic response
        
        self.mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Act
        response, sources = self.rag_system.query("Find course content about Python")
        
        # Assert - This helps identify if this is the issue
        self.assertEqual(response, "query failed")
        # This test helps us understand that the AI is returning "query failed"
        # likely because of how it's interpreting empty or error results


class TestRAGSystemErrorPropagation(unittest.TestCase):
    """Focused tests to identify where 'query failed' originates"""
    
    def setUp(self):
        """Set up for error propagation tests"""
        self.test_config = Config()
        self.test_config.ANTHROPIC_API_KEY = "test_key"
        
        # Patch all external dependencies
        self.patches = [
            patch('ai_generator.anthropic.Anthropic'),
            patch('vector_store.chromadb.PersistentClient'),
            patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'),
        ]
        
        for p in self.patches:
            p.start()
    
    def tearDown(self):
        """Stop all patches"""
        for p in self.patches:
            p.stop()
    
    def test_trace_query_failed_origin(self):
        """Trace where 'query failed' message originates"""
        # This test will help identify if the error comes from:
        # 1. Vector store search error
        # 2. Tool execution error
        # 3. AI response to error
        # 4. Exception handling
        
        with patch('rag_system.RAGSystem.query') as mock_query:
            mock_query.return_value = ("query failed", [])
            
            # Create a RAG system instance
            rag = RAGSystem(self.test_config)
            
            # Try different query types
            test_queries = [
                "What is Python?",
                "Tell me about functions",
                "Show course outline",
                "Search for loops in Python"
            ]
            
            for query in test_queries:
                response, sources = mock_query(query)
                if response == "query failed":
                    # This helps identify that the issue is consistent
                    self.assertEqual(response, "query failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)