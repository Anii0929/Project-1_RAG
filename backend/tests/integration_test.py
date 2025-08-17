#!/usr/bin/env python3
"""
Integration test for the RAG system.
Tests the system with real components (but mocked external APIs).
"""

import unittest
from unittest.mock import patch, Mock
import sys
import os

# Add backend directory to path for imports
backend_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_path)

from rag_system import RAGSystem
from config import Config
from models import Course, Lesson, CourseChunk
import anthropic


class IntegrationTest(unittest.TestCase):
    """Integration tests using real system components"""
    
    @patch('anthropic.Anthropic')
    def test_end_to_end_query_processing(self, mock_anthropic_class):
        """Test complete query flow with mocked external dependencies"""
        
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock a response that includes tool use
        mock_tool_response = Mock()
        mock_tool_response.content = [Mock()]
        mock_tool_response.content[0].type = "tool_use"
        mock_tool_response.content[0].name = "search_course_content"
        mock_tool_response.content[0].input = {"query": "Python basics"}
        mock_tool_response.content[0].id = "tool_123"
        mock_tool_response.stop_reason = "tool_use"
        
        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Python is a high-level programming language..."
        
        # Configure mock to return tool response then final response
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Create config with fake API key
        config = Config()
        config.ANTHROPIC_API_KEY = "fake_key_for_testing"
        
        # Initialize RAG system
        with patch('chromadb.PersistentClient') as mock_chroma:
            # Mock ChromaDB collections
            mock_collection = Mock()
            mock_collection.query.return_value = {
                'documents': [["Python is a programming language used for various applications."]],
                'metadatas': [[{'course_title': 'Python Fundamentals', 'lesson_number': 1}]],
                'distances': [[0.2]]
            }
            mock_collection.get.return_value = {
                'ids': ['Python Fundamentals'],
                'metadatas': [{
                    'title': 'Python Fundamentals',
                    'instructor': 'John Doe',
                    'course_link': 'https://example.com/python',
                    'lessons_json': '[{"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": "https://example.com/python/lesson1"}]'
                }]
            }
            
            mock_chroma_client = Mock()
            mock_chroma_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_chroma_client
            
            # Initialize system
            rag_system = RAGSystem(config)
            
            # Execute query
            response, sources = rag_system.query("What is Python?")
            
            # Verify response was generated
            self.assertIsNotNone(response)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            
            # Verify tool was called through the flow
            self.assertEqual(mock_client.messages.create.call_count, 2)
            
            print(f"[PASS] Integration test passed!")
            print(f"   Response: {response}")
            print(f"   Sources: {len(sources)} found")


if __name__ == '__main__':
    # Run the integration test
    unittest.main(verbosity=2)