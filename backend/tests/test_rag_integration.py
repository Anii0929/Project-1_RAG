"""
Integration tests for RAG system end-to-end functionality.

This module tests the complete RAG system flow from query input to response output,
including document processing, tool registration, and query handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import os
import sys

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rag_system import RAGSystem
from models import Course, Lesson
from config import Config


class TestRAGSystemInitialization:
    """Test suite for RAG system initialization."""
    
    @pytest.mark.integration
    def test_rag_system_init_anthropic(self):
        """Test RAG system initialization with Anthropic provider."""
        config = Mock(spec=Config)
        config.AI_PROVIDER = "anthropic"
        config.ANTHROPIC_API_KEY = "test_key"
        config.ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):
            
            rag = RAGSystem(config)
            
            assert rag.config == config
            mock_ai.assert_called_once_with("test_key", "claude-3-sonnet-20240229")
            assert rag.tool_manager is not None
            assert rag.search_tool is not None
            assert rag.outline_tool is not None
    
    @pytest.mark.integration
    def test_rag_system_init_ollama(self):
        """Test RAG system initialization with Ollama provider."""
        config = Mock(spec=Config)
        config.AI_PROVIDER = "ollama"
        config.OLLAMA_MODEL = "llama2"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.OllamaGenerator') as mock_ai, \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):
            
            rag = RAGSystem(config)
            
            mock_ai.assert_called_once_with("llama2")
    
    @pytest.mark.integration
    def test_rag_system_init_search_only(self):
        """Test RAG system initialization with search-only provider."""
        config = Mock(spec=Config)
        config.AI_PROVIDER = "search_only"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.SimpleSearchOnlyGenerator') as mock_ai, \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):
            
            rag = RAGSystem(config)
            
            mock_ai.assert_called_once()
    
    @pytest.mark.integration
    def test_rag_system_init_unknown_provider(self):
        """Test RAG system initialization fails with unknown provider."""
        config = Mock(spec=Config)
        config.AI_PROVIDER = "unknown_provider"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):
            
            with pytest.raises(ValueError, match="Unknown AI provider: unknown_provider"):
                RAGSystem(config)
    
    @pytest.mark.integration
    def test_tools_registration(self):
        """Test that both search and outline tools are registered."""
        config = Mock(spec=Config)
        config.AI_PROVIDER = "search_only"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.SimpleSearchOnlyGenerator'), \
             patch('rag_system.SessionManager'):
            
            rag = RAGSystem(config)
            
            tool_definitions = rag.tool_manager.get_tool_definitions()
            assert len(tool_definitions) == 3
            
            tool_names = [tool["name"] for tool in tool_definitions]
            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names
            assert "list_all_courses" in tool_names


class TestRAGSystemQuery:
    """Test suite for RAG system query processing."""
    
    def setup_mock_rag_system(self, config):
        """Helper to create a mocked RAG system for testing."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.SimpleSearchOnlyGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):
            
            return RAGSystem(config)
    
    @pytest.mark.integration
    def test_query_successful_with_tools(self):
        """Test successful query processing with tool execution."""
        config = Mock(spec=Config)
        config.AI_PROVIDER = "search_only"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        
        rag = self.setup_mock_rag_system(config)
        
        # Mock AI generator response
        rag.ai_generator.generate_response.return_value = "Based on the search results, MCP is a protocol for AI applications."
        
        # Mock tool manager sources
        test_sources = [
            {"text": "MCP Course - Lesson 1", "link": "https://example.com/lesson1"}
        ]
        rag.tool_manager.get_last_sources.return_value = test_sources
        
        response, sources = rag.query("What is MCP?")
        
        assert response == "Based on the search results, MCP is a protocol for AI applications."
        assert sources == test_sources
        
        # Verify AI generator was called with correct parameters
        rag.ai_generator.generate_response.assert_called_once()
        call_args = rag.ai_generator.generate_response.call_args
        assert "What is MCP?" in call_args[1]["query"]
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] == rag.tool_manager
    
    @pytest.mark.integration
    def test_query_with_session_management(self):
        """Test query processing with session management."""
        config = Mock(spec=Config)
        config.AI_PROVIDER = "search_only"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        
        rag = self.setup_mock_rag_system(config)
        
        # Mock session manager
        session_id = "test_session_123"
        conversation_history = "User: Previous question\nAssistant: Previous answer"
        rag.session_manager.get_conversation_history.return_value = conversation_history
        
        # Mock AI generator response
        rag.ai_generator.generate_response.return_value = "Response with context."
        rag.tool_manager.get_last_sources.return_value = []
        
        response, sources = rag.query("Follow-up question", session_id=session_id)
        
        assert response == "Response with context."
        
        # Verify session history was retrieved and used
        rag.session_manager.get_conversation_history.assert_called_once_with(session_id)
        
        # Verify AI generator received conversation history
        call_args = rag.ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == conversation_history
        
        # Verify conversation was updated
        rag.session_manager.add_exchange.assert_called_once_with(
            session_id, "Follow-up question", "Response with context."
        )
    
    @pytest.mark.integration
    def test_query_without_session(self):
        """Test query processing without session ID."""
        config = Mock(spec=Config)
        config.AI_PROVIDER = "search_only"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        
        rag = self.setup_mock_rag_system(config)
        
        rag.ai_generator.generate_response.return_value = "Response without session."
        rag.tool_manager.get_last_sources.return_value = []
        
        response, sources = rag.query("Standalone question")
        
        assert response == "Response without session."
        
        # Verify no session operations were performed
        rag.session_manager.get_conversation_history.assert_not_called()
        rag.session_manager.add_exchange.assert_not_called()
        
        # Verify AI generator received None for history
        call_args = rag.ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] is None
    
    @pytest.mark.integration
    def test_query_ai_generator_failure(self):
        """Test query handling when AI generator fails."""
        config = Mock(spec=Config)
        config.AI_PROVIDER = "anthropic"
        config.ANTHROPIC_API_KEY = "test_key"
        config.ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        
        rag = self.setup_mock_rag_system(config)
        
        # Mock AI generator failure by replacing the method
        rag.ai_generator.generate_response = Mock(side_effect=Exception("API key invalid"))
        
        with pytest.raises(Exception, match="API key invalid"):
            rag.query("What is MCP?")
    
    @pytest.mark.integration
    def test_sources_reset_after_query(self):
        """Test that sources are reset after being retrieved."""
        config = Mock(spec=Config)
        config.AI_PROVIDER = "search_only"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        
        rag = self.setup_mock_rag_system(config)
        
        rag.ai_generator.generate_response.return_value = "Test response"
        test_sources = [{"text": "Test Source", "link": None}]
        rag.tool_manager.get_last_sources.return_value = test_sources
        
        response, sources = rag.query("Test query")
        
        assert sources == test_sources
        
        # Verify sources were retrieved and reset
        rag.tool_manager.get_last_sources.assert_called_once()
        rag.tool_manager.reset_sources.assert_called_once()


class TestRAGSystemDocumentProcessing:
    """Test suite for RAG system document processing."""
    
    @pytest.mark.integration 
    def test_add_course_document_success(self):
        """Test successful course document addition."""
        config = Mock(spec=Config)
        config.AI_PROVIDER = "search_only"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        
        with patch('rag_system.DocumentProcessor') as mock_proc, \
             patch('rag_system.VectorStore') as mock_store, \
             patch('rag_system.SimpleSearchOnlyGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):
            
            # Mock course and chunks
            test_course = Course(
                title="Test Course",
                instructor="Test Instructor", 
                course_link="https://example.com",
                lessons=[Lesson(lesson_number=0, title="Introduction", lesson_link=None)]
            )
            test_chunks = ["chunk1", "chunk2", "chunk3"]
            
            mock_processor_instance = Mock()
            mock_processor_instance.process_course_document.return_value = (test_course, test_chunks)
            mock_proc.return_value = mock_processor_instance
            
            mock_store_instance = Mock()
            mock_store.return_value = mock_store_instance
            
            rag = RAGSystem(config)
            
            course, chunk_count = rag.add_course_document("test_file.txt")
            
            assert course == test_course
            assert chunk_count == 3
            
            # Verify document processor was called
            mock_processor_instance.process_course_document.assert_called_once_with("test_file.txt")
            
            # Verify vector store operations
            mock_store_instance.add_course_metadata.assert_called_once_with(test_course)
            mock_store_instance.add_course_content.assert_called_once_with(test_chunks)
    
    @pytest.mark.integration
    def test_add_course_document_failure(self):
        """Test course document addition failure handling."""
        config = Mock(spec=Config)
        config.AI_PROVIDER = "search_only"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        
        with patch('rag_system.DocumentProcessor') as mock_proc, \
             patch('rag_system.VectorStore'), \
             patch('rag_system.SimpleSearchOnlyGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):
            
            mock_processor_instance = Mock()
            mock_processor_instance.process_course_document.side_effect = Exception("File not found")
            mock_proc.return_value = mock_processor_instance
            
            rag = RAGSystem(config)
            
            course, chunk_count = rag.add_course_document("nonexistent.txt")
            
            assert course is None
            assert chunk_count == 0


class TestRAGSystemAnalytics:
    """Test suite for RAG system analytics functionality."""
    
    @pytest.mark.integration
    def test_get_course_analytics(self):
        """Test course analytics retrieval."""
        config = Mock(spec=Config)
        config.AI_PROVIDER = "search_only"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_store, \
             patch('rag_system.SimpleSearchOnlyGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):
            
            mock_store_instance = Mock()
            mock_store_instance.get_course_count.return_value = 3
            mock_store_instance.get_existing_course_titles.return_value = [
                "Course 1", "Course 2", "Course 3"
            ]
            mock_store.return_value = mock_store_instance
            
            rag = RAGSystem(config)
            analytics = rag.get_course_analytics()
            
            assert analytics["total_courses"] == 3
            assert analytics["course_titles"] == ["Course 1", "Course 2", "Course 3"]
            
            mock_store_instance.get_course_count.assert_called_once()
            mock_store_instance.get_existing_course_titles.assert_called_once()


class TestRAGSystemRealWorldScenarios:
    """Test suite for real-world usage scenarios."""
    
    @pytest.mark.integration
    def test_empty_database_query(self):
        """Test query behavior with empty database."""
        config = Mock(spec=Config)
        config.AI_PROVIDER = "search_only"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.SimpleSearchOnlyGenerator') as mock_gen, \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):
            
            rag = RAGSystem(config)
            
            # Mock search-only generator to return search results directly
            mock_gen_instance = Mock()
            mock_gen_instance.generate_response.return_value = "No relevant content found."
            rag.ai_generator = mock_gen_instance
            
            rag.tool_manager.get_last_sources.return_value = []
            
            response, sources = rag.query("What is MCP?")
            
            assert "No relevant content found" in response
            assert len(sources) == 0
    
    @pytest.mark.integration
    def test_multiple_queries_same_session(self):
        """Test multiple queries in the same session."""
        config = Mock(spec=Config)
        config.AI_PROVIDER = "search_only"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.SimpleSearchOnlyGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):
            
            rag = RAGSystem(config)
            session_id = "test_session"
            
            # Mock responses
            rag.ai_generator.generate_response.side_effect = [
                "First response about MCP",
                "Second response building on context"
            ]
            rag.tool_manager.get_last_sources.return_value = []
            
            # Mock session history evolution
            rag.session_manager.get_conversation_history.side_effect = [
                None,  # First query has no history
                "User: What is MCP?\nAssistant: First response about MCP"  # Second query has history
            ]
            
            # First query
            response1, sources1 = rag.query("What is MCP?", session_id=session_id)
            assert response1 == "First response about MCP"
            
            # Second query with context
            response2, sources2 = rag.query("Tell me more about it", session_id=session_id)
            assert response2 == "Second response building on context"
            
            # Verify session management was called correctly
            assert rag.session_manager.add_exchange.call_count == 2
            assert rag.session_manager.get_conversation_history.call_count == 2