import pytest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk

class TestRAGSystem:
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_initialization(self, mock_doc_processor_class, mock_vector_store_class, 
                          mock_ai_gen_class, mock_session_manager_class, mock_config):
        rag_system = RAGSystem(mock_config)
        
        mock_doc_processor_class.assert_called_once_with(
            mock_config.CHUNK_SIZE, 
            mock_config.CHUNK_OVERLAP
        )
        mock_vector_store_class.assert_called_once_with(
            mock_config.CHROMA_PATH, 
            mock_config.EMBEDDING_MODEL, 
            mock_config.MAX_RESULTS
        )
        mock_ai_gen_class.assert_called_once_with(
            mock_config.ANTHROPIC_API_KEY, 
            mock_config.ANTHROPIC_MODEL
        )
        mock_session_manager_class.assert_called_once_with(mock_config.MAX_HISTORY)
        
        assert hasattr(rag_system, 'tool_manager')
        assert hasattr(rag_system, 'search_tool')
        assert hasattr(rag_system, 'outline_tool')
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_document_successful(self, mock_doc_processor_class, mock_vector_store_class,
                                           mock_ai_gen_class, mock_session_manager_class, 
                                           mock_config, sample_courses, sample_course_chunks):
        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document.return_value = (
            sample_courses[0], sample_course_chunks[:2]
        )
        mock_doc_processor_class.return_value = mock_doc_processor
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        rag_system = RAGSystem(mock_config)
        course, chunk_count = rag_system.add_course_document("/path/to/course.txt")
        
        assert course == sample_courses[0]
        assert chunk_count == 2
        
        mock_doc_processor.process_course_document.assert_called_once_with("/path/to/course.txt")
        mock_vector_store.add_course_metadata.assert_called_once_with(sample_courses[0])
        mock_vector_store.add_course_content.assert_called_once_with(sample_course_chunks[:2])
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_document_with_error(self, mock_doc_processor_class, mock_vector_store_class,
                                           mock_ai_gen_class, mock_session_manager_class, mock_config):
        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document.side_effect = Exception("Processing error")
        mock_doc_processor_class.return_value = mock_doc_processor
        
        rag_system = RAGSystem(mock_config)
        course, chunk_count = rag_system.add_course_document("/path/to/bad_file.txt")
        
        assert course is None
        assert chunk_count == 0
    
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.isfile')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_folder_new_courses(self, mock_doc_processor_class, mock_vector_store_class,
                                          mock_ai_gen_class, mock_session_manager_class, 
                                          mock_isfile, mock_listdir, mock_exists,
                                          mock_config, sample_courses, sample_course_chunks):
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.pdf", "readme.md"]
        mock_isfile.side_effect = lambda x: True
        
        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document.side_effect = [
            (sample_courses[0], sample_course_chunks[:2]),
            (sample_courses[1], sample_course_chunks[2:])
        ]
        mock_doc_processor_class.return_value = mock_doc_processor
        
        mock_vector_store = Mock()
        mock_vector_store.get_existing_course_titles.return_value = []
        mock_vector_store_class.return_value = mock_vector_store
        
        rag_system = RAGSystem(mock_config)
        total_courses, total_chunks = rag_system.add_course_folder("/docs")
        
        assert total_courses == 2
        assert total_chunks == 4
        assert mock_vector_store.add_course_metadata.call_count == 2
        assert mock_vector_store.add_course_content.call_count == 2
    
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('os.path.isfile')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_folder_skip_existing(self, mock_doc_processor_class, mock_vector_store_class,
                                            mock_ai_gen_class, mock_session_manager_class,
                                            mock_isfile, mock_listdir, mock_exists,
                                            mock_config, sample_courses):
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt"]
        mock_isfile.return_value = True
        
        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document.return_value = (sample_courses[0], [])
        mock_doc_processor_class.return_value = mock_doc_processor
        
        mock_vector_store = Mock()
        mock_vector_store.get_existing_course_titles.return_value = ["Introduction to MCP"]
        mock_vector_store_class.return_value = mock_vector_store
        
        rag_system = RAGSystem(mock_config)
        total_courses, total_chunks = rag_system.add_course_folder("/docs")
        
        assert total_courses == 0
        assert total_chunks == 0
        mock_vector_store.add_course_metadata.assert_not_called()
        mock_vector_store.add_course_content.assert_not_called()
    
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_folder_clear_existing(self, mock_doc_processor_class, mock_vector_store_class,
                                             mock_ai_gen_class, mock_session_manager_class,
                                             mock_listdir, mock_exists, mock_config):
        mock_exists.return_value = True
        mock_listdir.return_value = []
        
        mock_vector_store = Mock()
        mock_vector_store.get_existing_course_titles.return_value = []
        mock_vector_store_class.return_value = mock_vector_store
        
        rag_system = RAGSystem(mock_config)
        rag_system.add_course_folder("/docs", clear_existing=True)
        
        mock_vector_store.clear_all_data.assert_called_once()
    
    @patch('os.path.exists')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_folder_nonexistent(self, mock_doc_processor_class, mock_vector_store_class,
                                          mock_ai_gen_class, mock_session_manager_class,
                                          mock_exists, mock_config):
        mock_exists.return_value = False
        
        rag_system = RAGSystem(mock_config)
        total_courses, total_chunks = rag_system.add_course_folder("/nonexistent")
        
        assert total_courses == 0
        assert total_chunks == 0
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_without_session(self, mock_doc_processor_class, mock_vector_store_class,
                                  mock_ai_gen_class, mock_session_manager_class, mock_config):
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "MCP is a framework for building applications."
        mock_ai_gen_class.return_value = mock_ai_gen
        
        mock_session_manager = Mock()
        mock_session_manager_class.return_value = mock_session_manager
        
        rag_system = RAGSystem(mock_config)
        rag_system.tool_manager.get_last_sources = Mock(return_value=[
            {"text": "Introduction to MCP - Lesson 1", "url": "https://example.com/mcp/1"}
        ])
        rag_system.tool_manager.reset_sources = Mock()
        
        response, sources = rag_system.query("What is MCP?")
        
        assert response == "MCP is a framework for building applications."
        assert len(sources) == 1
        assert sources[0]["text"] == "Introduction to MCP - Lesson 1"
        
        mock_ai_gen.generate_response.assert_called_once()
        call_args = mock_ai_gen.generate_response.call_args
        assert "What is MCP?" in call_args.kwargs["query"]
        assert call_args.kwargs["conversation_history"] is None
        assert call_args.kwargs["tools"] is not None
        assert call_args.kwargs["tool_manager"] is not None
        
        mock_session_manager.get_conversation_history.assert_not_called()
        mock_session_manager.add_exchange.assert_not_called()
        rag_system.tool_manager.reset_sources.assert_called_once()
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_with_session(self, mock_doc_processor_class, mock_vector_store_class,
                               mock_ai_gen_class, mock_session_manager_class, mock_config):
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Based on our previous discussion, MCP is..."
        mock_ai_gen_class.return_value = mock_ai_gen
        
        mock_session_manager = Mock()
        mock_session_manager.get_conversation_history.return_value = "Previous context about MCP"
        mock_session_manager_class.return_value = mock_session_manager
        
        rag_system = RAGSystem(mock_config)
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()
        
        response, sources = rag_system.query("Tell me more", session_id="session123")
        
        assert response == "Based on our previous discussion, MCP is..."
        assert sources == []
        
        mock_session_manager.get_conversation_history.assert_called_once_with("session123")
        mock_session_manager.add_exchange.assert_called_once_with(
            "session123", "Tell me more", "Based on our previous discussion, MCP is..."
        )
        
        call_args = mock_ai_gen.generate_response.call_args
        assert call_args.kwargs["conversation_history"] == "Previous context about MCP"
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_with_multiple_sources(self, mock_doc_processor_class, mock_vector_store_class,
                                        mock_ai_gen_class, mock_session_manager_class, mock_config):
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Comprehensive answer"
        mock_ai_gen_class.return_value = mock_ai_gen
        
        rag_system = RAGSystem(mock_config)
        
        multiple_sources = [
            {"text": "Course A - Lesson 1", "url": "https://example.com/a/1"},
            {"text": "Course B - Lesson 2", "url": "https://example.com/b/2"},
            {"text": "Course C", "url": None}
        ]
        rag_system.tool_manager.get_last_sources = Mock(return_value=multiple_sources)
        rag_system.tool_manager.reset_sources = Mock()
        
        response, sources = rag_system.query("Complex question")
        
        assert response == "Comprehensive answer"
        assert len(sources) == 3
        assert sources[0]["text"] == "Course A - Lesson 1"
        assert sources[1]["url"] == "https://example.com/b/2"
        assert sources[2]["url"] is None
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_get_course_analytics(self, mock_doc_processor_class, mock_vector_store_class,
                                 mock_ai_gen_class, mock_session_manager_class, mock_config):
        mock_vector_store = Mock()
        mock_vector_store.get_course_count.return_value = 5
        mock_vector_store.get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3", "Course 4", "Course 5"
        ]
        mock_vector_store_class.return_value = mock_vector_store
        
        rag_system = RAGSystem(mock_config)
        analytics = rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_tool_registration(self, mock_doc_processor_class, mock_vector_store_class,
                              mock_ai_gen_class, mock_session_manager_class, mock_config):
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        rag_system = RAGSystem(mock_config)
        
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None
        assert "search_course_content" in rag_system.tool_manager.tools
        assert "get_course_outline" in rag_system.tool_manager.tools
        
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        assert len(tool_definitions) == 2
        tool_names = [t["name"] for t in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names