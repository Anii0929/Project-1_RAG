import os
import sys
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager


class TestRAGSystem:
    """Test suite for RAGSystem"""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        config = Mock(spec=Config)
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_chroma"
        config.EMBEDDING_MODEL = "test-model"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test-api-key"
        config.ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
        config.MAX_HISTORY = 2
        return config

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all RAG system dependencies"""
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_proc,
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
        ):

            yield {
                "document_processor": mock_doc_proc,
                "vector_store": mock_vector_store,
                "ai_generator": mock_ai_gen,
                "session_manager": mock_session_mgr,
            }

    @pytest.fixture
    def rag_system(self, mock_config, mock_dependencies):
        """Create RAGSystem with mocked dependencies"""
        system = RAGSystem(mock_config)

        # Set up mocked instances
        system.document_processor = mock_dependencies["document_processor"].return_value
        system.vector_store = mock_dependencies["vector_store"].return_value
        system.ai_generator = mock_dependencies["ai_generator"].return_value
        system.session_manager = mock_dependencies["session_manager"].return_value

        # Mock tool manager and tools
        system.tool_manager = Mock(spec=ToolManager)
        system.search_tool = Mock(spec=CourseSearchTool)
        system.outline_tool = Mock(spec=CourseOutlineTool)

        return system

    def test_init(self, mock_config, mock_dependencies):
        """Test RAGSystem initialization"""
        system = RAGSystem(mock_config)

        # Verify all components were initialized
        mock_dependencies["document_processor"].assert_called_once_with(800, 100)
        mock_dependencies["vector_store"].assert_called_once_with(
            "./test_chroma", "test-model", 5
        )
        mock_dependencies["ai_generator"].assert_called_once_with(
            "test-api-key", "claude-3-5-sonnet-20241022"
        )
        mock_dependencies["session_manager"].assert_called_once_with(2)

        # Verify tool manager exists
        assert hasattr(system, "tool_manager")
        assert hasattr(system, "search_tool")
        assert hasattr(system, "outline_tool")

    def test_add_course_document_success(self, rag_system):
        """Test successful course document addition"""
        # Mock successful document processing
        mock_course = Mock(spec=Course)
        mock_course.title = "Test Course"
        mock_chunks = [Mock(spec=CourseChunk), Mock(spec=CourseChunk)]

        rag_system.document_processor.process_course_document.return_value = (
            mock_course,
            mock_chunks,
        )

        course, chunk_count = rag_system.add_course_document("/path/to/course.txt")

        assert course is mock_course
        assert chunk_count == 2

        # Verify vector store operations
        rag_system.vector_store.add_course_metadata.assert_called_once_with(mock_course)
        rag_system.vector_store.add_course_content.assert_called_once_with(mock_chunks)

    def test_add_course_document_error(self, rag_system):
        """Test course document addition with error"""
        rag_system.document_processor.process_course_document.side_effect = Exception(
            "Processing error"
        )

        course, chunk_count = rag_system.add_course_document("/path/to/invalid.txt")

        assert course is None
        assert chunk_count == 0

    def test_add_course_folder_success(self, rag_system):
        """Test successful course folder addition"""
        # Mock file system
        with (
            patch("rag_system.os.path.exists", return_value=True),
            patch(
                "rag_system.os.listdir",
                return_value=["course1.txt", "course2.pdf", "other.xyz"],
            ),
            patch("rag_system.os.path.isfile", return_value=True),
        ):

            # Mock existing courses
            rag_system.vector_store.get_existing_course_titles.return_value = []

            # Mock document processing
            mock_course1 = Mock(spec=Course)
            mock_course1.title = "Course 1"
            mock_course2 = Mock(spec=Course)
            mock_course2.title = "Course 2"

            rag_system.document_processor.process_course_document.side_effect = [
                (mock_course1, [Mock(), Mock()]),  # 2 chunks
                (mock_course2, [Mock(), Mock(), Mock()]),  # 3 chunks
            ]

            courses, chunks = rag_system.add_course_folder("/docs")

            assert courses == 2
            assert chunks == 5

            # Verify two files were processed (txt and pdf, not xyz)
            assert rag_system.document_processor.process_course_document.call_count == 2

    def test_add_course_folder_skip_existing(self, rag_system):
        """Test skipping existing courses in folder addition"""
        with (
            patch("rag_system.os.path.exists", return_value=True),
            patch("rag_system.os.listdir", return_value=["course1.txt"]),
            patch("rag_system.os.path.isfile", return_value=True),
        ):

            # Mock existing course
            rag_system.vector_store.get_existing_course_titles.return_value = [
                "Existing Course"
            ]

            # Mock document processing returning existing course
            mock_course = Mock(spec=Course)
            mock_course.title = "Existing Course"

            rag_system.document_processor.process_course_document.return_value = (
                mock_course,
                [Mock()],
            )

            courses, chunks = rag_system.add_course_folder("/docs")

            assert courses == 0  # No new courses added
            assert chunks == 0

            # Verify course was not added to vector store
            rag_system.vector_store.add_course_metadata.assert_not_called()
            rag_system.vector_store.add_course_content.assert_not_called()

    def test_add_course_folder_nonexistent(self, rag_system):
        """Test adding course folder that doesn't exist"""
        with patch("rag_system.os.path.exists", return_value=False):
            courses, chunks = rag_system.add_course_folder("/nonexistent")

            assert courses == 0
            assert chunks == 0

    def test_query_without_session(self, rag_system):
        """Test query processing without session ID"""
        # Mock AI response
        rag_system.ai_generator.generate_response.return_value = (
            "Test response about Python"
        )

        # Mock tool manager
        rag_system.tool_manager.get_tool_definitions.return_value = [
            {"name": "search_tool"}
        ]
        rag_system.tool_manager.get_last_sources.return_value = [
            {"text": "Python Course - Lesson 1"}
        ]

        response, sources = rag_system.query("What is Python?")

        assert response == "Test response about Python"
        assert sources == [{"text": "Python Course - Lesson 1"}]

        # Verify AI generator was called correctly
        rag_system.ai_generator.generate_response.assert_called_once()
        call_args = rag_system.ai_generator.generate_response.call_args
        assert (
            "Answer this question about course materials: What is Python?"
            in call_args[1]["query"]
        )
        assert call_args[1]["conversation_history"] is None
        assert call_args[1]["tools"] == [{"name": "search_tool"}]
        assert call_args[1]["tool_manager"] is rag_system.tool_manager

    def test_query_with_session(self, rag_system):
        """Test query processing with session ID"""
        # Mock session manager
        rag_system.session_manager.get_conversation_history.return_value = (
            "Previous conversation"
        )

        # Mock AI response
        rag_system.ai_generator.generate_response.return_value = "Response with context"

        # Mock tool manager
        rag_system.tool_manager.get_tool_definitions.return_value = []
        rag_system.tool_manager.get_last_sources.return_value = []

        response, sources = rag_system.query(
            "Follow up question", session_id="session_123"
        )

        assert response == "Response with context"

        # Verify session management
        rag_system.session_manager.get_conversation_history.assert_called_once_with(
            "session_123"
        )
        rag_system.session_manager.add_exchange.assert_called_once_with(
            "session_123", "Follow up question", "Response with context"
        )

        # Verify conversation history was passed
        call_args = rag_system.ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == "Previous conversation"

    def test_query_source_management(self, rag_system):
        """Test source tracking and reset in query processing"""
        # Mock AI response
        rag_system.ai_generator.generate_response.return_value = "Test response"

        # Mock tool manager with sources
        sources = [{"text": "Source 1"}, {"text": "Source 2"}]
        rag_system.tool_manager.get_tool_definitions.return_value = []
        rag_system.tool_manager.get_last_sources.return_value = sources

        response, returned_sources = rag_system.query("Test query")

        assert returned_sources == sources

        # Verify sources were reset after retrieval
        rag_system.tool_manager.reset_sources.assert_called_once()

    def test_get_course_analytics(self, rag_system):
        """Test course analytics retrieval"""
        # Mock vector store analytics
        rag_system.vector_store.get_course_count.return_value = 5
        rag_system.vector_store.get_existing_course_titles.return_value = [
            "Course A",
            "Course B",
            "Course C",
            "Course D",
            "Course E",
        ]

        analytics = rag_system.get_course_analytics()

        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course A" in analytics["course_titles"]


class TestRAGSystemIntegration:
    """Integration tests for RAG system with more complex scenarios"""

    def test_full_query_flow_simulation(self):
        """Simulate a complete query flow to identify potential issues"""
        # Create a more realistic test setup
        config = Mock(spec=Config)
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_chroma"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "sk-test-key"
        config.ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
        config.MAX_HISTORY = 2

        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):

            system = RAGSystem(config)

            # Mock the complete query flow
            system.ai_generator.generate_response.return_value = (
                "Python is a programming language..."
            )
            system.tool_manager.get_tool_definitions.return_value = [
                {"name": "search_course_content"},
                {"name": "get_course_outline"},
            ]
            system.tool_manager.get_last_sources.return_value = [
                {"text": "Python Fundamentals - Lesson 1", "link": "http://lesson1"}
            ]

            # Execute query
            response, sources = system.query("What is Python programming?")

            # Verify the flow worked
            assert isinstance(response, str)
            assert isinstance(sources, list)
            assert len(sources) >= 0

    def test_tool_registration_completeness(self):
        """Test that all required tools are properly registered"""
        config = Mock(spec=Config)
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_chroma"
        config.EMBEDDING_MODEL = "test-model"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test-key"
        config.ANTHROPIC_MODEL = "test-model"
        config.MAX_HISTORY = 2

        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):

            system = RAGSystem(config)

            # Verify tools are registered
            assert hasattr(system, "tool_manager")
            assert hasattr(system, "search_tool")
            assert hasattr(system, "outline_tool")

            # The tool manager should have both tools registered
            # (We can't easily test this without mocking the register_tool calls)
            assert system.tool_manager is not None

    def test_error_handling_in_query(self):
        """Test error handling in query processing"""
        config = Mock(spec=Config)
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_chroma"
        config.EMBEDDING_MODEL = "test-model"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test-key"
        config.ANTHROPIC_MODEL = "test-model"
        config.MAX_HISTORY = 2

        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):

            system = RAGSystem(config)

            # Mock AI generator to raise an exception
            system.ai_generator.generate_response.side_effect = Exception("API error")
            system.tool_manager.get_tool_definitions.return_value = []

            # The query should handle the exception gracefully
            # In the current implementation, exceptions are not caught in query()
            # This test will help identify if error handling needs improvement
            with pytest.raises(Exception) as exc_info:
                system.query("Test query")

            assert "API error" in str(exc_info.value)


class TestRAGSystemDiagnostics:
    """Diagnostic tests to help identify the "query failed" issue"""

    def test_component_initialization_diagnostic(self):
        """Diagnostic test to verify all components initialize correctly"""
        config = Mock(spec=Config)
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_chroma"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "sk-test-key"
        config.ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
        config.MAX_HISTORY = 2

        print("\n=== RAG SYSTEM INITIALIZATION DIAGNOSTIC ===")

        try:
            with (
                patch("rag_system.DocumentProcessor") as mock_doc,
                patch("rag_system.VectorStore") as mock_vs,
                patch("rag_system.AIGenerator") as mock_ai,
                patch("rag_system.SessionManager") as mock_session,
            ):

                system = RAGSystem(config)

                print("✓ RAGSystem created successfully")
                print(f"✓ Document Processor: {mock_doc.called}")
                print(f"✓ Vector Store: {mock_vs.called}")
                print(f"✓ AI Generator: {mock_ai.called}")
                print(f"✓ Session Manager: {mock_session.called}")

                # Check tool initialization
                print(f"✓ Tool Manager exists: {hasattr(system, 'tool_manager')}")
                print(f"✓ Search Tool exists: {hasattr(system, 'search_tool')}")
                print(f"✓ Outline Tool exists: {hasattr(system, 'outline_tool')}")

        except Exception as e:
            print(f"✗ RAGSystem initialization failed: {e}")
            raise

        print("==========================================\n")

    def test_query_flow_diagnostic(self):
        """Diagnostic test to trace query flow and identify failure points"""
        config = Mock(spec=Config)
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_chroma"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "sk-test-key"
        config.ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
        config.MAX_HISTORY = 2

        print("\n=== QUERY FLOW DIAGNOSTIC ===")

        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):

            system = RAGSystem(config)

            # Mock components for tracing
            system.ai_generator.generate_response.return_value = "Test response"
            system.tool_manager.get_tool_definitions.return_value = [
                {"name": "search_course_content"},
                {"name": "get_course_outline"},
            ]
            system.tool_manager.get_last_sources.return_value = []

            try:
                print("1. Starting query processing...")
                response, sources = system.query("What is Python?")

                print(f"2. Query completed successfully")
                print(f"   Response type: {type(response)}")
                print(f"   Sources type: {type(sources)}")
                print(f"   Response preview: {response[:50]}...")

                # Verify method calls
                print("3. Verifying method calls...")
                print(
                    f"   AI generator called: {system.ai_generator.generate_response.called}"
                )
                print(
                    f"   Tool definitions retrieved: {system.tool_manager.get_tool_definitions.called}"
                )
                print(
                    f"   Sources retrieved: {system.tool_manager.get_last_sources.called}"
                )
                print(f"   Sources reset: {system.tool_manager.reset_sources.called}")

            except Exception as e:
                print(f"✗ Query processing failed: {e}")
                print(f"   Exception type: {type(e)}")
                raise

        print("==============================\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements
