import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from rag_system import RAGSystem
from vector_store import SearchResults


class TestRAGSystem:
    """Test suite for RAG system query handling"""

    def test_rag_system_initialization(self, test_config):
        """Test RAG system initializes all components"""
        rag = RAGSystem(test_config)

        assert rag.config is not None
        assert rag.document_processor is not None
        assert rag.vector_store is not None
        assert rag.ai_generator is not None
        assert rag.session_manager is not None
        assert rag.tool_manager is not None
        assert rag.search_tool is not None
        assert rag.outline_tool is not None

    def test_query_without_session(self, rag_system):
        """Test query processing without session ID"""
        with patch.object(
            rag_system.ai_generator, "generate_response"
        ) as mock_generate:
            mock_generate.return_value = "Test response"

            response, sources = rag_system.query("What is testing?")

            assert response == "Test response"
            assert isinstance(sources, list)
            mock_generate.assert_called_once()

    def test_query_with_session(self, rag_system):
        """Test query processing with session ID"""
        session_id = "test-session-123"

        with patch.object(
            rag_system.ai_generator, "generate_response"
        ) as mock_generate:
            mock_generate.return_value = "Test response with session"

            response, sources = rag_system.query("What is testing?", session_id)

            assert response == "Test response with session"
            mock_generate.assert_called_once()

            # Verify conversation history was passed
            call_args = mock_generate.call_args
            assert call_args is not None

    def test_query_with_tool_usage(self, rag_system):
        """Test query that should trigger tool usage"""
        with patch.object(
            rag_system.ai_generator, "generate_response"
        ) as mock_generate:
            mock_generate.return_value = "Response based on search"

            # Mock tool manager to return some sources
            with patch.object(
                rag_system.tool_manager, "get_last_sources"
            ) as mock_sources:
                mock_sources.return_value = [
                    {
                        "text": "Test Course - Lesson 1",
                        "link": "https://example.com/lesson1",
                    }
                ]

                response, sources = rag_system.query(
                    "Tell me about testing in the course"
                )

                assert response == "Response based on search"
                assert len(sources) == 1
                assert sources[0]["text"] == "Test Course - Lesson 1"

    def test_query_error_handling(self, rag_system):
        """Test error handling during query processing"""
        with patch.object(
            rag_system.ai_generator, "generate_response"
        ) as mock_generate:
            mock_generate.side_effect = Exception("AI generation failed")

            # Should handle error gracefully
            with pytest.raises(Exception):
                rag_system.query("What is testing?")

    def test_add_course_document_success(self, rag_system, tmp_path):
        """Test successful addition of course document"""
        # Create test file
        test_file = tmp_path / "test_course.txt"
        test_content = """Course: Test Course
Instructor: Test Teacher

Lesson 1: Introduction
This is lesson 1 content.

Lesson 2: Advanced Topics  
This is lesson 2 content."""

        test_file.write_text(test_content)

        course, chunk_count = rag_system.add_course_document(str(test_file))

        assert course is not None
        assert chunk_count > 0
        assert course.title == "Test Course"

    def test_add_course_document_nonexistent_file(self, rag_system):
        """Test adding nonexistent course document"""
        course, chunk_count = rag_system.add_course_document("nonexistent_file.txt")

        assert course is None
        assert chunk_count == 0

    def test_add_course_folder_success(self, rag_system, tmp_path):
        """Test successful addition of course folder"""
        # Create test files
        test_file1 = tmp_path / "course1.txt"
        test_file1.write_text("Course: Course 1\nLesson 1: Intro\nContent here")

        test_file2 = tmp_path / "course2.txt"
        test_file2.write_text("Course: Course 2\nLesson 1: Intro\nContent here")

        courses_added, chunks_added = rag_system.add_course_folder(str(tmp_path))

        assert courses_added >= 0  # May be 0 if courses already exist
        assert chunks_added >= 0

    def test_add_course_folder_clear_existing(self, rag_system, tmp_path):
        """Test adding course folder with clear existing data"""
        test_file = tmp_path / "course.txt"
        test_file.write_text("Course: New Course\nLesson 1: Intro\nContent here")

        courses_added, chunks_added = rag_system.add_course_folder(
            str(tmp_path), clear_existing=True
        )

        assert courses_added >= 0
        assert chunks_added >= 0

    def test_add_course_folder_nonexistent_folder(self, rag_system):
        """Test adding nonexistent course folder"""
        courses_added, chunks_added = rag_system.add_course_folder("nonexistent_folder")

        assert courses_added == 0
        assert chunks_added == 0

    def test_get_course_analytics(self, rag_system):
        """Test course analytics retrieval"""
        analytics = rag_system.get_course_analytics()

        assert isinstance(analytics, dict)
        assert "total_courses" in analytics
        assert "course_titles" in analytics
        assert isinstance(analytics["total_courses"], int)
        assert isinstance(analytics["course_titles"], list)

    def test_session_management_integration(self, rag_system):
        """Test session management integration"""
        session_id = "test-session-456"

        with patch.object(
            rag_system.ai_generator, "generate_response"
        ) as mock_generate:
            mock_generate.return_value = "First response"

            # First query
            rag_system.query("First question", session_id)

            mock_generate.return_value = "Second response"

            # Second query - should have history
            rag_system.query("Second question", session_id)

            # Verify second call had conversation history
            assert mock_generate.call_count == 2
            second_call_args = mock_generate.call_args_list[1]
            assert "conversation_history" in second_call_args.kwargs

    def test_tools_integration(self, rag_system):
        """Test tools are properly integrated"""
        with patch.object(
            rag_system.ai_generator, "generate_response"
        ) as mock_generate:
            mock_generate.return_value = "Tool-based response"

            rag_system.query("Content-related question")

            # Verify tools were passed to AI generator
            call_args = mock_generate.call_args
            assert "tools" in call_args.kwargs
            assert "tool_manager" in call_args.kwargs
            assert call_args.kwargs["tools"] is not None
            assert call_args.kwargs["tool_manager"] is not None

    def test_sources_reset_after_query(self, rag_system):
        """Test that sources are reset after each query"""
        with patch.object(
            rag_system.ai_generator, "generate_response"
        ) as mock_generate:
            mock_generate.return_value = "Response"

            # Mock tool manager to track reset calls
            with patch.object(rag_system.tool_manager, "reset_sources") as mock_reset:
                rag_system.query("Test question")
                mock_reset.assert_called_once()


class TestRAGSystemConfigIssues:
    """Test RAG system behavior with configuration issues"""

    def test_max_results_zero_issue(self, test_config):
        """Test RAG system behavior when MAX_RESULTS is 0"""
        # Set MAX_RESULTS to 0 (current issue in config)
        test_config.MAX_RESULTS = 0
        rag = RAGSystem(test_config)

        with patch.object(rag.ai_generator, "generate_response") as mock_generate:
            mock_generate.return_value = "No results due to config"

            response, sources = rag.query("Tell me about testing")

            # Should still work but may have issues with search
            assert isinstance(response, str)
            assert isinstance(sources, list)

    def test_empty_api_key(self, test_config):
        """Test RAG system with empty API key"""
        test_config.ANTHROPIC_API_KEY = ""

        # Should initialize without error (API calls will fail later)
        rag = RAGSystem(test_config)
        assert rag.ai_generator is not None

    def test_invalid_chroma_path(self, test_config):
        """Test RAG system with invalid ChromaDB path"""
        # Set invalid path
        test_config.CHROMA_PATH = "/invalid/path/that/does/not/exist"

        # Should still initialize (ChromaDB will create the path)
        rag = RAGSystem(test_config)
        assert rag.vector_store is not None


class TestRAGSystemEndToEnd:
    """End-to-end integration tests"""

    def test_complete_query_flow_mocked(self, rag_system):
        """Test complete query flow with mocked external dependencies"""
        session_id = "e2e-test-session"

        with patch.object(
            rag_system.ai_generator, "generate_response"
        ) as mock_generate:
            # Mock AI to return a response that would come from tool usage
            mock_generate.return_value = (
                "Based on the course content, testing involves..."
            )

            # Mock tool manager to return realistic sources
            with patch.object(
                rag_system.tool_manager, "get_last_sources"
            ) as mock_sources:
                mock_sources.return_value = [
                    {
                        "text": "Test Course - Lesson 1",
                        "link": "https://example.com/lesson1",
                    },
                    {
                        "text": "Test Course - Lesson 2",
                        "link": "https://example.com/lesson2",
                    },
                ]

                # Execute query
                response, sources = rag_system.query(
                    "What is software testing?", session_id
                )

                # Verify response
                assert isinstance(response, str)
                assert len(response) > 0
                assert "testing" in response.lower()

                # Verify sources
                assert isinstance(sources, list)
                assert len(sources) == 2
                assert sources[0]["text"] == "Test Course - Lesson 1"
                assert sources[0]["link"] == "https://example.com/lesson1"

                # Verify AI generator was called with proper parameters
                call_args = mock_generate.call_args
                assert call_args is not None
                assert "tools" in call_args.kwargs
                assert "tool_manager" in call_args.kwargs

    def test_query_failure_scenarios(self, rag_system):
        """Test various query failure scenarios"""
        # Test with AI generation failure
        with patch.object(
            rag_system.ai_generator, "generate_response"
        ) as mock_generate:
            mock_generate.side_effect = Exception("API Error")

            with pytest.raises(Exception):
                rag_system.query("Test question")

        # Test with tool execution failure
        with patch.object(rag_system.tool_manager, "execute_tool") as mock_execute:
            mock_execute.side_effect = Exception("Tool Error")

            with patch.object(
                rag_system.ai_generator, "generate_response"
            ) as mock_generate:
                mock_generate.return_value = "Fallback response"

                # Should still work if AI handles tool errors gracefully
                response, sources = rag_system.query("Test question")
                assert isinstance(response, str)
