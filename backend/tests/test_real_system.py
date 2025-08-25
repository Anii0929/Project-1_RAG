"""
Real system tests to identify actual issues causing 'query failed' errors.

This module tests the actual components without extensive mocking to identify
where the real failures are occurring in the production system.
"""

import os
import shutil
import sys
import tempfile

import pytest

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import config
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from search_tools import CourseSearchTool, ToolManager
from vector_store import VectorStore


class TestRealSystemDiagnostics:
    """Test real system components to identify actual failures."""

    @pytest.mark.integration
    def test_vector_store_initialization(self):
        """Test if VectorStore can be initialized properly."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Test basic VectorStore initialization
            store = VectorStore(temp_dir, "all-MiniLM-L6-v2", max_results=5)

            assert store.course_catalog is not None
            assert store.course_content is not None
            assert store.max_results == 5

            print("✓ VectorStore initialization successful")

        except Exception as e:
            print(f"✗ VectorStore initialization failed: {e}")
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_vector_store_empty_search(self):
        """Test VectorStore search with empty database."""
        temp_dir = tempfile.mkdtemp()
        try:
            store = VectorStore(temp_dir, "all-MiniLM-L6-v2")

            results = store.search("test query")

            assert not results.error, f"Search should not error, got: {results.error}"
            assert results.is_empty(), "Empty database should return empty results"

            print("✓ VectorStore empty search works correctly")

        except Exception as e:
            print(f"✗ VectorStore empty search failed: {e}")
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_vector_store_with_sample_data(self):
        """Test VectorStore with real sample data."""
        temp_dir = tempfile.mkdtemp()
        try:
            store = VectorStore(temp_dir, "all-MiniLM-L6-v2")

            # Create sample course
            sample_course = Course(
                title="Test MCP Course",
                instructor="Test Instructor",
                course_link="https://example.com/course",
                lessons=[
                    Lesson(
                        lesson_number=0,
                        title="Introduction to MCP",
                        lesson_link="https://example.com/lesson0",
                    ),
                    Lesson(
                        lesson_number=1,
                        title="Building MCP Apps",
                        lesson_link="https://example.com/lesson1",
                    ),
                ],
            )

            # Add course metadata
            store.add_course_metadata(sample_course)

            # Create sample chunks
            sample_chunks = [
                CourseChunk(
                    content="MCP (Model Context Protocol) is a protocol for AI applications to access external data.",
                    course_title="Test MCP Course",
                    lesson_number=0,
                    chunk_index=0,
                ),
                CourseChunk(
                    content="Building MCP applications requires understanding client-server architecture.",
                    course_title="Test MCP Course",
                    lesson_number=1,
                    chunk_index=1,
                ),
            ]

            # Add course content
            store.add_course_content(sample_chunks)

            # Test search
            results = store.search("MCP protocol")

            assert not results.error, f"Search failed with error: {results.error}"
            assert not results.is_empty(), "Search should find results"
            assert len(results.documents) > 0, "Should have documents"
            assert "MCP" in results.documents[0], "Results should contain MCP"

            print(
                f"✓ VectorStore with sample data works: found {len(results.documents)} results"
            )

        except Exception as e:
            print(f"✗ VectorStore with sample data failed: {e}")
            import traceback

            traceback.print_exc()
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_course_search_tool_with_real_data(self):
        """Test CourseSearchTool with real VectorStore."""
        temp_dir = tempfile.mkdtemp()
        try:
            store = VectorStore(temp_dir, "all-MiniLM-L6-v2")
            tool = CourseSearchTool(store)

            # Test with empty database
            result = tool.execute("test query")
            assert "No relevant content found" in result
            print("✓ CourseSearchTool handles empty database correctly")

            # Add sample data
            sample_course = Course(
                title="Real MCP Course",
                instructor="Anthropic",
                lessons=[Lesson(lesson_number=0, title="MCP Introduction")],
            )

            store.add_course_metadata(sample_course)
            store.add_course_content(
                [
                    CourseChunk(
                        content="MCP enables AI applications to connect to external data sources securely.",
                        course_title="Real MCP Course",
                        lesson_number=0,
                        chunk_index=0,
                    )
                ]
            )

            # Test search with data
            result = tool.execute("MCP applications")
            assert "Real MCP Course" in result
            assert "MCP enables AI applications" in result
            print("✓ CourseSearchTool works with real data")

        except Exception as e:
            print(f"✗ CourseSearchTool test failed: {e}")
            import traceback

            traceback.print_exc()
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_rag_system_initialization_current_config(self):
        """Test RAG system initialization with current configuration."""
        try:
            print(f"Testing with AI_PROVIDER: {config.AI_PROVIDER}")

            # This should work regardless of AI provider
            rag = RAGSystem(config)

            assert rag.vector_store is not None
            assert rag.ai_generator is not None
            assert rag.tool_manager is not None
            assert rag.search_tool is not None

            # Test tool registration (now includes 3 tools after adding CourseListTool)
            tool_definitions = rag.tool_manager.get_tool_definitions()
            assert len(tool_definitions) == 3

            tool_names = [tool["name"] for tool in tool_definitions]
            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names
            assert "list_all_courses" in tool_names

            print("✓ RAG system initialization successful")

        except Exception as e:
            print(f"✗ RAG system initialization failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    @pytest.mark.integration
    def test_real_query_flow_with_search_only(self):
        """Test actual query processing with search_only provider."""
        # Temporarily override config for this test
        original_provider = config.AI_PROVIDER
        config.AI_PROVIDER = "search_only"

        try:
            rag = RAGSystem(config)

            # Test with empty database
            response, sources = rag.query("What is MCP?")

            # Should not crash, should return some response
            assert isinstance(response, str)
            assert isinstance(sources, list)

            print(f"✓ Query with empty database returned: '{response[:100]}...'")

        except Exception as e:
            print(f"✗ Real query flow failed: {e}")
            import traceback

            traceback.print_exc()
            raise
        finally:
            config.AI_PROVIDER = original_provider

    @pytest.mark.integration
    def test_check_existing_course_data(self):
        """Check if there's existing course data in the system."""
        try:
            # Use the actual configured ChromaDB path
            store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL)

            course_count = store.get_course_count()
            course_titles = store.get_existing_course_titles()

            print(f"Found {course_count} existing courses:")
            for title in course_titles:
                print(f"  - {title}")

            if course_count == 0:
                print("⚠️  No courses found in database - this could be the issue!")
            else:
                # Test search on existing data
                results = store.search("MCP protocol")
                if results.error:
                    print(f"⚠️  Search error on existing data: {results.error}")
                elif results.is_empty():
                    print("⚠️  Search returned no results on existing data")
                else:
                    print(
                        f"✓ Search found {len(results.documents)} results in existing data"
                    )

        except Exception as e:
            print(f"✗ Failed to check existing course data: {e}")
            import traceback

            traceback.print_exc()
            raise
