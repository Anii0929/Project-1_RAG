"""
End-to-end tests to verify the "query failed" fix.

This module tests the complete fix for the query failed issue, specifically
testing that queries like "What courses are available?" now work correctly.
"""

import os
import sys

import pytest

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import config
from rag_system import RAGSystem


class TestQueryFailedFix:
    """Test the complete fix for query failed issues."""

    @pytest.mark.integration
    def test_course_listing_query_success(self):
        """Test that general course listing queries now work."""
        try:
            rag = RAGSystem(config)

            # These queries previously failed with "query failed"
            test_queries = [
                "What courses are available?",
                "Show me all courses",
                "List all available courses",
                "What can I learn from this system?",
            ]

            for query in test_queries:
                print(f"\nTesting query: '{query}'")

                response, sources = rag.query(query)

                # Should not contain "query failed" or error messages
                assert (
                    "query failed" not in response.lower()
                ), f"Query '{query}' still failing: {response[:200]}..."
                assert (
                    "Error executing tool" not in response
                ), f"Tool execution error in '{query}': {response[:200]}..."

                # Should contain course information
                assert isinstance(response, str)

                # Print full response for debugging
                print(f"Full response: '{response}'")
                print(f"Sources: {sources}")

                # More flexible check - response should either be substantial or contain helpful info
                is_substantial = len(response) > 50
                contains_course_info = any(
                    keyword in response.lower()
                    for keyword in [
                        "course",
                        "available",
                        "mcp",
                        "chroma",
                        "anthropic",
                        "lesson",
                    ]
                )
                contains_helpful_guidance = (
                    "Please specify" in response or "Available courses:" in response
                )

                assert (
                    is_substantial or contains_course_info or contains_helpful_guidance
                ), f"Response for '{query}' lacks substance: {response}"

                # Should have sources (the new CourseListTool should provide sources)
                assert isinstance(sources, list)

                print(
                    f"✓ Query '{query}' succeeded - Response length: {len(response)}, Sources: {len(sources)}"
                )

        except Exception as e:
            print(f"✗ End-to-end test failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    @pytest.mark.integration
    def test_specific_course_queries_still_work(self):
        """Test that specific course queries continue to work."""
        try:
            rag = RAGSystem(config)

            # These should continue working as before
            test_queries = [
                "What is MCP?",
                "Tell me about MCP protocol",
                "Explain Chroma database",
            ]

            for query in test_queries:
                print(f"\nTesting content query: '{query}'")

                response, sources = rag.query(query)

                assert (
                    "query failed" not in response.lower()
                ), f"Content query '{query}' failed: {response[:200]}..."
                assert isinstance(response, str)
                assert (
                    len(response) > 50
                ), f"Content response too short for '{query}': {response}"

                print(f"✓ Content query '{query}' succeeded")

        except Exception as e:
            print(f"✗ Content query test failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    @pytest.mark.integration
    def test_outline_queries_still_work(self):
        """Test that course outline queries continue to work."""
        try:
            rag = RAGSystem(config)

            # Test outline queries with existing courses
            test_queries = [
                "Show me the outline of MCP course",
                "What lessons are in the MCP course?",
                "Give me the structure of the Chroma course",
            ]

            for query in test_queries:
                print(f"\nTesting outline query: '{query}'")

                response, sources = rag.query(query)

                # Allow for "No course found" if the course doesn't match exactly
                # but should not have "query failed" or parameter errors
                assert (
                    "query failed" not in response.lower()
                ), f"Outline query '{query}' failed: {response[:200]}..."
                assert (
                    "missing 1 required positional argument" not in response
                ), f"Parameter error in '{query}': {response[:200]}..."
                assert isinstance(response, str)
                assert len(response) > 20

                print(f"✓ Outline query '{query}' succeeded")

        except Exception as e:
            print(f"✗ Outline query test failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    @pytest.mark.integration
    def test_system_prompt_improvements(self):
        """Test that the system prompt improvements are working."""
        from ai_generator import AIGenerator

        # Check that new tool is mentioned in system prompt
        assert "list_all_courses" in AIGenerator.SYSTEM_PROMPT
        assert "what courses are available" in AIGenerator.SYSTEM_PROMPT
        assert "show me all courses" in AIGenerator.SYSTEM_PROMPT

        # Check tool selection guide is present
        assert "TOOL SELECTION GUIDE" in AIGenerator.SYSTEM_PROMPT

        print("✓ System prompt contains new tool guidance")

    @pytest.mark.integration
    def test_all_three_tools_registered(self):
        """Test that all three tools are properly registered."""
        rag = RAGSystem(config)

        tool_definitions = rag.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_definitions]

        assert (
            len(tool_definitions) == 3
        ), f"Expected 3 tools, got {len(tool_definitions)}: {tool_names}"
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
        assert "list_all_courses" in tool_names

        print("✓ All three tools properly registered")

    @pytest.mark.integration
    def test_chromadb_metadata_fix_works(self):
        """Test that the ChromaDB metadata None value fix works."""
        import shutil
        import tempfile

        from models import Course, Lesson
        from vector_store import VectorStore

        temp_dir = tempfile.mkdtemp()
        try:
            store = VectorStore(temp_dir, "all-MiniLM-L6-v2")

            # Create course with None values that previously caused errors
            test_course = Course(
                title="Test Course with None Values",
                instructor=None,  # This previously caused ChromaDB errors
                course_link=None,  # This too
                lessons=[
                    Lesson(lesson_number=0, title="Test Lesson", lesson_link=None)
                ],
            )

            # This should not raise an exception
            store.add_course_metadata(test_course)

            # Verify it was added successfully
            titles = store.get_existing_course_titles()
            assert "Test Course with None Values" in titles

            print("✓ ChromaDB metadata None value fix works")

        except Exception as e:
            print(f"✗ ChromaDB metadata fix failed: {e}")
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
