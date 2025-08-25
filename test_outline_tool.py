#!/usr/bin/env python3
"""
Simple test script to verify that the outline tool is properly registered
and has the correct definition.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

import json
from unittest.mock import Mock


# Mock the VectorStore to avoid chromadb dependency
class MockVectorStore:
    """Mock vector store for testing tool functionality without ChromaDB dependency."""

    def _resolve_course_name(self, course_name):
        """Mock course name resolution for testing.

        Args:
            course_name: Course name to resolve.

        Returns:
            str: Mock course title if 'MCP' is in the name, None otherwise.
        """
        return (
            "MCP: Build Rich-Context AI Apps with Anthropic"
            if "MCP" in course_name
            else None
        )

    @property
    def course_catalog(self):
        """Mock course catalog collection for testing.

        Returns:
            Mock: Mock collection object with predefined course metadata.
        """
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "metadatas": [
                {
                    "title": "MCP: Build Rich-Context AI Apps with Anthropic",
                    "instructor": "Anthropic",
                    "course_link": "https://example.com/mcp-course",
                    "lessons_json": json.dumps(
                        [
                            {
                                "lesson_number": 0,
                                "lesson_title": "Introduction to MCP",
                                "lesson_link": None,
                            },
                            {
                                "lesson_number": 1,
                                "lesson_title": "Building Apps",
                                "lesson_link": None,
                            },
                        ]
                    ),
                }
            ]
        }
        return mock_collection


def test_tool_definitions():
    """Test that both search and outline tools are properly defined and functional.

    Verifies that tools can be registered, provide valid definitions,
    and execute without errors using mock data.
    """
    from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager

    # Create tools with mock vector store
    mock_store = MockVectorStore()
    tm = ToolManager()
    search_tool = CourseSearchTool(mock_store)
    outline_tool = CourseOutlineTool(mock_store)

    tm.register_tool(search_tool)
    tm.register_tool(outline_tool)

    # Get tool definitions
    tools = tm.get_tool_definitions()

    print("=== TOOL DEFINITIONS ===")
    for tool in tools:
        print(json.dumps(tool, indent=2))
        print("---")

    # Test outline tool execution
    print("=== TESTING OUTLINE TOOL ===")
    result = tm.execute_tool("get_course_outline", course_name="MCP")
    print("Outline tool result:")
    print(result)
    print("---")


def test_tool_selection_logic():
    """Test and display keyword patterns for tool selection.

    Shows which query patterns should trigger the outline tool versus
    the content search tool to help with AI prompt engineering.
    """
    outline_keywords = [
        "what is the outline of",
        "course outline",
        "course structure",
        "what lessons are in",
        "syllabus",
        "course overview",
        "what's covered in",
    ]

    content_keywords = [
        "explain how to",
        "what is MCP protocol",
        "how does client-server work",
        "implementation details",
    ]

    print("=== TOOL SELECTION GUIDANCE ===")
    print("These queries should trigger get_course_outline:")
    for keyword in outline_keywords:
        print(f"  - '{keyword} [course name]'")

    print("\nThese queries should trigger search_course_content:")
    for keyword in content_keywords:
        print(f"  - '{keyword}'")
    print("---")


if __name__ == "__main__":
    test_tool_definitions()
    test_tool_selection_logic()
    print(
        "\nIf you see tool definitions above and outline tool result, the implementation is working!"
    )
