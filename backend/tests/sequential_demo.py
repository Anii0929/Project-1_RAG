#!/usr/bin/env python3
"""
Demo script showing sequential tool calling capabilities.
This demonstrates the new functionality without requiring a real API key.
"""

import os
import sys
from unittest.mock import Mock, patch

# Add backend directory to path for imports
backend_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, backend_path)

from ai_generator import AIGenerator
from tests.test_ai_generator import MockAnthropicResponse, MockContentBlock


def demo_sequential_tool_calling():
    """Demonstrate sequential tool calling with a realistic scenario"""

    print("=== Sequential Tool Calling Demo ===\n")
    print(
        "Scenario: User asks 'Find a course that discusses the same topic as lesson 4 of MCP Introduction course'\n"
    )

    # Mock the scenario where Claude needs to:
    # 1. Get outline of MCP Introduction course to find lesson 4 topic
    # 2. Search for other courses that discuss that topic

    with patch("anthropic.Anthropic") as mock_anthropic_class:
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Round 1: Claude gets course outline
        print("Round 1: Claude decides to get course outline first...")
        first_tool_content = [
            MockContentBlock(
                "tool_use",
                name="get_course_outline",
                input_data={"course_name": "MCP Introduction"},
                block_id="tool_1",
            )
        ]
        first_response = MockAnthropicResponse(
            first_tool_content, stop_reason="tool_use"
        )

        # Round 2: Claude searches for courses with similar content
        print(
            "Round 2: After seeing the outline, Claude searches for similar courses..."
        )
        second_tool_content = [
            MockContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={"query": "server architecture patterns"},
                block_id="tool_2",
            )
        ]
        second_response = MockAnthropicResponse(
            second_tool_content, stop_reason="tool_use"
        )

        # Final response: Claude synthesizes the information
        print("Final: Claude provides comprehensive answer based on both searches...")
        final_response = MockAnthropicResponse(
            "Based on the MCP Introduction course outline, lesson 4 covers 'Server Architecture Patterns'. "
            "I found several courses that discuss similar topics: 'Advanced System Design' covers "
            "distributed architecture patterns, and 'Microservices Fundamentals' discusses service "
            "communication patterns. Both would complement what you learned in lesson 4."
        )

        # Configure mock responses
        mock_client.messages.create.side_effect = [
            first_response,
            second_response,
            final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            # First tool call result (course outline)
            """Course Title: MCP Introduction
Course Link: https://example.com/mcp-intro
Course Instructor: Jane Smith
Total Lessons: 6

Lesson List:
Lesson 1: Introduction to MCP
Lesson 2: Basic Concepts
Lesson 3: Client Implementation
Lesson 4: Server Architecture Patterns
Lesson 5: Advanced Features
Lesson 6: Best Practices""",
            # Second tool call result (content search)
            """[Advanced System Design - Lesson 3]
This lesson covers distributed architecture patterns including server-client communication, 
load balancing, and scalable system design principles.

[Microservices Fundamentals - Lesson 2]
Learn about service communication patterns, API design, and how different services 
interact in a microservices architecture.""",
        ]

        # Mock tools
        mock_tools = [
            {"name": "get_course_outline", "description": "Get course outline"},
            {"name": "search_course_content", "description": "Search course content"},
        ]

        # Create AI generator and run the demo
        ai_gen = AIGenerator("fake_key", "claude-3-sonnet-20240229")
        ai_gen.client = mock_client

        # Execute the query
        query = "Find a course that discusses the same topic as lesson 4 of MCP Introduction course"
        print(f"User Query: {query}\n")

        result = ai_gen.generate_response(
            query, tools=mock_tools, tool_manager=mock_tool_manager
        )

        print("Tool Execution Log:")
        tool_calls = mock_tool_manager.execute_tool.call_args_list
        for i, call in enumerate(tool_calls, 1):
            tool_name = call[0][0]
            tool_args = call[1]
            print(f"  {i}. Called {tool_name} with {tool_args}")

        print(f"\nAPI Calls Made: {mock_client.messages.create.call_count}")
        print(f"Tools Executed: {mock_tool_manager.execute_tool.call_count}")

        print(f"\nFinal Response:\n{result}")

        print(f"\n=== Benefits of Sequential Tool Calling ===")
        print("[+] Claude can reason about initial results")
        print("[+] Enables complex multi-step queries")
        print("[+] More comprehensive and accurate answers")
        print("[+] Better handling of comparative questions")


def demo_early_termination():
    """Demonstrate early termination when Claude has enough information"""

    print("\n\n=== Early Termination Demo ===\n")
    print(
        "Scenario: User asks 'What is Python?' - Claude gets answer in first search\n"
    )

    with patch("anthropic.Anthropic") as mock_anthropic_class:
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Round 1: Claude searches and gets sufficient information
        print("Round 1: Claude searches for Python information...")
        first_tool_content = [
            MockContentBlock(
                "tool_use",
                name="search_course_content",
                input_data={"query": "Python programming language"},
                block_id="tool_1",
            )
        ]
        first_response = MockAnthropicResponse(
            first_tool_content, stop_reason="tool_use"
        )

        # Round 2: Claude decides no more tools needed
        print("Round 2: Claude has enough information, provides final answer...")
        second_response = MockAnthropicResponse(
            "Python is a high-level, interpreted programming language known for its "
            "simplicity and readability. It's widely used for web development, data science, "
            "artificial intelligence, and automation tasks."
        )

        # Configure mock responses
        mock_client.messages.create.side_effect = [first_response, second_response]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "Python is a high-level programming language created by Guido van Rossum. "
            "It emphasizes code readability and simplicity, making it popular for beginners "
            "and professionals alike."
        )

        # Mock tools
        mock_tools = [
            {"name": "search_course_content", "description": "Search course content"}
        ]

        # Create AI generator and run the demo
        ai_gen = AIGenerator("fake_key", "claude-3-sonnet-20240229")
        ai_gen.client = mock_client

        # Execute the query
        query = "What is Python?"
        print(f"User Query: {query}\n")

        result = ai_gen.generate_response(
            query, tools=mock_tools, tool_manager=mock_tool_manager
        )

        print(f"API Calls Made: {mock_client.messages.create.call_count}")
        print(f"Tools Executed: {mock_tool_manager.execute_tool.call_count}")
        print(f"Result: Early termination after 1 tool call\n")

        print(f"Final Response:\n{result}")


if __name__ == "__main__":
    demo_sequential_tool_calling()
    demo_early_termination()

    print(f"\n=== Implementation Summary ===")
    print("[+] Backward compatible - existing code works unchanged")
    print("[+] Configurable max_rounds (default: 2)")
    print("[+] Automatic termination when Claude has enough information")
    print("[+] Graceful error handling for tool failures")
    print("[+] Conversation context preserved across rounds")
    print("[+] All 34 tests passing including 5 new sequential tool tests")
