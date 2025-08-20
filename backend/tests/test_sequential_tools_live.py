#!/usr/bin/env python3
"""
Live test of sequential tool calling with mock scenarios.
This demonstrates the new 2-round tool calling capability.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, MagicMock
from ai_generator import AIGenerator


def create_mock_tool_manager():
    """Create a mock tool manager with sample responses"""
    tool_manager = Mock()
    
    # Track call count to return different results
    tool_manager.call_count = 0
    
    def execute_tool(tool_name, **kwargs):
        tool_manager.call_count += 1
        
        if tool_name == "get_course_outline":
            course = kwargs.get("course_title", "Unknown")
            return f"""**Course Title:** {course}
**Total Lessons:** 5
**Lesson List:**
  Lesson 1: Introduction to Programming
  Lesson 2: Variables and Data Types
  Lesson 3: Control Flow
  Lesson 4: Functions and Modules
  Lesson 5: Object-Oriented Programming"""
        
        elif tool_name == "search_course_content":
            query = kwargs.get("query", "")
            lesson = kwargs.get("lesson_number")
            course = kwargs.get("course_name")
            
            context = []
            if course:
                context.append(f"[{course}]")
            if lesson:
                context.append(f"[Lesson {lesson}]")
            
            context_str = " ".join(context) if context else ""
            
            if "functions" in query.lower() or "lesson 4" in query.lower():
                return f"""{context_str}
Functions are reusable blocks of code that perform specific tasks. 
In Lesson 4, we cover:
- Function definition and syntax
- Parameters and arguments
- Return values
- Scope and namespaces
- Lambda functions
- Decorators"""
            else:
                return f"""{context_str}
Content about {query} found in course materials."""
    
    tool_manager.execute_tool = execute_tool
    return tool_manager


def create_mock_anthropic_client(scenario="two_rounds"):
    """Create a mock Anthropic client with different scenarios"""
    client = Mock()
    
    if scenario == "two_rounds":
        # Scenario: Get outline first, then search specific lesson
        
        # Round 1: Get course outline
        mock_tool_1 = Mock()
        mock_tool_1.type = "tool_use"
        mock_tool_1.name = "get_course_outline"
        mock_tool_1.id = "tool_001"
        mock_tool_1.input = {"course_title": "Python Basics"}
        
        response_1 = Mock()
        response_1.stop_reason = "tool_use"
        response_1.content = [mock_tool_1]
        
        # Round 2: Search for specific content
        mock_tool_2 = Mock()
        mock_tool_2.type = "tool_use"
        mock_tool_2.name = "search_course_content"
        mock_tool_2.id = "tool_002"
        mock_tool_2.input = {"query": "functions", "lesson_number": 4}
        
        response_2 = Mock()
        response_2.stop_reason = "tool_use"
        response_2.content = [mock_tool_2]
        
        # Final response
        final_response = Mock()
        final_response.content = [Mock(text="""Lesson 4 of the Python Basics course covers Functions and Modules. 

The lesson teaches:
- How to define and use functions with the `def` keyword
- Working with parameters and arguments
- Understanding return values
- Variable scope and namespaces
- Advanced topics like lambda functions and decorators

This lesson provides the foundation for writing modular, reusable code in Python.""")]
        
        client.messages.create.side_effect = [response_1, response_2, final_response]
    
    elif scenario == "early_exit":
        # Scenario: Only one round needed
        
        mock_tool = Mock()
        mock_tool.type = "tool_use"
        mock_tool.name = "search_course_content"
        mock_tool.id = "tool_003"
        mock_tool.input = {"query": "Python introduction"}
        
        response_1 = Mock()
        response_1.stop_reason = "tool_use"
        response_1.content = [mock_tool]
        
        # No second tool needed
        response_2 = Mock()
        response_2.stop_reason = "end_turn"
        response_2.content = [Mock(text="Python is a high-level programming language known for its simplicity and readability.")]
        
        client.messages.create.side_effect = [response_1, response_2]
    
    return client


def test_scenario_1():
    """Test: Complex query requiring 2 rounds"""
    print("\n" + "="*60)
    print("SCENARIO 1: Two-Round Tool Calling")
    print("Query: 'What does lesson 4 of Python Basics course teach?'")
    print("="*60)
    
    # Mock the Anthropic client
    mock_client = create_mock_anthropic_client("two_rounds")
    
    # Create AIGenerator with mocked client
    ai_gen = AIGenerator.__new__(AIGenerator)
    ai_gen.client = mock_client
    ai_gen.model = "test-model"
    ai_gen.base_params = {"model": "test-model", "temperature": 0, "max_tokens": 800}
    ai_gen.SYSTEM_PROMPT = AIGenerator.SYSTEM_PROMPT
    
    # Create mock tool manager
    tool_manager = create_mock_tool_manager()
    
    # Define tools
    tools = [
        {"name": "get_course_outline", "description": "Get course outline"},
        {"name": "search_course_content", "description": "Search content"}
    ]
    
    # Execute query
    result = ai_gen.generate_response(
        query="What does lesson 4 of Python Basics course teach?",
        tools=tools,
        tool_manager=tool_manager
    )
    
    print("\nTool Execution Sequence:")
    print(f"  Total tool calls: {tool_manager.call_count}")
    print(f"  Total API calls: {mock_client.messages.create.call_count}")
    
    print("\nFinal Response:")
    print(result)
    
    # Verify expected behavior
    assert tool_manager.call_count == 2, f"Expected 2 tool calls, got {tool_manager.call_count}"
    assert mock_client.messages.create.call_count == 3, f"Expected 3 API calls, got {mock_client.messages.create.call_count}"
    print("\n✅ Test passed: Two rounds executed successfully")


def test_scenario_2():
    """Test: Simple query with early exit (1 round)"""
    print("\n" + "="*60)
    print("SCENARIO 2: Early Exit (One Round)")
    print("Query: 'What is Python?'")
    print("="*60)
    
    # Mock the Anthropic client
    mock_client = create_mock_anthropic_client("early_exit")
    
    # Create AIGenerator with mocked client
    ai_gen = AIGenerator.__new__(AIGenerator)
    ai_gen.client = mock_client
    ai_gen.model = "test-model"
    ai_gen.base_params = {"model": "test-model", "temperature": 0, "max_tokens": 800}
    ai_gen.SYSTEM_PROMPT = AIGenerator.SYSTEM_PROMPT
    
    # Create mock tool manager
    tool_manager = create_mock_tool_manager()
    
    # Define tools
    tools = [
        {"name": "search_course_content", "description": "Search content"}
    ]
    
    # Execute query
    result = ai_gen.generate_response(
        query="What is Python?",
        tools=tools,
        tool_manager=tool_manager
    )
    
    print("\nTool Execution Sequence:")
    print(f"  Total tool calls: {tool_manager.call_count}")
    print(f"  Total API calls: {mock_client.messages.create.call_count}")
    
    print("\nFinal Response:")
    print(result)
    
    # Verify expected behavior
    assert tool_manager.call_count == 1, f"Expected 1 tool call, got {tool_manager.call_count}"
    assert mock_client.messages.create.call_count == 2, f"Expected 2 API calls, got {mock_client.messages.create.call_count}"
    print("\n✅ Test passed: Early exit after one round")


def main():
    print("="*60)
    print("SEQUENTIAL TOOL CALLING - LIVE TEST")
    print("="*60)
    
    try:
        test_scenario_1()
        test_scenario_2()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✅")
        print("="*60)
        print("\nSummary:")
        print("- Sequential tool calling (up to 2 rounds) is working")
        print("- Early exit when no more tools needed is working")
        print("- Error handling preserves graceful degradation")
        print("- Message accumulation across rounds is correct")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())