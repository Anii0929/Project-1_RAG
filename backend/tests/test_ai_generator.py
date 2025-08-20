import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator


class MockAnthropicResponse:
    """Mock response from Anthropic API"""

    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class MockToolUseContent:
    """Mock tool use content block"""

    def __init__(self, name, input_params, tool_id="tool_123"):
        self.type = "tool_use"
        self.name = name
        self.input = input_params
        self.id = tool_id


class MockTextContent:
    """Mock text content block"""

    def __init__(self, text):
        self.type = "text"
        self.text = text


class TestAIGenerator:
    """Test suite for AIGenerator"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create mock Anthropic client"""
        return Mock()

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create AIGenerator with mocked client"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            generator = AIGenerator("test-api-key", "claude-3-5-sonnet-20241022")
            generator.client = mock_anthropic_client
            return generator

    def test_init(self):
        """Test AIGenerator initialization"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            generator = AIGenerator("test-key", "test-model")

            mock_anthropic.assert_called_once_with(api_key="test-key")
            assert generator.model == "test-model"
            assert generator.base_params["model"] == "test-model"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800

    def test_generate_response_without_tools(self, ai_generator, mock_anthropic_client):
        """Test basic response generation without tools"""
        # Mock API response
        mock_response = MockAnthropicResponse(
            [MockTextContent("This is a test response")]
        )
        mock_anthropic_client.messages.create.return_value = mock_response

        result = ai_generator.generate_response("What is Python?")

        assert result == "This is a test response"

        # Verify API call parameters
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["model"] == "claude-3-5-sonnet-20241022"
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 800
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["messages"][0]["role"] == "user"
        assert call_args[1]["messages"][0]["content"] == "What is Python?"

    def test_generate_response_with_conversation_history(
        self, ai_generator, mock_anthropic_client
    ):
        """Test response generation with conversation history"""
        mock_response = MockAnthropicResponse(
            [MockTextContent("Response with history")]
        )
        mock_anthropic_client.messages.create.return_value = mock_response

        history = "User: Hello\nAssistant: Hi there!"
        result = ai_generator.generate_response(
            "Follow up question", conversation_history=history
        )

        assert result == "Response with history"

        # Check that history was included in system prompt
        call_args = mock_anthropic_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:" in system_content
        assert "User: Hello" in system_content

    def test_generate_response_with_tools(self, ai_generator, mock_anthropic_client):
        """Test response generation with tools available"""
        mock_response = MockAnthropicResponse([MockTextContent("Response using tools")])
        mock_anthropic_client.messages.create.return_value = mock_response

        tools = [
            {
                "name": "search_course_content",
                "description": "Search for course content",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

        result = ai_generator.generate_response(
            "Search for Python tutorials", tools=tools
        )

        assert result == "Response using tools"

        # Verify tools were included in API call
        call_args = mock_anthropic_client.messages.create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"]["type"] == "auto"

    def test_tool_execution_flow(self, ai_generator, mock_anthropic_client):
        """Test the complete tool execution flow"""
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "Search results: Python is a programming language"
        )

        # First response with tool use
        tool_use_content = MockToolUseContent(
            name="search_course_content", input_params={"query": "Python basics"}
        )
        initial_response = MockAnthropicResponse(
            [tool_use_content], stop_reason="tool_use"
        )

        # Second response with final answer
        final_response = MockAnthropicResponse(
            [
                MockTextContent(
                    "Based on the search results, Python is a programming language..."
                )
            ]
        )

        # Set up mock responses
        mock_anthropic_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        tools = [{"name": "search_course_content", "description": "Test tool"}]

        result = ai_generator.generate_response(
            "Tell me about Python", tools=tools, tool_manager=mock_tool_manager
        )

        assert (
            result == "Based on the search results, Python is a programming language..."
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="Python basics"
        )

        # Verify two API calls were made
        assert mock_anthropic_client.messages.create.call_count == 2

    def test_tool_execution_with_multiple_tools(
        self, ai_generator, mock_anthropic_client
    ):
        """Test execution of multiple tools in one response"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Search result 1",
            "Search result 2",
        ]

        # Response with multiple tool uses
        tool_use_1 = MockToolUseContent("search_tool", {"query": "query1"}, "tool_1")
        tool_use_2 = MockToolUseContent("search_tool", {"query": "query2"}, "tool_2")

        initial_response = MockAnthropicResponse(
            [tool_use_1, tool_use_2], stop_reason="tool_use"
        )

        final_response = MockAnthropicResponse(
            [MockTextContent("Combined results from both tools")]
        )

        mock_anthropic_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        result = ai_generator.generate_response(
            "Complex query",
            tools=[{"name": "search_tool"}],
            tool_manager=mock_tool_manager,
        )

        assert result == "Combined results from both tools"
        assert mock_tool_manager.execute_tool.call_count == 2

    def test_system_prompt_content(self, ai_generator):
        """Test that system prompt contains expected guidelines"""
        prompt = ai_generator.SYSTEM_PROMPT

        # Check for key guidelines
        assert "Content Search Tool" in prompt
        assert "Course Outline Tool" in prompt
        assert "One tool call per query maximum" in prompt
        assert "Course content questions" in prompt
        assert "Course outline/structure questions" in prompt

    def test_handle_tool_execution_error(self, ai_generator, mock_anthropic_client):
        """Test handling of tool execution errors"""
        # Mock tool manager that returns error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Error: Tool execution failed"

        tool_use_content = MockToolUseContent("failing_tool", {"param": "value"})
        initial_response = MockAnthropicResponse(
            [tool_use_content], stop_reason="tool_use"
        )

        final_response = MockAnthropicResponse(
            [MockTextContent("I encountered an error while searching")]
        )

        mock_anthropic_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        result = ai_generator.generate_response(
            "Test query",
            tools=[{"name": "failing_tool"}],
            tool_manager=mock_tool_manager,
        )

        assert result == "I encountered an error while searching"

        # Verify the error was passed to the second API call
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]

        # Should have: user message, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        assert "Error: Tool execution failed" in str(messages[2]["content"])

    def test_tool_execution_message_structure(
        self, ai_generator, mock_anthropic_client
    ):
        """Test that tool execution creates proper message structure"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        tool_use_content = MockToolUseContent(
            "test_tool", {"param": "value"}, "tool_123"
        )
        initial_response = MockAnthropicResponse(
            [tool_use_content], stop_reason="tool_use"
        )
        final_response = MockAnthropicResponse([MockTextContent("Final answer")])

        mock_anthropic_client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        ai_generator.generate_response(
            "Test query", tools=[{"name": "test_tool"}], tool_manager=mock_tool_manager
        )

        # Check the structure of the second API call
        second_call = mock_anthropic_client.messages.create.call_args_list[1]
        messages = second_call[1]["messages"]

        # Should have 3 messages: user, assistant (tool use), user (tool results)
        assert len(messages) == 3

        # First message: original user query
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test query"

        # Second message: assistant's tool use response
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == [tool_use_content]

        # Third message: tool results
        assert messages[2]["role"] == "user"
        tool_result_content = messages[2]["content"]
        assert len(tool_result_content) == 1
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "tool_123"
        assert tool_result_content[0]["content"] == "Tool result"

    def test_no_tool_manager_with_tool_use(self, ai_generator, mock_anthropic_client):
        """Test behavior when tool use is requested but no tool manager provided"""
        tool_use_content = MockToolUseContent("test_tool", {"param": "value"})
        response = MockAnthropicResponse([tool_use_content], stop_reason="tool_use")
        mock_anthropic_client.messages.create.return_value = response

        # This should not call _handle_tool_execution since no tool_manager provided
        result = ai_generator.generate_response(
            "Test query",
            tools=[{"name": "test_tool"}],
            # Note: no tool_manager parameter
        )

        # Should return the tool use content directly (this might not be ideal behavior)
        # In a real scenario, this might need different handling
        assert mock_anthropic_client.messages.create.call_count == 1


class TestAIGeneratorIntegration:
    """Integration tests for AI Generator with realistic scenarios"""

    def test_realistic_content_search_scenario(self):
        """Test a realistic content search scenario"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            generator = AIGenerator("test-key", "claude-3-5-sonnet-20241022")
            generator.client = mock_client

            # Mock realistic tool use scenario
            tool_use = MockToolUseContent(
                "search_course_content",
                {"query": "Python functions", "course_name": "Python Basics"},
            )

            initial_response = MockAnthropicResponse([tool_use], stop_reason="tool_use")
            final_response = MockAnthropicResponse(
                [
                    MockTextContent(
                        "In Python, functions are defined using the 'def' keyword..."
                    )
                ]
            )

            mock_client.messages.create.side_effect = [initial_response, final_response]

            # Mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = """
            [Python Basics - Lesson 3]
            Functions in Python are defined using the 'def' keyword followed by the function name and parentheses.
            """

            tools = [
                {
                    "name": "search_course_content",
                    "description": "Search course materials",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "course_name": {"type": "string"},
                        },
                    },
                }
            ]

            result = generator.generate_response(
                "How do I create functions in Python?",
                tools=tools,
                tool_manager=mock_tool_manager,
            )

            assert "def' keyword" in result
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="Python functions",
                course_name="Python Basics",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
