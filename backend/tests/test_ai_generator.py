from unittest.mock import MagicMock, Mock, patch

import anthropic
import pytest
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test suite for AIGenerator tool calling functionality"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client for testing"""
        mock_client = Mock(spec=anthropic.Anthropic)
        return mock_client

    @pytest.fixture
    def ai_generator_with_mock(self, mock_anthropic_client, test_config):
        """AI generator with mocked client"""
        with patch("anthropic.Anthropic", return_value=mock_anthropic_client):
            generator = AIGenerator(
                api_key=test_config.ANTHROPIC_API_KEY, model=test_config.ANTHROPIC_MODEL
            )
            generator.client = mock_anthropic_client
        return generator

    def test_generate_response_without_tools(self, ai_generator_with_mock):
        """Test basic response generation without tools"""
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.stop_reason = "end_turn"
        ai_generator_with_mock.client.messages.create.return_value = mock_response

        result = ai_generator_with_mock.generate_response("What is AI?")

        assert result == "Test response"
        ai_generator_with_mock.client.messages.create.assert_called_once()

    def test_generate_response_with_tools_no_use(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test response generation with tools available but not used"""
        # Mock response without tool use
        mock_response = Mock()
        mock_response.content = [Mock(text="General knowledge response")]
        mock_response.stop_reason = "end_turn"
        ai_generator_with_mock.client.messages.create.return_value = mock_response

        tools = tool_manager.get_tool_definitions()
        result = ai_generator_with_mock.generate_response(
            "What is artificial intelligence?", tools=tools, tool_manager=tool_manager
        )

        assert result == "General knowledge response"

    def test_generate_response_with_tool_use(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test response generation that triggers tool use"""
        # Mock initial response with tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "testing"}

        initial_response = Mock()
        initial_response.content = [mock_tool_block]
        initial_response.stop_reason = "tool_use"

        # Mock final response after tool execution
        final_response = Mock()
        final_response.content = [
            Mock(text="Based on the search results, testing is important.")
        ]

        ai_generator_with_mock.client.messages.create.side_effect = [
            initial_response,
            final_response,
        ]

        tools = tool_manager.get_tool_definitions()
        result = ai_generator_with_mock.generate_response(
            "Tell me about testing", tools=tools, tool_manager=tool_manager
        )

        assert result == "Based on the search results, testing is important."
        # Should be called twice - initial and final
        assert ai_generator_with_mock.client.messages.create.call_count == 2

    def test_handle_tool_execution(self, ai_generator_with_mock, tool_manager):
        """Test tool execution handling"""
        # Create mock tool use block
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "testing"}

        initial_response = Mock()
        initial_response.content = [mock_tool_block]

        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Tool execution complete")]
        ai_generator_with_mock.client.messages.create.return_value = final_response

        base_params = {
            "messages": [{"role": "user", "content": "Tell me about testing"}],
            "system": "Test system prompt",
        }

        result = ai_generator_with_mock._handle_tool_execution(
            initial_response, base_params, tool_manager
        )

        assert result == "Tool execution complete"

    def test_tool_execution_with_multiple_tools(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test handling multiple tool calls in one response"""
        # Create multiple mock tool blocks
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "search_course_content"
        mock_tool_block1.id = "tool_123"
        mock_tool_block1.input = {"query": "testing"}

        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "search_course_content"
        mock_tool_block2.id = "tool_456"
        mock_tool_block2.input = {"query": "development"}

        initial_response = Mock()
        initial_response.content = [mock_tool_block1, mock_tool_block2]

        final_response = Mock()
        final_response.content = [Mock(text="Multiple tools executed")]
        ai_generator_with_mock.client.messages.create.return_value = final_response

        base_params = {
            "messages": [
                {"role": "user", "content": "Tell me about testing and development"}
            ],
            "system": "Test system prompt",
        }

        result = ai_generator_with_mock._handle_tool_execution(
            initial_response, base_params, tool_manager
        )

        assert result == "Multiple tools executed"

    def test_conversation_history_integration(self, ai_generator_with_mock):
        """Test that conversation history is properly integrated"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with history")]
        mock_response.stop_reason = "end_turn"
        ai_generator_with_mock.client.messages.create.return_value = mock_response

        history = "User: Hello\nAssistant: Hi there!"
        result = ai_generator_with_mock.generate_response(
            "How are you?", conversation_history=history
        )

        assert result == "Response with history"

        # Verify system prompt includes history
        call_args = ai_generator_with_mock.client.messages.create.call_args
        system_content = call_args.kwargs["system"]
        assert history in system_content

    def test_api_parameters_structure(self, ai_generator_with_mock, tool_manager):
        """Test that API parameters are structured correctly"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.stop_reason = "end_turn"
        ai_generator_with_mock.client.messages.create.return_value = mock_response

        tools = tool_manager.get_tool_definitions()
        ai_generator_with_mock.generate_response(
            "Test query", tools=tools, tool_manager=tool_manager
        )

        call_args = ai_generator_with_mock.client.messages.create.call_args
        params = call_args.kwargs

        # Verify required parameters
        assert "model" in params
        assert "messages" in params
        assert "system" in params
        assert "temperature" in params
        assert "max_tokens" in params

        # Verify tools parameters when tools are provided
        assert "tools" in params
        assert "tool_choice" in params
        assert params["tool_choice"]["type"] == "auto"

    def test_sequential_tool_calling_two_rounds(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test sequential tool calling across two rounds"""
        # Mock tool blocks for round 1
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "get_course_outline"
        mock_tool_block1.id = "tool_1"
        mock_tool_block1.input = {"course_name": "MCP Basics"}

        # Mock tool blocks for round 2
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "search_course_content"
        mock_tool_block2.id = "tool_2"
        mock_tool_block2.input = {"query": "lesson 4 topic"}

        # Round 1 response with tool use
        round1_response = Mock()
        round1_response.content = [mock_tool_block1]
        round1_response.stop_reason = "tool_use"

        # Round 2 response with tool use
        round2_response = Mock()
        round2_response.content = [mock_tool_block2]
        round2_response.stop_reason = "tool_use"

        # Final response without tool use
        final_response = Mock()
        final_response.content = [
            Mock(text="Found courses discussing the same topic as lesson 4")
        ]

        ai_generator_with_mock.client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        tools = tool_manager.get_tool_definitions()
        result = ai_generator_with_mock.generate_response(
            "Find courses discussing the same topic as lesson 4 of MCP Basics",
            tools=tools,
            tool_manager=tool_manager,
        )

        assert result == "Found courses discussing the same topic as lesson 4"
        # Should be called 3 times: round 1, round 2, final response
        assert ai_generator_with_mock.client.messages.create.call_count == 3

    def test_sequential_tool_calling_early_termination(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test that conversation stops early when Claude doesn't use tools"""
        # Round 1 response without tool use
        round1_response = Mock()
        round1_response.content = [Mock(text="This is a general knowledge question")]
        round1_response.stop_reason = "end_turn"

        ai_generator_with_mock.client.messages.create.return_value = round1_response

        tools = tool_manager.get_tool_definitions()
        result = ai_generator_with_mock.generate_response(
            "What is artificial intelligence?", tools=tools, tool_manager=tool_manager
        )

        assert result == "This is a general knowledge question"
        # Should only be called once
        assert ai_generator_with_mock.client.messages.create.call_count == 1

    def test_sequential_tool_calling_max_rounds(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test that conversation stops after max rounds (2)"""
        # Mock tool block
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "testing"}

        # Both round 1 and round 2 responses use tools
        tool_response = Mock()
        tool_response.content = [mock_tool_block]
        tool_response.stop_reason = "tool_use"

        # Final response after max rounds
        final_response = Mock()
        final_response.content = [Mock(text="Final response after 2 tool rounds")]

        ai_generator_with_mock.client.messages.create.side_effect = [
            tool_response,
            tool_response,
            final_response,
        ]

        tools = tool_manager.get_tool_definitions()
        result = ai_generator_with_mock.generate_response(
            "Complex query requiring multiple tools",
            tools=tools,
            tool_manager=tool_manager,
        )

        assert result == "Final response after 2 tool rounds"
        # Should be called 3 times: 2 tool rounds + 1 final
        assert ai_generator_with_mock.client.messages.create.call_count == 3

    def test_sequential_tool_calling_with_tool_error(
        self, ai_generator_with_mock, tool_manager
    ):
        """Test graceful handling of tool errors in sequential calling"""
        # Mock tool block
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "testing"}

        # Round 1 with tool use
        round1_response = Mock()
        round1_response.content = [mock_tool_block]
        round1_response.stop_reason = "tool_use"

        # Final response
        final_response = Mock()
        final_response.content = [Mock(text="Response despite tool error")]

        # Mock tool manager to raise exception
        tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        ai_generator_with_mock.client.messages.create.side_effect = [
            round1_response,
            final_response,
        ]

        tools = tool_manager.get_tool_definitions()
        result = ai_generator_with_mock.generate_response(
            "Query with failing tools", tools=tools, tool_manager=tool_manager
        )

        assert result == "Response despite tool error"
        # Should continue despite tool error
        assert ai_generator_with_mock.client.messages.create.call_count == 2

    def test_error_handling_in_tool_execution(self, ai_generator_with_mock):
        """Test error handling during tool execution"""
        # Create mock tool manager that raises exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "testing"}

        initial_response = Mock()
        initial_response.content = [mock_tool_block]

        final_response = Mock()
        final_response.content = [Mock(text="Error handled")]
        ai_generator_with_mock.client.messages.create.return_value = final_response

        base_params = {
            "messages": [{"role": "user", "content": "Test"}],
            "system": "Test system",
        }

        # This should not raise an exception
        try:
            result = ai_generator_with_mock._handle_tool_execution(
                initial_response, base_params, mock_tool_manager
            )
            # If we get here, the error was handled gracefully
            assert result == "Error handled"
        except Exception as e:
            # If an exception is raised, it should be handled properly
            pytest.fail(f"Tool execution error was not handled: {e}")


class TestAIGeneratorIntegration:
    """Integration tests for AIGenerator with real tool manager"""

    def test_real_tool_integration(self, tool_manager):
        """Test AIGenerator with real tools (mocking only Anthropic API)"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            # Setup mock responses for tool use scenario
            mock_tool_block = Mock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "search_course_content"
            mock_tool_block.id = "tool_123"
            mock_tool_block.input = {"query": "testing"}

            initial_response = Mock()
            initial_response.content = [mock_tool_block]
            initial_response.stop_reason = "tool_use"

            final_response = Mock()
            final_response.content = [Mock(text="Integration test complete")]

            mock_client = Mock()
            mock_client.messages.create.side_effect = [initial_response, final_response]
            mock_anthropic.return_value = mock_client

            generator = AIGenerator("test-key", "test-model")
            tools = tool_manager.get_tool_definitions()

            result = generator.generate_response(
                "Tell me about testing", tools=tools, tool_manager=tool_manager
            )

            assert result == "Integration test complete"
            assert mock_client.messages.create.call_count == 2
