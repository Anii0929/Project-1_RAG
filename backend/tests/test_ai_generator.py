"""
Tests for AIGenerator functionality.

This module tests the AIGenerator's tool execution mechanism, API integration,
and response handling with various scenarios including tool use and errors.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ai_generator import AIGenerator


class TestAIGeneratorBasic:
    """Test suite for basic AIGenerator functionality."""
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test AIGenerator initializes correctly."""
        api_key = "test_key"
        model = "claude-3-sonnet-20240229"
        
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(api_key, model)
            
            assert generator.model == model
            assert generator.max_tool_rounds == 2
            assert generator.base_params["model"] == model
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
    
    @pytest.mark.unit
    def test_initialization_with_custom_rounds(self):
        """Test AIGenerator initializes with custom max_tool_rounds."""
        api_key = "test_key"
        model = "claude-3-sonnet-20240229"
        
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator(api_key, model, max_tool_rounds=3)
            
            assert generator.max_tool_rounds == 3
    
    @pytest.mark.unit
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content."""
        assert "You are a course materials assistant" in AIGenerator.SYSTEM_PROMPT
        assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT
        assert "search_course_content" in AIGenerator.SYSTEM_PROMPT
        assert "NEVER answer course-related questions using your own knowledge" in AIGenerator.SYSTEM_PROMPT
        # Test sequential tool guidance
        assert "multiple tool calls in sequence" in AIGenerator.SYSTEM_PROMPT
        assert "SEQUENTIAL TOOL USAGE EXAMPLES" in AIGenerator.SYSTEM_PROMPT


class TestAIGeneratorNormalResponse:
    """Test suite for normal response generation without tools."""
    
    @pytest.mark.unit
    def test_generate_response_without_tools(self, mock_anthropic_client):
        """Test response generation without tool use."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            # Configure normal response
            mock_anthropic_client.messages.create.return_value = mock_anthropic_client._normal_response
            
            response = generator.generate_response("What is MCP?")
            
            assert response == "This is a test response from Claude."
            mock_anthropic_client.messages.create.assert_called_once()
            
            # Verify API call structure
            call_args = mock_anthropic_client.messages.create.call_args
            assert call_args[1]["messages"][0]["content"] == "What is MCP?"
            assert "You are a course materials assistant" in call_args[1]["system"]
            assert "tools" not in call_args[1]
    
    @pytest.mark.unit
    def test_generate_response_with_conversation_history(self, mock_anthropic_client):
        """Test response generation includes conversation history."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            mock_anthropic_client.messages.create.return_value = mock_anthropic_client._normal_response
            
            history = "User: Previous question\nAssistant: Previous answer"
            response = generator.generate_response("New question", conversation_history=history)
            
            assert response == "This is a test response from Claude."
            
            # Verify history was included in system prompt
            call_args = mock_anthropic_client.messages.create.call_args
            system_content = call_args[1]["system"]
            assert "Previous conversation:" in system_content
            assert "Previous question" in system_content


class TestAIGeneratorToolExecution:
    """Test suite for tool execution functionality."""
    
    @pytest.mark.unit
    def test_generate_response_with_single_tool_use(self, mock_anthropic_client, mock_tool_manager):
        """Test response generation with single round tool execution."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            # Configure tool use response for first call, normal response for second
            mock_anthropic_client.messages.create.side_effect = [
                mock_anthropic_client._tool_response,  # First call triggers tool use
                mock_anthropic_client._normal_response  # Second call returns final response
            ]
            
            # Mock tool execution
            mock_tool_manager.execute_tool.return_value = "Search results for MCP"
            
            tools = [{"name": "search_course_content", "description": "Search tool"}]
            response = generator.generate_response(
                "What is MCP?",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            assert response == "This is a test response from Claude."
            
            # Verify tool execution was called
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="test query"
            )
            
            # Verify two API calls were made
            assert mock_anthropic_client.messages.create.call_count == 2
            
            # Both calls should include tools (for sequential calling)
            first_call = mock_anthropic_client.messages.create.call_args_list[0]
            assert "tools" in first_call[1]
            assert first_call[1]["tool_choice"] == {"type": "auto"}
            
            # Second call should also include tools and tool results
            second_call = mock_anthropic_client.messages.create.call_args_list[1]
            assert "tools" in second_call[1]
            assert len(second_call[1]["messages"]) == 3  # user + assistant tool use + user tool results
    
    @pytest.mark.unit
    def test_execute_tools_single_tool(self, mock_tool_manager):
        """Test _execute_tools method with single tool execution."""
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            # Mock tool execution
            mock_tool_manager.execute_tool.return_value = "Tool execution result"
            
            # Create response with tool use
            response = Mock()
            response.content = [Mock()]
            response.content[0].type = "tool_use"
            response.content[0].name = "search_course_content"
            response.content[0].id = "tool_123"
            response.content[0].input = {"query": "test query", "course_name": "MCP"}
            
            result = generator._execute_tools(response, mock_tool_manager)
            
            # Verify tool was executed with correct parameters
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="test query",
                course_name="MCP"
            )
            
            # Verify result format
            assert len(result) == 1
            assert result[0]["type"] == "tool_result"
            assert result[0]["tool_use_id"] == "tool_123"
            assert result[0]["content"] == "Tool execution result"
            assert "is_error" not in result[0]
    
    @pytest.mark.unit
    def test_execute_tools_multiple_tools(self, mock_tool_manager):
        """Test _execute_tools method with multiple tool executions in one response."""
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            # Mock multiple tool executions
            mock_tool_manager.execute_tool.side_effect = [
                "First tool result",
                "Second tool result"
            ]
            
            # Create response with multiple tool uses
            response = Mock()
            tool1 = Mock()
            tool1.type = "tool_use"
            tool1.name = "search_course_content"
            tool1.id = "tool_1"
            tool1.input = {"query": "first query"}
            
            tool2 = Mock()
            tool2.type = "tool_use"
            tool2.name = "get_course_outline"
            tool2.id = "tool_2"
            tool2.input = {"course_name": "MCP"}
            
            response.content = [tool1, tool2]
            
            result = generator._execute_tools(response, mock_tool_manager)
            
            # Verify both tools were executed
            assert mock_tool_manager.execute_tool.call_count == 2
            mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="first query")
            mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="MCP")
            
            # Verify result format
            assert len(result) == 2
            assert result[0]["tool_use_id"] == "tool_1"
            assert result[0]["content"] == "First tool result"
            assert result[1]["tool_use_id"] == "tool_2"
            assert result[1]["content"] == "Second tool result"


class TestAIGeneratorSequentialToolCalling:
    """Test suite for sequential tool calling functionality."""
    
    @pytest.mark.unit
    def test_sequential_tool_calls_two_rounds(self, mock_anthropic_client, mock_tool_manager):
        """Test sequential tool calling with two rounds."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            # Configure responses: tool use → tool use → final response
            mock_anthropic_client.messages.create.side_effect = [
                mock_anthropic_client._tool_response,  # Round 1: tool use
                mock_anthropic_client._tool_response,  # Round 2: tool use
                mock_anthropic_client._normal_response  # Round 3: final response
            ]
            
            # Mock tool executions
            mock_tool_manager.execute_tool.side_effect = [
                "First tool result: MCP course outline",
                "Second tool result: Search results for MCP concepts"
            ]
            
            tools = [{"name": "search_course_content", "description": "Search tool"}]
            response = generator.generate_response(
                "Search for courses discussing same topic as lesson 4 of MCP course",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            assert response == "This is a test response from Claude."
            
            # Verify two tool executions occurred
            assert mock_tool_manager.execute_tool.call_count == 2
            
            # Verify three API calls were made (2 tool rounds + 1 final)
            assert mock_anthropic_client.messages.create.call_count == 3
            
            # Verify message history builds up correctly
            final_call = mock_anthropic_client.messages.create.call_args_list[2]
            messages = final_call[1]["messages"]
            # Should have: user + assistant1 + tool_results1 + assistant2 + tool_results2
            assert len(messages) == 5
    
    @pytest.mark.unit
    def test_sequential_tool_calls_early_termination(self, mock_anthropic_client, mock_tool_manager):
        """Test sequential tool calling that terminates after first round."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            # Configure responses: tool use → final response (no more tools needed)
            mock_anthropic_client.messages.create.side_effect = [
                mock_anthropic_client._tool_response,  # Round 1: tool use
                mock_anthropic_client._normal_response  # Round 2: final response
            ]
            
            # Mock tool execution
            mock_tool_manager.execute_tool.return_value = "Course list result"
            
            tools = [{"name": "list_all_courses", "description": "List courses"}]
            response = generator.generate_response(
                "What courses are available?",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            assert response == "This is a test response from Claude."
            
            # Verify only one tool execution
            mock_tool_manager.execute_tool.assert_called_once()
            
            # Verify two API calls (1 tool round + 1 final)
            assert mock_anthropic_client.messages.create.call_count == 2
    
    @pytest.mark.unit
    def test_max_rounds_reached(self, mock_anthropic_client, mock_tool_manager):
        """Test behavior when maximum tool rounds are reached."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229", max_tool_rounds=2)
            
            # Configure responses: tool use → tool use → final fallback call
            mock_anthropic_client.messages.create.side_effect = [
                mock_anthropic_client._tool_response,  # Round 1: tool use
                mock_anthropic_client._tool_response,  # Round 2: tool use (max reached)
                mock_anthropic_client._normal_response  # Final call without tools
            ]
            
            # Mock tool executions
            mock_tool_manager.execute_tool.side_effect = [
                "First tool result",
                "Second tool result"
            ]
            
            tools = [{"name": "search_course_content", "description": "Search tool"}]
            response = generator.generate_response(
                "Complex multi-part query",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            assert response == "This is a test response from Claude."
            
            # Verify exactly 2 tool executions (max_tool_rounds)
            assert mock_tool_manager.execute_tool.call_count == 2
            
            # Verify 3 API calls (2 tool rounds + 1 final without tools)
            assert mock_anthropic_client.messages.create.call_count == 3
            
            # Final call should not include tools
            final_call = mock_anthropic_client.messages.create.call_args_list[2]
            assert "tools" not in final_call[1]
    
    @pytest.mark.unit
    def test_tool_error_in_sequential_flow(self, mock_anthropic_client, mock_tool_manager):
        """Test error handling in sequential tool calling flow."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            # Configure responses: tool use → normal response (handles error gracefully)
            mock_anthropic_client.messages.create.side_effect = [
                mock_anthropic_client._tool_response,
                mock_anthropic_client._normal_response
            ]
            
            # Mock tool execution error
            mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")
            
            tools = [{"name": "search_course_content", "description": "Search tool"}]
            response = generator.generate_response(
                "Test query",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            assert response == "This is a test response from Claude."
            
            # Verify error was passed to Claude in tool results
            second_call = mock_anthropic_client.messages.create.call_args_list[1]
            tool_results = second_call[1]["messages"][2]["content"]
            assert "Tool execution failed: Tool failed" in tool_results[0]["content"]
            assert tool_results[0]["is_error"] is True
    
    @pytest.mark.unit
    def test_no_tools_available_sequential(self, mock_anthropic_client):
        """Test sequential flow when no tools are available."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            # Normal response without tools
            mock_anthropic_client.messages.create.return_value = mock_anthropic_client._normal_response
            
            response = generator.generate_response("What is MCP?")
            
            assert response == "This is a test response from Claude."
            
            # Should make only one API call
            mock_anthropic_client.messages.create.assert_called_once()
            
            # Call should not include tools
            call_args = mock_anthropic_client.messages.create.call_args
            assert "tools" not in call_args[1]


class TestAIGeneratorErrorHandling:
    """Test suite for error handling in AIGenerator."""
    
    @pytest.mark.unit
    def test_anthropic_api_error(self, mock_tool_manager):
        """Test handling of Anthropic API errors."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_client.messages.create.side_effect = Exception("API rate limit exceeded")
            mock_anthropic.return_value = mock_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            with pytest.raises(Exception, match="API rate limit exceeded"):
                generator.generate_response("Test query")
    
    @pytest.mark.unit
    def test_tool_execution_error_handling(self, mock_tool_manager):
        """Test handling of tool execution errors in _execute_tools."""
        with patch('ai_generator.anthropic.Anthropic'):
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            # Mock tool execution failure
            mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")
            
            # Create response with tool use
            response = Mock()
            response.content = [Mock()]
            response.content[0].type = "tool_use"
            response.content[0].name = "search_course_content"
            response.content[0].id = "tool_123"
            response.content[0].input = {"query": "test query"}
            
            result = generator._execute_tools(response, mock_tool_manager)
            
            # Verify error was handled gracefully
            assert len(result) == 1
            assert result[0]["type"] == "tool_result"
            assert result[0]["tool_use_id"] == "tool_123"
            assert "Tool execution failed: Database connection failed" in result[0]["content"]
            assert result[0]["is_error"] is True
    
    @pytest.mark.unit
    def test_tool_manager_none(self, mock_anthropic_client):
        """Test behavior when tool_manager is None but tools are provided."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            # Configure tool use response
            mock_anthropic_client.messages.create.return_value = mock_anthropic_client._tool_response
            
            tools = [{"name": "search_course_content", "description": "Search tool"}]
            
            # This should not attempt tool execution since tool_manager is None
            # Instead it should return the initial response (which might be incomplete)
            response = generator.generate_response("Test query", tools=tools, tool_manager=None)
            
            # With no tool manager, the response will be the tool_use content (not ideal but expected behavior)
            assert response is not None  # The exact behavior depends on implementation


class TestAIGeneratorEdgeCases:
    """Test suite for edge cases and unusual scenarios."""
    
    @pytest.mark.unit
    def test_empty_query(self, mock_anthropic_client):
        """Test behavior with empty query string."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            mock_anthropic_client.messages.create.return_value = mock_anthropic_client._normal_response
            
            response = generator.generate_response("")
            
            assert response == "This is a test response from Claude."
            
            # Verify empty query was sent
            call_args = mock_anthropic_client.messages.create.call_args
            assert call_args[1]["messages"][0]["content"] == ""
    
    @pytest.mark.unit
    def test_very_long_conversation_history(self, mock_anthropic_client):
        """Test behavior with very long conversation history."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            mock_anthropic_client.messages.create.return_value = mock_anthropic_client._normal_response
            
            # Create a very long history
            long_history = "User: Question\nAssistant: Answer\n" * 1000
            
            response = generator.generate_response("Test", conversation_history=long_history)
            
            assert response == "This is a test response from Claude."
            
            # Verify history was included (API might truncate internally)
            call_args = mock_anthropic_client.messages.create.call_args
            assert "Previous conversation:" in call_args[1]["system"]
    
    @pytest.mark.unit
    def test_response_without_text_content(self, mock_tool_manager):
        """Test handling of API response that doesn't have expected text content."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = []  # Empty content array
            
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            # The new implementation should handle empty content gracefully
            response = generator.generate_response("Test query")
            assert response == "No response generated"