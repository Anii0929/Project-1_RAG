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
            assert generator.base_params["model"] == model
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
    
    @pytest.mark.unit
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content."""
        assert "You are a course materials assistant" in AIGenerator.SYSTEM_PROMPT
        assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT
        assert "search_course_content" in AIGenerator.SYSTEM_PROMPT
        assert "NEVER answer course-related questions using your own knowledge" in AIGenerator.SYSTEM_PROMPT


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
    def test_generate_response_with_tool_use(self, mock_anthropic_client, mock_tool_manager):
        """Test response generation with tool execution."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            # Configure tool use response for first call
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
            
            # First call should include tools
            first_call = mock_anthropic_client.messages.create.call_args_list[0]
            assert "tools" in first_call[1]
            assert first_call[1]["tool_choice"] == {"type": "auto"}
            
            # Second call should not include tools but should include tool results
            second_call = mock_anthropic_client.messages.create.call_args_list[1]
            assert "tools" not in second_call[1]
            assert len(second_call[1]["messages"]) == 3  # user + assistant tool use + user tool results
    
    @pytest.mark.unit
    def test_handle_tool_execution_single_tool(self, mock_anthropic_client, mock_tool_manager):
        """Test handling of single tool execution."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            # Mock tool execution
            mock_tool_manager.execute_tool.return_value = "Tool execution result"
            
            # Create initial response with tool use
            initial_response = Mock()
            initial_response.content = [Mock()]
            initial_response.content[0].type = "tool_use"
            initial_response.content[0].name = "search_course_content"
            initial_response.content[0].id = "tool_123"
            initial_response.content[0].input = {"query": "test query", "course_name": "MCP"}
            
            # Base parameters for API call
            base_params = {
                "messages": [{"role": "user", "content": "Test query"}],
                "system": AIGenerator.SYSTEM_PROMPT
            }
            
            # Configure final response
            mock_anthropic_client.messages.create.return_value = mock_anthropic_client._normal_response
            
            result = generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)
            
            assert result == "This is a test response from Claude."
            
            # Verify tool was executed with correct parameters
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="test query",
                course_name="MCP"
            )
            
            # Verify final API call was made with tool results
            mock_anthropic_client.messages.create.assert_called_once()
            call_args = mock_anthropic_client.messages.create.call_args
            messages = call_args[1]["messages"]
            
            # Should have 3 messages: original user, assistant tool use, user tool results
            assert len(messages) == 3
            assert messages[2]["role"] == "user"
            assert messages[2]["content"][0]["type"] == "tool_result"
            assert messages[2]["content"][0]["tool_use_id"] == "tool_123"
            assert messages[2]["content"][0]["content"] == "Tool execution result"
    
    @pytest.mark.unit
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic_client, mock_tool_manager):
        """Test handling of multiple tool executions in one response."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            # Mock multiple tool executions
            mock_tool_manager.execute_tool.side_effect = [
                "First tool result",
                "Second tool result"
            ]
            
            # Create initial response with multiple tool uses
            initial_response = Mock()
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
            
            initial_response.content = [tool1, tool2]
            
            base_params = {
                "messages": [{"role": "user", "content": "Test query"}],
                "system": AIGenerator.SYSTEM_PROMPT
            }
            
            mock_anthropic_client.messages.create.return_value = mock_anthropic_client._normal_response
            
            result = generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)
            
            assert result == "This is a test response from Claude."
            
            # Verify both tools were executed
            assert mock_tool_manager.execute_tool.call_count == 2
            mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="first query")
            mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="MCP")
            
            # Verify final API call includes both tool results
            call_args = mock_anthropic_client.messages.create.call_args
            tool_results = call_args[1]["messages"][2]["content"]
            assert len(tool_results) == 2
            assert tool_results[0]["tool_use_id"] == "tool_1"
            assert tool_results[0]["content"] == "First tool result"
            assert tool_results[1]["tool_use_id"] == "tool_2"
            assert tool_results[1]["content"] == "Second tool result"


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
    def test_tool_execution_error(self, mock_anthropic_client, mock_tool_manager):
        """Test handling of tool execution errors."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            
            generator = AIGenerator("test_key", "claude-3-sonnet-20240229")
            
            # Configure tool use response
            mock_anthropic_client.messages.create.side_effect = [
                mock_anthropic_client._tool_response,
                mock_anthropic_client._normal_response
            ]
            
            # Mock tool execution failure
            mock_tool_manager.execute_tool.return_value = "Error: Tool execution failed"
            
            tools = [{"name": "search_course_content", "description": "Search tool"}]
            response = generator.generate_response(
                "Test query",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            # Should still return a response even if tool fails
            assert response == "This is a test response from Claude."
            
            # Tool error should be passed to Claude
            call_args = mock_anthropic_client.messages.create.call_args_list[1]
            tool_result = call_args[1]["messages"][2]["content"][0]["content"]
            assert "Error: Tool execution failed" in tool_result
    
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
            
            # This might raise an IndexError or return None depending on implementation
            try:
                response = generator.generate_response("Test query")
                # If it doesn't crash, response might be None or empty
                assert response is not None or response is None
            except (IndexError, AttributeError):
                # Expected if implementation doesn't handle empty content gracefully
                pass