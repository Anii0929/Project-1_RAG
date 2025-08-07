import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator

class TestAIGenerator:
    
    def test_initialization(self, mock_config):
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic_class:
            ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            mock_anthropic_class.assert_called_once_with(api_key=mock_config.ANTHROPIC_API_KEY)
            assert ai_gen.model == mock_config.ANTHROPIC_MODEL
            assert ai_gen.base_params["model"] == mock_config.ANTHROPIC_MODEL
            assert ai_gen.base_params["temperature"] == 0
            assert ai_gen.base_params["max_tokens"] == 800
    
    def test_generate_response_without_tools(self, mock_config, mock_anthropic_client):
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            response = ai_gen.generate_response(
                query="What is MCP?",
                conversation_history=None,
                tools=None,
                tool_manager=None
            )
            
            assert response == "This is a test response from Claude."
            
            call_args = mock_anthropic_client.messages.create.call_args
            assert call_args.kwargs["model"] == mock_config.ANTHROPIC_MODEL
            assert call_args.kwargs["messages"][0]["role"] == "user"
            assert call_args.kwargs["messages"][0]["content"] == "What is MCP?"
            assert "tools" not in call_args.kwargs
    
    def test_generate_response_with_conversation_history(self, mock_config, mock_anthropic_client):
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            response = ai_gen.generate_response(
                query="Tell me more",
                conversation_history="Previous context about MCP",
                tools=None,
                tool_manager=None
            )
            
            call_args = mock_anthropic_client.messages.create.call_args
            expected_system = AIGenerator.SYSTEM_PROMPT + "\n\nPrevious conversation:\nPrevious context about MCP"
            assert call_args.kwargs["system"] == expected_system
    
    def test_generate_response_with_tools_no_usage(self, mock_config, mock_anthropic_client, mock_tool_manager):
        tool_definitions = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]
        
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            response = ai_gen.generate_response(
                query="What is 2+2?",
                conversation_history=None,
                tools=tool_definitions,
                tool_manager=mock_tool_manager
            )
            
            assert response == "This is a test response from Claude."
            
            call_args = mock_anthropic_client.messages.create.call_args
            assert call_args.kwargs["tools"] == tool_definitions
            assert call_args.kwargs["tool_choice"] == {"type": "auto"}
    
    def test_generate_response_with_tool_usage(self, mock_config, mock_anthropic_client_with_tool_use, mock_tool_manager):
        tool_definitions = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]
        
        mock_tool_manager.execute_tool.return_value = "MCP is a powerful framework for building applications."
        
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client_with_tool_use):
            ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            response = ai_gen.generate_response(
                query="What is MCP framework?",
                conversation_history=None,
                tools=tool_definitions,
                tool_manager=mock_tool_manager
            )
            
            assert response == "Based on the search, MCP is a powerful framework."
            
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="MCP framework",
                course_name="Introduction to MCP"
            )
            
            assert mock_anthropic_client_with_tool_use.messages.create.call_count == 2
    
    def test_handle_tool_execution_single_tool(self, mock_config, mock_tool_manager):
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "neural networks"}
        
        initial_response.content = [tool_block]
        
        base_params = {
            "messages": [{"role": "user", "content": "Tell me about neural networks"}],
            "system": "System prompt",
            "model": "claude-sonnet-4-20250514",
            "temperature": 0,
            "max_tokens": 800
        }
        
        mock_tool_manager.execute_tool.return_value = "Neural networks are computational models."
        
        final_response = Mock()
        final_response.content = [Mock(text="Neural networks are fascinating computational models.")]
        
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_client.messages.create.return_value = final_response
            mock_anthropic_class.return_value = mock_client
            
            ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            result = ai_gen._handle_tool_execution(initial_response, base_params, mock_tool_manager)
            
            assert result == "Neural networks are fascinating computational models."
            
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="neural networks"
            )
            
            final_call_args = mock_client.messages.create.call_args
            assert len(final_call_args.kwargs["messages"]) == 3
            assert final_call_args.kwargs["messages"][2]["role"] == "user"
            assert final_call_args.kwargs["messages"][2]["content"][0]["type"] == "tool_result"
    
    def test_handle_tool_execution_multiple_tools(self, mock_config, mock_tool_manager):
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.id = "tool_1"
        tool_block1.input = {"query": "MCP basics"}
        
        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "get_course_outline"
        tool_block2.id = "tool_2"
        tool_block2.input = {"course_title": "Introduction to MCP"}
        
        initial_response.content = [tool_block1, tool_block2]
        
        base_params = {
            "messages": [{"role": "user", "content": "Tell me about MCP course"}],
            "system": "System prompt",
            "model": "claude-sonnet-4-20250514",
            "temperature": 0,
            "max_tokens": 800
        }
        
        mock_tool_manager.execute_tool.side_effect = [
            "MCP is a framework.",
            "Course outline: 3 lessons"
        ]
        
        final_response = Mock()
        final_response.content = [Mock(text="Combined response about MCP.")]
        
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_client.messages.create.return_value = final_response
            mock_anthropic_class.return_value = mock_client
            
            ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            result = ai_gen._handle_tool_execution(initial_response, base_params, mock_tool_manager)
            
            assert result == "Combined response about MCP."
            assert mock_tool_manager.execute_tool.call_count == 2
            
            mock_tool_manager.execute_tool.assert_any_call(
                "search_course_content",
                query="MCP basics"
            )
            mock_tool_manager.execute_tool.assert_any_call(
                "get_course_outline",
                course_title="Introduction to MCP"
            )
    
    def test_system_prompt_content(self):
        expected_keywords = [
            "search_course_content",
            "get_course_outline",
            "Tool Usage Guidelines",
            "Response Protocol",
            "Brief, Concise and focused",
            "Educational"
        ]
        
        for keyword in expected_keywords:
            assert keyword in AIGenerator.SYSTEM_PROMPT
    
    def test_generate_response_with_empty_tools_list(self, mock_config, mock_anthropic_client):
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            response = ai_gen.generate_response(
                query="Test query",
                conversation_history=None,
                tools=[],
                tool_manager=None
            )
            
            call_args = mock_anthropic_client.messages.create.call_args
            assert "tools" not in call_args.kwargs
    
    def test_handle_tool_execution_no_tool_blocks(self, mock_config, mock_tool_manager):
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Some text"
        
        initial_response.content = [text_block]
        
        base_params = {
            "messages": [{"role": "user", "content": "Query"}],
            "system": "System prompt",
            "model": "claude-sonnet-4-20250514",
            "temperature": 0,
            "max_tokens": 800
        }
        
        final_response = Mock()
        final_response.content = [Mock(text="Final response")]
        
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_client.messages.create.return_value = final_response
            mock_anthropic_class.return_value = mock_client
            
            ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            result = ai_gen._handle_tool_execution(initial_response, base_params, mock_tool_manager)
            
            assert result == "Final response"
            mock_tool_manager.execute_tool.assert_not_called()