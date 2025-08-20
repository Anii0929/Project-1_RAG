import unittest
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class TestAIGenerator(unittest.TestCase):
    """Test suite for AIGenerator tool calling functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the Anthropic client
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            self.mock_client = Mock()
            mock_anthropic.return_value = self.mock_client
            self.ai_generator = AIGenerator(api_key="test_key", model="test_model")
    
    def test_initialization(self):
        """Test AIGenerator initialization"""
        self.assertEqual(self.ai_generator.model, "test_model")
        self.assertEqual(self.ai_generator.base_params["model"], "test_model")
        self.assertEqual(self.ai_generator.base_params["temperature"], 0)
        self.assertEqual(self.ai_generator.base_params["max_tokens"], 800)
    
    def test_generate_response_without_tools(self):
        """Test generating response without tools"""
        # Arrange
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="This is a simple response")]
        self.mock_client.messages.create.return_value = mock_response
        
        # Act
        result = self.ai_generator.generate_response(
            query="What is Python?",
            conversation_history=None,
            tools=None,
            tool_manager=None
        )
        
        # Assert
        self.assertEqual(result, "This is a simple response")
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args[1]
        self.assertEqual(call_args["messages"][0]["content"], "What is Python?")
        self.assertNotIn("tools", call_args)
    
    def test_generate_response_with_tools_no_usage(self):
        """Test generating response with tools available but not used"""
        # Arrange
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Direct answer without tools")]
        self.mock_client.messages.create.return_value = mock_response
        
        mock_tools = [{"name": "search", "description": "Search tool"}]
        
        # Act
        result = self.ai_generator.generate_response(
            query="What is 2+2?",
            tools=mock_tools,
            tool_manager=Mock()
        )
        
        # Assert
        self.assertEqual(result, "Direct answer without tools")
        call_args = self.mock_client.messages.create.call_args[1]
        self.assertEqual(call_args["tools"], mock_tools)
        self.assertEqual(call_args["tool_choice"], {"type": "auto"})
    
    def test_generate_response_with_tool_usage(self):
        """Test generating response that uses tools"""
        # Arrange
        # First response - tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "Python basics"}
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_block]
        
        # Final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Based on search: Python is a programming language")]
        
        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results about Python"
        
        mock_tools = [{
            "name": "search_course_content",
            "description": "Search course content"
        }]
        
        # Act
        result = self.ai_generator.generate_response(
            query="Tell me about Python",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Assert
        self.assertEqual(result, "Based on search: Python is a programming language")
        self.assertEqual(self.mock_client.messages.create.call_count, 2)
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics"
        )
    
    def test_handle_tool_execution(self):
        """Test the _handle_tool_execution method directly"""
        # Arrange
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "get_course_outline"
        mock_tool_block.id = "tool_456"
        mock_tool_block.input = {"course_title": "Python Course"}
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]
        
        # First call after tool execution (with tools, checking for more)
        mock_second_response = Mock()
        mock_second_response.stop_reason = "end_turn"  # No more tools needed
        mock_second_response.content = [Mock(text="Course outline: Lesson 1, Lesson 2")]
        
        self.mock_client.messages.create.return_value = mock_second_response
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Course outline data"
        
        base_params = {
            "messages": [{"role": "user", "content": "Get Python course outline"}],
            "system": "System prompt",
            "tools": [{"name": "get_course_outline"}]  # Include tools in base_params
        }
        
        # Act
        result = self.ai_generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )
        
        # Assert
        self.assertEqual(result, "Course outline: Lesson 1, Lesson 2")
        mock_tool_manager.execute_tool.assert_called_once_with(
            "get_course_outline",
            course_title="Python Course"
        )
        
        # Check the API call - since stop_reason != "tool_use", it returns early
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args[1]
        self.assertIn("tools", call_args)  # Tools are still present in round 1
    
    def test_multiple_tool_calls_in_response(self):
        """Test handling multiple tool calls in a single response"""
        # Arrange
        mock_tool1 = Mock()
        mock_tool1.type = "tool_use"
        mock_tool1.name = "search_course_content"
        mock_tool1.id = "tool_1"
        mock_tool1.input = {"query": "Python"}
        
        mock_tool2 = Mock()
        mock_tool2.type = "tool_use"
        mock_tool2.name = "get_course_outline"
        mock_tool2.id = "tool_2"
        mock_tool2.input = {"course_title": "Python"}
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool1, mock_tool2]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Combined results from both tools")]
        
        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Search result 1",
            "Outline result 2"
        ]
        
        # Act
        result = self.ai_generator.generate_response(
            query="Tell me about Python course",
            tools=[{"name": "search"}, {"name": "outline"}],
            tool_manager=mock_tool_manager
        )
        
        # Assert
        self.assertEqual(result, "Combined results from both tools")
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
    
    def test_conversation_history_inclusion(self):
        """Test that conversation history is included in system prompt"""
        # Arrange
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response with history")]
        self.mock_client.messages.create.return_value = mock_response
        
        conversation_history = "User: Hello\nAssistant: Hi there!"
        
        # Act
        result = self.ai_generator.generate_response(
            query="How are you?",
            conversation_history=conversation_history
        )
        
        # Assert
        self.assertEqual(result, "Response with history")
        call_args = self.mock_client.messages.create.call_args[1]
        self.assertIn("Previous conversation:", call_args["system"])
        self.assertIn(conversation_history, call_args["system"])
    
    def test_tool_execution_exception_handling(self):
        """Test handling of exceptions during tool execution"""
        # Arrange
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_error"
        mock_tool_block.input = {"query": "test"}
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_block]
        
        # After error, system continues with error message in tool result
        mock_second_response = Mock()
        mock_second_response.stop_reason = "end_turn"
        mock_second_response.content = [Mock(text="I encountered an error while searching")]
        
        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_second_response
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        # Act - Should not raise exception, but handle it gracefully
        result = self.ai_generator.generate_response(
            query="Test query",
            tools=[{"name": "search"}],
            tool_manager=mock_tool_manager
        )
        
        # Assert - Error is handled, response is returned
        self.assertEqual(result, "I encountered an error while searching")
        self.assertEqual(self.mock_client.messages.create.call_count, 2)
        
        # Verify error message was passed to Claude
        second_call = self.mock_client.messages.create.call_args_list[1]
        messages = second_call[1]["messages"]
        # Find the tool result message
        tool_result_msg = messages[-1]
        self.assertEqual(tool_result_msg["role"], "user")
        self.assertIn("Tool execution failed", str(tool_result_msg["content"]))
    
    def test_empty_tool_results_handling(self):
        """Test handling when tool returns empty result"""
        # Arrange
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_empty"
        mock_tool_block.input = {"query": "nonexistent"}
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_block]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="No results found")]
        
        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = ""  # Empty result
        
        # Act
        result = self.ai_generator.generate_response(
            query="Find something that doesn't exist",
            tools=[{"name": "search"}],
            tool_manager=mock_tool_manager
        )
        
        # Assert
        self.assertEqual(result, "No results found")
        mock_tool_manager.execute_tool.assert_called_once()
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        # Arrange
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Test response")]
        self.mock_client.messages.create.return_value = mock_response
        
        # Act
        self.ai_generator.generate_response(query="Test")
        
        # Assert
        call_args = self.mock_client.messages.create.call_args[1]
        system_prompt = call_args["system"]
        self.assertIn("search_course_content", system_prompt)
        self.assertIn("get_course_outline", system_prompt)
        self.assertIn("Tool Usage Guidelines", system_prompt)
    
    def test_sequential_tool_calls_two_rounds(self):
        """Test sequential tool calling across two rounds"""
        # Arrange
        # Round 1: Initial tool use (get_course_outline)
        mock_tool_block_1 = Mock()
        mock_tool_block_1.type = "tool_use"
        mock_tool_block_1.name = "get_course_outline"
        mock_tool_block_1.id = "tool_001"
        mock_tool_block_1.input = {"course_title": "Python Course"}
        
        mock_response_1 = Mock()
        mock_response_1.stop_reason = "tool_use"
        mock_response_1.content = [mock_tool_block_1]
        
        # Round 2: Second tool use after seeing first results (search_course_content)
        mock_tool_block_2 = Mock()
        mock_tool_block_2.type = "tool_use"
        mock_tool_block_2.name = "search_course_content"
        mock_tool_block_2.id = "tool_002"
        mock_tool_block_2.input = {"query": "functions", "lesson_number": 4}
        
        mock_response_2 = Mock()
        mock_response_2.stop_reason = "tool_use"
        mock_response_2.content = [mock_tool_block_2]
        
        # Final response after both tool rounds
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Lesson 4 of Python Course covers functions including...")]
        
        # Set up API call sequence
        self.mock_client.messages.create.side_effect = [
            mock_response_1,  # Initial API call returns first tool use
            mock_response_2,  # Round 2 API call returns second tool use
            mock_final_response  # Final API call returns synthesized response
        ]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline: Lesson 1: Intro, Lesson 4: Functions",  # Round 1 result
            "Functions are reusable blocks of code..."  # Round 2 result
        ]
        
        mock_tools = [
            {"name": "get_course_outline"},
            {"name": "search_course_content"}
        ]
        
        # Act
        result = self.ai_generator.generate_response(
            query="What does lesson 4 of Python Course teach?",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Assert
        self.assertEqual(result, "Lesson 4 of Python Course covers functions including...")
        self.assertEqual(self.mock_client.messages.create.call_count, 3)  # 3 API calls total
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)  # 2 tool executions
        
        # Verify tool execution order
        tool_calls = mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(tool_calls[0][0][0], "get_course_outline")
        self.assertEqual(tool_calls[1][0][0], "search_course_content")
    
    def test_sequential_tool_calls_early_exit(self):
        """Test early exit when no more tools needed after first round"""
        # Arrange
        # Round 1: Tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_003"
        mock_tool_block.input = {"query": "Python basics"}
        
        mock_response_1 = Mock()
        mock_response_1.stop_reason = "tool_use"
        mock_response_1.content = [mock_tool_block]
        
        # Round 2: No more tools needed (early exit)
        mock_response_2 = Mock()
        mock_response_2.stop_reason = "end_turn"  # Not "tool_use"
        mock_response_2.content = [Mock(text="Python is a high-level programming language...")]
        
        self.mock_client.messages.create.side_effect = [
            mock_response_1,  # Initial call
            mock_response_2   # Second call - no more tools
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Python basics content found"
        
        # Act
        result = self.ai_generator.generate_response(
            query="What is Python?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Assert
        self.assertEqual(result, "Python is a high-level programming language...")
        self.assertEqual(self.mock_client.messages.create.call_count, 2)  # Only 2 calls (early exit)
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 1)  # Only 1 tool execution
    
    def test_sequential_tool_error_recovery(self):
        """Test error handling in first round continues to second round"""
        # Arrange
        # Round 1: Tool use with error
        mock_tool_block_1 = Mock()
        mock_tool_block_1.type = "tool_use"
        mock_tool_block_1.name = "get_course_outline"
        mock_tool_block_1.id = "tool_error"
        mock_tool_block_1.input = {"course_title": "NonExistent"}
        
        mock_response_1 = Mock()
        mock_response_1.stop_reason = "tool_use"
        mock_response_1.content = [mock_tool_block_1]
        
        # Round 2: Try different approach
        mock_tool_block_2 = Mock()
        mock_tool_block_2.type = "tool_use"
        mock_tool_block_2.name = "search_course_content"
        mock_tool_block_2.id = "tool_recover"
        mock_tool_block_2.input = {"query": "Python basics"}
        
        mock_response_2 = Mock()
        mock_response_2.stop_reason = "tool_use"
        mock_response_2.content = [mock_tool_block_2]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Found information about Python basics")]
        
        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_final_response
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            Exception("Course not found"),  # First tool fails
            "Python basics content"  # Second tool succeeds
        ]
        
        # Act
        result = self.ai_generator.generate_response(
            query="Tell me about Python",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Assert
        self.assertEqual(result, "Found information about Python basics")
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
    
    def test_api_call_sequence_for_two_rounds(self):
        """Test the exact sequence of API calls for 2-round scenario"""
        # Arrange
        mock_tool_1 = Mock(type="tool_use", name="tool1", id="id1", input={})
        mock_tool_2 = Mock(type="tool_use", name="tool2", id="id2", input={})
        
        mock_response_1 = Mock(stop_reason="tool_use", content=[mock_tool_1])
        mock_response_2 = Mock(stop_reason="tool_use", content=[mock_tool_2])
        mock_response_3 = Mock(content=[Mock(text="Final answer")])
        
        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_response_3
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        # Act
        result = self.ai_generator.generate_response(
            query="Complex query",
            tools=[{"name": "tool1"}, {"name": "tool2"}],
            tool_manager=mock_tool_manager
        )
        
        # Assert
        self.assertEqual(result, "Final answer")
        
        # Verify API call parameters
        api_calls = self.mock_client.messages.create.call_args_list
        
        # First call: Has tools
        self.assertIn("tools", api_calls[0][1])
        
        # Second call: Still has tools (for potential round 2)
        self.assertIn("tools", api_calls[1][1])
        
        # Third call: No tools (final synthesis)
        self.assertNotIn("tools", api_calls[2][1])
    
    def test_message_accumulation_across_rounds(self):
        """Test that messages accumulate correctly across rounds"""
        # Arrange
        mock_tool_1 = Mock(type="tool_use", name="tool1", id="id1", input={"param": "value1"})
        mock_tool_2 = Mock(type="tool_use", name="tool2", id="id2", input={"param": "value2"})
        
        mock_response_1 = Mock(stop_reason="tool_use", content=[mock_tool_1])
        mock_response_2 = Mock(stop_reason="tool_use", content=[mock_tool_2])
        mock_response_3 = Mock(content=[Mock(text="Final")])
        
        self.mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_response_3
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        # Act
        self.ai_generator.generate_response(
            query="Query",
            tools=[{"name": "tool1"}, {"name": "tool2"}],
            tool_manager=mock_tool_manager
        )
        
        # Assert - Check final API call has all accumulated messages
        final_call = self.mock_client.messages.create.call_args_list[2]
        messages = final_call[1]["messages"]
        
        # Should have: user query, assistant tool1, user result1, assistant tool2, user result2
        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[0]["role"], "user")  # Original query
        self.assertEqual(messages[1]["role"], "assistant")  # Tool 1 request
        self.assertEqual(messages[2]["role"], "user")  # Tool 1 result
        self.assertEqual(messages[3]["role"], "assistant")  # Tool 2 request
        self.assertEqual(messages[4]["role"], "user")  # Tool 2 result
    
    def test_max_rounds_limit(self):
        """Test that tool calling stops after max_rounds (2)"""
        # Arrange - Set up responses that would continue beyond 2 rounds
        mock_tool_block = Mock(type="tool_use", name="search", id="id", input={})
        mock_response_tool_use = Mock(stop_reason="tool_use", content=[mock_tool_block])
        mock_response_final = Mock(content=[Mock(text="Stopped at 2 rounds")])
        
        # Even if we set up more tool responses, it should stop at 2
        self.mock_client.messages.create.side_effect = [
            mock_response_tool_use,  # Round 1
            mock_response_tool_use,  # Round 2
            mock_response_final      # Final (after 2 rounds)
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # Act
        result = self.ai_generator.generate_response(
            query="Query requiring many tools",
            tools=[{"name": "search"}],
            tool_manager=mock_tool_manager
        )
        
        # Assert
        self.assertEqual(result, "Stopped at 2 rounds")
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)  # Exactly 2 rounds


if __name__ == "__main__":
    unittest.main(verbosity=2)