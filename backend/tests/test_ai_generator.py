import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend directory to path for imports
backend_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_path)

from ai_generator import AIGenerator


class MockContentBlock:
    """Mock content block for simulating Anthropic response"""
    def __init__(self, block_type, text=None, name=None, input_data=None, block_id=None):
        self.type = block_type
        self.text = text
        self.name = name
        self.input = input_data or {}
        self.id = block_id or "mock_id"


class MockAnthropicResponse:
    """Mock Anthropic API response"""
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content if isinstance(content, list) else [MockContentBlock("text", content)]
        self.stop_reason = stop_reason


class TestAIGenerator(unittest.TestCase):
    """Test cases for AIGenerator tool calling functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.ai_generator = AIGenerator("fake_api_key", "claude-3-sonnet-20240229")
        self.mock_tool_manager = Mock()
        
        # Mock tools list
        self.mock_tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]
    
    @patch('anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic_class):
        """Test basic response generation without tool usage"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock response
        mock_response = MockAnthropicResponse("This is a direct response")
        mock_client.messages.create.return_value = mock_response
        
        # Create new instance to use mocked client
        ai_gen = AIGenerator("fake_key", "claude-3-sonnet-20240229")
        ai_gen.client = mock_client
        
        result = ai_gen.generate_response("What is Python?")
        
        # Verify response
        self.assertEqual(result, "This is a direct response")
        
        # Verify API was called correctly
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        self.assertEqual(call_args['messages'][0]['content'], "What is Python?")
        self.assertNotIn('tools', call_args)
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_class):
        """Test response generation with tools available but not used"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock response without tool use
        mock_response = MockAnthropicResponse("General knowledge response")
        mock_client.messages.create.return_value = mock_response
        
        # Create new instance to use mocked client
        ai_gen = AIGenerator("fake_key", "claude-3-sonnet-20240229")
        ai_gen.client = mock_client
        
        result = ai_gen.generate_response(
            "What is 2+2?", 
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify response
        self.assertEqual(result, "General knowledge response")
        
        # Verify API was called with tools
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        self.assertIn('tools', call_args)
        self.assertEqual(call_args['tools'], self.mock_tools)
        self.assertEqual(call_args['tool_choice'], {"type": "auto"})
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic_class):
        """Test response generation with actual tool usage"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock initial response with tool use
        tool_use_content = [
            MockContentBlock("tool_use", name="search_course_content", 
                           input_data={"query": "Python basics"}, block_id="tool_1")
        ]
        initial_response = MockAnthropicResponse(tool_use_content, stop_reason="tool_use")
        
        # Mock final response after tool execution
        final_response = MockAnthropicResponse("Based on the search results, Python is...")
        
        # Configure mock to return different responses on subsequent calls
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Mock tool manager
        self.mock_tool_manager.execute_tool.return_value = "Search results about Python"
        
        # Create new instance to use mocked client
        ai_gen = AIGenerator("fake_key", "claude-3-sonnet-20240229")
        ai_gen.client = mock_client
        
        result = ai_gen.generate_response(
            "Tell me about Python basics",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify final response
        self.assertEqual(result, "Based on the search results, Python is...")
        
        # Verify tool was executed
        self.mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", 
            query="Python basics"
        )
        
        # Verify two API calls were made
        self.assertEqual(mock_client.messages.create.call_count, 2)
        
        # Verify second call included tool results
        second_call_args = mock_client.messages.create.call_args_list[1][1]
        self.assertEqual(len(second_call_args['messages']), 3)  # user, assistant, tool_results
        self.assertEqual(second_call_args['messages'][2]['role'], 'user')
        self.assertIn('tool_result', str(second_call_args['messages'][2]['content']))
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test response generation with conversation history"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        mock_response = MockAnthropicResponse("Response with context")
        mock_client.messages.create.return_value = mock_response
        
        # Create new instance to use mocked client
        ai_gen = AIGenerator("fake_key", "claude-3-sonnet-20240229")
        ai_gen.client = mock_client
        
        history = "Previous conversation about Python"
        result = ai_gen.generate_response(
            "Continue the discussion",
            conversation_history=history
        )
        
        # Verify response
        self.assertEqual(result, "Response with context")
        
        # Verify history was included in system prompt
        call_args = mock_client.messages.create.call_args[1]
        system_content = call_args['system']
        self.assertIn(history, system_content)
        self.assertIn("Previous conversation", system_content)
    
    @patch('anthropic.Anthropic')
    def test_generate_response_multiple_tool_calls(self, mock_anthropic_class):
        """Test response generation with multiple tool calls in one response"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock response with multiple tool uses
        tool_use_content = [
            MockContentBlock("tool_use", name="search_course_content", 
                           input_data={"query": "Python"}, block_id="tool_1"),
            MockContentBlock("tool_use", name="search_course_content", 
                           input_data={"query": "JavaScript"}, block_id="tool_2")
        ]
        initial_response = MockAnthropicResponse(tool_use_content, stop_reason="tool_use")
        final_response = MockAnthropicResponse("Comparison of Python and JavaScript")
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Mock tool manager to return different results
        self.mock_tool_manager.execute_tool.side_effect = [
            "Python search results", 
            "JavaScript search results"
        ]
        
        # Create new instance to use mocked client
        ai_gen = AIGenerator("fake_key", "claude-3-sonnet-20240229")
        ai_gen.client = mock_client
        
        result = ai_gen.generate_response(
            "Compare Python and JavaScript",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify final response
        self.assertEqual(result, "Comparison of Python and JavaScript")
        
        # Verify both tools were executed
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
        
        # Verify tool calls with correct parameters
        tool_calls = self.mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(tool_calls[0][0], ("search_course_content",))
        self.assertEqual(tool_calls[0][1], {"query": "Python"})
        self.assertEqual(tool_calls[1][0], ("search_course_content",))
        self.assertEqual(tool_calls[1][1], {"query": "JavaScript"})
    
    @patch('anthropic.Anthropic')
    def test_generate_response_tool_execution_failure(self, mock_anthropic_class):
        """Test response generation when tool execution fails"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock response with tool use
        tool_use_content = [
            MockContentBlock("tool_use", name="search_course_content", 
                           input_data={"query": "test"}, block_id="tool_1")
        ]
        initial_response = MockAnthropicResponse(tool_use_content, stop_reason="tool_use")
        final_response = MockAnthropicResponse("I couldn't find information about that")
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Mock tool manager to return error
        self.mock_tool_manager.execute_tool.return_value = "Tool execution failed: Database error"
        
        # Create new instance to use mocked client
        ai_gen = AIGenerator("fake_key", "claude-3-sonnet-20240229")
        ai_gen.client = mock_client
        
        result = ai_gen.generate_response(
            "Search for something",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Should still return a response even with tool failure
        self.assertEqual(result, "I couldn't find information about that")
        
        # Verify tool was still called
        self.mock_tool_manager.execute_tool.assert_called_once()
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        system_prompt = AIGenerator.SYSTEM_PROMPT
        
        # Verify key instruction elements are present
        self.assertIn("Tool Usage Guidelines", system_prompt)
        self.assertIn("Content Search Tool", system_prompt)
        self.assertIn("Course Outline Tool", system_prompt)
        self.assertIn("Sequential Tool Calling", system_prompt)
        self.assertIn("up to 2 rounds of interaction", system_prompt)
        self.assertIn("No meta-commentary", system_prompt)
        
        # Verify response quality requirements
        self.assertIn("Brief, Concise and focused", system_prompt)
        self.assertIn("Educational", system_prompt)
        self.assertIn("Clear", system_prompt)
    
    def test_base_params_configuration(self):
        """Test that base API parameters are configured correctly"""
        ai_gen = AIGenerator("test_key", "test_model")
        
        expected_params = {
            "model": "test_model",
            "temperature": 0,
            "max_tokens": 800
        }
        
        self.assertEqual(ai_gen.base_params, expected_params)
    
    @patch('anthropic.Anthropic')
    def test_sequential_tool_calling_two_rounds(self, mock_anthropic_class):
        """Test sequential tool calling across two rounds"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock first round - tool use response
        first_tool_content = [
            MockContentBlock("tool_use", name="get_course_outline", 
                           input_data={"course_name": "Python Course"}, block_id="tool_1")
        ]
        first_response = MockAnthropicResponse(first_tool_content, stop_reason="tool_use")
        
        # Mock second round - another tool use response
        second_tool_content = [
            MockContentBlock("tool_use", name="search_course_content", 
                           input_data={"query": "variables"}, block_id="tool_2")
        ]
        second_response = MockAnthropicResponse(second_tool_content, stop_reason="tool_use")
        
        # Mock final response after max rounds
        final_response = MockAnthropicResponse("Based on the course outline and content search, here's what I found...")
        
        # Configure mock to return responses in sequence
        mock_client.messages.create.side_effect = [first_response, second_response, final_response]
        
        # Mock tool manager responses
        self.mock_tool_manager.execute_tool.side_effect = [
            "Course outline for Python Course...",
            "Variables are used to store data..."
        ]
        
        # Create new instance to use mocked client
        ai_gen = AIGenerator("fake_key", "claude-3-sonnet-20240229")
        ai_gen.client = mock_client
        
        result = ai_gen.generate_response(
            "Find a course similar to lesson 1 of Python Course",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify final response
        self.assertEqual(result, "Based on the course outline and content search, here's what I found...")
        
        # Verify two tools were executed
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 2)
        
        # Verify three API calls were made (2 rounds + final without tools)
        self.assertEqual(mock_client.messages.create.call_count, 3)
        
        # Verify tool calls were correct
        tool_calls = self.mock_tool_manager.execute_tool.call_args_list
        self.assertEqual(tool_calls[0][0], ("get_course_outline",))
        self.assertEqual(tool_calls[0][1], {"course_name": "Python Course"})
        self.assertEqual(tool_calls[1][0], ("search_course_content",))
        self.assertEqual(tool_calls[1][1], {"query": "variables"})
    
    @patch('anthropic.Anthropic')
    def test_sequential_tool_calling_early_termination(self, mock_anthropic_class):
        """Test early termination when Claude decides no more tools needed"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock first round - tool use response
        first_tool_content = [
            MockContentBlock("tool_use", name="search_course_content", 
                           input_data={"query": "Python basics"}, block_id="tool_1")
        ]
        first_response = MockAnthropicResponse(first_tool_content, stop_reason="tool_use")
        
        # Mock second round - direct response (no tools)
        second_response = MockAnthropicResponse("Python is a programming language...")
        
        # Configure mock to return responses in sequence
        mock_client.messages.create.side_effect = [first_response, second_response]
        
        # Mock tool manager response
        self.mock_tool_manager.execute_tool.return_value = "Python basics content..."
        
        # Create new instance to use mocked client
        ai_gen = AIGenerator("fake_key", "claude-3-sonnet-20240229")
        ai_gen.client = mock_client
        
        result = ai_gen.generate_response(
            "What is Python?",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify final response
        self.assertEqual(result, "Python is a programming language...")
        
        # Verify one tool was executed
        self.assertEqual(self.mock_tool_manager.execute_tool.call_count, 1)
        
        # Verify two API calls were made (tool round + final response)
        self.assertEqual(mock_client.messages.create.call_count, 2)
    
    @patch('anthropic.Anthropic')
    def test_sequential_tool_calling_tool_failure(self, mock_anthropic_class):
        """Test handling of tool execution failure in sequential calling"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock first round - tool use response
        first_tool_content = [
            MockContentBlock("tool_use", name="search_course_content", 
                           input_data={"query": "test"}, block_id="tool_1")
        ]
        first_response = MockAnthropicResponse(first_tool_content, stop_reason="tool_use")
        
        # Configure mock to return tool use response
        mock_client.messages.create.return_value = first_response
        
        # Mock tool manager to raise exception
        self.mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")
        
        # Create new instance to use mocked client
        ai_gen = AIGenerator("fake_key", "claude-3-sonnet-20240229")
        ai_gen.client = mock_client
        
        result = ai_gen.generate_response(
            "Search for something",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify error message is returned
        self.assertEqual(result, "I encountered an error while processing your request.")
        
        # Verify tool was attempted
        self.mock_tool_manager.execute_tool.assert_called_once()
        
        # Verify only one API call was made (the failed tool round)
        self.assertEqual(mock_client.messages.create.call_count, 1)
    
    @patch('anthropic.Anthropic')
    def test_sequential_tool_calling_max_rounds_parameter(self, mock_anthropic_class):
        """Test custom max_rounds parameter"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock single round with direct response
        mock_response = MockAnthropicResponse("Direct response without tools")
        mock_client.messages.create.return_value = mock_response
        
        # Create new instance to use mocked client
        ai_gen = AIGenerator("fake_key", "claude-3-sonnet-20240229")
        ai_gen.client = mock_client
        
        result = ai_gen.generate_response(
            "What is 2+2?",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager,
            max_rounds=1  # Custom max rounds
        )
        
        # Verify response
        self.assertEqual(result, "Direct response without tools")
        
        # Verify one API call was made
        self.assertEqual(mock_client.messages.create.call_count, 1)
    
    @patch('anthropic.Anthropic')
    def test_sequential_tool_calling_conversation_context(self, mock_anthropic_class):
        """Test that conversation context is preserved across rounds"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock first round - tool use response
        first_tool_content = [
            MockContentBlock("tool_use", name="search_course_content", 
                           input_data={"query": "Python"}, block_id="tool_1")
        ]
        first_response = MockAnthropicResponse(first_tool_content, stop_reason="tool_use")
        
        # Mock second round - direct response
        second_response = MockAnthropicResponse("Python is a programming language...")
        
        # Configure mock to return responses in sequence
        mock_client.messages.create.side_effect = [first_response, second_response]
        
        # Mock tool manager response
        self.mock_tool_manager.execute_tool.return_value = "Python content..."
        
        # Create new instance to use mocked client
        ai_gen = AIGenerator("fake_key", "claude-3-sonnet-20240229")
        ai_gen.client = mock_client
        
        # Test with conversation history
        history = "Previous discussion about programming languages"
        result = ai_gen.generate_response(
            "Tell me about Python",
            conversation_history=history,
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify final response
        self.assertEqual(result, "Python is a programming language...")
        
        # Verify conversation history was included in system prompt for both calls
        call_args_list = mock_client.messages.create.call_args_list
        for call_args in call_args_list:
            system_content = call_args[1]['system']
            self.assertIn(history, system_content)
            self.assertIn("Previous conversation", system_content)


if __name__ == '__main__':
    unittest.main()