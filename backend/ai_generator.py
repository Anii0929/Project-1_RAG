import anthropic
import openai
import tiktoken
from typing import List, Optional, Dict, Any

class OpenAIGenerator:
    """Handles interactions with company's OpenAI-compatible LLM API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
        
        # Initialize tokenizer for usage tracking
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")  # Use gpt-4 encoding as fallback
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")  # Default encoding
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare messages for OpenAI format
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Convert Anthropic tools to OpenAI functions format
        if tools:
            api_params["functions"] = self._convert_tools_to_functions(tools)
            api_params["function_call"] = "auto"
        
        # Get response from company LLM
        response = self.client.chat.completions.create(**api_params)
        
        # Handle function execution if needed
        if response.choices[0].finish_reason == "function_call" and tool_manager:
            return self._handle_function_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.choices[0].message.content
    
    def _convert_tools_to_functions(self, tools: List) -> List[Dict]:
        """Convert Anthropic tool format to OpenAI function format"""
        functions = []
        for tool in tools:
            function = {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"]
            }
            functions.append(function)
        return functions
    
    def _handle_function_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of function calls and get follow-up response.
        
        Args:
            initial_response: The response containing function call requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after function execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Get the function call from response
        function_call = initial_response.choices[0].message.function_call
        
        # Add AI's function call response
        messages.append({
            "role": "assistant", 
            "content": None,
            "function_call": {
                "name": function_call.name,
                "arguments": function_call.arguments
            }
        })
        
        # Execute function call
        import json
        function_args = json.loads(function_call.arguments)
        function_result = tool_manager.execute_tool(function_call.name, **function_args)
        
        # Add function result
        messages.append({
            "role": "function",
            "name": function_call.name,
            "content": str(function_result)
        })
        
        # Prepare final API call without functions
        final_params = {
            **self.base_params,
            "messages": messages
        }
        
        # Get final response
        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text