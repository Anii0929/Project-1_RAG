import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage Guidelines:
- **Content Search Tool**: Use for questions about specific course content or detailed educational materials
- **Course Outline Tool**: Use for questions about course structure, lesson lists, course overviews, or when users ask "what's in this course"
- **Sequential Tool Calling**: You can make multiple tool calls across up to 2 rounds of interaction to gather comprehensive information
- **Round 1**: Use tools to gather initial information
- **Round 2**: Use additional tools if needed to gather more context, compare information, or clarify details
- **Reasoning**: After each tool call, analyze results and determine if additional information is needed for a complete answer
- Synthesize all tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use content search tool first, then answer
- **Course outline/structure questions**: Use outline tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the outline"

For outline queries, always include:
- Course title and link
- Course instructor
- Complete lesson list with numbers and titles

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
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with sequential tool usage support and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum sequential tool calls (default: 2)
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Start with the original user query
        current_messages = [{"role": "user", "content": query}]
        
        # Sequential tool calling loop
        for round_num in range(max_rounds):
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": current_messages.copy(),
                "system": system_content
            }
            
            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}
            
            # Get response from Claude
            response = self.client.messages.create(**api_params)
            
            # If no tool use, we're done
            if response.stop_reason != "tool_use" or not tool_manager:
                return response.content[0].text
            
            # Handle tool execution and update messages
            current_messages = self._handle_tool_execution_sequential(
                response, current_messages, tool_manager
            )
            
            # If tool execution failed, return error message
            if current_messages is None:
                return "I encountered an error while processing your request."
        
        # If we've completed max rounds with tools, make final call without tools
        final_params = {
            **self.base_params,
            "messages": current_messages,
            "system": system_content
        }
        
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
    
    def _handle_tool_execution_sequential(self, response, messages: List, tool_manager):
        """
        Handle tool execution for sequential calling and return updated messages.
        
        Args:
            response: The response containing tool use requests
            messages: Current message history
            tool_manager: Manager to execute tools
            
        Returns:
            Updated messages list or None if tool execution fails
        """
        try:
            # Add AI's tool use response to messages
            messages.append({"role": "assistant", "content": response.content})
            
            # Execute all tool calls and collect results
            tool_results = []
            for content_block in response.content:
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
            
            # Add tool results as user message
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            
            return messages
            
        except Exception as e:
            # Log error and return None to indicate failure
            print(f"Tool execution error: {e}")
            return None
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Original single tool execution method - kept for backward compatibility.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Use the sequential method but return just the final response
        messages = base_params["messages"].copy()
        updated_messages = self._handle_tool_execution_sequential(
            initial_response, messages, tool_manager
        )
        
        if updated_messages is None:
            return "I encountered an error while processing your request."
        
        # Make final call to get response
        final_params = {
            **self.base_params,
            "messages": updated_messages,
            "system": base_params["system"]
        }
        
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text