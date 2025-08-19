import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage Guidelines:
- **Content Search Tool**: Use for questions about specific course content or detailed educational materials
- **Course Outline Tool**: Use for questions about course structure, lesson lists, course overview, or when users ask "what lessons are in..." or "show me the outline of..."
- **One tool call per query maximum**
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Query Types:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course content questions**: Use content search tool first, then answer
- **Course outline/structure questions**: Use outline tool to get course title, course link, instructor, and complete lesson list with numbers and titles
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the tool results"

Response Requirements:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
5. **Complete information** - For outline queries, include course title, course link, instructor, and all lessons with their numbers and titles

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
        
        # Get response from Claude with error handling
        try:
            response = self.client.messages.create(**api_params)
        except Exception as e:
            # Log the error and return a meaningful message
            print(f"Error calling Anthropic API: {type(e).__name__}: {e}")
            return f"I apologize, but I'm experiencing technical difficulties. Please try your question again."
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            try:
                return self._handle_tool_execution(response, api_params, tool_manager)
            except Exception as e:
                print(f"Error in tool execution: {type(e).__name__}: {e}")
                return "I encountered an issue while searching for information. Please try rephrasing your question."
        
        # Return direct response
        try:
            return response.content[0].text
        except (IndexError, AttributeError) as e:
            print(f"Error parsing response content: {type(e).__name__}: {e}")
            return "I received an unexpected response format. Please try your question again."
    
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
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                except Exception as e:
                    print(f"Error executing tool {content_block.name}: {type(e).__name__}: {e}")
                    tool_result = f"Error executing search: {str(e)}"
                
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
        
        # Get final response with error handling
        try:
            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text
        except Exception as e:
            print(f"Error in final API call: {type(e).__name__}: {e}")
            return "I encountered an issue while processing the search results. Please try your question again."