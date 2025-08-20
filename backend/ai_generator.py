import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **search_course_content**: Search for specific content within course materials
   - Use for questions about topics, concepts, or specific lesson content
   - Can filter by course name and lesson number
   
2. **get_course_outline**: Get the complete structure of a course
   - Use for questions about course organization, lesson lists, or course overviews
   - Returns course title, link, and complete lesson list with numbers and titles

Multi-Step Tool Usage:
- You can use tools sequentially across up to 2 rounds to gather comprehensive information
- Round 1: Gather initial information (e.g., get course outline to see lesson structure)
- Round 2: Use results from round 1 for targeted actions (e.g., search specific lesson content)
- Each round allows you to reason about previous results before making the next tool call

Sequential Tool Patterns:
1. **Exploration → Refinement**: First get overview, then search for specifics
   - Example: Get course outline → Search for content in specific lesson
2. **Comparison**: Search different courses/lessons to compare content
   - Example: Search topic in course A → Search same topic in course B
3. **Multi-part questions**: Address each part with appropriate tool calls
   - Example: Find lesson title → Search for related content in other courses

Tool Usage Examples:
- "What does lesson 4 of course X teach?" → get_course_outline (find lesson 4 title) → search_course_content (search that topic)
- "Compare Python basics across courses" → search_course_content for course A → search_course_content for course B
- "Find courses covering similar topics as course X lesson 3" → get_course_outline → search_course_content for identified topics

Tool Usage Guidelines:
- **Course outline/structure questions**: Use get_course_outline tool
- **Content-specific questions**: Use search_course_content tool
- **Complex queries**: Use multiple tool calls sequentially (up to 2 rounds)
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course-specific questions**: Use appropriate tool(s), then provide comprehensive answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"

Formatting Course Outlines:
- Present course title, link, and lesson count clearly
- List lessons with their numbers and titles in order
- Keep formatting clean and readable

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required but not found in environment. Please add it to your .env file.")
        
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize Anthropic client: {e}")
        
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
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager, max_rounds: int = 2):
        """
        Handle sequential tool execution with up to max_rounds of tool calls.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters including tools
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool-calling rounds (default: 2)
            
        Returns:
            Final response text after all tool executions
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        current_response = initial_response
        tools = base_params.get("tools", [])
        
        for round_num in range(1, max_rounds + 1):
            # Add assistant's tool use response
            messages.append({"role": "assistant", "content": current_response.content})
            
            # Execute all tool calls and collect results
            tool_results = []
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name,
                            **content_block.input
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result
                        })
                    except Exception as e:
                        # Handle tool execution errors gracefully
                        print(f"Tool execution error in round {round_num}: {e}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution failed: {str(e)}"
                        })
            
            # Add tool results to messages
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            
            # Check if we've reached max rounds or should make final call
            if round_num >= max_rounds:
                # Final round reached - make final call without tools
                break
            
            # Prepare next API call WITH tools still available for potential round 2
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
                "tools": tools,
                "tool_choice": {"type": "auto"}
            }
            
            # Get next response to see if Claude wants to use more tools
            current_response = self.client.messages.create(**next_params)
            
            # Check if Claude wants to use more tools
            if current_response.stop_reason != "tool_use":
                # No more tool use needed, return the response
                return current_response.content[0].text
        
        # Final API call without tools to get the synthesized answer
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text