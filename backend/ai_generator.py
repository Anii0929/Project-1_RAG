import anthropic
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are a course materials assistant. You MUST use the provided tools to answer questions about course content and structure.

CRITICAL TOOL USAGE RULES:
- For general questions about available courses → ALWAYS use list_all_courses tool
- For specific course outline/structure (when course name is mentioned) → use get_course_outline tool  
- For questions about course content/concepts → use search_course_content tool
- NEVER answer course-related questions using your own knowledge - ALWAYS use the appropriate tool first
- You can make multiple tool calls in sequence if needed to answer complex questions
- For complex queries, use results from one tool to inform the next tool call

TOOL SELECTION GUIDE:
- list_all_courses: "what courses are available", "show me all courses", "list courses", "what can I learn"
- get_course_outline: "outline of MCP course", "structure of [specific course]", "lessons in [course name]"
- search_course_content: "what is MCP", "explain concepts", "how does X work", "tell me about [topic]"

SEQUENTIAL TOOL USAGE EXAMPLES:
- "Search for courses discussing same topic as lesson 4 of MCP course" → get_course_outline(MCP) → search_course_content(lesson 4 topic)
- "Compare content between two courses" → search_course_content(course A) → search_course_content(course B)

FORMAT: Always use tools first, then provide the response based on tool results only."""

    def __init__(self, api_key: str, model: str, max_tool_rounds: int = 2):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tool_rounds = max_tool_rounds

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
        Generate AI response with sequential tool calling support.

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

        # Initialize conversation context
        messages = [{"role": "user", "content": query}]
        round_count = 0

        # Sequential tool calling loop
        while round_count < self.max_tool_rounds:
            round_count += 1

            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            print(f"Round {round_count} API params:\n {api_params}")
            response = self.client.messages.create(**api_params)
            print(f"Round {round_count} response:\n{response}")

            # Handle different response types
            if response.stop_reason == "tool_use" and tool_manager:
                # Execute tools and prepare for next round
                messages.append({"role": "assistant", "content": response.content})
                
                tool_results = self._execute_tools(response, tool_manager)
                messages.append({"role": "user", "content": tool_results})
                
                # Continue to next round
                continue
                
            elif response.stop_reason in ["end_turn", "max_tokens"]:
                # Final response received - no more tools needed
                return response.content[0].text if response.content else "No response generated"
                
            else:
                # Unexpected stop reason - return what we have
                return response.content[0].text if response.content else "No response generated"
        
        # Max rounds reached - make final call without tools to get response
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text if final_response.content else "Maximum tool rounds exceeded."

    def _execute_tools(self, response, tool_manager) -> List[Dict[str, Any]]:
        """
        Execute tool calls from Claude's response.

        Args:
            response: The response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            List of tool results formatted for API
        """
        tool_results = []
        for content_block in response.content:
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
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution failed: {str(e)}",
                        "is_error": True
                    })

        return tool_results
