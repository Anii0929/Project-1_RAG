"""
Alternative AI generators for the RAG system
Choose based on your preferences:
1. OllamaGenerator - Best local option, no internet required
2. HuggingFaceGenerator - Free cloud-based option
3. OpenAICompatibleGenerator - For local OpenAI-compatible servers
"""

from typing import Any, Dict, List, Optional


class OllamaGenerator:
    """Local LLM using Ollama - No API costs, runs offline"""

    def __init__(self, model: str = "llama2"):
        """Initialize Ollama generator with specified model.

        Args:
            model: Name of the Ollama model to use (default: "llama2").

        Raises:
            ImportError: If ollama package is not installed.
        """
        self.model = model
        try:
            import ollama

            self.client = ollama
            print(f"✓ Ollama initialized with model: {model}")
        except ImportError:
            raise ImportError("Install ollama: pip install ollama")

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """Generate response using Ollama with tool support.

        Args:
            query: User's question or request.
            conversation_history: Previous conversation context.
            tools: Available tools (for compatibility, not directly used).
            tool_manager: Tool manager for executing tools.

        Returns:
            str: Generated response, potentially enhanced with tool results.

        Raises:
            Exception: If Ollama model is not available or request fails.
        """

        # Build system prompt with tool instructions (similar to Anthropic approach)
        system_prompt = """You are a course materials assistant. You have access to tools to help answer questions.

Available tools:
- get_course_outline: Use for questions about course structure, outline, syllabus, lessons, overview
- search_course_content: Use for specific questions about course content and topics

Instructions:
1. Analyze the user's question
2. If they ask about course outline/structure/lessons → use get_course_outline with course_name="course title"
3. If they ask about specific content/topics → use search_course_content with query="search terms"
4. Format your tool usage as: TOOL_USE: tool_name(parameter_name="value")
5. After using a tool, provide a clear answer based on the results

Answer the following question using the appropriate tool if needed."""

        # Build the full prompt
        if conversation_history:
            full_prompt = f"{system_prompt}\n\nPrevious conversation:\n{conversation_history}\n\nQuestion: {query}"
        else:
            full_prompt = f"{system_prompt}\n\nQuestion: {query}"

        # Get initial response from Ollama
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                options={"temperature": 0},
            )
            initial_response = response["message"]["content"]

            # Check if the model wants to use a tool
            if tool_manager and "TOOL_USE:" in initial_response:
                return self._handle_ollama_tool_execution(
                    initial_response, query, tool_manager
                )

            return initial_response

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _handle_ollama_tool_execution(
        self, initial_response: str, original_query: str, tool_manager
    ) -> str:
        """Parse Ollama response for tool usage and execute tools.

        Implements simplified tool calling by parsing TOOL_USE patterns
        from Ollama's text responses.

        Args:
            initial_response: Ollama's initial response potentially containing tool usage.
            original_query: The original user query.
            tool_manager: Tool manager for executing tools.

        Returns:
            str: Final response after tool execution and synthesis.
        """

        # Parse tool usage from the response
        import re

        # Look for TOOL_USE: pattern
        tool_match = re.search(r"TOOL_USE:\s*(\w+)\((.*?)\)", initial_response)
        if not tool_match:
            # Fallback: try to determine tool from query content
            return self._fallback_tool_selection(original_query, tool_manager)

        tool_name = tool_match.group(1)
        params_str = tool_match.group(2)

        # Parse parameters (simple key="value" format)
        params = {}
        param_matches = re.findall(r'(\w+)=["\']([^"\']+)["\']', params_str)
        for key, value in param_matches:
            params[key] = value

        # Execute the tool
        try:
            tool_result = tool_manager.execute_tool(tool_name, **params)

            # Generate final response with tool results
            final_prompt = f"""Based on this tool result:
{tool_result}

Please provide a clear, concise answer to the original question: {original_query}"""

            final_response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": final_prompt}],
                options={"temperature": 0},
            )

            return final_response["message"]["content"]

        except Exception as e:
            return f"Error executing tool: {str(e)}"

    def _fallback_tool_selection(self, query: str, tool_manager) -> str:
        """Intelligently select and execute appropriate tool based on query keywords.

        Used when tool parsing fails, applies heuristics to determine
        whether to use outline or content search tools.

        Args:
            query: User's query to analyze.
            tool_manager: Tool manager for executing tools.

        Returns:
            str: Response generated after tool execution.
        """

        outline_keywords = [
            "outline",
            "structure",
            "syllabus",
            "lessons",
            "overview",
            "what's covered",
            "what's in",
        ]
        is_outline_query = any(keyword in query.lower() for keyword in outline_keywords)

        if is_outline_query:
            # Extract course name from query
            import re

            course_match = re.search(r'"([^"]+)"|\b([A-Z]{2,}[^a-z]*)', query)
            if course_match:
                course_name = course_match.group(1) or course_match.group(2)
                tool_results = tool_manager.execute_tool(
                    "get_course_outline", course_name=course_name
                )
            else:
                tool_results = tool_manager.execute_tool(
                    "search_course_content", query=query
                )
        else:
            tool_results = tool_manager.execute_tool(
                "search_course_content", query=query
            )

        # Generate response with tool results
        final_prompt = f"""Based on this information:
{tool_results}

Please provide a clear answer to: {query}"""

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": final_prompt}],
                options={"temperature": 0},
            )
            return response["message"]["content"]
        except Exception as e:
            return f"Error generating final response: {str(e)}"


class HuggingFaceGenerator:
    """Free Hugging Face transformers - No API costs"""

    def __init__(self):
        """Initialize Hugging Face text generation pipeline.

        Loads a conversational model suitable for educational content Q&A.

        Raises:
            ImportError: If transformers or torch packages are not installed.
            Exception: If model loading fails.
        """
        try:
            from transformers import pipeline

            # Use a good conversational model
            self.generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=-1,  # Use CPU
            )
            print("✓ Hugging Face model loaded")
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """Generate response using Hugging Face transformers with search integration.

        Args:
            query: User's question or request.
            conversation_history: Previous conversation context (not used).
            tools: Available tools (not used directly).
            tool_manager: Tool manager for performing course content search.

        Returns:
            str: Generated response based on search results and model inference.
        """

        # Build prompt with search results
        if tool_manager:
            search_results = tool_manager.execute_tool(
                "search_course_content", query=query
            )
            prompt = (
                f"Course Information: {search_results}\n\nQuestion: {query}\nAnswer:"
            )
        else:
            prompt = f"Question: {query}\nAnswer:"

        try:
            response = self.generator(
                prompt,
                max_length=len(prompt) + 150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id,
            )

            # Extract only the generated part
            generated_text = response[0]["generated_text"]
            answer = generated_text[len(prompt) :].strip()

            return answer if answer else "I couldn't generate a proper response."
        except Exception as e:
            return f"Error generating response: {str(e)}"


class OpenAICompatibleGenerator:
    """For local OpenAI-compatible servers (like text-generation-webui)"""

    def __init__(
        self, base_url: str = "http://localhost:5000/v1", model: str = "local-model"
    ):
        """Initialize OpenAI-compatible client for local inference servers.

        Args:
            base_url: Base URL of the OpenAI-compatible API server.
            model: Model name to use for requests.

        Raises:
            ImportError: If openai package is not installed.
        """
        self.base_url = base_url
        self.model = model
        try:
            import openai

            self.client = openai
            self.client.api_base = base_url
            self.client.api_key = "not-needed"
            print(f"✓ OpenAI-compatible client initialized: {base_url}")
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """Generate response using OpenAI-compatible local server.

        Args:
            query: User's question or request.
            conversation_history: Previous conversation context.
            tools: Available tools (not used directly).
            tool_manager: Tool manager for performing course content search.

        Returns:
            str: Generated response from the local model server.

        Raises:
            Exception: If connection to local server fails.
        """

        messages = []

        # Add system message with search results
        if tool_manager:
            search_results = tool_manager.execute_tool(
                "search_course_content", query=query
            )
            messages.append(
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. Use this course information to answer questions:\n{search_results}",
                }
            )

        # Add conversation history
        if conversation_history:
            messages.append(
                {
                    "role": "system",
                    "content": f"Previous conversation context:\n{conversation_history}",
                }
            )

        # Add user query
        messages.append({"role": "user", "content": query})

        try:
            response = self.client.ChatCompletion.create(
                model=self.model, messages=messages, temperature=0, max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error connecting to local server: {str(e)}"


class SimpleSearchOnlyGenerator:
    """Fallback: Just return search results without AI generation"""

    def __init__(self):
        """Initialize search-only generator (no AI inference).

        This fallback option returns formatted search results without
        any AI generation, useful when AI providers are unavailable.
        """
        print("✓ Search-only generator initialized (no AI generation)")

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """Return formatted search results without AI generation.

        Args:
            query: User's question or request.
            conversation_history: Previous conversation context (not used).
            tools: Available tools (not used directly).
            tool_manager: Tool manager for performing searches.

        Returns:
            str: Formatted search results with appropriate tool selection.
        """

        if tool_manager:
            # Decide which tool to use based on query keywords
            outline_keywords = [
                "outline",
                "structure",
                "syllabus",
                "lessons",
                "overview",
                "what's covered",
                "what's in",
            ]
            is_outline_query = any(
                keyword in query.lower() for keyword in outline_keywords
            )

            if is_outline_query:
                # Extract course name from query for outline tool
                import re

                course_match = re.search(r'"([^"]+)"|\b([A-Z]{2,}[^a-z]*)', query)
                if course_match:
                    course_name = course_match.group(1) or course_match.group(2)
                    tool_results = tool_manager.execute_tool(
                        "get_course_outline", course_name=course_name
                    )
                    return f"Here is the course outline information:\n\n{tool_results}"
                else:
                    # Fallback to search if we can't extract course name
                    tool_results = tool_manager.execute_tool(
                        "search_course_content", query=query
                    )
                    return f"Based on your query '{query}', here's what I found in the course materials:\n\n{tool_results}"
            else:
                # Use search tool for content questions
                tool_results = tool_manager.execute_tool(
                    "search_course_content", query=query
                )
                return f"Based on your query '{query}', here's what I found in the course materials:\n\n{tool_results}"
        else:
            return "Search functionality is not available. Please check the system configuration."
