"""
Alternative AI generators for the RAG system
Choose based on your preferences:
1. OllamaGenerator - Best local option, no internet required
2. HuggingFaceGenerator - Free cloud-based option
3. OpenAICompatibleGenerator - For local OpenAI-compatible servers
"""

from typing import List, Optional, Dict, Any

class OllamaGenerator:
    """Local LLM using Ollama - No API costs, runs offline"""
    
    def __init__(self, model: str = "llama2"):
        self.model = model
        try:
            import ollama
            self.client = ollama
            print(f"✓ Ollama initialized with model: {model}")
        except ImportError:
            raise ImportError("Install ollama: pip install ollama")
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        
        # Build prompt with search results if available
        if tool_manager:
            # Use search tool to get relevant content
            search_results = tool_manager.execute_tool("search_course_content", query=query)
            prompt = f"Based on this course information:\n{search_results}\n\nPlease answer: {query}"
        else:
            prompt = f"Answer this question about course materials: {query}"
        
        # Add conversation history
        if conversation_history:
            prompt = f"Previous conversation:\n{conversation_history}\n\n{prompt}"
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating response: {str(e)}"


class HuggingFaceGenerator:
    """Free Hugging Face transformers - No API costs"""
    
    def __init__(self):
        try:
            from transformers import pipeline
            # Use a good conversational model
            self.generator = pipeline(
                "text-generation", 
                model="microsoft/DialoGPT-medium",
                device=-1  # Use CPU
            )
            print("✓ Hugging Face model loaded")
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        
        # Build prompt with search results
        if tool_manager:
            search_results = tool_manager.execute_tool("search_course_content", query=query)
            prompt = f"Course Information: {search_results}\n\nQuestion: {query}\nAnswer:"
        else:
            prompt = f"Question: {query}\nAnswer:"
        
        try:
            response = self.generator(
                prompt, 
                max_length=len(prompt) + 150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            # Extract only the generated part
            generated_text = response[0]['generated_text']
            answer = generated_text[len(prompt):].strip()
            
            return answer if answer else "I couldn't generate a proper response."
        except Exception as e:
            return f"Error generating response: {str(e)}"


class OpenAICompatibleGenerator:
    """For local OpenAI-compatible servers (like text-generation-webui)"""
    
    def __init__(self, base_url: str = "http://localhost:5000/v1", model: str = "local-model"):
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
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        
        messages = []
        
        # Add system message with search results
        if tool_manager:
            search_results = tool_manager.execute_tool("search_course_content", query=query)
            messages.append({
                "role": "system", 
                "content": f"You are a helpful assistant. Use this course information to answer questions:\n{search_results}"
            })
        
        # Add conversation history
        if conversation_history:
            messages.append({
                "role": "system",
                "content": f"Previous conversation context:\n{conversation_history}"
            })
        
        # Add user query
        messages.append({
            "role": "user",
            "content": query
        })
        
        try:
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error connecting to local server: {str(e)}"


class SimpleSearchOnlyGenerator:
    """Fallback: Just return search results without AI generation"""
    
    def __init__(self):
        print("✓ Search-only generator initialized (no AI generation)")
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        
        if tool_manager:
            # Get search results
            search_results = tool_manager.execute_tool("search_course_content", query=query)
            
            # Format nicely
            return f"Based on your query '{query}', here's what I found in the course materials:\n\n{search_results}"
        else:
            return "Search functionality is not available. Please check the system configuration."