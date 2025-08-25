"""
AI Provider diagnosis tests to identify tool execution issues.

These tests focus on the AI provider configuration and tool execution flow
to identify why queries are failing.
"""

import os
import sys

import pytest

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import config
from rag_system import RAGSystem
from search_tools import CourseSearchTool, ToolManager
from vector_store import VectorStore


class TestAIProviderDiagnosis:
    """Test AI provider configuration and tool execution."""

    @pytest.mark.integration
    def test_current_ai_provider_info(self):
        """Display current AI provider configuration."""
        print(f"Current AI_PROVIDER: {config.AI_PROVIDER}")
        print(f"ANTHROPIC_API_KEY set: {'Yes' if config.ANTHROPIC_API_KEY else 'No'}")
        print(f"OLLAMA_MODEL: {config.OLLAMA_MODEL}")
        print(f"CHROMA_PATH: {config.CHROMA_PATH}")
        print(f"EMBEDDING_MODEL: {config.EMBEDDING_MODEL}")

    @pytest.mark.integration
    def test_tool_execution_direct(self):
        """Test tool execution directly without AI layer."""
        try:
            # Use real VectorStore with existing data
            store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL)
            tool = CourseSearchTool(store)

            # Test direct tool execution
            result = tool.execute("MCP protocol")

            print(f"Direct tool execution result: {result[:200]}...")

            assert isinstance(result, str)
            if "No relevant content found" in result:
                print("⚠️  Tool found no content - might be search issue")
            elif "Search error" in result:
                print("⚠️  Tool had search error - vector store issue")
                assert False, f"Tool had search error: {result}"
            else:
                print("✓ Tool execution successful")

        except Exception as e:
            print(f"✗ Direct tool execution failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    @pytest.mark.integration
    def test_tool_manager_execution(self):
        """Test tool execution through ToolManager."""
        try:
            store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL)
            search_tool = CourseSearchTool(store)

            tool_manager = ToolManager()
            tool_manager.register_tool(search_tool)

            # Test execution through manager
            result = tool_manager.execute_tool(
                "search_course_content", query="MCP applications", course_name="MCP"
            )

            print(f"ToolManager execution result: {result[:200]}...")

            assert isinstance(result, str)
            if "No relevant content found" in result:
                print("⚠️  ToolManager found no content")
            elif "Search error" in result:
                print("⚠️  ToolManager had search error")
                assert False, f"ToolManager search error: {result}"
            else:
                print("✓ ToolManager execution successful")

        except Exception as e:
            print(f"✗ ToolManager execution failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    @pytest.mark.integration
    def test_ai_generator_with_different_providers(self):
        """Test AI generators with different providers."""
        providers_to_test = ["search_only"]  # Safe provider to test

        for provider in providers_to_test:
            print(f"\nTesting provider: {provider}")

            # Temporarily override provider
            original_provider = config.AI_PROVIDER
            config.AI_PROVIDER = provider

            try:
                rag = RAGSystem(config)

                # Test basic query
                response, sources = rag.query("What is MCP protocol?")

                print(f"Response: {response[:200]}...")
                print(f"Sources: {len(sources)}")

                assert isinstance(response, str)
                assert isinstance(sources, list)

                if "query failed" in response.lower():
                    print(f"⚠️  Provider {provider} returned 'query failed'")
                else:
                    print(f"✓ Provider {provider} working")

            except Exception as e:
                print(f"✗ Provider {provider} failed: {e}")
                import traceback

                traceback.print_exc()
                raise
            finally:
                config.AI_PROVIDER = original_provider

    @pytest.mark.integration
    def test_ollama_availability(self):
        """Test if Ollama is available and working."""
        if config.AI_PROVIDER != "ollama":
            pytest.skip("Not using Ollama provider")

        try:
            # Test Ollama availability
            import subprocess

            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)

            if result.returncode != 0:
                print("⚠️  Ollama command failed")
                print(f"Error: {result.stderr}")
            else:
                print("✓ Ollama is available")
                print(f"Available models: {result.stdout}")

                # Check if the configured model is available
                if config.OLLAMA_MODEL in result.stdout:
                    print(f"✓ Model {config.OLLAMA_MODEL} is available")
                else:
                    print(f"⚠️  Model {config.OLLAMA_MODEL} not found in Ollama")

        except FileNotFoundError:
            print("⚠️  Ollama not installed or not in PATH")
        except Exception as e:
            print(f"✗ Ollama availability check failed: {e}")

    @pytest.mark.integration
    def test_anthropic_api_key_validation(self):
        """Test Anthropic API key if using Anthropic provider."""
        if config.AI_PROVIDER != "anthropic":
            pytest.skip("Not using Anthropic provider")

        if not config.ANTHROPIC_API_KEY:
            print("⚠️  No Anthropic API key configured")
            return

        if not config.ANTHROPIC_API_KEY.startswith("sk-ant-"):
            print("⚠️  Anthropic API key format looks incorrect")
            print(f"Key starts with: {config.ANTHROPIC_API_KEY[:10]}...")
        else:
            print("✓ Anthropic API key format looks correct")

    @pytest.mark.integration
    def test_end_to_end_with_real_query(self):
        """Test end-to-end query with the actual configured provider."""
        try:
            rag = RAGSystem(config)

            # Test queries that should work with existing data
            test_queries = [
                "What is MCP?",
                "Tell me about Chroma",
                "What courses are available?",
                "Anthropic building",
            ]

            for query in test_queries:
                print(f"\nTesting query: '{query}'")

                try:
                    response, sources = rag.query(query)

                    print(f"Response length: {len(response)}")
                    print(f"Sources count: {len(sources)}")
                    print(f"Response preview: {response[:100]}...")

                    if "query failed" in response.lower():
                        print(f"⚠️  Query '{query}' returned 'query failed'")
                    elif "error" in response.lower():
                        print(f"⚠️  Query '{query}' returned error")
                    else:
                        print(f"✓ Query '{query}' succeeded")

                except Exception as e:
                    print(f"✗ Query '{query}' threw exception: {e}")
                    import traceback

                    traceback.print_exc()

        except Exception as e:
            print(f"✗ End-to-end test setup failed: {e}")
            import traceback

            traceback.print_exc()
            raise
