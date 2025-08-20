#!/usr/bin/env python3
"""
Live diagnostic tests to identify the actual 'query failed' issue.
Run this to test against the real system and identify failures.
"""

import sys
import os
import traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from rag_system import RAGSystem
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator


def test_config():
    """Test 1: Check configuration"""
    print("\n=== TEST 1: Configuration Check ===")
    try:
        config = Config()
        print(f"✓ Config loaded")
        print(f"  - API Key present: {'Yes' if config.ANTHROPIC_API_KEY else 'No'}")
        print(f"  - Model: {config.ANTHROPIC_MODEL}")
        print(f"  - ChromaDB path: {config.CHROMA_PATH}")
        print(f"  - Embedding model: {config.EMBEDDING_MODEL}")
        
        if not config.ANTHROPIC_API_KEY:
            print("❌ ERROR: No API key found!")
            return False
        return True
    except Exception as e:
        print(f"❌ Config Error: {e}")
        traceback.print_exc()
        return False


def test_vector_store():
    """Test 2: Check VectorStore initialization and basic operations"""
    print("\n=== TEST 2: VectorStore Check ===")
    try:
        config = Config()
        vector_store = VectorStore(
            config.CHROMA_PATH,
            config.EMBEDDING_MODEL,
            config.MAX_RESULTS
        )
        print("✓ VectorStore initialized")
        
        # Test getting course count
        course_count = vector_store.get_course_count()
        print(f"  - Courses in database: {course_count}")
        
        # Test getting course titles
        titles = vector_store.get_existing_course_titles()
        print(f"  - Course titles: {titles[:3] if titles else 'None'}...")
        
        if course_count == 0:
            print("⚠️  WARNING: No courses in database!")
        
        return True
    except Exception as e:
        print(f"❌ VectorStore Error: {e}")
        traceback.print_exc()
        return False


def test_search_functionality():
    """Test 3: Check search functionality directly"""
    print("\n=== TEST 3: Search Functionality ===")
    try:
        config = Config()
        vector_store = VectorStore(
            config.CHROMA_PATH,
            config.EMBEDDING_MODEL,
            config.MAX_RESULTS
        )
        
        # Test basic search
        print("Testing basic search for 'Python'...")
        results = vector_store.search(query="Python")
        
        if results.error:
            print(f"❌ Search returned error: {results.error}")
            return False
        
        if results.is_empty():
            print("⚠️  Search returned no results")
        else:
            print(f"✓ Search returned {len(results.documents)} results")
            print(f"  - First result preview: {results.documents[0][:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Search Error: {e}")
        traceback.print_exc()
        return False


def test_course_search_tool():
    """Test 4: Check CourseSearchTool execution"""
    print("\n=== TEST 4: CourseSearchTool ===")
    try:
        config = Config()
        vector_store = VectorStore(
            config.CHROMA_PATH,
            config.EMBEDDING_MODEL,
            config.MAX_RESULTS
        )
        
        tool = CourseSearchTool(vector_store)
        
        # Test tool definition
        definition = tool.get_tool_definition()
        print(f"✓ Tool definition created: {definition['name']}")
        
        # Test execute method
        print("Testing tool.execute('Python basics')...")
        result = tool.execute(query="Python basics")
        
        if "error" in result.lower() or "failed" in result.lower():
            print(f"❌ Tool execution returned error: {result}")
            return False
        
        print(f"✓ Tool executed successfully")
        print(f"  - Result preview: {result[:200]}...")
        print(f"  - Sources tracked: {tool.last_sources}")
        
        return True
    except Exception as e:
        print(f"❌ CourseSearchTool Error: {e}")
        traceback.print_exc()
        return False


def test_ai_generator():
    """Test 5: Check AIGenerator without tools"""
    print("\n=== TEST 5: AIGenerator (No Tools) ===")
    try:
        config = Config()
        
        if not config.ANTHROPIC_API_KEY:
            print("⚠️  Skipping - No API key")
            return False
        
        ai_gen = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        
        # Test simple query without tools
        print("Testing simple query without tools...")
        response = ai_gen.generate_response(
            query="What is 2 + 2?",
            conversation_history=None,
            tools=None,
            tool_manager=None
        )
        
        print(f"✓ AI responded: {response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ AIGenerator Error: {e}")
        traceback.print_exc()
        return False


def test_ai_with_tools():
    """Test 6: Check AIGenerator with tools"""
    print("\n=== TEST 6: AIGenerator with Tools ===")
    try:
        config = Config()
        
        if not config.ANTHROPIC_API_KEY:
            print("⚠️  Skipping - No API key")
            return False
        
        # Set up components
        vector_store = VectorStore(
            config.CHROMA_PATH,
            config.EMBEDDING_MODEL,
            config.MAX_RESULTS
        )
        
        ai_gen = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        
        # Set up tool manager
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(vector_store)
        tool_manager.register_tool(search_tool)
        
        # Test query with tools
        print("Testing query with tools: 'What is Python?'...")
        response = ai_gen.generate_response(
            query="What is Python?",
            conversation_history=None,
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        if "query failed" in response.lower():
            print(f"❌ Response contains 'query failed': {response}")
            return False
        
        print(f"✓ AI responded with tools: {response[:200]}...")
        return True
    except Exception as e:
        print(f"❌ AI with Tools Error: {e}")
        traceback.print_exc()
        return False


def test_full_rag_system():
    """Test 7: Check complete RAG system"""
    print("\n=== TEST 7: Complete RAG System ===")
    try:
        config = Config()
        
        if not config.ANTHROPIC_API_KEY:
            print("⚠️  Skipping - No API key")
            return False
        
        rag_system = RAGSystem(config)
        
        # Test query
        print("Testing RAG query: 'What is Python?'...")
        response, sources = rag_system.query("What is Python?")
        
        if "query failed" in response.lower():
            print(f"❌ RAG returned 'query failed': {response}")
            print(f"  - Sources: {sources}")
            return False
        
        print(f"✓ RAG responded: {response[:200]}...")
        print(f"  - Sources: {sources}")
        return True
    except Exception as e:
        print(f"❌ RAG System Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests"""
    print("=" * 60)
    print("RAG CHATBOT DIAGNOSTIC TESTS")
    print("=" * 60)
    
    tests = [
        test_config,
        test_vector_store,
        test_search_functionality,
        test_course_search_tool,
        test_ai_generator,
        test_ai_with_tools,
        test_full_rag_system
    ]
    
    results = []
    for test_func in tests:
        try:
            success = test_func()
            results.append((test_func.__name__, success))
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_func.__name__}: {e}")
            results.append((test_func.__name__, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    failed_count = sum(1 for _, success in results if not success)
    if failed_count > 0:
        print(f"\n⚠️  {failed_count} test(s) failed!")
        print("\nPOSSIBLE ISSUES:")
        
        if not results[0][1]:  # Config test failed
            print("1. Missing ANTHROPIC_API_KEY in .env file")
        
        if not results[1][1]:  # Vector store failed
            print("2. ChromaDB initialization issue")
        
        if not results[2][1]:  # Search failed
            print("3. No documents indexed or search error")
        
        if results[5][1] and not results[6][1]:  # AI works but not RAG
            print("4. Integration issue between components")
    else:
        print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()