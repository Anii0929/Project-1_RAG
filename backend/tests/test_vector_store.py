"""
Tests for VectorStore functionality.

This module tests the VectorStore's search methods, course resolution,
filtering logic, and ChromaDB integration with various scenarios.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vector_store import VectorStore, SearchResults


class TestVectorStoreSearch:
    """Test suite for VectorStore search functionality."""
    
    @pytest.mark.unit
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_successful(self, mock_embedding_fn, mock_client):
        """Test successful search with results."""
        # Setup mocks
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [["Document 1", "Document 2"]],
            'metadatas': [[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 2}
            ]],
            'distances': [[0.1, 0.2]]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        # Create VectorStore
        store = VectorStore("./test_db", "test-model", max_results=5)
        store.course_content = mock_collection
        
        # Execute search
        results = store.search("test query")
        
        # Assertions
        assert not results.error
        assert len(results.documents) == 2
        assert results.documents == ["Document 1", "Document 2"]
        assert len(results.metadata) == 2
        assert results.metadata[0]["course_title"] == "Test Course"
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=None
        )
    
    @pytest.mark.unit
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_course_name_filter(self, mock_embedding_fn, mock_client):
        """Test search with course name filtering."""
        # Setup mocks
        mock_content_collection = Mock()
        mock_catalog_collection = Mock()
        
        # Mock course name resolution
        mock_catalog_collection.query.return_value = {
            'documents': [["MCP Course"]],
            'metadatas': [[{"title": "MCP: Build Rich-Context AI Apps with Anthropic"}]]
        }
        
        # Mock content search
        mock_content_collection.query.return_value = {
            'documents': [["MCP content"]],
            'metadatas': [[{"course_title": "MCP: Build Rich-Context AI Apps with Anthropic", "lesson_number": 1}]],
            'distances': [[0.1]]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [
            mock_catalog_collection,  # course_catalog
            mock_content_collection   # course_content
        ]
        mock_client.return_value = mock_client_instance
        
        # Create VectorStore
        store = VectorStore("./test_db", "test-model")
        
        # Execute search
        results = store.search("test query", course_name="MCP")
        
        # Assertions
        assert not results.error
        assert len(results.documents) == 1
        assert results.documents[0] == "MCP content"
        
        # Verify course resolution was called
        mock_catalog_collection.query.assert_called_once_with(
            query_texts=["MCP"],
            n_results=1
        )
        
        # Verify content search with filter
        mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"course_title": "MCP: Build Rich-Context AI Apps with Anthropic"}
        )
    
    @pytest.mark.unit
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_lesson_number_filter(self, mock_embedding_fn, mock_client):
        """Test search with lesson number filtering."""
        # Setup mocks
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [["Lesson 2 content"]],
            'metadatas': [[{"course_title": "Test Course", "lesson_number": 2}]],
            'distances': [[0.1]]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        # Create VectorStore
        store = VectorStore("./test_db", "test-model")
        store.course_content = mock_collection
        
        # Execute search
        results = store.search("test query", lesson_number=2)
        
        # Assertions
        assert not results.error
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"lesson_number": 2}
        )
    
    @pytest.mark.unit
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_both_filters(self, mock_embedding_fn, mock_client):
        """Test search with both course name and lesson number filters."""
        # Setup mocks
        mock_content_collection = Mock()
        mock_catalog_collection = Mock()
        
        # Mock course name resolution
        mock_catalog_collection.query.return_value = {
            'documents': [["Test Course"]],
            'metadatas': [[{"title": "Test Course"}]]
        }
        
        # Mock content search
        mock_content_collection.query.return_value = {
            'documents': [["Filtered content"]],
            'metadatas': [[{"course_title": "Test Course", "lesson_number": 3}]],
            'distances': [[0.1]]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            mock_content_collection
        ]
        mock_client.return_value = mock_client_instance
        
        # Create VectorStore
        store = VectorStore("./test_db", "test-model")
        
        # Execute search
        results = store.search("test query", course_name="Test", lesson_number=3)
        
        # Assertions
        assert not results.error
        mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"$and": [{"course_title": "Test Course"}, {"lesson_number": 3}]}
        )
    
    @pytest.mark.unit
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_course_not_found(self, mock_embedding_fn, mock_client):
        """Test search when course name cannot be resolved."""
        # Setup mocks
        mock_catalog_collection = Mock()
        mock_catalog_collection.query.return_value = {
            'documents': [[]],  # Empty results
            'metadatas': [[]]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_catalog_collection
        mock_client.return_value = mock_client_instance
        
        # Create VectorStore
        store = VectorStore("./test_db", "test-model")
        
        # Execute search
        results = store.search("test query", course_name="NonexistentCourse")
        
        # Assertions
        assert results.error == "No course found matching 'NonexistentCourse'"
        assert results.is_empty()
    
    @pytest.mark.unit
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_chromadb_exception(self, mock_embedding_fn, mock_client):
        """Test search handles ChromaDB exceptions."""
        # Setup mocks
        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("Database connection failed")
        
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        # Create VectorStore
        store = VectorStore("./test_db", "test-model")
        store.course_content = mock_collection
        
        # Execute search
        results = store.search("test query")
        
        # Assertions
        assert results.error == "Search error: Database connection failed"
        assert results.is_empty()
    
    @pytest.mark.unit
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_empty_results(self, mock_embedding_fn, mock_client):
        """Test search with no matching documents."""
        # Setup mocks
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [[]],  # No documents found
            'metadatas': [[]],
            'distances': [[]]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        # Create VectorStore
        store = VectorStore("./test_db", "test-model")
        store.course_content = mock_collection
        
        # Execute search
        results = store.search("nonexistent query")
        
        # Assertions
        assert not results.error
        assert results.is_empty()
        assert len(results.documents) == 0


class TestVectorStoreCourseResolution:
    """Test suite for course name resolution logic."""
    
    @pytest.mark.unit
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_resolve_course_name_found(self, mock_embedding_fn, mock_client):
        """Test successful course name resolution."""
        # Setup mocks
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [["MCP Course"]],
            'metadatas': [[{"title": "MCP: Build Rich-Context AI Apps with Anthropic"}]]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        # Create VectorStore
        store = VectorStore("./test_db", "test-model")
        
        # Test resolution
        resolved = store._resolve_course_name("MCP")
        
        assert resolved == "MCP: Build Rich-Context AI Apps with Anthropic"
        mock_collection.query.assert_called_once_with(
            query_texts=["MCP"],
            n_results=1
        )
    
    @pytest.mark.unit
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_resolve_course_name_not_found(self, mock_embedding_fn, mock_client):
        """Test course name resolution when not found."""
        # Setup mocks
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        # Create VectorStore
        store = VectorStore("./test_db", "test-model")
        
        # Test resolution
        resolved = store._resolve_course_name("NonexistentCourse")
        
        assert resolved is None
    
    @pytest.mark.unit
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_resolve_course_name_exception(self, mock_embedding_fn, mock_client):
        """Test course name resolution handles exceptions."""
        # Setup mocks
        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("Connection error")
        
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        # Create VectorStore
        store = VectorStore("./test_db", "test-model")
        
        # Test resolution
        resolved = store._resolve_course_name("TestCourse")
        
        assert resolved is None


class TestVectorStoreFiltering:
    """Test suite for ChromaDB filter building logic."""
    
    @pytest.fixture
    def store_instance(self):
        """Create a VectorStore instance for filter testing."""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            return VectorStore("./test_db", "test-model")
    
    @pytest.mark.unit
    def test_build_filter_none(self, store_instance):
        """Test filter building with no parameters."""
        filter_dict = store_instance._build_filter(None, None)
        assert filter_dict is None
    
    @pytest.mark.unit
    def test_build_filter_course_only(self, store_instance):
        """Test filter building with only course title."""
        filter_dict = store_instance._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}
    
    @pytest.mark.unit
    def test_build_filter_lesson_only(self, store_instance):
        """Test filter building with only lesson number."""
        filter_dict = store_instance._build_filter(None, 5)
        assert filter_dict == {"lesson_number": 5}
    
    @pytest.mark.unit
    def test_build_filter_both(self, store_instance):
        """Test filter building with both course and lesson."""
        filter_dict = store_instance._build_filter("Test Course", 3)
        expected = {"$and": [
            {"course_title": "Test Course"},
            {"lesson_number": 3}
        ]}
        assert filter_dict == expected
    
    @pytest.mark.unit
    def test_build_filter_lesson_zero(self, store_instance):
        """Test filter building with lesson number 0 (should be treated as valid)."""
        filter_dict = store_instance._build_filter("Test Course", 0)
        expected = {"$and": [
            {"course_title": "Test Course"},
            {"lesson_number": 0}
        ]}
        assert filter_dict == expected


class TestSearchResults:
    """Test suite for SearchResults utility class."""
    
    @pytest.mark.unit
    def test_from_chroma_success(self):
        """Test SearchResults creation from ChromaDB results."""
        chroma_results = {
            'documents': [["Doc 1", "Doc 2"]],
            'metadatas': [[{"course": "Test"}, {"course": "Test2"}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ["Doc 1", "Doc 2"]
        assert results.metadata == [{"course": "Test"}, {"course": "Test2"}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
    
    @pytest.mark.unit
    def test_from_chroma_empty(self):
        """Test SearchResults creation from empty ChromaDB results."""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None
    
    @pytest.mark.unit
    def test_empty_with_error(self):
        """Test empty SearchResults with error message."""
        results = SearchResults.empty("Test error message")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error message"
        assert results.is_empty() is True
    
    @pytest.mark.unit
    def test_is_empty_false(self):
        """Test is_empty returns False when there are documents."""
        results = SearchResults(
            documents=["Doc 1"],
            metadata=[{"test": "data"}],
            distances=[0.1]
        )
        
        assert results.is_empty() is False
    
    @pytest.mark.unit
    def test_is_empty_true(self):
        """Test is_empty returns True when no documents."""
        results = SearchResults(
            documents=[],
            metadata=[],
            distances=[]
        )
        
        assert results.is_empty() is True


# Integration tests with temporary ChromaDB
class TestVectorStoreIntegration:
    """Integration tests with temporary ChromaDB instance."""
    
    @pytest.mark.integration
    def test_end_to_end_search_no_data(self, temp_chroma_db):
        """Test search with empty database returns empty results."""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2")
        
        results = store.search("test query")
        
        assert not results.error
        assert results.is_empty()
    
    @pytest.mark.integration
    def test_initialization_success(self, temp_chroma_db):
        """Test VectorStore initializes successfully with temporary DB."""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2")
        
        assert store.course_catalog is not None
        assert store.course_content is not None
        assert store.max_results == 5