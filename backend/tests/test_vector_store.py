import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import json

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test suite for SearchResults class"""

    def test_from_chroma_with_data(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'key1': 'value1'}, {'key2': 'value2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'key1': 'value1'}, {'key2': 'value2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error"""
        results = SearchResults.empty("Test error message")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error message"

    def test_is_empty(self):
        """Test is_empty method"""
        empty_results = SearchResults([], [], [])
        non_empty_results = SearchResults(['doc'], [{}], [0.1])
        
        assert empty_results.is_empty()
        assert not non_empty_results.is_empty()


class TestVectorStore:
    """Test suite for VectorStore class"""

    @pytest.fixture
    def mock_chroma_client(self):
        """Create mock ChromaDB client"""
        with patch('vector_store.chromadb.PersistentClient') as mock_client:
            yield mock_client

    @pytest.fixture
    def mock_embedding_function(self):
        """Create mock embedding function"""
        with patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_emb:
            yield mock_emb

    @pytest.fixture
    def vector_store(self, mock_chroma_client, mock_embedding_function):
        """Create VectorStore instance with mocked dependencies"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            mock_content_collection
        ]
        
        store = VectorStore("./test_chroma", "test-model", max_results=5)
        store.course_catalog = mock_catalog_collection
        store.course_content = mock_content_collection
        return store

    def test_init(self, mock_chroma_client, mock_embedding_function):
        """Test VectorStore initialization"""
        mock_client_instance = Mock()
        mock_chroma_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        store = VectorStore("./test_path", "test-model", max_results=10)
        
        # Verify ChromaDB client was created with correct parameters
        mock_chroma_client.assert_called_once()
        call_args = mock_chroma_client.call_args
        assert "./test_path" in str(call_args)
        
        # Verify embedding function was created
        mock_embedding_function.assert_called_once_with(model_name="test-model")
        
        assert store.max_results == 10

    def test_search_successful(self, vector_store):
        """Test successful search operation"""
        # Mock course content query
        mock_query_result = {
            'documents': [['Sample document content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        vector_store.course_content.query.return_value = mock_query_result
        
        result = vector_store.search("test query")
        
        assert not result.is_empty()
        assert result.error is None
        assert result.documents == ['Sample document content']
        assert result.metadata == [{'course_title': 'Test Course', 'lesson_number': 1}]

    def test_search_with_course_name_resolution(self, vector_store):
        """Test search with course name that needs resolution"""
        # Mock course name resolution
        vector_store._resolve_course_name = Mock(return_value="Resolved Course Title")
        
        # Mock successful content search
        mock_query_result = {
            'documents': [['Content from resolved course']],
            'metadatas': [[{'course_title': 'Resolved Course Title'}]],
            'distances': [[0.2]]
        }
        vector_store.course_content.query.return_value = mock_query_result
        
        result = vector_store.search("test query", course_name="Partial Course Name")
        
        # Verify course name resolution was called
        vector_store._resolve_course_name.assert_called_once_with("Partial Course Name")
        
        # Verify search was performed with resolved name
        vector_store.course_content.query.assert_called_once()
        call_kwargs = vector_store.course_content.query.call_args[1]
        assert call_kwargs['where'] == {'course_title': 'Resolved Course Title'}

    def test_search_course_not_found(self, vector_store):
        """Test search when course name cannot be resolved"""
        vector_store._resolve_course_name = Mock(return_value=None)
        
        result = vector_store.search("test query", course_name="Nonexistent Course")
        
        assert result.is_empty()
        assert "No course found matching 'Nonexistent Course'" in result.error

    def test_search_with_filters(self, vector_store):
        """Test search with course title and lesson number filters"""
        mock_query_result = {
            'documents': [['Filtered content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 2}]],
            'distances': [[0.15]]
        }
        vector_store.course_content.query.return_value = mock_query_result
        
        result = vector_store.search(
            "test query",
            course_name="Test Course",
            lesson_number=2
        )
        
        # Verify the correct filter was applied
        call_kwargs = vector_store.course_content.query.call_args[1]
        expected_filter = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 2}
            ]
        }
        assert call_kwargs['where'] == expected_filter

    def test_search_exception_handling(self, vector_store):
        """Test search exception handling"""
        vector_store.course_content.query.side_effect = Exception("Database error")
        
        result = vector_store.search("test query")
        
        assert result.is_empty()
        assert "Search error: Database error" in result.error

    def test_resolve_course_name_success(self, vector_store):
        """Test successful course name resolution"""
        mock_query_result = {
            'documents': [['Course Title']],
            'metadatas': [[{'title': 'Exact Course Title'}]],
            'distances': [[0.1]]
        }
        vector_store.course_catalog.query.return_value = mock_query_result
        
        result = vector_store._resolve_course_name("Partial Title")
        
        assert result == "Exact Course Title"
        vector_store.course_catalog.query.assert_called_once_with(
            query_texts=["Partial Title"],
            n_results=1
        )

    def test_resolve_course_name_not_found(self, vector_store):
        """Test course name resolution when no match found"""
        mock_query_result = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        vector_store.course_catalog.query.return_value = mock_query_result
        
        result = vector_store._resolve_course_name("Nonexistent Course")
        
        assert result is None

    def test_build_filter_combinations(self, vector_store):
        """Test different filter combinations"""
        # No filters
        assert vector_store._build_filter(None, None) is None
        
        # Course title only
        result = vector_store._build_filter("Course A", None)
        assert result == {"course_title": "Course A"}
        
        # Lesson number only
        result = vector_store._build_filter(None, 3)
        assert result == {"lesson_number": 3}
        
        # Both filters
        result = vector_store._build_filter("Course B", 2)
        expected = {
            "$and": [
                {"course_title": "Course B"},
                {"lesson_number": 2}
            ]
        }
        assert result == expected

    def test_add_course_metadata(self, vector_store):
        """Test adding course metadata to catalog"""
        # Create test course
        lessons = [
            Lesson(lesson_number=1, title="Introduction", content="Intro content", lesson_link="http://lesson1"),
            Lesson(lesson_number=2, title="Advanced Topics", content="Advanced content", lesson_link="http://lesson2")
        ]
        course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="http://course",
            lessons=lessons
        )
        
        vector_store.add_course_metadata(course)
        
        # Verify course catalog was called correctly
        vector_store.course_catalog.add.assert_called_once()
        call_args = vector_store.course_catalog.add.call_args
        
        # Check documents parameter
        assert call_args[1]['documents'] == ["Test Course"]
        
        # Check IDs parameter
        assert call_args[1]['ids'] == ["Test Course"]
        
        # Check metadata
        metadata = call_args[1]['metadatas'][0]
        assert metadata['title'] == "Test Course"
        assert metadata['instructor'] == "Test Instructor"
        assert metadata['course_link'] == "http://course"
        assert metadata['lesson_count'] == 2
        
        # Check lessons JSON
        lessons_data = json.loads(metadata['lessons_json'])
        assert len(lessons_data) == 2
        assert lessons_data[0]['lesson_number'] == 1
        assert lessons_data[0]['lesson_title'] == "Introduction"
        assert lessons_data[0]['lesson_link'] == "http://lesson1"

    def test_add_course_content(self, vector_store):
        """Test adding course content chunks"""
        chunks = [
            CourseChunk(
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
                content="First chunk content"
            ),
            CourseChunk(
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1,
                content="Second chunk content"
            )
        ]
        
        vector_store.add_course_content(chunks)
        
        # Verify course content was called correctly
        vector_store.course_content.add.assert_called_once()
        call_args = vector_store.course_content.add.call_args
        
        # Check documents
        expected_docs = ["First chunk content", "Second chunk content"]
        assert call_args[1]['documents'] == expected_docs
        
        # Check metadata
        expected_metadata = [
            {'course_title': 'Test Course', 'lesson_number': 1, 'chunk_index': 0},
            {'course_title': 'Test Course', 'lesson_number': 1, 'chunk_index': 1}
        ]
        assert call_args[1]['metadatas'] == expected_metadata
        
        # Check IDs
        expected_ids = ["Test_Course_0", "Test_Course_1"]
        assert call_args[1]['ids'] == expected_ids

    def test_get_existing_course_titles(self, vector_store):
        """Test getting existing course titles"""
        mock_result = {
            'ids': ['Course A', 'Course B', 'Course C']
        }
        vector_store.course_catalog.get.return_value = mock_result
        
        titles = vector_store.get_existing_course_titles()
        
        assert titles == ['Course A', 'Course B', 'Course C']

    def test_get_existing_course_titles_error(self, vector_store):
        """Test error handling in get_existing_course_titles"""
        vector_store.course_catalog.get.side_effect = Exception("Database error")
        
        titles = vector_store.get_existing_course_titles()
        
        assert titles == []

    def test_get_course_count(self, vector_store):
        """Test getting course count"""
        mock_result = {
            'ids': ['Course A', 'Course B']
        }
        vector_store.course_catalog.get.return_value = mock_result
        
        count = vector_store.get_course_count()
        
        assert count == 2

    def test_get_all_courses_metadata(self, vector_store):
        """Test getting all courses metadata with JSON parsing"""
        lessons_json = json.dumps([
            {"lesson_number": 1, "lesson_title": "Lesson 1", "lesson_link": "http://lesson1"},
            {"lesson_number": 2, "lesson_title": "Lesson 2", "lesson_link": "http://lesson2"}
        ])
        
        mock_result = {
            'metadatas': [
                {
                    'title': 'Course A',
                    'instructor': 'Instructor A',
                    'course_link': 'http://coursea',
                    'lessons_json': lessons_json,
                    'lesson_count': 2
                }
            ]
        }
        vector_store.course_catalog.get.return_value = mock_result
        
        metadata = vector_store.get_all_courses_metadata()
        
        assert len(metadata) == 1
        course_meta = metadata[0]
        assert course_meta['title'] == 'Course A'
        assert 'lessons_json' not in course_meta  # Should be removed after parsing
        assert 'lessons' in course_meta
        assert len(course_meta['lessons']) == 2
        assert course_meta['lessons'][0]['lesson_title'] == 'Lesson 1'

    def test_get_course_link(self, vector_store):
        """Test getting course link by title"""
        mock_result = {
            'metadatas': [{'course_link': 'http://example.com/course'}]
        }
        vector_store.course_catalog.get.return_value = mock_result
        
        link = vector_store.get_course_link("Test Course")
        
        assert link == 'http://example.com/course'
        vector_store.course_catalog.get.assert_called_once_with(ids=["Test Course"])

    def test_get_lesson_link(self, vector_store):
        """Test getting lesson link by course title and lesson number"""
        lessons_json = json.dumps([
            {"lesson_number": 1, "lesson_title": "Lesson 1", "lesson_link": "http://lesson1"},
            {"lesson_number": 2, "lesson_title": "Lesson 2", "lesson_link": "http://lesson2"}
        ])
        
        mock_result = {
            'metadatas': [{'lessons_json': lessons_json}]
        }
        vector_store.course_catalog.get.return_value = mock_result
        
        link = vector_store.get_lesson_link("Test Course", 2)
        
        assert link == "http://lesson2"

    def test_get_lesson_link_not_found(self, vector_store):
        """Test getting lesson link when lesson number not found"""
        lessons_json = json.dumps([
            {"lesson_number": 1, "lesson_title": "Lesson 1", "lesson_link": "http://lesson1"}
        ])
        
        mock_result = {
            'metadatas': [{'lessons_json': lessons_json}]
        }
        vector_store.course_catalog.get.return_value = mock_result
        
        link = vector_store.get_lesson_link("Test Course", 5)
        
        assert link is None

    def test_clear_all_data(self, vector_store, mock_chroma_client):
        """Test clearing all data from collections"""
        mock_client_instance = mock_chroma_client.return_value
        
        # Mock new collections after deletion
        new_catalog = Mock()
        new_content = Mock() 
        mock_client_instance.get_or_create_collection.side_effect = [new_catalog, new_content]
        
        vector_store.clear_all_data()
        
        # Verify collections were deleted
        mock_client_instance.delete_collection.assert_any_call("course_catalog")
        mock_client_instance.delete_collection.assert_any_call("course_content")
        
        # Verify new collections were created
        assert vector_store.course_catalog is new_catalog
        assert vector_store.course_content is new_content


class TestVectorStoreIntegration:
    """Integration tests for VectorStore with more realistic scenarios"""

    def test_realistic_search_scenario(self):
        """Test realistic search scenario with proper data"""
        with patch('vector_store.chromadb.PersistentClient') as mock_client_class:
            with patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
                
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                
                mock_catalog = Mock()
                mock_content = Mock()
                mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
                
                # Create vector store
                store = VectorStore("./test", "model", 5)
                
                # Mock realistic search result
                realistic_query_result = {
                    'documents': [['Python functions are defined using the def keyword...']],
                    'metadatas': [[{
                        'course_title': 'Python Programming Fundamentals',
                        'lesson_number': 3,
                        'chunk_index': 0
                    }]],
                    'distances': [[0.12]]
                }
                mock_content.query.return_value = realistic_query_result
                
                result = store.search("How to define functions in Python")
                
                assert not result.is_empty()
                assert "Python functions are defined" in result.documents[0]
                assert result.metadata[0]['course_title'] == 'Python Programming Fundamentals'
                assert result.metadata[0]['lesson_number'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])