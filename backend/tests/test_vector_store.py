import pytest
from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


class TestVectorStore:
    """Test suite for VectorStore search functionality"""

    def test_search_with_normal_max_results(self, vector_store):
        """Test search with normal max_results setting"""
        # Ensure max_results is set to a reasonable value
        vector_store.max_results = 5

        # Add some test data
        course = Course(
            title="Search Test Course",
            lessons=[Lesson(lesson_number=1, title="Test Lesson")],
        )
        chunks = [
            CourseChunk(
                content="This is test content about testing",
                course_title="Search Test Course",
                lesson_number=1,
                chunk_index=0,
            )
        ]

        vector_store.add_course_metadata(course)
        vector_store.add_course_content(chunks)

        # Perform search
        results = vector_store.search("testing")

        assert not results.is_empty()
        assert len(results.documents) > 0
        assert results.error is None

    def test_search_with_zero_max_results(self, vector_store):
        """Test search with zero max_results (current config issue)"""
        # Set max_results to 0 to simulate the config issue
        vector_store.max_results = 0

        # Add some test data
        course = Course(
            title="Zero Results Test Course",
            lessons=[Lesson(lesson_number=1, title="Test Lesson")],
        )
        chunks = [
            CourseChunk(
                content="This is test content",
                course_title="Zero Results Test Course",
                lesson_number=1,
                chunk_index=0,
            )
        ]

        vector_store.add_course_metadata(course)
        vector_store.add_course_content(chunks)

        # Perform search - should return empty results due to limit=0
        results = vector_store.search("test")

        # With max_results=0, search should return no results
        assert results.is_empty() or len(results.documents) == 0

    def test_search_with_explicit_limit(self, vector_store):
        """Test search with explicitly provided limit"""
        # Add test data
        course = Course(
            title="Explicit Limit Test Course",
            lessons=[Lesson(lesson_number=1, title="Test Lesson")],
        )
        chunks = [
            CourseChunk(
                content=f"Test content {i}",
                course_title="Explicit Limit Test Course",
                lesson_number=1,
                chunk_index=i,
            )
            for i in range(10)
        ]

        vector_store.add_course_metadata(course)
        vector_store.add_course_content(chunks)

        # Search with explicit limit should override max_results
        results = vector_store.search("test", limit=3)

        assert not results.is_empty()
        assert len(results.documents) <= 3

    def test_search_course_name_resolution(self, populated_vector_store):
        """Test course name resolution in search"""
        results = populated_vector_store.search("test", course_name="Test Course")

        # Should find the course and return results
        if not results.is_empty():
            for metadata in results.metadata:
                assert metadata.get("course_title") == "Test Course"

    def test_search_nonexistent_course(self, populated_vector_store):
        """Test search for nonexistent course"""
        results = populated_vector_store.search(
            "test", course_name="Nonexistent Course"
        )

        # Should return error about course not found
        assert results.error is not None
        assert "No course found matching" in results.error

    def test_search_with_lesson_filter(self, populated_vector_store):
        """Test search with lesson number filter"""
        results = populated_vector_store.search("test", lesson_number=1)

        if not results.is_empty():
            for metadata in results.metadata:
                assert metadata.get("lesson_number") == 1

    def test_get_existing_course_titles(self, populated_vector_store):
        """Test getting existing course titles"""
        titles = populated_vector_store.get_existing_course_titles()

        assert isinstance(titles, list)
        assert "Test Course" in titles

    def test_get_course_count(self, populated_vector_store):
        """Test getting course count"""
        count = populated_vector_store.get_course_count()

        assert isinstance(count, int)
        assert count >= 1  # At least the test course

    def test_get_lesson_link(self, populated_vector_store):
        """Test getting lesson link"""
        link = populated_vector_store.get_lesson_link("Test Course", 1)

        # Should return the link or None
        assert link is None or isinstance(link, str)

    def test_clear_all_data(self, vector_store):
        """Test clearing all vector store data"""
        # Add some data first
        course = Course(title="Temp Course", lessons=[])
        vector_store.add_course_metadata(course)

        # Verify data exists
        count_before = vector_store.get_course_count()
        assert count_before > 0

        # Clear data
        vector_store.clear_all_data()

        # Verify data is cleared
        count_after = vector_store.get_course_count()
        assert count_after == 0


class TestSearchResults:
    """Test SearchResults utility class"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"key": "value1"}, {"key": "value2"}]],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert len(results.documents) == 2
        assert len(results.metadata) == 2
        assert len(results.distances) == 2
        assert results.error is None

    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = SearchResults.from_chroma(chroma_results)

        assert len(results.documents) == 0
        assert len(results.metadata) == 0
        assert len(results.distances) == 0
        assert results.error is None

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error"""
        results = SearchResults.empty("Test error message")

        assert results.is_empty()
        assert results.error == "Test error message"
        assert len(results.documents) == 0

    def test_is_empty_method(self):
        """Test is_empty method"""
        # Empty results
        empty_results = SearchResults([], [], [])
        assert empty_results.is_empty()

        # Non-empty results
        non_empty_results = SearchResults(["doc"], [{"key": "value"}], [0.1])
        assert not non_empty_results.is_empty()
