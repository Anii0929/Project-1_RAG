import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from fastapi import HTTPException
from fastapi.testclient import TestClient


class TestQueryEndpoint:
    """Test suite for /api/query endpoint"""

    @pytest.mark.api
    def test_query_with_session_id(self, client, mock_rag_system, api_test_data):
        """Test query endpoint with provided session_id"""
        # Setup mock response
        mock_rag_system.query.return_value = (
            "Python is a high-level programming language known for its simplicity.",
            [{"text": "Python Basics - Lesson 1", "link": "https://example.com/lesson1"}]
        )
        
        response = client.post("/api/query", json=api_test_data["valid_query"])
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Verify content
        assert data["session_id"] == "test_session_123"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Python Basics - Lesson 1"
        
        # Verify mock was called correctly
        mock_rag_system.query.assert_called_once_with(
            "What is Python programming?", 
            "test_session_123"
        )

    @pytest.mark.api
    def test_query_without_session_id(self, client, mock_rag_system, api_test_data):
        """Test query endpoint without session_id (auto-generation)"""
        mock_rag_system.query.return_value = (
            "Variables in Python are containers for storing data.",
            [{"text": "Python Basics - Lesson 2", "link": "https://example.com/lesson2"}]
        )
        
        response = client.post("/api/query", json=api_test_data["query_without_session"])
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify session was auto-created
        assert data["session_id"] == "test_session_123"
        
        # Verify session creation was called
        mock_rag_system.session_manager.create_session.assert_called_once()
        
        # Verify query was processed
        mock_rag_system.query.assert_called_once_with(
            "Explain variables in Python",
            "test_session_123"
        )

    @pytest.mark.api
    def test_query_empty_string(self, client):
        """Test query endpoint with empty query string"""
        response = client.post("/api/query", json={"query": ""})
        
        assert response.status_code == 200
        # Should still process empty queries (let RAG system handle validation)

    @pytest.mark.api
    def test_query_missing_field(self, client):
        """Test query endpoint with missing required field"""
        response = client.post("/api/query", json={"session_id": "test"})
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.api
    def test_query_invalid_json(self, client):
        """Test query endpoint with invalid JSON"""
        response = client.post(
            "/api/query", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

    @pytest.mark.api
    def test_query_rag_system_error(self, client, mock_rag_system, error_scenarios):
        """Test query endpoint when RAG system throws exception"""
        mock_rag_system.query.side_effect = error_scenarios["rag_system_error"]
        
        response = client.post("/api/query", json={"query": "test query"})
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "RAG system processing failed" in data["detail"]

    @pytest.mark.api
    def test_query_session_error(self, client, mock_rag_system, error_scenarios):
        """Test query endpoint when session creation fails"""
        mock_rag_system.session_manager.create_session.side_effect = error_scenarios["session_error"]
        
        response = client.post("/api/query", json={"query": "test query"})
        
        assert response.status_code == 500

    @pytest.mark.api
    def test_query_long_text(self, client, mock_rag_system):
        """Test query endpoint with very long query text"""
        long_query = "What is Python? " * 1000  # Very long query
        
        mock_rag_system.query.return_value = (
            "Python is a programming language.",
            []
        )
        
        response = client.post("/api/query", json={"query": long_query})
        
        assert response.status_code == 200
        # Verify the long query was passed through
        mock_rag_system.query.assert_called_once()
        called_query = mock_rag_system.query.call_args[0][0]
        assert len(called_query) == len(long_query)

    @pytest.mark.api
    def test_query_special_characters(self, client, mock_rag_system):
        """Test query endpoint with special characters"""
        special_query = "What about 'Python' & \"programming\"? 🐍 <script>alert('test')</script>"
        
        mock_rag_system.query.return_value = (
            "Python is safe from injection attacks.",
            []
        )
        
        response = client.post("/api/query", json={"query": special_query})
        
        assert response.status_code == 200
        # Verify special characters were preserved
        mock_rag_system.query.assert_called_once_with(special_query, "test_session_123")


class TestCoursesEndpoint:
    """Test suite for /api/courses endpoint"""

    @pytest.mark.api
    def test_get_courses_success(self, client, mock_rag_system, api_test_data):
        """Test successful course statistics retrieval"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Verify content matches expected data
        expected = api_test_data["expected_stats"]
        assert data["total_courses"] == expected["total_courses"]
        assert data["course_titles"] == expected["course_titles"]
        
        # Verify mock was called
        mock_rag_system.get_course_analytics.assert_called_once()

    @pytest.mark.api
    def test_get_courses_empty_result(self, client, mock_rag_system):
        """Test courses endpoint with no courses"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    @pytest.mark.api
    def test_get_courses_rag_system_error(self, client, mock_rag_system, error_scenarios):
        """Test courses endpoint when RAG system throws exception"""
        mock_rag_system.get_course_analytics.side_effect = error_scenarios["rag_system_error"]
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "RAG system processing failed" in data["detail"]

    @pytest.mark.api
    def test_get_courses_many_courses(self, client, mock_rag_system):
        """Test courses endpoint with many courses"""
        many_courses = [f"Course {i}" for i in range(100)]
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 100,
            "course_titles": many_courses
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 100
        assert len(data["course_titles"]) == 100


class TestRootEndpoint:
    """Test suite for root endpoint /"""

    @pytest.mark.api
    def test_root_endpoint(self, client):
        """Test root endpoint returns expected message"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Test RAG System API"

    @pytest.mark.api
    def test_root_endpoint_method_not_allowed(self, client):
        """Test root endpoint with POST method"""
        response = client.post("/")
        
        assert response.status_code == 405  # Method not allowed


class TestRequestValidation:
    """Test suite for request validation and edge cases"""

    @pytest.mark.api
    def test_query_request_validation(self, client):
        """Test query request model validation"""
        # Test with extra fields (should be ignored)
        response = client.post("/api/query", json={
            "query": "test query",
            "session_id": "test_session",
            "extra_field": "should be ignored"
        })
        
        assert response.status_code == 200

    @pytest.mark.api
    def test_query_null_session_id(self, client, mock_rag_system):
        """Test query with null session_id"""
        mock_rag_system.query.return_value = ("Response", [])
        
        response = client.post("/api/query", json={
            "query": "test query",
            "session_id": None
        })
        
        assert response.status_code == 200
        # Should auto-create session when null
        mock_rag_system.session_manager.create_session.assert_called_once()

    @pytest.mark.api
    def test_content_type_validation(self, client):
        """Test endpoint with different content types"""
        # Test with form data instead of JSON
        response = client.post(
            "/api/query",
            data={"query": "test"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code == 422  # Should expect JSON


class TestResponseValidation:
    """Test suite for response validation and structure"""

    @pytest.mark.api
    def test_query_response_structure(self, client, mock_rag_system):
        """Test that query response matches expected structure"""
        mock_rag_system.query.return_value = (
            "Test answer",
            [
                {"text": "Source 1", "link": "https://example.com/1"},
                {"text": "Source 2", "link": None}  # Test with null link
            ]
        )
        
        response = client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields are present
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Verify source structure
        for source in data["sources"]:
            assert "text" in source
            assert "link" in source
            assert isinstance(source["text"], str)
            # link can be string or null

    @pytest.mark.api
    def test_courses_response_structure(self, client, mock_rag_system):
        """Test that courses response matches expected structure"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 5,
            "course_titles": ["Course A", "Course B", "Course C", "Course D", "Course E"]
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify structure
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert all(isinstance(title, str) for title in data["course_titles"])
        assert data["total_courses"] == len(data["course_titles"])


class TestErrorHandling:
    """Test suite for comprehensive error handling"""

    @pytest.mark.api
    def test_404_endpoints(self, client):
        """Test non-existent endpoints return 404"""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
        
        response = client.post("/api/invalid")
        assert response.status_code == 404

    @pytest.mark.api
    def test_method_not_allowed(self, client):
        """Test wrong HTTP methods return 405"""
        response = client.delete("/api/query")
        assert response.status_code == 405
        
        response = client.put("/api/courses")
        assert response.status_code == 405

    @pytest.mark.api
    def test_large_payload(self, client, mock_rag_system):
        """Test handling of very large request payloads"""
        # Create a very large query
        large_query = "x" * (1024 * 1024)  # 1MB query
        
        mock_rag_system.query.return_value = ("Response", [])
        
        response = client.post("/api/query", json={"query": large_query})
        
        # Should handle large payloads gracefully
        assert response.status_code in [200, 413, 422]  # Success, payload too large, or validation error


class TestIntegrationScenarios:
    """Integration test scenarios"""

    @pytest.mark.integration
    @pytest.mark.api
    def test_query_to_courses_flow(self, client, mock_rag_system):
        """Test typical user flow: query -> get courses"""
        # First, make a query
        mock_rag_system.query.return_value = (
            "Python is great for beginners.",
            [{"text": "Python Course - Lesson 1", "link": "https://example.com/lesson1"}]
        )
        
        query_response = client.post("/api/query", json={"query": "Tell me about Python"})
        assert query_response.status_code == 200
        
        # Then, get course information  
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Python Programming Basics"]
        }
        
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == 200
        
        # Verify both responses are consistent
        query_data = query_response.json()
        courses_data = courses_response.json()
        
        assert "Python" in query_data["sources"][0]["text"]
        assert "Python" in courses_data["course_titles"][0]

    @pytest.mark.integration
    @pytest.mark.api
    def test_multiple_queries_same_session(self, client, mock_rag_system):
        """Test multiple queries with the same session"""
        session_id = "persistent_session_123"
        
        # Mock consistent session behavior
        mock_rag_system.session_manager.create_session.return_value = session_id
        mock_rag_system.query.return_value = ("Response", [])
        
        # First query
        response1 = client.post("/api/query", json={
            "query": "What is Python?",
            "session_id": session_id
        })
        
        # Second query with same session
        response2 = client.post("/api/query", json={
            "query": "Tell me more about variables",
            "session_id": session_id
        })
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Both should use the same session
        assert response1.json()["session_id"] == session_id
        assert response2.json()["session_id"] == session_id
        
        # Verify RAG system was called with correct session
        assert mock_rag_system.query.call_count == 2
        for call in mock_rag_system.query.call_args_list:
            assert call[0][1] == session_id  # Second argument is session_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])