import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch

@pytest.mark.api
class TestAPIEndpoints:
    """Test suite for API endpoints"""

    def test_root_endpoint(self, test_client):
        """Test the root endpoint returns correct response"""
        response = test_client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Course Materials RAG System Test API"}

    def test_create_new_session(self, test_client):
        """Test creating a new session"""
        response = test_client.post("/api/session/new")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"

    def test_create_new_session_with_previous_id(self, test_client, mock_rag_system):
        """Test creating a new session while clearing previous one"""
        response = test_client.post("/api/session/new?prev_session_id=old-session-456")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        
        # Verify clear_session was called with the previous session ID
        mock_rag_system.session_manager.clear_session.assert_called_once_with("old-session-456")

    def test_query_endpoint_with_session_id(self, test_client):
        """Test querying with provided session ID"""
        query_data = {
            "query": "What is machine learning?",
            "session_id": "existing-session-789"
        }
        response = test_client.post("/api/query", json=query_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "Test answer from the RAG system"
        assert data["session_id"] == "existing-session-789"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Test source content"
        assert data["sources"][0]["link"] == "https://example.com/test"

    def test_query_endpoint_without_session_id(self, test_client, mock_rag_system):
        """Test querying without session ID (should create new session)"""
        query_data = {
            "query": "Explain neural networks"
        }
        response = test_client.post("/api/query", json=query_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"  # From mock
        
        # Verify create_session was called
        mock_rag_system.session_manager.create_session.assert_called()

    def test_query_endpoint_invalid_data(self, test_client):
        """Test query endpoint with invalid data"""
        # Missing required query field
        response = test_client.post("/api/query", json={})
        assert response.status_code == 422  # Validation error

    def test_query_endpoint_empty_query(self, test_client):
        """Test query endpoint with empty query string"""
        query_data = {"query": ""}
        response = test_client.post("/api/query", json=query_data)
        assert response.status_code == 200  # Should still work with empty query

    def test_courses_endpoint(self, test_client):
        """Test getting course statistics"""
        response = test_client.get("/api/courses")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Test Course 1", "Test Course 2"]

    @patch('app.rag_system')
    def test_query_endpoint_error_handling(self, mock_rag_patch, test_client):
        """Test error handling in query endpoint"""
        # Make the mock RAG system raise an exception
        test_client.app.state.rag_system.query.side_effect = Exception("Test error")
        
        query_data = {"query": "What is AI?"}
        response = test_client.post("/api/query", json=query_data)
        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]

    @patch('app.rag_system')
    def test_courses_endpoint_error_handling(self, mock_rag_patch, test_client):
        """Test error handling in courses endpoint"""
        # Make the mock RAG system raise an exception
        test_client.app.state.rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client.get("/api/courses")
        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]

    @patch('app.rag_system')
    def test_session_endpoint_error_handling(self, mock_rag_patch, test_client):
        """Test error handling in session creation endpoint"""
        # Make the session manager raise an exception
        test_client.app.state.rag_system.session_manager.create_session.side_effect = Exception("Session error")
        
        response = test_client.post("/api/session/new")
        assert response.status_code == 500
        assert "Session error" in response.json()["detail"]

    def test_query_with_string_sources(self, test_client, mock_rag_system):
        """Test query handling when sources are strings (legacy format)"""
        # Configure mock to return string sources instead of dict sources
        mock_rag_system.query.return_value = (
            "Test answer",
            ["String source 1", "String source 2"]
        )
        
        query_data = {"query": "Test query"}
        response = test_client.post("/api/query", json=query_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["sources"]) == 2
        assert data["sources"][0]["text"] == "String source 1"
        assert data["sources"][0]["link"] is None
        assert data["sources"][1]["text"] == "String source 2"
        assert data["sources"][1]["link"] is None

    def test_query_with_mixed_sources(self, test_client, mock_rag_system):
        """Test query handling with mixed source formats"""
        # Configure mock to return mixed sources
        mock_rag_system.query.return_value = (
            "Test answer",
            [
                {"text": "Dict source with link", "link": "https://example.com"},
                "String source without link",
                {"text": "Dict source without link"}
            ]
        )
        
        query_data = {"query": "Test query"}
        response = test_client.post("/api/query", json=query_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["sources"]) == 3
        assert data["sources"][0]["text"] == "Dict source with link"
        assert data["sources"][0]["link"] == "https://example.com"
        assert data["sources"][1]["text"] == "String source without link"
        assert data["sources"][1]["link"] is None
        assert data["sources"][2]["text"] == "Dict source without link"
        assert data["sources"][2]["link"] is None

    def test_content_type_validation(self, test_client):
        """Test that endpoints require proper content type"""
        # Test with form data instead of JSON
        response = test_client.post("/api/query", data={"query": "test"})
        assert response.status_code == 422  # Should fail validation
        
        # Test with proper JSON
        response = test_client.post("/api/query", json={"query": "test"})
        assert response.status_code == 200

    def test_response_models(self, test_client):
        """Test that responses match expected Pydantic models"""
        # Test query response structure
        query_data = {"query": "Test query"}
        response = test_client.post("/api/query", json=query_data)
        data = response.json()
        
        required_fields = {"answer", "sources", "session_id"}
        assert required_fields.issubset(data.keys())
        
        # Test courses response structure
        response = test_client.get("/api/courses")
        data = response.json()
        
        required_fields = {"total_courses", "course_titles"}
        assert required_fields.issubset(data.keys())
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)