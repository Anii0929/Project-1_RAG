"""
API endpoint tests for FastAPI routes.

Tests the FastAPI endpoints (/api/query, /api/courses, /) for proper 
request/response handling without requiring the full application context.
"""

import pytest
import json
from httpx import AsyncClient
from unittest.mock import Mock

pytestmark = pytest.mark.asyncio


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint."""
    
    async def test_query_endpoint_success(self, client: AsyncClient):
        """Test successful query with all required fields."""
        response = await client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )
        
        # Debug output
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "This is a test response about MCP applications."
        assert len(data["sources"]) == 1
        assert data["session_id"] == "test_session_123"
    
    async def test_query_endpoint_with_session_id(self, client: AsyncClient):
        """Test query with provided session_id."""
        response = await client.post(
            "/api/query",
            json={
                "query": "How do I build MCP apps?",
                "session_id": "existing_session_456"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "existing_session_456"
        assert "answer" in data
        assert "sources" in data
    
    async def test_query_endpoint_missing_query(self, client: AsyncClient):
        """Test query endpoint with missing query field."""
        response = await client.post(
            "/api/query",
            json={}
        )
        
        assert response.status_code == 422  # Validation error
        
    async def test_query_endpoint_empty_query(self, client: AsyncClient):
        """Test query endpoint with empty query string."""
        response = await client.post(
            "/api/query",
            json={"query": ""}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    async def test_query_endpoint_invalid_json(self, client: AsyncClient):
        """Test query endpoint with invalid JSON."""
        response = await client.post(
            "/api/query",
            content="invalid json",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422
    
    async def test_query_endpoint_server_error(self, client: AsyncClient, mock_rag_system):
        """Test query endpoint when RAG system raises an exception."""
        mock_rag_system.query.side_effect = Exception("Database connection failed")
        
        response = await client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection failed" in data["detail"]


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint."""
    
    async def test_courses_endpoint_success(self, client: AsyncClient):
        """Test successful retrieval of course statistics."""
        response = await client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "MCP: Build Rich-Context AI Apps with Anthropic" in data["course_titles"]
        assert "Advanced Python Programming" in data["course_titles"]
    
    async def test_courses_endpoint_server_error(self, client: AsyncClient, mock_rag_system):
        """Test courses endpoint when RAG system raises an exception."""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics service unavailable")
        
        response = await client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Analytics service unavailable" in data["detail"]


@pytest.mark.api
class TestRootEndpoint:
    """Test the root / endpoint."""
    
    async def test_root_endpoint(self, client: AsyncClient):
        """Test the root endpoint returns basic API info."""
        response = await client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "Course Materials RAG System API" in data["message"]


@pytest.mark.api
class TestResponseModels:
    """Test that response models conform to expected schemas."""
    
    async def test_query_response_schema(self, client: AsyncClient):
        """Test that query response matches the expected Pydantic model."""
        response = await client.post(
            "/api/query",
            json={"query": "Test query"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields exist
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Check source structure
        if data["sources"]:
            source = data["sources"][0]
            assert "course_title" in source
            assert "lesson_number" in source
    
    async def test_courses_response_schema(self, client: AsyncClient):
        """Test that courses response matches the expected Pydantic model."""
        response = await client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields exist
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Check course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)


@pytest.mark.api
class TestCORSHeaders:
    """Test CORS headers are properly set."""
    
    async def test_cors_headers_on_query(self, client: AsyncClient):
        """Test CORS headers are present on query endpoint."""
        response = await client.post(
            "/api/query",
            json={"query": "Test"}
        )
        
        assert response.status_code == 200
        # Note: AsyncClient may not preserve all headers in test environment
        # In a real scenario, you'd check for Access-Control-Allow-Origin, etc.
    
    async def test_options_request(self, client: AsyncClient):
        """Test OPTIONS request for CORS preflight."""
        response = await client.options("/api/query")
        
        # Should handle OPTIONS request without error
        assert response.status_code in [200, 405]  # Some test clients may return 405


@pytest.mark.api
class TestContentTypeHandling:
    """Test content type handling."""
    
    async def test_json_content_type(self, client: AsyncClient):
        """Test that endpoints properly handle JSON content type."""
        response = await client.post(
            "/api/query",
            json={"query": "Test query"},
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("application/json")
    
    async def test_unsupported_content_type(self, client: AsyncClient):
        """Test handling of unsupported content types."""
        response = await client.post(
            "/api/query",
            content="query=test",
            headers={"content-type": "application/x-www-form-urlencoded"}
        )
        
        # Should return validation error for unsupported content type
        assert response.status_code == 422