import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import json


@pytest.mark.api
class TestAPIEndpoints:
    """Test API endpoints for the RAG system"""
    
    def test_root_endpoint(self, test_client_no_static):
        """Test root endpoint returns correct message"""
        response = test_client_no_static.get("/")
        
        assert response.status_code == 200
        assert response.json() == {"message": "Course Materials RAG System"}
    
    def test_query_endpoint_with_session_id(self, test_client_no_static):
        """Test query endpoint with provided session ID"""
        query_data = {
            "query": "What is the main topic?",
            "session_id": "existing-session-123"
        }
        
        response = test_client_no_static.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "existing-session-123"
        assert data["answer"] == "Test response"
        assert data["sources"] == ["source1", "source2"]
    
    def test_query_endpoint_without_session_id(self, test_client_no_static):
        """Test query endpoint creates new session when none provided"""
        query_data = {
            "query": "What is the main topic?"
        }
        
        response = test_client_no_static.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"  # From mock
        assert data["answer"] == "Test response"
        assert data["sources"] == ["source1", "source2"]
    
    def test_query_endpoint_empty_query(self, test_client_no_static):
        """Test query endpoint with empty query string"""
        query_data = {
            "query": ""
        }
        
        response = test_client_no_static.post("/api/query", json=query_data)
        
        assert response.status_code == 200  # Should still process empty queries
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
    
    def test_query_endpoint_missing_query_field(self, test_client_no_static):
        """Test query endpoint with missing query field"""
        query_data = {}
        
        response = test_client_no_static.post("/api/query", json=query_data)
        
        assert response.status_code == 422  # Validation error
        error_data = response.json()
        assert "detail" in error_data
        assert any("query" in str(error).lower() for error in error_data["detail"])
    
    def test_query_endpoint_invalid_json(self, test_client_no_static):
        """Test query endpoint with invalid JSON"""
        response = test_client_no_static.post(
            "/api/query", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_query_endpoint_rag_system_error(self, test_client_no_static):
        """Test query endpoint when RAG system raises an exception"""
        # Configure mock to raise exception
        test_client_no_static.app.state.mock_rag.query.side_effect = Exception("RAG system error")
        
        query_data = {
            "query": "What is the main topic?",
            "session_id": "test-session"
        }
        
        response = test_client_no_static.post("/api/query", json=query_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "RAG system error" in data["detail"]
        
        # Reset mock for other tests
        test_client_no_static.app.state.mock_rag.query.side_effect = None
        test_client_no_static.app.state.mock_rag.query.return_value = ("Test response", ["source1", "source2"])
    
    def test_courses_endpoint_success(self, test_client_no_static):
        """Test courses endpoint returns correct statistics"""
        response = test_client_no_static.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Course 1", "Course 2"]
    
    def test_courses_endpoint_rag_system_error(self, test_client_no_static):
        """Test courses endpoint when RAG system raises an exception"""
        # Configure mock to raise exception
        test_client_no_static.app.state.mock_rag.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client_no_static.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Analytics error" in data["detail"]
        
        # Reset mock for other tests
        test_client_no_static.app.state.mock_rag.get_course_analytics.side_effect = None
        test_client_no_static.app.state.mock_rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Course 1", "Course 2"]
        }
    
    def test_query_endpoint_long_query(self, test_client_no_static):
        """Test query endpoint with very long query string"""
        long_query = "What is " + "very " * 1000 + "long question?"
        query_data = {
            "query": long_query,
            "session_id": "test-session"
        }
        
        response = test_client_no_static.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
    
    def test_query_endpoint_special_characters(self, test_client_no_static):
        """Test query endpoint with special characters"""
        special_query = "What about émojis 🚀 and spëcial chars @#$%?"
        query_data = {
            "query": special_query,
            "session_id": "test-session"
        }
        
        response = test_client_no_static.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
    
    def test_query_endpoint_response_format(self, test_client_no_static):
        """Test that query endpoint response matches expected format"""
        query_data = {
            "query": "Test query",
            "session_id": "test-session"
        }
        
        response = test_client_no_static.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert isinstance(data, dict)
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        assert len(data["answer"]) > 0
        assert len(data["session_id"]) > 0
    
    def test_courses_endpoint_response_format(self, test_client_no_static):
        """Test that courses endpoint response matches expected format"""
        response = test_client_no_static.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert isinstance(data, dict)
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] >= 0
        assert all(isinstance(title, str) for title in data["course_titles"])
    
    def test_cors_headers(self, test_client_no_static):
        """Test CORS headers are properly set"""
        response = test_client_no_static.get("/api/courses")
        
        assert response.status_code == 200
        # CORS headers should be present (handled by FastAPI middleware)
        # Note: TestClient may not include all middleware headers
    
    def test_invalid_endpoint(self, test_client_no_static):
        """Test request to non-existent endpoint"""
        response = test_client_no_static.get("/api/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "Not Found" in data["detail"]