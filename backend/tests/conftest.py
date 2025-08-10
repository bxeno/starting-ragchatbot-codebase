import pytest
import tempfile
import shutil
import os
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from typing import Generator, Dict, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from models import Course, Lesson, CourseChunk
from vector_store import VectorStore
from rag_system import RAGSystem
from session_manager import SessionManager


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return Config(
        anthropic_api_key="test-key",
        chunk_size=500,
        chunk_overlap=50,
        max_search_results=3,
        embedding_model="all-MiniLM-L6-v2",
        claude_model="claude-sonnet-4-20250514",
        max_history=2
    )


@pytest.fixture
def temp_db_path():
    """Temporary database path for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_course():
    """Sample course data for testing"""
    return Course(
        title="Test Course",
        instructor="Test Instructor", 
        lessons=[
            Lesson(number=0, title="Introduction", content="This is lesson 0 content"),
            Lesson(number=1, title="Advanced Topics", content="This is lesson 1 content")
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is lesson 0 content",
            course_title="Test Course",
            lesson_number=0,
            chunk_index=0,
            metadata={"instructor": "Test Instructor"}
        ),
        CourseChunk(
            content="This is lesson 1 content", 
            course_title="Test Course",
            lesson_number=1,
            chunk_index=0,
            metadata={"instructor": "Test Instructor"}
        )
    ]


@pytest.fixture
def mock_vector_store(mock_config, temp_db_path, sample_course_chunks):
    """Mock vector store with test data"""
    with patch('vector_store.chromadb') as mock_chromadb:
        # Setup mock ChromaDB client
        mock_client = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Setup mock collections
        mock_course_collection = MagicMock()
        mock_chunk_collection = MagicMock()
        
        mock_client.get_or_create_collection.side_effect = lambda name, **kwargs: {
            "course_metadata": mock_course_collection,
            "course_chunks": mock_chunk_collection
        }[name]
        
        # Mock search results
        mock_chunk_collection.query.return_value = {
            "documents": [[chunk.content for chunk in sample_course_chunks]],
            "metadatas": [[chunk.metadata for chunk in sample_course_chunks]],
            "distances": [[0.1, 0.2]]
        }
        
        vector_store = VectorStore(mock_config, temp_db_path)
        return vector_store


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    session_manager = MagicMock(spec=SessionManager)
    session_manager.create_session.return_value = "test-session-123"
    session_manager.add_message.return_value = None
    session_manager.get_conversation_history.return_value = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"}
    ]
    return session_manager


@pytest.fixture
def mock_ai_generator():
    """Mock AI generator for testing"""
    with patch('rag_system.AIGenerator') as mock_ai_class:
        mock_ai = AsyncMock()
        mock_ai.generate_response.return_value = "Test AI response"
        mock_ai_class.return_value = mock_ai
        yield mock_ai


@pytest.fixture
def mock_rag_system(mock_config, mock_vector_store, mock_session_manager, mock_ai_generator):
    """Mock RAG system for testing"""
    with patch('rag_system.VectorStore', return_value=mock_vector_store), \
         patch('rag_system.SessionManager', return_value=mock_session_manager), \
         patch('rag_system.DocumentProcessor') as mock_doc_processor:
        
        # Setup document processor mock
        mock_doc_instance = MagicMock()
        mock_doc_processor.return_value = mock_doc_instance
        mock_doc_instance.process_file.return_value = (1, ["Test Course"])
        
        rag_system = RAGSystem(mock_config)
        rag_system.session_manager = mock_session_manager
        return rag_system


@pytest.fixture
def test_client_no_static():
    """FastAPI test client without static file mounting to avoid path issues"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create a test-specific app without static file mounting
    app = FastAPI(title="Test Course Materials RAG System")
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"], 
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models (same as main app)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Mock RAG system for test client
    mock_rag = MagicMock()
    mock_rag.query.return_value = ("Test response", ["source1", "source2"])
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course 1", "Course 2"]
    }
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag.session_manager.create_session()
            
            answer, sources = mock_rag.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System"}
    
    # Attach mock for access in tests
    app.state.mock_rag = mock_rag
    
    return TestClient(app)


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for AI testing"""
    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        # Mock message response
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Test AI response"
        mock_client.messages.create.return_value = mock_response
        
        yield mock_client


@pytest.fixture(scope="session", autouse=True)
def suppress_warnings():
    """Suppress warnings during tests"""
    import warnings
    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)