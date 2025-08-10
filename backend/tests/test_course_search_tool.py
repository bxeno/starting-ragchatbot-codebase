import pytest
import os
import sys
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil

# Add the backend directory to Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, ToolManager
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk
from config import Config


class TestCourseSearchTool:
    """Test CourseSearchTool functionality"""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing"""
        mock_store = Mock(spec=VectorStore)
        return mock_store
    
    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create CourseSearchTool with mock vector store"""
        return CourseSearchTool(mock_vector_store)
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing"""
        return SearchResults(
            documents=["Course content about machine learning basics", 
                      "Advanced ML concepts and algorithms"],
            metadata=[
                {"course_title": "Machine Learning Course", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "Machine Learning Course", "lesson_number": 2, "chunk_index": 1}
            ],
            distances=[0.1, 0.2]
        )
    
    def test_get_tool_definition(self, search_tool):
        """Test that tool definition is correctly formatted"""
        definition = search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        
        # Check properties
        properties = definition["input_schema"]["properties"]
        assert "query" in properties
        assert "course_name" in properties  
        assert "lesson_number" in properties
    
    def test_execute_successful_search(self, search_tool, mock_vector_store, sample_search_results):
        """Test successful search execution"""
        mock_vector_store.search.return_value = sample_search_results
        
        result = search_tool.execute("machine learning", "ML Course")
        
        # Verify search was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name="ML Course",
            lesson_number=None
        )
        
        # Verify result format
        assert "[Machine Learning Course - Lesson 1]" in result
        assert "[Machine Learning Course - Lesson 2]" in result
        assert "Course content about machine learning basics" in result
        assert "Advanced ML concepts and algorithms" in result
    
    def test_execute_with_lesson_filter(self, search_tool, mock_vector_store, sample_search_results):
        """Test search with lesson number filter"""
        mock_vector_store.search.return_value = sample_search_results
        
        result = search_tool.execute("algorithms", "ML Course", 2)
        
        mock_vector_store.search.assert_called_once_with(
            query="algorithms",
            course_name="ML Course", 
            lesson_number=2
        )
    
    def test_execute_search_error(self, search_tool, mock_vector_store):
        """Test handling of search errors"""
        error_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="ChromaDB connection failed"
        )
        mock_vector_store.search.return_value = error_results
        
        result = search_tool.execute("test query")
        
        assert result == "ChromaDB connection failed"
    
    def test_execute_empty_results(self, search_tool, mock_vector_store):
        """Test handling of empty search results"""
        empty_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[]
        )
        mock_vector_store.search.return_value = empty_results
        
        result = search_tool.execute("nonexistent topic", "Unknown Course")
        
        assert "No relevant content found" in result
        assert "Unknown Course" in result
    
    def test_execute_empty_results_with_lesson_filter(self, search_tool, mock_vector_store):
        """Test empty results with lesson filter"""
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = search_tool.execute("test", "Course", 5)
        
        assert "No relevant content found" in result
        assert "in course 'Course'" in result
        assert "in lesson 5" in result
    
    def test_format_results_with_sources(self, search_tool, mock_vector_store):
        """Test that sources are tracked correctly"""
        results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        
        formatted = search_tool._format_results(results)
        
        # Check that sources were stored - sources are strings, not dicts
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0] == "Test Course - Lesson 1"
    
    def test_search_tool_basic_functionality(self, search_tool):
        """Test that search tool has required methods and attributes"""
        assert hasattr(search_tool, 'execute')
        assert hasattr(search_tool, 'get_tool_definition')
        assert hasattr(search_tool, 'last_sources')
        assert hasattr(search_tool, 'store')
    
    def test_search_tool_stores_sources_correctly(self, search_tool):
        """Test that sources are stored in the correct format"""
        # Initialize empty sources
        search_tool.last_sources = []
        
        results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        
        search_tool._format_results(results)
        
        assert len(search_tool.last_sources) == 2
        assert "Course A - Lesson 1" in search_tool.last_sources
        assert "Course B - Lesson 2" in search_tool.last_sources


class TestToolManager:
    """Test ToolManager functionality"""
    
    @pytest.fixture
    def tool_manager(self):
        """Create a ToolManager instance"""
        return ToolManager()
    
    @pytest.fixture 
    def mock_tool(self):
        """Create a mock tool for testing"""
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "Test tool"
        }
        mock_tool.execute.return_value = "Test result"
        return mock_tool
    
    def test_register_tool(self, tool_manager, mock_tool):
        """Test tool registration"""
        tool_manager.register_tool(mock_tool)
        
        assert "test_tool" in tool_manager.tools
        assert tool_manager.tools["test_tool"] == mock_tool
    
    def test_get_tool_definitions(self, tool_manager, mock_tool):
        """Test getting all tool definitions"""
        tool_manager.register_tool(mock_tool)
        
        definitions = tool_manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "test_tool"
    
    def test_execute_tool_success(self, tool_manager, mock_tool):
        """Test successful tool execution"""
        tool_manager.register_tool(mock_tool)
        
        result = tool_manager.execute_tool("test_tool", param1="value1")
        
        mock_tool.execute.assert_called_once_with(param1="value1")
        assert result == "Test result"
    
    def test_execute_tool_not_found(self, tool_manager):
        """Test execution of non-existent tool"""
        result = tool_manager.execute_tool("nonexistent_tool")
        
        assert "Tool 'nonexistent_tool' not found" in result
    
    def test_get_last_sources(self, tool_manager):
        """Test getting sources from tools"""
        # Create mock tool with sources
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"name": "search_tool"}
        mock_tool.last_sources = [{"text": "Source 1"}, {"text": "Source 2"}]
        
        tool_manager.register_tool(mock_tool)
        
        sources = tool_manager.get_last_sources()
        assert len(sources) == 2
        assert sources[0]["text"] == "Source 1"
    
    def test_reset_sources(self, tool_manager):
        """Test resetting sources from tools"""
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"name": "search_tool"}
        mock_tool.last_sources = [{"text": "Source 1"}]
        
        tool_manager.register_tool(mock_tool)
        tool_manager.reset_sources()
        
        assert mock_tool.last_sources == []


class TestCourseSearchToolIntegration:
    """Integration tests with actual vector store"""
    
    @pytest.fixture
    def temp_chroma_path(self):
        """Create temporary ChromaDB path for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def vector_store(self, temp_chroma_path):
        """Create real vector store for integration testing"""
        config = Config()
        config.CHROMA_PATH = temp_chroma_path
        return VectorStore(temp_chroma_path, config.EMBEDDING_MODEL, config.MAX_RESULTS)
    
    @pytest.fixture
    def search_tool_with_real_store(self, vector_store):
        """Create CourseSearchTool with real vector store"""
        return CourseSearchTool(vector_store)
    
    def test_search_with_empty_store(self, search_tool_with_real_store):
        """Test search with empty vector store"""
        result = search_tool_with_real_store.execute("test query")
        
        assert "No relevant content found" in result
    
    def test_search_after_adding_content(self, search_tool_with_real_store, vector_store):
        """Test search after adding sample content"""
        # Add sample course and content
        course = Course(
            title="Test Course",
            instructor="Test Instructor"
        )
        course.lessons.append(Lesson(lesson_number=1, title="Lesson 1"))
        
        chunks = [
            CourseChunk(
                content="This is lesson 1 about machine learning basics",
                course_title="Test Course", 
                lesson_number=1,
                chunk_index=0
            )
        ]
        
        vector_store.add_course_metadata(course)
        vector_store.add_course_content(chunks)
        
        result = search_tool_with_real_store.execute("machine learning")
        
        assert "Test Course" in result
        assert "machine learning basics" in result
        assert len(search_tool_with_real_store.last_sources) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])