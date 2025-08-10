import pytest
import os
import sys
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Add the backend directory to Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator
from search_tools import ToolManager


@dataclass
class MockContentBlock:
    """Mock Anthropic content block for testing"""
    type: str
    text: str = ""
    name: str = ""
    input: dict = None
    id: str = "test_id"


@dataclass  
class MockResponse:
    """Mock Anthropic response for testing"""
    content: list
    stop_reason: str = "end_turn"


class TestAIGenerator:
    """Test AIGenerator functionality"""
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Create mock Anthropic client"""
        mock_client = Mock()
        return mock_client
    
    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create AIGenerator with mock client"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
            generator.client = mock_anthropic_client
            return generator
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager"""
        mock_manager = Mock(spec=ToolManager)
        mock_manager.get_tool_definitions.return_value = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        ]
        mock_manager.execute_tool.return_value = "Search results: ML course content"
        return mock_manager
    
    def test_init(self, ai_generator):
        """Test AIGenerator initialization"""
        assert ai_generator.model == "claude-sonnet-4-20250514"
        assert ai_generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert ai_generator.base_params["temperature"] == 0
        assert ai_generator.base_params["max_tokens"] == 800
    
    def test_generate_response_without_tools(self, ai_generator, mock_anthropic_client):
        """Test response generation without tools"""
        # Mock response without tool use
        mock_response = MockResponse(
            content=[MockContentBlock(type="text", text="This is a direct answer without tools.")],
            stop_reason="end_turn"
        )
        mock_anthropic_client.messages.create.return_value = mock_response
        
        result = ai_generator.generate_response("What is machine learning?")
        
        # Verify API call
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["messages"][0]["content"] == "What is machine learning?"
        assert "tools" not in call_args  # No tools provided
        
        assert result == "This is a direct answer without tools."
    
    def test_generate_response_with_tools_no_use(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test response generation with tools available but not used"""
        # Mock response without tool use
        mock_response = MockResponse(
            content=[MockContentBlock(type="text", text="General answer without using tools.")],
            stop_reason="end_turn"
        )
        mock_anthropic_client.messages.create.return_value = mock_response
        
        result = ai_generator.generate_response(
            "What is AI?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Verify tools were provided but not used
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tool_choice"] == {"type": "auto"}
        
        # Tool manager should not be called
        mock_tool_manager.execute_tool.assert_not_called()
        
        assert result == "General answer without using tools."
    
    def test_generate_response_with_tool_use(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test response generation with tool use"""
        # Mock initial response with tool use
        initial_response = MockResponse(
            content=[MockContentBlock(
                type="tool_use",
                name="search_course_content",
                input={"query": "machine learning"},
                id="test_tool_id"
            )],
            stop_reason="tool_use"
        )
        
        # Mock final response after tool execution
        final_response = MockResponse(
            content=[MockContentBlock(
                type="text", 
                text="Based on the search, machine learning is..."
            )],
            stop_reason="end_turn"
        )
        
        mock_anthropic_client.messages.create.side_effect = [initial_response, final_response]
        
        result = ai_generator.generate_response(
            "What is machine learning?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning"
        )
        
        # Verify two API calls (initial + final)
        assert mock_anthropic_client.messages.create.call_count == 2
        
        assert result == "Based on the search, machine learning is..."
    
    def test_generate_response_with_conversation_history(self, ai_generator, mock_anthropic_client):
        """Test response generation with conversation history"""
        mock_response = MockResponse(
            content=[MockContentBlock(type="text", text="Response with context.")],
            stop_reason="end_turn"
        )
        mock_anthropic_client.messages.create.return_value = mock_response
        
        history = "User: Previous question\nAssistant: Previous answer"
        
        result = ai_generator.generate_response(
            "Follow-up question",
            conversation_history=history
        )
        
        # Check that history was included in system prompt
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_args["system"]
        assert history in call_args["system"]
    
    def test_handle_tool_execution_multiple_tools(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test handling multiple tool calls in one response"""
        # Mock response with multiple tool uses
        initial_response = MockResponse(
            content=[
                MockContentBlock(
                    type="tool_use",
                    name="search_course_content", 
                    input={"query": "machine learning"},
                    id="tool_1"
                ),
                MockContentBlock(
                    type="tool_use",
                    name="search_course_content",
                    input={"query": "deep learning"},
                    id="tool_2" 
                )
            ],
            stop_reason="tool_use"
        )
        
        final_response = MockResponse(
            content=[MockContentBlock(type="text", text="Combined results...")],
            stop_reason="end_turn"
        )
        
        mock_anthropic_client.messages.create.side_effect = [initial_response, final_response]
        
        # Mock base params for tool execution
        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "test system"
        }
        
        result = ai_generator._handle_tool_execution(
            initial_response,
            base_params,
            mock_tool_manager
        )
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="machine learning")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="deep learning")
    
    def test_handle_tool_execution_error(self, ai_generator, mock_tool_manager):
        """Test tool execution error handling - currently doesn't handle errors gracefully"""
        # Mock tool manager to raise exception
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        initial_response = MockResponse(
            content=[MockContentBlock(
                type="tool_use",
                name="search_course_content",
                input={"query": "test"},
                id="test_id"
            )],
            stop_reason="tool_use"
        )
        
        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "test system"
        }
        
        # Currently the AI generator doesn't handle tool execution errors
        # This test documents the current behavior - it should crash
        with pytest.raises(Exception) as excinfo:
            ai_generator._handle_tool_execution(
                initial_response,
                base_params, 
                mock_tool_manager
            )
        
        assert "Tool execution failed" in str(excinfo.value)
    
    def test_system_prompt_content(self, ai_generator):
        """Test system prompt contains expected instructions"""
        system_prompt = ai_generator.SYSTEM_PROMPT
        
        # Check for key instruction elements
        assert "course materials" in system_prompt.lower()
        assert "search tool" in system_prompt.lower()
        assert "tool" in system_prompt.lower()
        assert "course-specific questions" in system_prompt.lower()
    
    def test_api_parameters_structure(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test API parameters are structured correctly"""
        mock_response = MockResponse(
            content=[MockContentBlock(type="text", text="test response")],
            stop_reason="end_turn"
        )
        mock_anthropic_client.messages.create.return_value = mock_response
        
        ai_generator.generate_response(
            "test query",
            conversation_history="test history",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )
        
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        # Verify required parameters
        assert "model" in call_args
        assert "messages" in call_args  
        assert "system" in call_args
        assert "tools" in call_args
        assert "tool_choice" in call_args
        assert "temperature" in call_args
        assert "max_tokens" in call_args
        
        # Verify message structure
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "test query"


class TestAIGeneratorIntegration:
    """Integration tests for AIGenerator with real tool scenarios"""
    
    @pytest.fixture
    def real_tool_manager(self):
        """Create real tool manager with mock tools"""
        from search_tools import ToolManager, Tool
        
        class MockSearchTool(Tool):
            def get_tool_definition(self):
                return {
                    "name": "search_course_content",
                    "description": "Search course materials",
                    "input_schema": {
                        "type": "object", 
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            
            def execute(self, **kwargs):
                return f"Mock search results for: {kwargs.get('query', 'unknown')}"
        
        manager = ToolManager()
        manager.register_tool(MockSearchTool())
        return manager
    
    @pytest.mark.skip(reason="Requires actual API key - use for manual testing")
    def test_real_api_call_without_tools(self):
        """Test with real API (requires API key) - skip by default"""
        import os
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")
            
        generator = AIGenerator(api_key, "claude-sonnet-4-20250514")
        response = generator.generate_response("What is 2+2?")
        
        assert response is not None
        assert len(response) > 0
    
    def test_tool_definitions_format(self, real_tool_manager):
        """Test that tool definitions are properly formatted for Anthropic API"""
        definitions = real_tool_manager.get_tool_definitions()
        
        assert len(definitions) == 1
        definition = definitions[0]
        
        # Check Anthropic API format requirements
        assert "name" in definition
        assert "description" in definition  
        assert "input_schema" in definition
        assert "type" in definition["input_schema"]
        assert "properties" in definition["input_schema"]
        assert "required" in definition["input_schema"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])