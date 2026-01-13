import pytest
from unittest.mock import Mock, patch
import dataclasses
from omegaconf import DictConfig

from src.llm.providers.gpt5_openai_client import GPT5OpenAIClient


class MockResponse:
    def __init__(self, choices=None, usage=None):
        self.choices = choices or []
        self.usage = usage


class MockChoice:
    def __init__(self, finish_reason="stop", message_content="Test response"):
        self.finish_reason = finish_reason
        self.message = Mock()
        self.message.content = message_content
        self.message.role = "assistant"


class TestGPT5OpenAIClientProcessLLMResponse:
    """Test suite for GPT5OpenAIClient.process_llm_response method"""

    def setup_method(self):
        """Setup test fixture"""
        # Create mock config
        mock_cfg = Mock(spec=DictConfig)
        mock_cfg.llm = Mock()
        mock_cfg.llm.provider_class = "openai"
        mock_cfg.llm.model_name = "gpt-5"
        mock_cfg.llm.temperature = 0.7
        mock_cfg.llm.top_p = 0.9
        mock_cfg.llm.min_p = 0.1
        mock_cfg.llm.top_k = 40
        mock_cfg.llm.max_tokens = 4096
        mock_cfg.llm.max_context_length = 128000
        mock_cfg.llm.oai_tool_thinking = True
        mock_cfg.llm.async_client = False
        mock_cfg.llm.use_tool_calls = True
        mock_cfg.llm.openrouter_provider = ""
        mock_cfg.llm.disable_cache_control = False
        mock_cfg.llm.reasoning_effort = "medium"
        mock_cfg.llm.repetition_penalty = 1.0

        # Create client instance
        self.client = GPT5OpenAIClient(
            task_id="test_task",
            cfg=mock_cfg
        )

    def test_process_llm_response_empty_response(self):
        """Test processing empty LLM response"""
        # Arrange
        llm_response = None
        message_history = []

        # Act
        result, should_exit = self.client.process_llm_response(
            llm_response, message_history
        )

        # Assert
        assert result == ""
        assert should_exit is True
        assert len(message_history) == 0

    def test_process_llm_response_no_choices(self):
        """Test processing response with no choices"""
        # Arrange
        llm_response = MockResponse(choices=[])
        message_history = []

        # Act
        result, should_exit = self.client.process_llm_response(
            llm_response, message_history
        )

        # Assert
        assert result == ""
        assert should_exit is True
        assert len(message_history) == 0

    def test_process_llm_response_stop_finish_reason(self):
        """Test processing response with stop finish reason"""
        # Arrange
        llm_response = MockResponse(choices=[MockChoice(finish_reason="stop")])
        message_history = []

        # Act
        result, should_exit = self.client.process_llm_response(
            llm_response, message_history
        )

        # Assert
        assert result == "Test response"
        assert should_exit is False
        assert len(message_history) == 1
        assert message_history[0]["role"] == "assistant"
        assert message_history[0]["content"] == "Test response"

    def test_process_llm_response_stop_finish_reason_empty_content(self):
        """Test processing stop finish reason with empty content"""
        # Arrange
        choice = MockChoice(finish_reason="stop", message_content="")
        llm_response = MockResponse(choices=[choice])
        message_history = []

        # Act
        result, should_exit = self.client.process_llm_response(
            llm_response, message_history
        )

        # Assert
        assert result == ""
        assert should_exit is False
        assert len(message_history) == 1
        assert message_history[0]["role"] == "assistant"
        assert message_history[0]["content"] == ""

    def test_process_llm_response_length_finish_reason_non_empty(self):
        """Test processing length finish reason with non-empty content"""
        # Arrange
        llm_response = MockResponse(choices=[MockChoice(finish_reason="length")])
        message_history = []

        # Act
        result, should_exit = self.client.process_llm_response(
            llm_response, message_history
        )

        # Assert
        assert result == "Test response"
        assert should_exit is False
        assert len(message_history) == 1
        assert message_history[0]["role"] == "assistant"
        assert message_history[0]["content"] == "Test response"

    def test_process_llm_response_length_finish_reason_empty_content(self):
        """Test processing length finish reason with empty content"""
        # Arrange
        choice = MockChoice(finish_reason="length", message_content="")
        llm_response = MockResponse(choices=[choice])
        message_history = []

        # Act
        result, should_exit = self.client.process_llm_response(
            llm_response, message_history
        )

        # Assert
        assert result == "LLM response is empty. This is likely due to thinking block used up all tokens."
        assert should_exit is False
        assert len(message_history) == 1
        assert message_history[0]["role"] == "assistant"
        assert message_history[0]["content"] == "LLM response is empty. This is likely due to thinking block used up all tokens."

    def test_process_llm_response_unsupported_finish_reason(self):
        """Test processing unsupported finish reason"""
        # Arrange
        llm_response = MockResponse(choices=[MockChoice(finish_reason="tool_calls")])
        message_history = []

        # Act
        result, should_exit = self.client.process_llm_response(
            llm_response, message_history
        )

        # Assert
        expected_msg = "Successful response, but unsupported finish reason: tool_calls"
        assert result == expected_msg
        assert should_exit is False
        assert len(message_history) == 1
        assert message_history[0]["role"] == "assistant"
        assert message_history[0]["content"] == expected_msg

    def test_process_llm_response_none_finish_reason(self):
        """Test processing None finish reason"""
        # Arrange
        choice = MockChoice(finish_reason=None)
        llm_response = MockResponse(choices=[choice])
        message_history = []

        # Act
        result, should_exit = self.client.process_llm_response(
            llm_response, message_history
        )

        # Assert
        expected_msg = "Successful response, but unsupported finish reason: None"
        assert result == expected_msg
        assert should_exit is False
        assert len(message_history) == 1
        assert message_history[0]["role"] == "assistant"
        assert message_history[0]["content"] == expected_msg

    def test_process_llm_response_empty_string_finish_reason(self):
        """Test processing empty string finish reason"""
        # Arrange
        choice = MockChoice(finish_reason="")
        llm_response = MockResponse(choices=[choice])
        message_history = []

        # Act
        result, should_exit = self.client.process_llm_response(
            llm_response, message_history
        )

        # Assert
        expected_msg = "Successful response, but unsupported finish reason: "
        assert result == expected_msg
        assert should_exit is False
        assert len(message_history) == 1
        assert message_history[0]["role"] == "assistant"
        assert message_history[0]["content"] == expected_msg

    def test_process_llm_response_multiple_choices_first_choice_used(self):
        """Test that only first choice is processed when multiple choices exist"""
        # Arrange
        choice1 = MockChoice(finish_reason="stop", message_content="First choice")
        choice2 = MockChoice(finish_reason="stop", message_content="Second choice")
        llm_response = MockResponse(choices=[choice1, choice2])
        message_history = []

        # Act
        result, should_exit = self.client.process_llm_response(
            llm_response, message_history
        )

        # Assert
        assert result == "First choice"
        assert should_exit is False
        assert len(message_history) == 1
        assert message_history[0]["role"] == "assistant"
        assert message_history[0]["content"] == "First choice"

    def test_process_llm_response_with_existing_message_history(self):
        """Test processing response with existing message history"""
        # Arrange
        llm_response = MockResponse(choices=[MockChoice(finish_reason="stop")])
        message_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]

        # Act
        result, should_exit = self.client.process_llm_response(
            llm_response, message_history
        )

        # Assert
        assert result == "Test response"
        assert should_exit is False
        assert len(message_history) == 3
        assert message_history[2]["role"] == "assistant"
        assert message_history[2]["content"] == "Test response"

    def test_process_llm_response_different_agent_type(self):
        """Test processing response with different agent type"""
        # Arrange
        llm_response = MockResponse(choices=[MockChoice(finish_reason="stop")])
        message_history = []

        # Act
        result, should_exit = self.client.process_llm_response(
            llm_response, message_history, agent_type="tool"
        )

        # Assert
        assert result == "Test response"
        assert should_exit is False
        assert len(message_history) == 1
        assert message_history[0]["role"] == "assistant"
        assert message_history[0]["content"] == "Test response"

    @patch('src.llm.providers.gpt5_openai_client.logger')
    def test_process_llm_response_logging_on_error(self, mock_logger):
        """Test that errors are logged appropriately"""
        # Arrange
        llm_response = None
        message_history = []

        # Act
        result, should_exit = self.client.process_llm_response(
            llm_response, message_history
        )

        # Assert
        mock_logger.error.assert_called_once_with("Should never happen: LLM did not return a valid response.")

    @patch('src.llm.providers.gpt5_openai_client.logger')
    def test_process_llm_response_logging_unsupported_reason(self, mock_logger):
        """Test that unsupported finish reasons are logged"""
        # Arrange
        llm_response = MockResponse(choices=[MockChoice(finish_reason="unknown_reason")])
        message_history = []

        # Act
        result, should_exit = self.client.process_llm_response(
            llm_response, message_history
        )

        # Assert
        mock_logger.error.assert_called_once_with("Unsupported finish reason: unknown_reason")

    @patch('src.llm.providers.gpt5_openai_client.logger')
    def test_process_llm_response_debug_logging(self, mock_logger):
        """Test that successful responses are debug logged"""
        # Arrange
        llm_response = MockResponse(choices=[MockChoice(finish_reason="stop")])
        message_history = []

        # Act
        result, should_exit = self.client.process_llm_response(
            llm_response, message_history
        )

        # Assert
        mock_logger.debug.assert_called_once_with("LLM Response: Test response")

    def test_process_llm_response_content_cleaning_called(self):
        """Test that _clean_user_content_from_response is called for stop reason"""
        # Arrange
        llm_response = MockResponse(choices=[MockChoice(finish_reason="stop", message_content="Test\n\nUser: content<use_mcp_tool>rest")])
        message_history = []
        
        # Mock the cleaning method
        with patch.object(self.client, '_clean_user_content_from_response') as mock_clean:
            mock_clean.return_value = "Cleaned response"

            # Act
            result, should_exit = self.client.process_llm_response(
                llm_response, message_history
            )

            # Assert
            mock_clean.assert_called_once_with("Test\n\nUser: content<use_mcp_tool>rest")
            assert result == "Cleaned response"
            assert message_history[0]["content"] == "Cleaned response"

    def test_process_llm_response_content_cleaning_called_for_length(self):
        """Test that _clean_user_content_from_response is called for length reason"""
        # Arrange
        llm_response = MockResponse(choices=[MockChoice(finish_reason="length", message_content="Test\n\nUser: content<use_mcp_tool>rest")])
        message_history = []
        
        # Mock the cleaning method
        with patch.object(self.client, '_clean_user_content_from_response') as mock_clean:
            mock_clean.return_value = "Cleaned response"

            # Act
            result, should_exit = self.client.process_llm_response(
                llm_response, message_history
            )

            # Assert
            mock_clean.assert_called_once_with("Test\n\nUser: content<use_mcp_tool>rest")
            assert result == "Cleaned response"
            assert message_history[0]["content"] == "Cleaned response"

    def test_process_llm_response_no_cleaning_for_empty_length(self):
        """Test that cleaning is not called for empty length responses"""
        # Arrange
        choice = MockChoice(finish_reason="length", message_content="")
        llm_response = MockResponse(choices=[choice])
        message_history = []
        
        # Mock the cleaning method
        with patch.object(self.client, '_clean_user_content_from_response') as mock_clean:
            # Act
            result, should_exit = self.client.process_llm_response(
                llm_response, message_history
            )

            # Assert
            mock_clean.assert_not_called()
            assert "LLM response is empty" in result

    def test_process_llm_response_no_cleaning_for_unsupported_reason(self):
        """Test that cleaning is not called for unsupported finish reasons"""
        # Arrange
        llm_response = MockResponse(choices=[MockChoice(finish_reason="unknown")])
        message_history = []
        
        # Mock the cleaning method
        with patch.object(self.client, '_clean_user_content_from_response') as mock_clean:
            # Act
            result, should_exit = self.client.process_llm_response(
                llm_response, message_history
            )

            # Assert
            mock_clean.assert_not_called()
            assert "unsupported finish reason" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])