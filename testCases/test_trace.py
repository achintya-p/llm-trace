"""
Pytest tests for llmtrace-lite character counting and edge cases.
"""
import time
import pytest
from llmTrace import trace


# Test functions
@trace
def call_llm(prompt, model="gpt-4o"):
    """Simulated LLM call."""
    time.sleep(0.01)  # Reduced sleep for faster tests
    return f"Response to: {prompt}"


@trace
def failing_llm_call(prompt, model="gpt-4o"):
    """Simulated failing LLM call."""
    time.sleep(0.01)
    raise ValueError("API rate limit exceeded")


@trace
def call_llm_with_kwargs(prompt="default", model="gpt-4o"):
    """LLM call with prompt as kwarg."""
    time.sleep(0.01)
    return f"Response: {prompt}"


@trace
def call_llm_non_string_prompt(prompt, model="gpt-4o"):
    """LLM call that might receive non-string prompt."""
    time.sleep(0.01)
    return {"response": str(prompt)}


@trace
def call_llm_non_string_output(prompt, model="gpt-4o"):
    """LLM call that returns non-string output."""
    time.sleep(0.01)
    return {"result": "data"}


@trace
def return_empty():
    """Return empty string."""
    time.sleep(0.01)
    return ""


@trace
def call_with_none(prompt=None):
    """Call with None prompt."""
    time.sleep(0.01)
    return "response"


class TestTraceCharacterCounting:
    """Test character counting edge cases."""
    
    def test_successful_call_with_model(self, capsys):
        """Test successful call with model specified."""
        result = call_llm("Explain quantum computing", model="gpt-4o")
        captured = capsys.readouterr()
        assert "call_llm" in captured.out
        assert "model: gpt-4o" in captured.out
        assert "prompt_chars: 25" in captured.out
        assert "output_chars: 38" in captured.out
        assert "status: success" in captured.out
        assert result == "Response to: Explain quantum computing"
    
    def test_failed_call(self, capsys):
        """Test failed call with exception."""
        with pytest.raises(ValueError, match="API rate limit exceeded"):
            failing_llm_call("This will fail", model="claude-3")
        captured = capsys.readouterr()
        assert "failing_llm_call" in captured.out
        assert "model: claude-3" in captured.out
        assert "prompt_chars: 14" in captured.out
        assert "status: error" in captured.out
        assert "error: API rate limit exceeded" in captured.out
    
    def test_no_model_specified(self, capsys):
        """Test call without model specified."""
        result = call_llm("Simple prompt")
        captured = capsys.readouterr()
        assert "call_llm" in captured.out
        # When no model kwarg is passed, it should use default, so model may or may not appear
        # The key check is that prompt_chars and status are present
        assert "prompt_chars: 13" in captured.out
        assert "status: success" in captured.out
    
    def test_empty_string_prompt(self, capsys):
        """Test with empty string prompt."""
        result = call_llm("", model="gpt-4o")
        captured = capsys.readouterr()
        assert "prompt_chars: 0" in captured.out
        assert "status: success" in captured.out
    
    def test_empty_string_output(self, capsys):
        """Test with empty string output."""
        result = return_empty()
        captured = capsys.readouterr()
        assert "output_chars: 0" in captured.out
        assert "status: success" in captured.out
        assert result == ""
    
    def test_unicode_and_emoji_characters(self, capsys):
        """Test with unicode and emoji characters."""
        prompt = "Hello 世界 🌍 🚀"
        result = call_llm(prompt, model="gpt-4o")
        captured = capsys.readouterr()
        # Unicode and emoji should count as characters (Python len counts code points)
        expected_chars = len(prompt)
        assert f"prompt_chars: {expected_chars}" in captured.out
        assert "status: success" in captured.out
    
    def test_very_long_string(self, capsys):
        """Test with very long string."""
        long_prompt = "A" * 1000
        result = call_llm(long_prompt, model="gpt-4o")
        captured = capsys.readouterr()
        assert "prompt_chars: 1000" in captured.out
        assert "status: success" in captured.out
    
    def test_newlines_and_special_characters(self, capsys):
        """Test with newlines and special characters."""
        multiline_prompt = "Line 1\nLine 2\nLine 3\tTabbed"
        result = call_llm(multiline_prompt, model="gpt-4o")
        captured = capsys.readouterr()
        expected_chars = len(multiline_prompt)
        assert f"prompt_chars: {expected_chars}" in captured.out
        assert "status: success" in captured.out
    
    def test_prompt_as_keyword_argument(self, capsys):
        """Test with prompt as keyword argument."""
        result = call_llm_with_kwargs(prompt="Keyword argument prompt", model="gpt-4o")
        captured = capsys.readouterr()
        assert "prompt_chars: 23" in captured.out  # "Keyword argument prompt" = 23 chars
        assert "status: success" in captured.out
    
    def test_non_string_return_value(self, capsys):
        """Test with non-string return value (should not count output_chars)."""
        result = call_llm_non_string_output("test prompt", model="gpt-4o")
        captured = capsys.readouterr()
        assert "prompt_chars: 11" in captured.out
        # Should not have output_chars for non-string return
        assert "output_chars:" not in captured.out.split("status")[0]
        assert "status: success" in captured.out
        assert isinstance(result, dict)
    
    def test_non_string_prompt(self, capsys):
        """Test with non-string prompt (should not count chars)."""
        result = call_llm_non_string_prompt(12345, model="gpt-4o")
        captured = capsys.readouterr()
        # Non-string prompt should not have prompt_chars
        assert "prompt_chars:" not in captured.out.split("status")[0]
        assert "status: success" in captured.out
    
    def test_none_prompt(self, capsys):
        """Test with None prompt (should not count chars)."""
        result = call_with_none(None)
        captured = capsys.readouterr()
        # None prompt should not have prompt_chars
        assert "prompt_chars:" not in captured.out.split("status")[0]
        assert "status: success" in captured.out
        assert result == "response"