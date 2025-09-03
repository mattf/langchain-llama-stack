"""Unit tests for LlamaStackSafety."""

from unittest.mock import MagicMock, patch

import pytest

from langchain_llama_stack.safety import (
    LlamaStackModerationTool,
    LlamaStackSafety,
    LlamaStackSafetyTool,
    SafetyResult,
)


class TestLlamaStackSafety:
    """Test LlamaStackSafety functionality."""

    def test_init(self):
        """Test LlamaStackSafety initialization."""
        # Test without API key (local usage)
        safety = LlamaStackSafety(
            base_url="http://test:8321", shield_type="llama_guard"
        )
        assert safety.base_url == "http://test:8321"
        assert safety.api_key is None
        assert safety.shield_type == "llama_guard"

    def test_init_with_api_key(self):
        """Test LlamaStackSafety initialization with API key."""
        safety = LlamaStackSafety(
            base_url="http://test:8321", api_key="test-key", shield_type="llama_guard"
        )
        assert safety.base_url == "http://test:8321"
        assert safety.api_key == "test-key"
        assert safety.shield_type == "llama_guard"

    @patch("langchain_llama_stack.safety.LlamaStackClient")
    def test_check_content_safety_safe(self, mock_client_class):
        """Test content safety check with safe content."""
        # Mock the client and response
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Create a class-based mock that hasattr() can work with
        class MockSafeResponse:
            def __init__(self):
                self.is_violation = False
                self.confidence_score = 0.95

        mock_response = MockSafeResponse()
        mock_client.safety.run_shield.return_value = mock_response

        safety = LlamaStackSafety()
        result = safety.check_content_safety("This is safe content")

        assert isinstance(result, SafetyResult)
        assert result.is_safe is True
        assert result.violations == []
        assert result.confidence_score == 0.95

    @patch("langchain_llama_stack.safety.LlamaStackClient")
    def test_check_content_safety_unsafe(self, mock_client_class):
        """Test content safety check with unsafe content."""
        # Mock the client and response
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Create a class-based mock that hasattr() can work with
        class MockUnsafeResponse:
            def __init__(self):
                self.is_violation = True
                self.violation_level = "high"
                self.metadata = {"category": "hate"}

        mock_response = MockUnsafeResponse()
        mock_client.safety.run_shield.return_value = mock_response

        safety = LlamaStackSafety()
        result = safety.check_content_safety("This is unsafe content")

        assert isinstance(result, SafetyResult)
        assert result.is_safe is False
        assert len(result.violations) == 1
        assert result.violations[0]["category"] == "safety_violation"
        assert result.violations[0]["level"] == "high"

    @patch("langchain_llama_stack.safety.LlamaStackClient")
    def test_moderate_content_safe(self, mock_client_class):
        """Test content moderation with safe content."""
        # Mock the client and response
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.flagged = False
        mock_result.categories = {}

        mock_response = MagicMock()
        mock_response.results = [mock_result]
        mock_client.moderations.create.return_value = mock_response

        safety = LlamaStackSafety()
        result = safety.moderate_content("This is safe content")

        assert isinstance(result, SafetyResult)
        assert result.is_safe is True
        assert result.violations == []

    @patch("langchain_llama_stack.safety.LlamaStackClient")
    def test_moderate_content_unsafe(self, mock_client_class):
        """Test content moderation with unsafe content."""
        # Mock the client and response
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_result = MagicMock()
        mock_result.flagged = True
        mock_result.categories = {"hate": True, "violence": False}
        mock_result.category_scores = MagicMock()
        mock_result.category_scores.hate = 0.85

        mock_response = MagicMock()
        mock_response.results = [mock_result]
        mock_client.moderations.create.return_value = mock_response

        safety = LlamaStackSafety()
        result = safety.moderate_content("This is unsafe content")

        assert isinstance(result, SafetyResult)
        assert result.is_safe is False
        assert len(result.violations) == 1
        assert result.violations[0]["category"] == "hate"
        assert result.violations[0]["flagged"] is True
        assert result.violations[0]["score"] == 0.85

    @patch("langchain_llama_stack.safety.LlamaStackClient")
    def test_check_content_safety_error_handling(self, mock_client_class):
        """Test error handling in content safety check."""
        # Mock the client to raise an exception
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.safety.run_shield.side_effect = Exception("API Error")

        safety = LlamaStackSafety()
        result = safety.check_content_safety("Test content")

        # Should return safe by default on error
        assert isinstance(result, SafetyResult)
        assert result.is_safe is True
        assert result.violations == []
        assert "Safety check failed" in result.explanation

    @patch("langchain_llama_stack.safety.LlamaStackClient")
    def test_moderate_content_error_handling(self, mock_client_class):
        """Test error handling in content moderation."""
        # Mock the client to raise an exception
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.moderations.create.side_effect = Exception("API Error")

        safety = LlamaStackSafety()
        result = safety.moderate_content("Test content")

        # Should return safe by default on error
        assert isinstance(result, SafetyResult)
        assert result.is_safe is True
        assert result.violations == []
        assert "Moderation failed" in result.explanation

    @pytest.mark.asyncio
    @patch("langchain_llama_stack.safety.AsyncLlamaStackClient")
    async def test_acheck_content_safety(self, mock_async_client_class):
        """Test async content safety check."""
        # Mock the async client and response
        mock_async_client = MagicMock()
        mock_async_client_class.return_value = mock_async_client

        # Create a simple class-based mock that hasattr() can work with
        class MockResponse:
            def __init__(self):
                self.is_violation = False
                self.confidence_score = 0.95

        mock_response = MockResponse()

        # Mock the async method call
        async def mock_run_shield(*args, **kwargs):
            return mock_response

        mock_async_client.safety.run_shield = mock_run_shield

        safety = LlamaStackSafety()
        result = await safety.acheck_content_safety("This is safe content")

        assert isinstance(result, SafetyResult)
        assert result.is_safe is True
        assert result.confidence_score == 0.95

    @pytest.mark.asyncio
    @patch("langchain_llama_stack.safety.AsyncLlamaStackClient")
    async def test_amoderate_content(self, mock_async_client_class):
        """Test async content moderation."""
        # Mock the async client and response
        mock_async_client = MagicMock()
        mock_async_client_class.return_value = mock_async_client

        mock_result = MagicMock()
        mock_result.flagged = False
        mock_result.categories = {}

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        # Mock the async method call
        async def mock_create(*args, **kwargs):
            return mock_response

        mock_async_client.moderations.create = mock_create

        safety = LlamaStackSafety()
        result = await safety.amoderate_content("This is safe content")

        assert isinstance(result, SafetyResult)
        assert result.is_safe is True
        assert result.violations == []

    def test_safety_result_model(self):
        """Test SafetyResult model."""
        result = SafetyResult(
            is_safe=False,
            violations=[{"category": "test", "score": 0.8}],
            confidence_score=0.9,
            explanation="Test explanation",
        )

        assert result.is_safe is False
        assert len(result.violations) == 1
        assert result.violations[0]["category"] == "test"
        assert result.confidence_score == 0.9
        assert result.explanation == "Test explanation"


class TestLlamaStackSafetyTool:
    """Test LlamaStackSafetyTool functionality."""

    def test_safety_tool_safe_content(self):
        """Test safety tool with safe content."""
        # Create a real safety client and mock its method
        safety = LlamaStackSafety()

        with patch.object(safety, "check_content_safety") as mock_check:
            mock_check.return_value = SafetyResult(is_safe=True, confidence_score=0.95)

            tool = LlamaStackSafetyTool(safety_client=safety)
            result = tool._run("This is safe content")

            assert "SAFE" in result
            assert "0.95" in result

    def test_safety_tool_unsafe_content(self):
        """Test safety tool with unsafe content."""
        # Create a real safety client and mock its method
        safety = LlamaStackSafety()

        with patch.object(safety, "check_content_safety") as mock_check:
            mock_check.return_value = SafetyResult(
                is_safe=False,
                violations=[{"category": "hate"}, {"category": "violence"}],
            )

            tool = LlamaStackSafetyTool(safety_client=safety)
            result = tool._run("This is unsafe content")

            assert "UNSAFE" in result
            assert "hate" in result
            assert "violence" in result

    @pytest.mark.asyncio
    async def test_safety_tool_async(self):
        """Test async safety tool."""
        # Create a real safety client and mock its method
        safety = LlamaStackSafety()

        with patch.object(safety, "acheck_content_safety") as mock_acheck:
            mock_acheck.return_value = SafetyResult(is_safe=True, confidence_score=0.85)

            tool = LlamaStackSafetyTool(safety_client=safety)
            result = await tool._arun("This is safe content")

            assert "SAFE" in result
            assert "0.85" in result


class TestLlamaStackModerationTool:
    """Test LlamaStackModerationTool functionality."""

    def test_moderation_tool_safe_content(self):
        """Test moderation tool with safe content."""
        # Create a real safety client and mock its method
        safety = LlamaStackSafety()

        with patch.object(safety, "moderate_content") as mock_moderate:
            mock_moderate.return_value = SafetyResult(is_safe=True, violations=[])

            tool = LlamaStackModerationTool(safety_client=safety)
            result = tool._run("This is safe content")

            assert "passed moderation checks" in result

    def test_moderation_tool_unsafe_content(self):
        """Test moderation tool with unsafe content."""
        # Create a real safety client and mock its method
        safety = LlamaStackSafety()

        with patch.object(safety, "moderate_content") as mock_moderate:
            mock_moderate.return_value = SafetyResult(
                is_safe=False,
                violations=[
                    {"category": "hate", "flagged": True, "score": 0.8},
                    {"category": "violence", "flagged": False, "score": 0.2},
                ],
            )

            tool = LlamaStackModerationTool(safety_client=safety)
            result = tool._run("This is unsafe content")

            assert "flagged for" in result
            assert "hate" in result
            assert "0.8" in result
            # Should not include violence since it's not flagged
            assert "violence" not in result

    @pytest.mark.asyncio
    async def test_moderation_tool_async(self):
        """Test async moderation tool."""
        # Create a real safety client and mock its method
        safety = LlamaStackSafety()

        with patch.object(safety, "amoderate_content") as mock_amoderate:
            mock_amoderate.return_value = SafetyResult(is_safe=True, violations=[])

            tool = LlamaStackModerationTool(safety_client=safety)
            result = await tool._arun("This is safe content")

            assert "passed moderation checks" in result
