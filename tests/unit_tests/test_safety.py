"""Clean unit tests for LlamaStackSafety using pytest async support."""

import os
from typing import Any
from unittest.mock import Mock, patch

import pytest

from langchain_llama_stack.safety import LlamaStackSafety, SafetyResult


class TestLlamaStackSafety:
    """Test class for LlamaStackSafety."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.base_url = "http://test-server.com"
        self.init_kwargs = {
            "base_url": self.base_url,
            "model": "test-model",
        }

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        with patch.dict(os.environ, {}, clear=True):
            safety = LlamaStackSafety()

            assert safety.base_url == "http://localhost:8321"
            assert safety.model == "ollama/llama-guard3:8b"
            assert safety.timeout == 30.0
            assert safety.max_retries == 2

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        safety = LlamaStackSafety(**self.init_kwargs)

        assert safety.base_url == self.base_url
        assert safety.model == "test-model"

    @patch("langchain_llama_stack.safety.LlamaStackClient")
    def test_check_content_safety_success(self, mock_client_class: Any) -> None:
        """Test successful content safety check."""
        mock_client = Mock()

        # Create a simple mock response that matches the expected structure
        class MockResult:
            flagged = False
            categories = {}
            category_scores = {}
            user_message = "Safe content"
            metadata = {}

        class MockResponse:
            id = "mod-123"
            model = "test-model"
            results = [MockResult()]

        mock_response = MockResponse()
        mock_client.moderations.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(**self.init_kwargs)
        result = safety.check_content_safety("Hello world")

        assert isinstance(result, SafetyResult)
        assert result.is_safe is True
        assert result.explanation == "Safe content"

    @patch("langchain_llama_stack.safety.LlamaStackClient")
    def test_check_content_safety_with_violation(self, mock_client_class: Any) -> None:
        """Test content safety check with violation."""
        mock_client = Mock()

        # Create a mock response that indicates a violation
        class MockResult:
            flagged = True  # This should make is_safe = False
            categories = {"harassment": True, "violence": False}
            category_scores = {"harassment": 0.95, "violence": 0.1}
            user_message = None
            metadata = {}

        class MockResponse:
            id = "mod-456"
            model = "test-model"
            results = [MockResult()]

        mock_response = MockResponse()
        mock_client.moderations.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(**self.init_kwargs)
        result = safety.check_content_safety("Harmful content")

        assert result.is_safe is False  # Should be False because flagged=True
        assert len(result.violations) == 1
        assert result.violations[0]["category"] == "harassment"

    @patch("langchain_llama_stack.safety.LlamaStackClient")
    def test_check_content_safety_error(self, mock_client_class: Any) -> None:
        """Test content safety check with error."""
        mock_client = Mock()
        mock_client.moderations.create.side_effect = Exception("API error")
        mock_client_class.return_value = mock_client

        safety = LlamaStackSafety(**self.init_kwargs)
        result = safety.check_content_safety("Test content")

        assert result.is_safe is True  # Fails open
        assert "Safety check failed: API error" in result.explanation

    @pytest.mark.asyncio
    @patch("langchain_llama_stack.safety.AsyncLlamaStackClient")
    async def test_acheck_content_safety_success(
        self, mock_async_client_class: Any
    ) -> None:
        """Test successful async content safety check."""
        mock_async_client = Mock()

        # Create a simple mock response that matches the expected structure
        class MockResult:
            flagged = False
            categories = {}
            category_scores = {}
            user_message = "Safe async content"
            metadata = {}

        class MockResponse:
            id = "mod-123"
            model = "test-model"
            results = [MockResult()]

        mock_response = MockResponse()

        # Make the async call return a coroutine that resolves to the mock response
        async def mock_moderations_create(*args, **kwargs):
            return mock_response

        mock_async_client.moderations.create = mock_moderations_create
        mock_async_client_class.return_value = mock_async_client

        safety = LlamaStackSafety(**self.init_kwargs)
        result = await safety.acheck_content_safety("Hello async world")

        assert result.is_safe is True
        assert result.explanation == "Safe async content"

    @pytest.mark.asyncio
    @patch("langchain_llama_stack.safety.AsyncLlamaStackClient")
    async def test_acheck_content_safety_with_violation(
        self, mock_async_client_class: Any
    ) -> None:
        """Test async content safety check with violation."""
        mock_async_client = Mock()

        # Create a mock response that indicates a violation
        class MockResult:
            flagged = True  # This should make is_safe = False
            categories = {"violence": True, "harassment": False}
            category_scores = {"violence": 0.85, "harassment": 0.1}
            user_message = None
            metadata = {}

        class MockResponse:
            id = "mod-789"
            model = "test-model"
            results = [MockResult()]

        mock_response = MockResponse()

        # Make the async call return a coroutine that resolves to the mock response
        async def mock_moderations_create(*args, **kwargs):
            return mock_response

        mock_async_client.moderations.create = mock_moderations_create
        mock_async_client_class.return_value = mock_async_client

        safety = LlamaStackSafety(**self.init_kwargs)
        result = await safety.acheck_content_safety("Violent content")

        assert result.is_safe is False  # Should be False because flagged=True
        assert len(result.violations) == 1
        assert result.violations[0]["category"] == "violence"

    @pytest.mark.asyncio
    @patch("langchain_llama_stack.safety.AsyncLlamaStackClient")
    async def test_acheck_content_safety_error(
        self, mock_async_client_class: Any
    ) -> None:
        """Test async content safety check with error."""
        mock_async_client = Mock()

        # Make the async call raise an exception
        async def mock_moderations_create_error(*args, **kwargs):
            raise Exception("Async API error")

        mock_async_client.moderations.create = mock_moderations_create_error
        mock_async_client_class.return_value = mock_async_client

        safety = LlamaStackSafety(**self.init_kwargs)
        result = await safety.acheck_content_safety("Test async content")

        assert result.is_safe is True  # Fails open
        assert "Async safety check failed: Async API error" in result.explanation
