"""Clean unit tests for LlamaStackSafety using pytest async support."""

import os
from typing import Any, Dict, List
from unittest.mock import Mock, patch

from langchain_llamastack.safety import LlamaStackSafety, SafetyResult


class TestLlamaStackSafety:
    """Test class for LlamaStackSafety."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.base_url: str = "http://test-server.com"
        self.model: str = "test-model"

    def test_init_with_base_url(self) -> None:
        """Test initialization with base URL."""
        safety = LlamaStackSafety(base_url=self.base_url, model=self.model)
        assert safety.base_url == self.base_url
        assert safety.model == self.model

    def test_init_with_environment_url(self) -> None:
        """Test initialization using environment variable."""
        with patch.dict(os.environ, {"LLAMA_STACK_BASE_URL": "http://env-server.com"}):
            safety = LlamaStackSafety(model=self.model)
            assert safety.base_url == "http://env-server.com"

    def test_init_with_model(self) -> None:
        """Test initialization with specific model."""
        safety = LlamaStackSafety(base_url=self.base_url, model="test-model")
        assert safety.model == "test-model"
        assert safety.base_url == self.base_url

    def test_list_shields_with_mock_client(self) -> None:
        """Test list_shields method with mock client."""
        safety = LlamaStackSafety(base_url=self.base_url, model=self.model)

        # Mock the client attribute directly
        mock_client = Mock()
        mock_shield = Mock()
        mock_shield.identifier = "test-shield"
        mock_client.shields.list.return_value = [mock_shield]
        safety.client = mock_client

        shields = safety.list_shields()
        assert shields == ["test-shield"]

    def test_check_content_safety_with_mock(self) -> None:
        """Test check_content_safety with mock response."""
        safety = LlamaStackSafety(base_url=self.base_url, model=self.model)

        # Create a properly structured mock result
        class MockResult:
            flagged: bool = False
            categories: Dict[str, bool] = {}
            category_scores: Dict[str, float] = {}
            user_message: str = "Content is safe"
            metadata: Dict[str, Any] = {}

        # Mock the client directly
        mock_client = Mock()
        mock_response = Mock()
        mock_response.results = [MockResult()]
        mock_client.moderations.create.return_value = mock_response
        safety.client = mock_client

        result = safety.check_content_safety("Safe content")

        assert result.is_safe is True
        assert result.explanation is not None
        assert "Content is safe" in result.explanation

    def test_check_content_safety_flagged_content(self) -> None:
        """Test check_content_safety with flagged content."""
        safety = LlamaStackSafety(base_url=self.base_url, model=self.model)

        class MockResult:
            flagged: bool = True
            categories: Dict[str, bool] = {"violence": True}
            category_scores: Dict[str, float] = {"violence": 0.9}
            user_message: str = "Content flagged"
            metadata: Dict[str, Any] = {}

        # Mock the client directly
        mock_client = Mock()
        mock_response = Mock()
        mock_response.results = [MockResult()]
        mock_client.moderations.create.return_value = mock_response
        safety.client = mock_client

        result = safety.check_content_safety("Violent content")

        assert result.is_safe is False
        assert len(result.violations) > 0

    def test_check_content_safety_with_http_fallback(self) -> None:
        """Test HTTP fallback when client is not available."""
        # This test simulates the case where LlamaStackClient is not available
        with patch("langchain_llamastack.safety.LlamaStackClient", None):
            safety = LlamaStackSafety(base_url=self.base_url, model=self.model)
            result = safety.check_content_safety("Test content")

            # Should return safe with explanation about client not being available
            assert result.is_safe is True
            assert result.explanation is not None
            assert "LlamaStackClient not available" in result.explanation

    def test_check_content_safety_http_error(self) -> None:
        """Test error handling when client fails."""
        # This test simulates complete failure case
        with patch("langchain_llamastack.safety.LlamaStackClient", None):
            safety = LlamaStackSafety(base_url=self.base_url, model=self.model)
            result = safety.check_content_safety("Test content")

            # Should return a safe result with error explanation
            assert result.is_safe is True
            assert result.explanation is not None
            assert "LlamaStackClient not available" in result.explanation

    def test_process_safety_result_safe_content(self) -> None:
        """Test _process_safety_result with safe content."""
        safety = LlamaStackSafety(base_url=self.base_url, model=self.model)

        class MockResult:
            flagged: bool = False
            categories: Dict[str, bool] = {}
            category_scores: Dict[str, float] = {}
            user_message: str = "Content is safe"
            metadata: Dict[str, Any] = {"model": "test-model"}

        result = safety._process_safety_result([MockResult()])

        assert result.is_safe is True
        assert result.confidence_score == 1.0
        assert "Content is safe" in result.explanation
        assert len(result.violations) == 0

    def test_process_safety_result_unsafe_content(self) -> None:
        """Test _process_safety_result with unsafe content."""
        safety = LlamaStackSafety(base_url=self.base_url, model=self.model)

        class MockResult:
            flagged: bool = True
            categories: Dict[str, bool] = {"violence": True, "hate": False}
            category_scores: Dict[str, float] = {"violence": 0.9, "hate": 0.1}
            user_message: str = "Content flagged for violence"
            metadata: Dict[str, Any] = {"model": "test-model"}

        result = safety._process_safety_result([MockResult()])

        assert result.is_safe is False
        # 1 - max_score (with tolerance)
        assert result.confidence_score is not None
        assert (
            abs(result.confidence_score - 0.1) < 0.01
        )  # 1 - max_score (with tolerance)
        assert result.explanation is not None
        assert "violence" in result.explanation
        assert len(result.violations) == 1
        assert result.violations[0]["category"] == "violence"

    def test_process_safety_result_multiple_violations(self) -> None:
        """Test _process_safety_result with multiple violations."""
        safety = LlamaStackSafety(base_url=self.base_url, model=self.model)

        class MockResult:
            flagged: bool = True
            categories: Dict[str, bool] = {
                "violence": True,
                "hate": True,
                "spam": False,
            }
            category_scores: Dict[str, float] = {
                "violence": 0.8,
                "hate": 0.7,
                "spam": 0.2,
            }
            user_message: str = "Multiple violations detected"
            metadata: Dict[str, Any] = {}

        result = safety._process_safety_result([MockResult()])

        assert result.is_safe is False
        assert len(result.violations) == 2  # Only flagged categories
        violation_categories = [v["category"] for v in result.violations]
        assert "violence" in violation_categories
        assert "hate" in violation_categories
        assert "spam" not in violation_categories


class TestSafetyResult:
    """Test class for SafetyResult model."""

    def test_safety_result_safe(self) -> None:
        """Test SafetyResult with safe content."""
        result = SafetyResult(
            is_safe=True,
            confidence_score=0.95,
            explanation="Content is safe",
            violations=[],
        )

        assert result.is_safe is True
        assert result.confidence_score == 0.95
        assert result.explanation == "Content is safe"
        assert len(result.violations) == 0

    def test_safety_result_unsafe(self) -> None:
        """Test SafetyResult with unsafe content."""
        violations: List[Dict[str, Any]] = [
            {"category": "violence", "score": 0.9, "flagged": True}
        ]

        result = SafetyResult(
            is_safe=False,
            confidence_score=0.1,
            explanation="Content contains violence",
            violations=violations,
        )

        assert result.is_safe is False
        assert result.confidence_score == 0.1
        assert "violence" in result.explanation
        assert len(result.violations) == 1

    def test_safety_result_string_representation(self) -> None:
        """Test string representation of SafetyResult."""
        result = SafetyResult(
            is_safe=True,
            confidence_score=0.95,
            explanation="Safe content",
            violations=[],
        )

        str_repr = str(result)
        assert "is_safe=True" in str_repr
        assert "confidence_score=0.95" in str_repr
        assert "Safe content" in str_repr
