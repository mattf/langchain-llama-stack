"""Clean unit tests for input_output_safety_moderation_hooks using pytest."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from langchain_llamastack.input_output_safety_moderation_hooks import (
    SafeLLMWrapper,
    create_safe_llm,
    create_safety_hook,
)
from langchain_llamastack.safety import LlamaStackSafety, SafetyResult


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self) -> None:
        """Initialize mock LLM."""
        self.model = "test-model"

    def invoke(self, input_text: str) -> AIMessage:
        """Mock invoke method."""
        return AIMessage(content=f"Response to: {input_text}")

    async def ainvoke(self, input_text: str) -> AIMessage:
        """Mock async invoke method."""
        return AIMessage(content=f"Async response to: {input_text}")


class MockAgent:
    """Mock agent for testing."""

    def invoke(self, input_text: str) -> str:
        """Mock invoke method."""
        return f"Agent response to: {input_text}"

    async def ainvoke(self, input_text: str) -> str:
        """Mock async invoke method."""
        return f"Async agent response to: {input_text}"


class TestSafeLLMWrapper:
    """Test class for SafeLLMWrapper."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.mock_llm = MockLLM()
        self.mock_safety = Mock(spec=LlamaStackSafety)
        self.mock_safety.check_content_safety.return_value = SafetyResult(
            is_safe=True, confidence_score=0.9, explanation="Safe", violations=[]
        )

    def test_init_with_llm(self) -> None:
        """Test initialization with LLM."""
        wrapper = SafeLLMWrapper(self.mock_llm, self.mock_safety)

        assert wrapper.runnable == self.mock_llm
        assert wrapper.safety_client == self.mock_safety
        assert wrapper.input_hook is None
        assert wrapper.output_hook is None

    def test_set_hooks(self) -> None:
        """Test setting input and output hooks."""
        wrapper = SafeLLMWrapper(self.mock_llm, self.mock_safety)

        def mock_hook(content: str) -> SafetyResult:
            return SafetyResult(
                is_safe=True, confidence_score=0.9, explanation="Safe", violations=[]
            )

        wrapper.set_input_hook(mock_hook)
        wrapper.set_output_hook(mock_hook)

        assert wrapper.input_hook == mock_hook
        assert wrapper.output_hook == mock_hook

    def test_invoke_with_safe_content_no_hooks(self) -> None:
        """Test invoke with safe content and no hooks."""
        wrapper = SafeLLMWrapper(self.mock_llm, self.mock_safety)
        result = wrapper.invoke("Hello world")

        # Should return the model output directly since no hooks are set
        assert result == "Response to: Hello world"

    def test_invoke_with_safe_content_with_hooks(self) -> None:
        """Test invoke with safe content and hooks enabled."""
        wrapper = SafeLLMWrapper(self.mock_llm, self.mock_safety)

        # Create safe hooks
        def safe_hook(content: str) -> SafetyResult:
            return SafetyResult(
                is_safe=True, confidence_score=0.9, explanation="Safe", violations=[]
            )

        wrapper.set_input_hook(safe_hook)
        wrapper.set_output_hook(safe_hook)

        result = wrapper.invoke("Hello world")
        assert result == "Response to: Hello world"

    def test_invoke_with_unsafe_input(self) -> None:
        """Test invoke with unsafe input content."""
        wrapper = SafeLLMWrapper(self.mock_llm, self.mock_safety)

        # Create unsafe input hook
        def unsafe_input_hook(content: str) -> SafetyResult:
            return SafetyResult(
                is_safe=False,
                confidence_score=0.8,
                explanation="Unsafe input",
                violations=[
                    {"category": "violence", "reason": "Violent content detected"}
                ],
            )

        wrapper.set_input_hook(unsafe_input_hook)

        result = wrapper.invoke("Harmful content")
        assert isinstance(result, str)
        assert "Input blocked by safety system" in result

    def test_invoke_with_unsafe_output(self) -> None:
        """Test invoke with unsafe output content."""
        wrapper = SafeLLMWrapper(self.mock_llm, self.mock_safety)

        # Create safe input hook and unsafe output hook
        def safe_input_hook(content: str) -> SafetyResult:
            return SafetyResult(
                is_safe=True, confidence_score=0.9, explanation="Safe", violations=[]
            )

        def unsafe_output_hook(content: str) -> SafetyResult:
            return SafetyResult(
                is_safe=False,
                confidence_score=0.8,
                explanation="Unsafe output",
                violations=[
                    {"category": "violence", "reason": "Violent content in output"}
                ],
            )

        wrapper.set_input_hook(safe_input_hook)
        wrapper.set_output_hook(unsafe_output_hook)

        result = wrapper.invoke("Hello world")
        assert isinstance(result, str)
        assert "Output blocked by safety system" in result

    @pytest.mark.asyncio
    async def test_ainvoke_with_safe_content(self) -> None:
        """Test async invoke with safe content."""
        wrapper = SafeLLMWrapper(self.mock_llm, self.mock_safety)

        # Create safe hooks
        def safe_hook(content: str) -> SafetyResult:
            return SafetyResult(
                is_safe=True, confidence_score=0.9, explanation="Safe", violations=[]
            )

        wrapper.set_input_hook(safe_hook)
        wrapper.set_output_hook(safe_hook)

        result = await wrapper.ainvoke("Hello async world")
        assert result == "Async response to: Hello async world"

    def test_invoke_with_agent(self) -> None:
        """Test invoke with agent (non-LLM runnable)."""
        mock_agent = MockAgent()
        wrapper = SafeLLMWrapper(mock_agent, self.mock_safety)

        # Create safe hooks
        def safe_hook(content: str) -> SafetyResult:
            return SafetyResult(
                is_safe=True, confidence_score=0.9, explanation="Safe", violations=[]
            )

        wrapper.set_input_hook(safe_hook)
        wrapper.set_output_hook(safe_hook)

        result = wrapper.invoke("Test agent input")
        assert result == "Agent response to: Test agent input"

    def test_invoke_with_callable(self) -> None:
        """Test invoke with simple callable."""

        def simple_func(input_text: str) -> str:
            return f"Function response: {input_text}"

        wrapper = SafeLLMWrapper(simple_func, self.mock_safety)

        # Create safe hooks
        def safe_hook(content: str) -> SafetyResult:
            return SafetyResult(
                is_safe=True, confidence_score=0.9, explanation="Safe", violations=[]
            )

        wrapper.set_input_hook(safe_hook)
        wrapper.set_output_hook(safe_hook)

        result = wrapper.invoke("Test function input")
        assert result == "Function response: Test function input"

    def test_hook_exception_handling(self) -> None:
        """Test handling of hook exceptions."""
        wrapper = SafeLLMWrapper(self.mock_llm, self.mock_safety)

        # Create hook that raises exception
        def failing_hook(content: str) -> SafetyResult:
            raise Exception("Hook failed")

        wrapper.set_input_hook(failing_hook)

        # Should raise the exception from the hook
        with pytest.raises(Exception, match="Hook failed"):
            wrapper.invoke("Test input")


class TestCreateSafetyHook:
    """Test class for create_safety_hook function."""

    def test_create_input_hook_safe(self) -> None:
        """Test creating input hook with safe content."""
        mock_safety = Mock(spec=LlamaStackSafety)
        mock_safety.check_content_safety.return_value = SafetyResult(
            is_safe=True, confidence_score=0.9, explanation="Safe", violations=[]
        )

        hook = create_safety_hook(mock_safety, "input")
        result = hook("safe content")

        assert result.is_safe is True
        mock_safety.check_content_safety.assert_called_once_with("safe content")

    def test_create_input_hook_fails_open(self) -> None:
        """Test that input hooks fail open on error."""
        mock_safety = Mock(spec=LlamaStackSafety)
        mock_safety.check_content_safety.side_effect = Exception("API Error")

        hook = create_safety_hook(mock_safety, "input")
        result = hook("test content")

        # Input hooks should fail open (allow content)
        assert result.is_safe is True
        assert "Safety check failed" in result.explanation

    def test_create_output_hook_fails_closed(self) -> None:
        """Test that output hooks fail closed on error."""
        mock_safety = Mock(spec=LlamaStackSafety)
        mock_safety.check_content_safety.side_effect = Exception("API Error")

        hook = create_safety_hook(mock_safety, "output")
        result = hook("test content")

        # Output hooks should fail closed (block content)
        assert result.is_safe is False
        assert "Safety check failed" in result.explanation


class TestCreateSafeLLM:
    """Test class for create_safe_llm factory function."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.mock_llm = MockLLM()
        self.mock_safety = Mock(spec=LlamaStackSafety)
        self.mock_safety.model = "test-shield"
        # Mock list_shields to return a list so the 'in' operator works
        self.mock_safety.list_shields.return_value = ["test-shield", "prompt-guard"]

    def test_create_safe_llm_default(self) -> None:
        """Test create_safe_llm with default parameters."""
        safe_llm = create_safe_llm(self.mock_llm, self.mock_safety)

        assert isinstance(safe_llm, SafeLLMWrapper)
        assert safe_llm.runnable == self.mock_llm
        assert safe_llm.safety_client == self.mock_safety

    def test_create_safe_llm_input_check_enabled(self) -> None:
        """Test create_safe_llm with input check enabled."""
        safe_llm = create_safe_llm(self.mock_llm, self.mock_safety, input_check=True)

        assert isinstance(safe_llm, SafeLLMWrapper)
        assert safe_llm.input_hook is not None

    def test_create_safe_llm_output_check_enabled(self) -> None:
        """Test create_safe_llm with output check enabled."""
        safe_llm = create_safe_llm(self.mock_llm, self.mock_safety, output_check=True)

        assert isinstance(safe_llm, SafeLLMWrapper)
        assert safe_llm.output_hook is not None

    def test_create_safe_llm_both_checks_disabled(self) -> None:
        """Test create_safe_llm with both checks disabled."""
        safe_llm = create_safe_llm(
            self.mock_llm, self.mock_safety, input_check=False, output_check=False
        )

        # When both checks are disabled,
        # should still return a SafeLLMWrapper
        #  but with no hooks
        assert isinstance(safe_llm, SafeLLMWrapper)
        assert safe_llm.input_hook is None
        assert safe_llm.output_hook is None

    def test_create_safe_llm_with_different_runnables(self) -> None:
        """Test create_safe_llm with different types of runnables."""
        # Test with LLM
        safe_llm = create_safe_llm(self.mock_llm, self.mock_safety)
        assert isinstance(safe_llm, SafeLLMWrapper)

        # Test with agent
        mock_agent = MockAgent()
        safe_agent = create_safe_llm(mock_agent, self.mock_safety)
        assert isinstance(safe_agent, SafeLLMWrapper)

        # Test with callable
        def simple_func(x: str) -> str:
            return f"Result: {x}"

        safe_func = create_safe_llm(simple_func, self.mock_safety)
        assert isinstance(safe_func, SafeLLMWrapper)
