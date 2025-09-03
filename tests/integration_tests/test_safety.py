"""Integration tests for LlamaStackSafety."""

import os

import pytest

from langchain_llama_stack.safety import (
    LlamaStackModerationTool,
    LlamaStackSafety,
    LlamaStackSafetyTool,
    SafetyResult,
)


class TestLlamaStackSafetyIntegration:
    """Integration tests for LlamaStackSafety."""

    @pytest.fixture
    def safety_client(self):
        """Create a LlamaStackSafety client for testing."""
        return LlamaStackSafety(
            base_url="http://localhost:8321",
            shield_type="llama_guard",
        )

    def test_check_content_safety_integration(self, safety_client):
        """Test actual content safety check (requires running Llama Stack server on localhost:8321)."""
        # Test safe content
        safe_content = "Hello, how are you today?"
        result = safety_client.check_content_safety(safe_content)

        assert isinstance(result, SafetyResult)
        assert isinstance(result.is_safe, bool)
        assert isinstance(result.violations, list)

    def test_moderate_content_integration(self, safety_client):
        """Test actual content moderation (requires running Llama Stack server on localhost:8321)."""
        # Test content moderation
        content = "This is a test message for moderation."
        result = safety_client.moderate_content(content)

        assert isinstance(result, SafetyResult)
        assert isinstance(result.is_safe, bool)
        assert isinstance(result.violations, list)

    @pytest.mark.asyncio
    async def test_async_safety_check_integration(self, safety_client):
        """Test async safety check (requires running Llama Stack server on localhost:8321)."""
        content = "Hello, this is a test for async safety check."
        result = await safety_client.acheck_content_safety(content)

        assert isinstance(result, SafetyResult)
        assert isinstance(result.is_safe, bool)
        assert isinstance(result.violations, list)

    @pytest.mark.asyncio
    async def test_async_moderation_integration(self, safety_client):
        """Test async moderation (requires running Llama Stack server on localhost:8321)."""
        content = "This is a test message for async moderation."
        result = await safety_client.amoderate_content(content)

        assert isinstance(result, SafetyResult)
        assert isinstance(result.is_safe, bool)
        assert isinstance(result.violations, list)

    def test_safety_client_initialization(self):
        """Test safety client can be initialized with different parameters."""
        # Test with custom parameters
        safety = LlamaStackSafety(
            base_url="http://custom:8321",
            api_key="custom-key",
            shield_type="custom_shield",
            moderation_model="custom_model",
            timeout=60.0,
            max_retries=5,
        )

        assert safety.base_url == "http://custom:8321"
        assert safety.api_key == "custom-key"
        assert safety.shield_type == "custom_shield"
        assert safety.moderation_model == "custom_model"
        assert safety.timeout == 60.0
        assert safety.max_retries == 5

    def test_safety_client_env_variables(self):
        """Test safety client uses environment variables."""
        # Test with environment variables (mocked)
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("LLAMA_STACK_BASE_URL", "http://env:8321")
            mp.setenv("LLAMA_STACK_API_KEY", "env-key")

            safety = LlamaStackSafety()
            assert safety.base_url == "http://env:8321"
            assert safety.api_key == "env-key"

    def test_safety_tool_integration(self, safety_client):
        """Test LlamaStackSafetyTool integration (requires running Llama Stack server on localhost:8321)."""
        tool = LlamaStackSafetyTool(safety_client=safety_client)

        # Test the tool interface
        assert tool.name == "llama_stack_safety"
        assert "safety" in tool.description.lower()

        # Test running the tool
        result = tool._run("Hello, this is a test message.")

        # Should return a string result
        assert isinstance(result, str)
        assert "SAFE" in result or "UNSAFE" in result

    def test_moderation_tool_integration(self, safety_client):
        """Test LlamaStackModerationTool integration (requires running Llama Stack server on localhost:8321)."""
        tool = LlamaStackModerationTool(safety_client=safety_client)

        # Test the tool interface
        assert tool.name == "llama_stack_moderation"
        assert "moderation" in tool.description.lower()

        # Test running the tool
        result = tool._run("Hello, this is a test message.")

        # Should return a string result
        assert isinstance(result, str)
        assert "passed" in result or "flagged" in result

    @pytest.mark.asyncio
    async def test_tools_async_integration(self, safety_client):
        """Test async tool functionality (requires running Llama Stack server on localhost:8321)."""
        safety_tool = LlamaStackSafetyTool(safety_client=safety_client)
        moderation_tool = LlamaStackModerationTool(safety_client=safety_client)

        # Test async safety tool
        safety_result = await safety_tool._arun("Hello, this is a test message.")
        assert isinstance(safety_result, str)
        assert "SAFE" in safety_result or "UNSAFE" in safety_result

        # Test async moderation tool
        moderation_result = await moderation_tool._arun(
            "Hello, this is a test message."
        )
        assert isinstance(moderation_result, str)
        assert "passed" in moderation_result or "flagged" in moderation_result
