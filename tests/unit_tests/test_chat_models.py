from typing import Any, AsyncIterator, Tuple, Type

import pytest
import pytest_mock
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ChatModelUnitTests
from llama_stack_client import NOT_GIVEN
from llama_stack_client.resources.inference import InferenceResource
from llama_stack_client.types import (
    ChatCompletionResponse,
    CompletionMessage,
)

from langchain_llama_stack.chat_models import ChatLlamaStack


class TestChatLlamaStackUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatLlamaStack]:
        return ChatLlamaStack

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "not-a-model",
        }

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        """(tuple) environment variables, additional initialization args, and expected
        instance attributes for testing initialization from environment variables."""
        return (
            {
                "LLAMA_STACK_API_KEY": "test-key",  # we need this for test_serdes
            },
            {
                "model": "test-a-model",
            },
            {
                # when using LLAMA_STACK_API_KEY env, the key is not stored on
                # ChatLlamaStack, it is stored in the underlying LlamaStackClient.
                # therefore, we expect the key to be None on ChatLlamaStack and
                # cannot reasonably check it.
                # "api_key": None,
            },
        )

    @pytest.mark.parametrize(
        "tool_choice, expected_tool_config",
        [
            ("auto", {"tool_choice": "auto"}),
            ("required", {"tool_choice": "required"}),
            ("none", {"tool_choice": "none"}),
            ("xyz", {"tool_choice": "xyz"}),
            ("any", {"tool_choice": "auto"}),
            (None, NOT_GIVEN),
        ],
        ids=[
            "auto",
            "required",
            "none",
            "xyz",
            "any",
            "None",
        ],
    )
    def test_bind_tools_tool_choice(
        self,
        model: BaseChatModel,
        mocker: pytest_mock.MockFixture,
        my_adder_tool: BaseTool,
        tool_choice: str,
        expected_tool_config: str,
    ) -> None:
        mock_chat_completion = mocker.patch.object(
            InferenceResource,
            "chat_completion",
            return_value=ChatCompletionResponse(
                completion_message=CompletionMessage(
                    role="assistant",
                    content="Hello, world!",
                    stop_reason="end_of_turn",
                ),
            ),
        )

        bound_model = model.bind_tools(tools=[my_adder_tool], tool_choice=tool_choice)
        bound_model.invoke("Hello, world!")

        mock_chat_completion.assert_called_once()
        _, kwargs = mock_chat_completion.call_args
        assert "tool_config" in kwargs
        assert kwargs["tool_config"] == expected_tool_config

    @pytest.mark.asyncio
    async def test_async_is_concurrent(
        self, mocker: Any, model: ChatLlamaStack
    ) -> None:
        """
        ChatLlamaStack's _generate is used for all sync and async calls. It is
        implemented using a sync LlamaStackClient.

        We test that ainvoke and astream calls are run concurrently.

        We accomplish this by patching LlamaStackClient's inference.chat_completion
        method to sleep for 1/5 second, starting 5 async calls, and making sure they
        all finish in less than 1/4 seconds.
        """
        import asyncio
        import time

        def sleepy_chat_completion(*args: Any, **kwargs: Any) -> ChatCompletionResponse:
            time.sleep(1 / 5)
            return ChatCompletionResponse(
                completion_message=CompletionMessage(
                    role="assistant",
                    content="hello back!",
                    stop_reason="end_of_turn",
                ),
            )

        mocker.patch(
            "llama_stack_client.resources.inference.InferenceResource.chat_completion",
            side_effect=sleepy_chat_completion,
        )

        async def collect(stream: AsyncIterator) -> list:
            return [msg async for msg in stream]

        start = time.time()
        await asyncio.gather(
            model.ainvoke("Hello, world! (ai0)"),
            collect(model.astream("Hello, world! (as0)")),
            model.ainvoke("Hello, world! (ai1)"),
            collect(model.astream("Hello, world! (as1)")),
            model.ainvoke("Hello, world! (ai2)"),
        )
        assert time.time() - start < 1 / 4
