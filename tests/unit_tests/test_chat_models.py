from typing import Type

from langchain_tests.unit_tests import ChatModelUnitTests

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

    # TODO(mf): re-enable when we can extend the test suite
    # import pytest_mock
    # import pytest
    # from llama_stack_client import NOT_GIVEN
    # from langchain_core.language_models import BaseChatModel
    # from langchain_core.tools import BaseTool
    # from llama_stack_client.resources.inference import InferenceResource
    # from llama_stack_client.types import (
    #     ChatCompletionResponse,
    #     CompletionMessage,
    # )
    # @pytest.mark.parametrize(
    #     "tool_choice, expected_tool_config",
    #     [
    #         ("auto", {"tool_choice": "auto"}),
    #         ("required", {"tool_choice": "required"}),
    #         ("none", {"tool_choice": "none"}),
    #         ("xyz", {"tool_choice": "xyz"}),
    #         ("any", {"tool_choice": "auto"}),
    #         (None, NOT_GIVEN),
    #     ],
    #     ids=[
    #         "auto",
    #         "required",
    #         "none",
    #         "xyz",
    #         "any",
    #         "None",
    #     ],
    # )
    # def test_bind_tools_tool_choice(
    #     self,
    #     model: BaseChatModel,
    #     mocker: pytest_mock.MockFixture,
    #     my_adder_tool: BaseTool,
    #     tool_choice: str,
    #     expected_tool_config: str,
    # ) -> None:
    #     mock_chat_completion = mocker.patch.object(
    #         InferenceResource,
    #         "chat_completion",
    #         return_value=ChatCompletionResponse(
    #             completion_message=CompletionMessage(
    #                 role="assistant",
    #                 content="Hello, world!",
    #                 stop_reason="end_of_turn",
    #             ),
    #         ),
    #     )

    #     bound_model = model.bind_tools(tools=[my_adder_tool], tool_choice=tool_choice)
    #     bound_model.invoke("Hello, world!")

    #     mock_chat_completion.assert_called_once()
    #     _, kwargs = mock_chat_completion.call_args
    #     assert "tool_config" in kwargs
    #     assert kwargs["tool_config"] == expected_tool_config
