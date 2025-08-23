from typing import Any, Type, cast

import llama_stack_client
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_llama_stack.chat_models import ChatLlamaStack


class TestChatLlamaStackIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatLlamaStack]:
        return ChatLlamaStack

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "meta-llama/Llama-3.3-70B-Instruct",
        }

    @property
    def image_model_params(self) -> dict:
        return {
            "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        }

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @pytest.fixture
    def image_model(self) -> BaseChatModel:
        return self.chat_model_class(
            **{
                **self.standard_chat_model_params,
                **self.image_model_params,
            }
        )

    #
    # handling test_tool_call_with_no_arguments -
    #
    # we know it will fail for Llama API, so we skip that case.
    # we want it to fail otherwise.
    #
    # the standard tests have a DO_NOT_OVERRIDE without xfail requirement.
    # so we mark the test as xfail and create a copy that only skips the
    # known issue w/ Llama API.
    #

    @pytest.mark.xfail(reason="Llama API does not support argument-less functions")
    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        super().test_tool_calling_with_no_arguments(model)

    def test_tool_calling_with_no_arguments_(self, model: BaseChatModel) -> None:
        try:
            super().test_tool_calling_with_no_arguments(model)
        except llama_stack_client.BadRequestError as e:
            message = str(e)
            if (
                "schema constraint" in message
                and "required" in message
                and "tools.0.function" in message
            ):
                pytest.skip(
                    "Llama API (https://www.llama.com/products/llama-api/) "
                    "does not support argument-less functions"
                )
            else:
                raise

    @pytest.mark.xfail(reason="Does not follow OpenAI tool call wire format")
    def test_tool_message_histories_string_content(self, *args: Any) -> None:
        super().test_tool_message_histories_string_content(*args)

    @pytest.mark.xfail(reason=("Does not follow Anthropic wire format"))
    def test_tool_message_histories_list_content(self, *args: Any) -> None:
        super().test_tool_message_histories_list_content(*args)

    @pytest.mark.xfail(reason=("Does not support tool call status"))
    def test_tool_message_error_status(self, *args: Any) -> None:
        super().test_tool_message_error_status(*args)

    @pytest.mark.xfail(reason=("Does not follow OpenAI tool call wire format"))
    def test_structured_few_shot_examples(self, *args: Any) -> None:
        # need to support messages = [
        #   HumanMessage(content='What is 1 + 2', ...),
        #   AIMessage(content='',
        #             tool_calls=[{'name': 'my_adder_tool',
        #                          'args': {'a': 1, 'b': 2},
        #                          'id': 'id0',
        #                          'type': 'tool_call'}],
        #             additional_kwargs={
        #               'tool_calls': [{
        #                 'id': 'id0',
        #                 'type': 'function',
        #                 'function': {
        #                   'name': 'my_adder_tool',
        #                   'arguments': '{"a":1,"b":2}'}}
        #               ]},
        #             ...)
        #   ToolMessage(content='{"result": 3}',
        #               tool_call_id='id0'),
        #   AIMessage(content='{"result": 3}', ...),
        #   HumanMessage(content='What is 3 + 4', ...)
        # ]
        super().test_structured_few_shot_examples(*args)

    @pytest.mark.xfail(reason=("Not all models / endpoints support logprobs"))
    def test_logprobs(self, model: BaseChatModel) -> None:
        logprobs_llm = model.bind(logprobs=True)
        ai_msg = logprobs_llm.invoke("Hello, how are you?")
        assert "logprobs" in ai_msg.response_metadata, "logprobs not present"

    def test_doc_json_mode(self, model: BaseChatModel) -> None:
        """
        Test structured output via JSON mode according to
         https://python.langchain.com/docs/concepts/structured_outputs/#json-mode

            ```
            from langchain_llama_stack import ChatLlamaStack
            model = ChatLlamaStack(
              model=...
            ).with_structured_output(method="json_mode")
            ai_msg = model.invoke(
                "Return a JSON object with key 'random_ints' "
                "and a value of 10 random ints in [0-99]"
            )
            ai_msg
            {'random_ints': [27, 84, 9, 56, 31, 57, 43, 68, 19, 74]}
            ```
        """
        ai_msg = (
            cast(ChatLlamaStack, model)
            .with_structured_output(method="json_mode")
            .invoke(
                "Return only a JSON object with key 'random_ints' "
                "and a value of 10 random ints in [0-99]. "
            )
        )
        assert isinstance(ai_msg, dict)
        assert "random_ints" in ai_msg
        assert isinstance(ai_msg["random_ints"], list)

    #
    # Special handling for image tests -
    #
    #  The chat model may not support image inputs and an image model may
    #  not support all chat features. To address this we introduce a
    #  new fixture configured for image model tests.
    #
    #  It is modeled after https://github.com/langchain-ai/langchain/pull/30395.
    #
    #
    #  To address this we'll override the image tests as xfail and then
    #  call them directly with the image model fixture.

    @pytest.mark.xfail(reason="Model may not support images")
    def test_image_inputs(self, image_model: BaseChatModel) -> None:
        super().test_image_inputs(image_model)

    #
    # End of special handling of image tests.
    #
