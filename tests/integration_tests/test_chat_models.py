from typing import Any, Type

import pytest
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_llama_stack.chat_models import ChatLlamaStack


class TestChatLlamaStackIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatLlamaStack]:
        return ChatLlamaStack

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "meta/llama-3.1-8b-instruct",
        }

    @property
    def image_model_params(self) -> dict:
        return {
            "model": "meta/llama-3.2-11b-vision-instruct",
        }

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def supports_image_inputs(self) -> bool:
        return True

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

    # TODO(mf): re-enable when we can extend the test suite
    # def test_logprobs(self, model: BaseChatModel) -> None:
    #     logprobs_llm = model.bind(logprobs=True)
    #     ai_msg = logprobs_llm.invoke("Hello, how are you?")
    #     assert "logprobs" in ai_msg.response_metadata, "logprobs not present"

    # TODO(mf): re-enable when we can extend the test suite
    # from langchain_core.language_models import BaseChatModel

    # def test_doc_json_mode(self, model: BaseChatModel) -> None:
    #     """
    #     Test structured output via JSON mode according to
    #      https://python.langchain.com/docs/concepts/structured_outputs/#json-mode

    #         ```
    #         from langchain_llama_stack import ChatLlamaStack
    #         model = ChatLlamaStack(
    #           model=...
    #         ).with_structured_output(method="json_mode")
    #         ai_msg = model.invoke(
    #             "Return a JSON object with key 'random_ints' "
    #             "and a value of 10 random ints in [0-99]"
    #         )
    #         ai_msg
    #         {'random_ints': [27, 84, 9, 56, 31, 57, 43, 68, 19, 74]}
    #         ```
    #     """
    #     ai_msg = (
    #         cast(ChatLlamaStack, model)
    #         .with_structured_output(method="json_mode")
    #         .invoke(
    #             "Return only a JSON object with key 'random_ints' "
    #             "and a value of 10 random ints in [0-99]. "
    #         )
    #     )
    #     assert isinstance(ai_msg, dict)
    #     assert "random_ints" in ai_msg
    #     assert isinstance(ai_msg["random_ints"], list)
