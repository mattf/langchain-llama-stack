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
