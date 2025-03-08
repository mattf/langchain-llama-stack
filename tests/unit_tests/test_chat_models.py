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
