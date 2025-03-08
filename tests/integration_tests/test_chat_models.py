from typing import Type

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
            "model": "meta/llama-3.1-8b-instruct",
        }

    @property
    def returns_usage_metadata(self) -> bool:
        return False

    @pytest.mark.xfail(reason="Produces full output, not chunks")
    def test_stream(self, model: BaseChatModel) -> None:
        self.test_stream(model)

    @pytest.mark.xfail(reason="Produces full output, not chunks")
    async def test_astream(self, model: BaseChatModel) -> None:
        await self.test_astream(model)
