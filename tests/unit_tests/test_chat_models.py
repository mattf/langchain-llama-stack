from typing import Tuple, Type

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_llamastack.chat_models import ChatLlamaStack


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
