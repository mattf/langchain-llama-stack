from typing import Type, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_llamastack.chat_models import ChatLlamaStack


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

    @pytest.mark.xfail(
        reason="not all models / endpoints support usage metadata (streaming)"
    )
    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        super().test_usage_metadata_streaming(model)

    @pytest.mark.xfail(
        reason=(
            "not all models / endpoints support json schema/pydantic handling "
            "during fallback"
        )
    )
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)

    @pytest.mark.xfail(reason="not all models / endpoints support optional params")
    def test_structured_output_optional_param(self, model: BaseChatModel) -> None:
        super().test_structured_output_optional_param(model)

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
