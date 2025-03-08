"""Test LlamaStack embeddings."""

from typing import Type

from langchain_llama_stack.embeddings import LlamaStackEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[LlamaStackEmbeddings]:
        return LlamaStackEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
