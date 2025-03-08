from typing import Type

from langchain_llama_stack.retrievers import LlamaStackRetriever
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)


class TestLlamaStackRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[LlamaStackRetriever]:
        """Get an empty vectorstore for unit tests."""
        return LlamaStackRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"k": 2}

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a dictionary representing the "args" of an example retriever call.
        """
        return "example query"
