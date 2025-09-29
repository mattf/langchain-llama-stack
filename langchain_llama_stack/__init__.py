from importlib import metadata

from langchain_llama_stack.chat_models import ChatLlamaStack

# Import safety hooks
from langchain_llama_stack.input_output_safety_moderation_hooks import (
    SafeLLMWrapper,
    create_safe_llm,
    create_safety_hook,
)
from langchain_llama_stack.safety import LlamaStackSafety, SafetyResult

# from langchain_llama_stack.document_loaders import LlamaStackLoader
# from langchain_llama_stack.embeddings import LlamaStackEmbeddings
# from langchain_llama_stack.retrievers import LlamaStackRetriever
# from langchain_llama_stack.toolkits import LlamaStackToolkit
# from langchain_llama_stack.tools import LlamaStackTool
# from langchain_llama_stack.vectorstores import LlamaStackVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatLlamaStack",
    "LlamaStackSafety",
    "SafetyResult",
    "SafeLLMWrapper",
    "create_safe_llm",
    "create_safety_hook",
    "__version__",
    # Future components (commented out until implemented):
    # "LlamaStackVectorStore",
    # "LlamaStackEmbeddings",
    # "LlamaStackTelemetry",
    # "LlamaStackLoader",
]
