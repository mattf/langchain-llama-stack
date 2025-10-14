from importlib import metadata

from langchain_llamastack.chat_models import ChatLlamaStack

# Import safety hooks
from langchain_llamastack.input_output_safety_moderation_hooks import (
    SafeLLMWrapper,
    create_safe_llm,
    create_safety_hook,
)
from langchain_llamastack.safety import LlamaStackSafety, SafetyResult

# from langchain_llamastack.document_loaders import LlamaStackLoader
# from langchain_llamastack.embeddings import LlamaStackEmbeddings
# from langchain_llamastack.retrievers import LlamaStackRetriever
# from langchain_llamastack.toolkits import LlamaStackToolkit
# from langchain_llamastack.tools import LlamaStackTool
# from langchain_llamastack.vectorstores import LlamaStackVectorStore

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
