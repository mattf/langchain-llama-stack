from importlib import metadata

from langchain_llama_stack.chat_models import ChatLlamaStack

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
    # "LlamaStackVectorStore",
    # "LlamaStackEmbeddings",
    # "LlamaStackLoader",
    # "LlamaStackRetriever",
    # "LlamaStackToolkit",
    # "LlamaStackTool",
    "__version__",
]
