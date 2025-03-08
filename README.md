# langchain-llama-stack

This package contains the LangChain integration with LlamaStack

## Installation

```bash
pip install -U langchain-llama-stack
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatLlamaStack` class exposes chat models from LlamaStack.

```python
from langchain_llama_stack import ChatLlamaStack

llm = ChatLlamaStack()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`LlamaStackEmbeddings` class exposes embeddings from LlamaStack.

```python
from langchain_llama_stack import LlamaStackEmbeddings

embeddings = LlamaStackEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`LlamaStackLLM` class exposes LLMs from LlamaStack.

```python
from langchain_llama_stack import LlamaStackLLM

llm = LlamaStackLLM()
llm.invoke("The meaning of life is")
```
