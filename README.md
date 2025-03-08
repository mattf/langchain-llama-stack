# langchain-llama-stack

This package contains the LangChain integration with [Llama Stack](https://github.com/meta-llama/llama-stack).

## Installation

```bash
pip install -U langchain-llama-stack
```

If your Llama Stack distribution requires credentials, use the `LLAMA_STACK_API_KEY` environment variable.

If your Llama Stack distribution server is not running on `http://localhost:8321`, use the `LLAMA_STACK_BASE_URL` environment variable.

## Chat Models

`ChatLlamaStack` class exposes chat models, which are hosted at `/v1/inference/chat-completion` on your Llama Stack distribution server.

```python
from langchain_llama_stack import ChatLlamaStack

llm = ChatLlamaStack(model="meta/llama-3.1-8b-instruct")
llm.invoke("Sing a ballad of LangChain.")
```
