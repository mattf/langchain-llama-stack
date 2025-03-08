import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from llama_stack_client.types import (
    ChatCompletionResponse,
)
from llama_stack_client.types.shared_params import (
    CompletionMessage as LlamaStackCompletionMessage,
)
from llama_stack_client.types.shared_params import (
    InterleavedContent as LlamaStackInterleavedContent,
)
from llama_stack_client.types.shared_params import (
    Message as LlamaStackMessage,
)
from llama_stack_client.types.shared_params import (
    SystemMessage as LlamaStackSystemMessage,
)
from llama_stack_client.types.shared_params import (
    ToolResponseMessage as LlamaStackToolResponseMessage,
)
from llama_stack_client.types.shared_params import (
    UserMessage as LlamaStackUserMessage,
)
from llama_stack_client.types.shared_params.interleaved_content_item import (
    TextContentItem as LlamaStackTextContentItem,
)

logger = logging.getLogger(__name__)


def _convert_content(
    content: str | list[str | dict[Any, Any]],
) -> LlamaStackInterleavedContent:
    """
    Convert LangChain content to Llama Stack interleaved content.

    TODO(mf): add support for image content
    TODO(mf): validate content schema

    Notes:
     - LangChain content allows for list of strings while Llama Stack does not. These
       are converted to lists of TextContentItem.

    Args:
        content: LangChain content.

    Returns:
        LlamaStackInterleavedContent: Llama Stack interleaved content.
    """
    if isinstance(content, list):
        return [
            LlamaStackTextContentItem(type="text", text=text)
            if isinstance(text, str)
            else LlamaStackTextContentItem(type="text", text=text["text"])
            for text in content
        ]

    # content is a string
    return content


def convert_message(message: BaseMessage) -> LlamaStackMessage:
    """
    Convert a LangChain message to a Llama Stack message.

    | LangChain message | Llama Stack message           | Notes                |
    |-------------------|-------------------------------|----------------------|
    | SystemMessage     | LlamaStackSystemMessage       |                      |
    | AIMessage         | LlamaStackCompletionMessage   | stop_reason mismatch |
    | HumanMessage      | LlamaStackUserMessage         |                      |
    | ToolMessage       | LlamaStackToolResponseMessage | tool_name mismatch   |

    Notes:
        - stop_reason defaulted to "end_of_turn" if not found

    Args:
        message: A LangChain message.

    Returns:
        LlamaStackMessage: A Llama Stack message.
    """
    if isinstance(message, SystemMessage):
        return LlamaStackSystemMessage(
            role="system",
            content=_convert_content(message.content),
        )
    elif isinstance(message, AIMessage):
        if "stop_reason" not in message.response_metadata:
            logger.warning(
                "no stop_reason found in response_metadata"
                f": {message.response_metadata}, "
                "will use end_of_turn as default."
            )
        return LlamaStackCompletionMessage(
            role="assistant",
            content=_convert_content(message.content),
            # this may fail is user is using messages from multiple LLMs,
            # e.g. Llama Stack wants end_of_turn/end_of_message/out_of_tokens
            # while other LLMs may use length in place of out_of_tokens.
            # also, other LLMs may use finish_reason instead of stop_reason.
            stop_reason=message.response_metadata.get("stop_reason", "end_of_turn"),
        )
    elif isinstance(message, HumanMessage):
        return LlamaStackUserMessage(
            role="user",
            content=_convert_content(message.content),
        )
    elif isinstance(message, ToolMessage):
        # ToolMessage.status has no parallel in Llama Stack Inference API
        return LlamaStackToolResponseMessage(
            role="tool",
            content=_convert_content(message.content),
            call_id=message.tool_call_id,
            # Llama Stack requires tool_name, LangChain does not provide it,
            # see https://github.com/meta-llama/llama-stack/issues/1421
            tool_name="use the call_id",
        )
    elif isinstance(message, FunctionMessage):
        raise ValueError("FunctionMessage is not supported, use ToolMessage instead.")
    elif isinstance(message, ChatMessage):
        raise ValueError(
            "ChatMessage is not supported, use AIMessage / HumanMessage instead."
        )
    else:
        raise ValueError(f"Unknown message type: {type(message)}")


def convert_response(response: ChatCompletionResponse) -> ChatResult:
    if isinstance(response.completion_message.content, str):
        content: str | list[str | dict] = response.completion_message.content
    elif isinstance(response.completion_message.content, list):
        content = [
            item if isinstance(item, str) else item.to_dict()
            for item in response.completion_message.content
        ]

    return ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(
                    content=content,
                    response_metadata={
                        "stop_reason": response.completion_message.stop_reason,
                    },
                    # TODO(md): add tool_calls
                    # TODO(mf): add usage_metadata
                    # TODO(mf): add logprobs
                )
            )
        ]
    )
