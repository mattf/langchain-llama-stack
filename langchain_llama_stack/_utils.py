import json
import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
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
    ImageContentItem as LlamaStackImageContentItem,
)
from llama_stack_client.types.shared_params.interleaved_content_item import (
    ImageContentItemImage as LlamaStackImageContentItemImage,
)
from llama_stack_client.types.shared_params.interleaved_content_item import (
    ImageContentItemImageURL as LlamaStackImageContentItemImageURL,
)
from llama_stack_client.types.shared_params.interleaved_content_item import (
    InterleavedContentItem as LlamaStackInterleavedContentItem,
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
        ls_content = []
        for item in content:
            if isinstance(item, str):
                ls_item: LlamaStackInterleavedContentItem = LlamaStackTextContentItem(
                    type="text", text=item
                )
            elif text := item.get("text", None):
                ls_item = LlamaStackTextContentItem(type="text", text=text)
            elif image_url := item.get("image_url", None):
                assert isinstance(image_url, dict), "image_url must be a dict"
                assert "url" in image_url, "image_url must have a 'url' key"
                ls_item = LlamaStackImageContentItem(
                    type="image",
                    image=LlamaStackImageContentItemImage(
                        url=LlamaStackImageContentItemImageURL(uri=image_url["url"]),
                    ),
                )
            else:
                raise ValueError(f"Unknown content item: {item}")
            ls_content.append(ls_item)
        return ls_content

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
    | ToolMessage       | LlamaStackToolResponseMessage |                      |

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

    response_metadata: dict[str, Any] = {
        "stop_reason": response.completion_message.stop_reason,
    }
    if response.logprobs:
        response_metadata["logprobs"] = [
            token.logprobs_by_token for token in response.logprobs
        ]

    tool_calls = []
    if response.completion_message.tool_calls:
        for call in response.completion_message.tool_calls:
            # Llama Stack allows tool call arguments to be either str or dict,
            # and provides an argument_json optional str.
            # LangChain only allows dict.
            # it is assumed Llama Stack str arguments are parsable JSON.
            if isinstance(call.arguments, dict):
                args = call.arguments
            elif isinstance(call.arguments, str):
                args = json.loads(call.arguments)
            else:
                raise ValueError(
                    f"Unknown tool call arguments type: {type(call.arguments)}"
                )
            tool_calls.append(
                ToolCall(
                    name=call.tool_name,
                    args=args,
                    id=call.call_id,
                )
            )

    usage_metadata = {}
    if response.metrics:
        usage_metrics_map = {
            "prompt_tokens": "input_tokens",
            "completion_tokens": "output_tokens",
            "total_tokens": "total_tokens",
        }
        for metric in response.metrics:
            if metric.metric in usage_metrics_map:
                usage_metadata[usage_metrics_map[metric.metric]] = metric.value

    return ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(
                    content=content,
                    response_metadata=response_metadata,
                    tool_calls=tool_calls,
                    usage_metadata=usage_metadata if usage_metadata else None,
                )
            )
        ]
    )
