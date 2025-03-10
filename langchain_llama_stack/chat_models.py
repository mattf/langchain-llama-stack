"""Llama Stack chat models."""

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import httpx
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_json_schema
from llama_stack_client import NOT_GIVEN, Client
from llama_stack_client.types import (
    ChatCompletionResponse,
)
from llama_stack_client.types.inference_chat_completion_params import (
    Tool as LlamaStackTool,
)
from llama_stack_client.types.inference_completion_params import (
    Logprobs as LlamaStackLogprobs,
)
from llama_stack_client.types.shared_params import (
    SamplingParams,
)
from llama_stack_client.types.shared_params.sampling_params import (
    StrategyGreedySamplingStrategy,
    StrategyTopPSamplingStrategy,
)
from llama_stack_client.types.shared_params.tool_param_definition import (
    ToolParamDefinition as LlamaStackToolParamDefinition,
)
from pydantic import Field, SecretStr, model_validator

from langchain_llama_stack._utils import convert_message, convert_response

logger = logging.getLogger(__name__)


class ChatLlamaStack(BaseChatModel):
    """
    Llama Stack chat model integration.

    Setup:
        Install ``langchain-llama-stack`` and set optional environment variable ``LLAMA_STACK_API_KEY`` or ``LLAMA_STACK_BASE_URL``.

        .. code-block:: bash

            pip install -U langchain-llama-stack
            export LLAMA_STACK_API_KEY="your-api-key"
            export LLAMA_STACK_BASE_URL="http://my-llama-stack-disto:8321

    Key init args — completion params:
        model: str
            Name of model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    Key init args — client params:
        base_url: Optional[str]
            If not passed in will be read from env var LLAMA_STACK_BASE_URL.
        api_key: Optional[str]
            If not passed in will be read from env var LLAMA_STACK_API_KEY.
        timeout: Optional[float]
            Timeout for requests.
        max_retries: int
            Max number of retries.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_llama_stack import ChatLlamaStack

            llm = ChatLlamaStack(
                base_url="...",
                api_key="...",
                model="...",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(content='"J\'adore programmer."', additional_kwargs={}, response_metadata={'stop_reason': 'end_of_turn'}, id='run-561341ff-ac7f-41fa-bc61-13dc1c9a2ec3-0')

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

        .. code-block:: python

            # TODO: Example output.

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            # TODO: Example output.

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in (await llm.astream(messages))

            # batch:
            # await llm.abatch([messages])

        .. code-block:: python

            AIMessage(content='"J\'adore programmer."', additional_kwargs={}, response_metadata={'stop_reason': 'end_of_turn'}, id='run-fe11469c-bc41-4086-85fb-80c37a9d7efe-0')

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state, e.g. Boston, MA")

            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(..., description="The city and state, e.g. Boston, MA")

            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
            ai_msg.tool_calls

        .. code-block:: python

            [{'name': 'GetWeather',
              'args': {'location': 'Los Angeles, CA'},
              'id': 'chatcmpl-tool-25d24ab4522845ed8b0fe9335c5911d3',
              'type': 'tool_call'}]

        See ``ChatLlamaStack.bind_tools()`` method for more.

    Structured output:  TODO(mf): add support for structured output
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field

            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")

            structured_llm = llm.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats")

        .. code-block:: python

            # TODO: Example output.

        See ``ChatLlamaStack.with_structured_output()`` for more.

    JSON mode:  TODO(mf): add support for JSON mode
        .. code-block:: python

            json_llm = llm.bind(response_format={"type": "json_object"})
            ai_msg = json_llm.invoke("Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99]")
            ai_msg.content

        .. code-block:: python

            # TODO: Example output.

    Image input:  TODO(mf): add support for image inputs
        .. code-block:: python

            import base64
            import httpx
            from langchain_core.messages import HumanMessage

            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "describe the weather in this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            )
            ai_msg = llm.invoke([message])
            ai_msg.content

        .. code-block:: python

            # TODO: Example output.

    Token usage:  TODO(mf): add support for token usage metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {'input_tokens': 28, 'output_tokens': 5, 'total_tokens': 33}

    Logprobs:
        .. code-block:: python

            logprobs_llm = llm.bind(logprobs=True)  # or logprobs=3 for top 3
            ai_msg = logprobs_llm.invoke(messages)
            ai_msg.response_metadata["logprobs"]

        .. code-block:: python

            [{'"': -2.4698164463043213},
             {'J': -0.31870245933532715},
             {"'": -0.15102243423461914},
             {'ad': -0.004451931454241276},
             {'ore': -0.0009182137437164783},
             {' programmer': -1.1367093324661255},
             {'."': -0.14114761352539062}]

    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {'stop_reason': 'end_of_turn'}

    """  # noqa: E501

    model_name: str = Field(alias="model")
    """The name of the model"""
    base_url: str | httpx.URL | None = None
    "Loaded from LLAMA_STACK_BASE_URL environment variable if not provided."
    api_key: SecretStr | None = None
    "Loaded from LLAMA_STACK_API_KEY environment variable if not provided."
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    max_retries: Optional[int] = None

    disable_streaming: bool = True  # TODO(mf): add streaming support

    client: Client | None = Field(default=None, exclude=True)
    """Internal Llama Stack client"""

    @model_validator(mode="after")
    def _initialize_client(self) -> "ChatLlamaStack":
        """Initialize the Llama Stack client with configuration parameters."""
        params: dict[str, Any] = {
            "base_url": self.base_url,
            "api_key": self.api_key.get_secret_value() if self.api_key else None,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        params = {k: v for k, v in params.items() if v is not None}
        self.client = Client(**params)
        return self

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-llama-stack"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }

    def _get_sampling_params(
        self, **kwargs: Any
    ) -> tuple[Optional[SamplingParams], Any]:
        """
        Get the sampling parameters.

        TODO(mf): allow kwargs to override the model's settings
        """
        sampling_params: SamplingParams | None = None
        if self.max_tokens is not None or self.temperature is not None:
            sampling_params = SamplingParams(
                strategy=StrategyGreedySamplingStrategy(type="greedy"),
            )
            if self.max_tokens is not None:
                sampling_params["max_tokens"] = self.max_tokens
            # Llama Stack Inference API does now allow temperature=0,
            # we convert that to Greedy Sampling by not changing the
            # sampling strategy.
            if self.temperature is not None and self.temperature > 0:
                sampling_params["strategy"] = StrategyTopPSamplingStrategy(
                    type="top_p",
                    temperature=self.temperature,
                )
        return sampling_params, kwargs

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        logprobs: Optional[bool | int] = None,
        tools: Optional[Iterable[LlamaStackTool]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response to the input messages.

        Args:
            messages: the prompt composed of a list of messages.
            stop: this parameter is not supported
            run_manager: A run manager with callbacks for the LLM.
        """
        assert self.client is not None, "client not initialized"  # satisfy mypy

        if stop:
            logging.warning(
                "ignoring stop words, not supported by Llama Stack Inference API"
            )

        sampling_params, kwargs = self._get_sampling_params(**kwargs)

        if kwargs:
            logging.warning(f"ignoring extra kwargs: {kwargs}")

        response: ChatCompletionResponse = self.client.inference.chat_completion(
            model_id=self.model_name,
            messages=[convert_message(message) for message in messages],
            sampling_params=sampling_params if sampling_params else NOT_GIVEN,
            logprobs=LlamaStackLogprobs(top_k=logprobs) if logprobs else NOT_GIVEN,
            tools=tools if tools else NOT_GIVEN,
        )

        return convert_response(response)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """
        Bind tools to the chat model.

        Args:
            tools: List of tools to bind to the chat model.

        Returns:
            A new chat model with the tools bound.

        TODO(mf): tool_choice parameter
        """
        #
        # we convert the tools to JSON schema, convert the JSON schema to
        # Llama Stack tool objects, and then bind the Llama Stack tools to
        # the chat model.
        #

        ls_tools = []
        for tool in tools:
            json_schema = convert_to_json_schema(tool)
            assert "title" in json_schema, "missing title (tool's name): {json_schema}"
            if "description" not in json_schema:
                logging.warning(f"missing description for tool: {json_schema['title']}")
            parameters: dict[str, LlamaStackToolParamDefinition] = {}
            if "properties" in json_schema:
                #
                # {
                #   "type": "object",
                #   "properties": {
                #     "location": {
                #       "type": "string",
                #       "description": "The city and state, e.g. Somerville, MA"
                #     },
                #     "rank": {
                #       "type": "integer",
                #       "description": "Rank of the location",
                #       "exclusiveMaximum": 100,
                #       "exclusiveMinimum": 0
                #     }
                #     "unit": {
                #       "type": "string",
                #       "enum": ["celsius", "fahrenheit"]
                #     }
                #   },
                #   "required": ["location"]
                # }
                #
                # becomes
                #
                # [
                #   ToolParamDefinition(
                #     name="location",
                #     type="string",
                #     description="The city and state, e.g. Somerville, MA",
                #     required=True
                #   ),
                #   ToolParamDefinition(
                #     name="rank",
                #     type="integer",
                #     description="Rank of the location",
                #     required=False
                #   ),
                #   ToolParamDefinition(
                #     name="unit",
                #     type="string",
                #     description="The unit of temperature",
                #     required=False
                #   )
                # ]
                #
                # TODO(mf): JSON schema can have limits, e.g. int lt/gt or enum for
                #           value, Llama Stack Inference API does not support these
                # TODO(mf): Llama Stack Inference API supports a default value, but
                #           the input JSON schema does not
                #
                required = json_schema.get("required", [])
                parameters = {}
                for name, param in json_schema.get("properties", {}).items():
                    assert "type" in param, f"missing type for parameter {name}"
                    if "description" not in param:
                        logging.warning(f"missing description for parameter: {name}")
                    parameters[name] = LlamaStackToolParamDefinition(
                        param_type=param["type"],
                        default=param.get("default", None),
                        description=param.get("description", ""),
                        required=name in required,
                    )
            ls_tools.append(
                LlamaStackTool(
                    tool_name=json_schema["title"],
                    description=json_schema.get("description", ""),
                    parameters=parameters,
                )
            )

        if kwargs:
            logging.warning(f"ignoring extra kwargs: {kwargs}")

        return self.bind(tools=ls_tools)

    # TODO: Implement native _stream.
    # def _stream(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[CallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> Iterator[ChatGenerationChunk]:

    # TODO: Implement native async streaming.
    # async def _astream(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> AsyncIterator[ChatGenerationChunk]:

    # TODO: Implement native async generation.
    # async def _agenerate(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> ChatResult:
