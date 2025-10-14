"""Llama Stack chat models."""

import os
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr


class ChatLlamaStack(ChatOpenAI):
    """
    Llama Stack chat model integration.

    Setup:
        Install ``langchain-llama-stack`` and set optional environment variable ``LLAMA_STACK_API_KEY`` or ``LLAMA_STACK_BASE_URL``.

        .. code-block:: bash

            pip install -U langchain-llama-stack
            export LLAMA_STACK_API_KEY="your-api-key"
            export LLAMA_STACK_BASE_URL="http://my-llama-stack-disto:8321/v1/openai/v1

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

            AIMessageChunk(content='"J\'adore programmer."', additional_kwargs={}, response_metadata={'stop_reason': 'end_of_turn'}, id='run-3fd45daf-7488-458d-8e8f-947f7d066ef4')

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

    Structured output:
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

            Joke(setup='Why did the cat join a band?', punchline='Because it wanted to be the purr-cussionist', rating=8)

        See ``ChatLlamaStack.with_structured_output()`` for more.

    JSON mode:
        .. code-block:: python

            json_llm = llm.with_structured_output(method="json_mode")
            ai_msg = json_llm.invoke("Return only a JSON object with key 'random_ints' and a value of 10 random ints in [0-99].")
            ai_msg

        .. code-block:: python

            {'random_ints': [83, 19, 91, 46, 75, 33, 28, 41, 59, 12]}

    Image input:
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

            'The weather in this image is sunny and warm, with a clear blue sky and fluffy white clouds. The sun is shining down on the boardwalk and the grass, casting a warm glow over the entire scene. The air is filled with a gentle breeze that rustles the leaves of the trees and carries the sweet scent of blooming flowers. Overall, the atmosphere is peaceful and serene, inviting the viewer to step into the idyllic world depicted in the image.'

    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {'input_tokens': 32, 'output_tokens': 17, 'total_tokens': 49}

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

    openai_api_base: str | None = Field(
        alias="base_url",
        default_factory=lambda: os.environ.get(
            "LLAMA_STACK_BASE_URL", "http://localhost:8321/v1/openai/v1"
        ),
    )
    "Loaded from LLAMA_STACK_BASE_URL environment variable if not provided."
    openai_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=lambda: SecretStr(
            os.environ.get("LLAMA_STACK_API_KEY", "NO_API_KEY")
        ),
    )
    "Loaded from LLAMA_STACK_API_KEY environment variable if not provided."

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "llama-stack-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
        }

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return False
