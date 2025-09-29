"""LlamaStack Safety integration for LangChain."""

import logging
import os
from typing import Any, Optional

from pydantic import BaseModel

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from llama_stack_client import (  # type: ignore
        AsyncLlamaStackClient,
        LlamaStackClient,
    )
except ImportError:
    AsyncLlamaStackClient = None  # type: ignore
    LlamaStackClient = None  # type: ignore


class SafetyResult(BaseModel):
    """Result from safety check."""

    is_safe: bool
    violations: list[dict[str, Any]] = []
    confidence_score: Optional[float] = None
    explanation: str = "No explanation provided"
    metadata: Optional[dict[str, Any]] = None


class LlamaStackSafety:
    """
    Llama Stack safety and moderation integration.

    Key init args — safety params:
        model: str
            Name of safety model to use for safety and moderation\
             (default: "llama-guard").

    Key init args — client params:
        base_url: str
            Llama Stack server URL (default: "http://localhost:8321").
        timeout: Optional[float]
            Timeout for requests (default: 30.0).
        max_retries: int
            Max number of retries (default: 2).

    Instantiate:
        .. code-block:: python

            from langchain_llama_stack import LlamaStackSafety

            # For local Llama Stack (no API key needed)
            safety = LlamaStackSafety(
                base_url="http://localhost:8321", shield_type="llama_guard"
            )

            # For remote Llama Stack (with API key)
            safety = LlamaStackSafety(
                base_url="http://remote-llama-stack:8321",
                shield_type="llama_guard",
            )

    Check content safety:
        .. code-block:: python

            result = safety.check_content_safety("This is some text to check")
            print(result.is_safe)
            print(result.violations)

        .. code-block:: python

            SafetyResult(is_safe=True, violations=[], confidence_score=0.95)

    Async:
        .. code-block:: python

            result = await safety.acheck_content_safety("Text to check")

        .. code-block:: python

            SafetyResult(is_safe=True, violations=[])
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "ollama/llama-guard3:8b",
        timeout: Optional[float] = 30.0,
        max_retries: int = 2,
    ):
        """Initialize LlamaStackSafety."""
        self.base_url = base_url or os.environ.get(
            "LLAMA_STACK_BASE_URL", "http://localhost:8321"
        )
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # Clients will be initialized lazily when needed
        self.client: Optional[LlamaStackClient] = None
        self.async_client: Optional[AsyncLlamaStackClient] = None

    def _get_client_kwargs(self) -> dict[str, Any]:
        """Get common client kwargs."""
        client_kwargs = {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        return client_kwargs

    def _initialize_client(self) -> None:
        """Initialize the Llama Stack client."""
        if self.client is None and LlamaStackClient is not None:
            self.client = LlamaStackClient(**self._get_client_kwargs())

    def _initialize_async_client(self) -> None:
        """Initialize the async Llama Stack client."""
        if self.async_client is None and AsyncLlamaStackClient is not None:
            self.async_client = AsyncLlamaStackClient(**self._get_client_kwargs())

    def list_shields(self) -> list[str]:
        """List all available shields from LlamaStack server."""
        try:
            if self.client is None:
                self._initialize_client()

            if self.client is None:
                return []

            # Get shields
            shields_response = self.client.shields.list()
            # Handle both list and object with .data attribute
            if hasattr(shields_response, "data"):
                shields_data = shields_response.data
            else:
                shields_data = shields_response

            # Return just the identifiers for easy checking
            shield_ids = [shield.identifier for shield in shields_data]
            return shield_ids

        except Exception as e:
            logger.error(f"Error listing shields: {e}")
            return []

    def _process_safety_result(self, results: list[Any]) -> SafetyResult:
        """Process safety results from client response."""
        if not results:
            return SafetyResult(
                is_safe=True,
                confidence_score=1.0,
                explanation="No results to process",
                violations=[],
            )

        result = results[0]
        is_flagged = getattr(result, "flagged", False)
        is_safe = not is_flagged
        violations = []

        # Extract user message if available
        explanation = getattr(result, "user_message", "Content processed")

        # Calculate confidence score
        confidence_score = 1.0 if is_safe else 0.1

        # Process violations if content is flagged
        if is_flagged and hasattr(result, "categories"):
            categories = result.categories
            category_scores = getattr(result, "category_scores", {})

            if isinstance(categories, dict):
                flagged_categories = [
                    cat for cat, flagged in categories.items() if flagged
                ]
            else:
                flagged_categories = [
                    attr
                    for attr in dir(categories)
                    if not attr.startswith("_") and getattr(categories, attr, False)
                ]

            for category in flagged_categories:
                score = None
                if isinstance(category_scores, dict):
                    score = category_scores.get(category)
                else:
                    score = getattr(category_scores, category, None)

                violations.append(
                    {"category": category, "score": score, "flagged": True}
                )

            # Update confidence score based on violations
            if violations and isinstance(category_scores, dict):
                max_score = max(
                    category_scores.get(v["category"], 0) for v in violations
                )
                confidence_score = 1.0 - max_score

        return SafetyResult(
            is_safe=is_safe,
            confidence_score=confidence_score,
            explanation=explanation,
            violations=violations,
        )

    def check_content_safety(
        self, content: str, content_type: str = "text", **kwargs: Any
    ) -> SafetyResult:
        """
        Check content safety using Llama Stack shields.

        Args:
            content: The content to check for safety
            content_type: Type of content (default: "text")
            **kwargs: Additional parameters for safety checking

        Returns:
            SafetyResult with safety assessment
        """
        # logger.info(f"Starting safety check for content:
        #  '{content[:50]}...'")
        # logger.info(f"Using shield_type: {self.shield_type}")
        # logger.info(f"Base URL: {self.base_url}")

        # Check if LlamaStackClient is available
        if LlamaStackClient is None:
            logger.error("LlamaStackClient is None - client not available")
            return SafetyResult(
                is_safe=True,
                violations=[],
                explanation=(
                    "LlamaStackClient not available - install llama-stack-client"
                ),
            )

        logger.info("LlamaStackClient is available")

        if self.client is None:
            logger.info("Client is None, initializing...")
            self._initialize_client()

        try:
            # Ensure client is not None after initialization
            if self.client is None:
                logger.error("Client is still None after initialization")
                return SafetyResult(
                    is_safe=True,
                    violations=[],
                    explanation="LlamaStack client not initialized",
                )

            logger.info("Client initialized successfully")
            logger.info(f"Making API call with model: {self.model}")

            # Use the moderations.create method
            response = self.client.moderations.create(
                input=content,
                model=self.model,
                **kwargs,
            )

            # Parse OpenAI-compatible moderation response
            is_safe = True
            violations = []
            confidence_score = None
            explanation = None
            metadata = {}

            # Extract top-level metadata
            if hasattr(response, "id"):
                metadata["id"] = response.id
            if hasattr(response, "model"):
                metadata["model"] = response.model

            # Process results array
            if hasattr(response, "results") and response.results:
                # Use the first result (typically there's only one)
                result = response.results[0]

                # Check if content is flagged
                is_flagged = getattr(result, "flagged", False)
                is_safe = not is_flagged

                # Extract user message if available
                if hasattr(result, "user_message"):
                    explanation = result.user_message

                # Extract metadata
                if hasattr(result, "metadata"):
                    result_metadata = result.metadata
                    if isinstance(result_metadata, dict):
                        # Ensure all values are strings for type safety
                        str_metadata = {k: str(v) for k, v in result_metadata.items()}
                        metadata.update(str_metadata)
                    else:
                        # Handle object-style metadata
                        for attr in dir(result_metadata):
                            if not attr.startswith("_"):
                                value = getattr(result_metadata, attr)
                                if not callable(value):
                                    metadata[attr] = str(value)

                # Process flagged categories
                if is_flagged and hasattr(result, "categories"):
                    categories = result.categories
                    category_scores = getattr(result, "category_scores", {})

                    if isinstance(categories, dict):
                        # Dictionary format: {"violence": True, "hate": False, ...}
                        flagged_categories = [
                            cat for cat, flagged in categories.items() if flagged
                        ]
                    else:
                        # Object format: result.categories.violence = True
                        flagged_categories = [
                            attr
                            for attr in dir(categories)
                            if not attr.startswith("_")
                            and getattr(categories, attr, False)
                        ]

                    # Create violations from flagged categories
                    for category in flagged_categories:
                        score = None
                        if isinstance(category_scores, dict):
                            score = category_scores.get(category)
                        else:
                            score = getattr(category_scores, category, None)

                        violations.append(
                            {"category": category, "score": score, "flagged": True}
                        )

                # Extract confidence score if available
                # (some implementations may have this)
                if hasattr(result, "confidence_score"):
                    confidence_score = result.confidence_score
            else:
                # Fallback: if no results, treat as safe
                logger.warning("No results found in moderation response")

            # logger.info(f"Final result - is_safe:
            # {is_safe},
            # violations: {violations}")

            return SafetyResult(
                is_safe=is_safe,
                violations=violations,
                confidence_score=confidence_score,
                explanation=explanation or "Safety check completed",
            )
        except Exception as e:
            logger.error(f"Exception occurred during safety check: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            # Return safe by default on error, but log the issue
            return SafetyResult(
                is_safe=True,
                violations=[],
                explanation=f"Safety check failed: {str(e)}",
            )

    async def acheck_content_safety(
        self, content: str, content_type: str = "text", **kwargs: Any
    ) -> SafetyResult:
        """
        Async check content safety using Llama Stack safety shields.

        Args:
            content: The content to check for safety
            content_type: Type of content (text, image, etc.)
            **kwargs: Additional parameters for safety check

        Returns:
            SafetyResult with safety assessment
        """
        if not self.async_client:
            self._initialize_async_client()

        try:
            # Ensure async client is not None
            if self.async_client is None:
                raise ValueError("LlamaStack async client not initialized")

            # Use the AsyncLlamaStackClient.moderations.create method
            response = await self.async_client.moderations.create(
                input=content,
                model=self.model,
                **kwargs,
            )

            # Parse OpenAI-compatible moderation response (same logic as sync version)
            is_safe = True
            violations = []
            confidence_score = None
            explanation = None
            metadata = {}

            # Extract top-level metadata
            if hasattr(response, "id"):
                metadata["id"] = response.id
            if hasattr(response, "model"):
                metadata["model"] = response.model

            # Process results array
            if hasattr(response, "results") and response.results:
                # Use the first result (typically there's only one)
                result = response.results[0]

                # Check if content is flagged
                is_flagged = getattr(result, "flagged", False)
                is_safe = not is_flagged

                # Extract user message if available
                if hasattr(result, "user_message"):
                    explanation = result.user_message

                # Extract metadata
                if hasattr(result, "metadata"):
                    result_metadata = result.metadata
                    if isinstance(result_metadata, dict):
                        # Ensure all values are strings for type safety
                        str_metadata = {k: str(v) for k, v in result_metadata.items()}
                        metadata.update(str_metadata)
                    else:
                        # Handle object-style metadata
                        for attr in dir(result_metadata):
                            if not attr.startswith("_"):
                                value = getattr(result_metadata, attr)
                                if not callable(value):
                                    metadata[attr] = str(value)

                # Process flagged categories
                if is_flagged and hasattr(result, "categories"):
                    categories = result.categories
                    category_scores = getattr(result, "category_scores", {})

                    if isinstance(categories, dict):
                        # Dictionary format: {"violence": True, "hate": False, ...}
                        flagged_categories = [
                            cat for cat, flagged in categories.items() if flagged
                        ]
                    else:
                        # Object format: result.categories.violence = True
                        flagged_categories = [
                            attr
                            for attr in dir(categories)
                            if not attr.startswith("_")
                            and getattr(categories, attr, False)
                        ]

                    # Create violations from flagged categories
                    for category in flagged_categories:
                        score = None
                        if isinstance(category_scores, dict):
                            score = category_scores.get(category)
                        else:
                            score = getattr(category_scores, category, None)

                        violations.append(
                            {"category": category, "score": score, "flagged": True}
                        )

                # Extract confidence score if available
                # (some implementations may have this)
                if hasattr(result, "confidence_score"):
                    confidence_score = result.confidence_score
            else:
                # Fallback: if no results, treat as safe
                logger.warning("No results found in async moderation response")

            return SafetyResult(
                is_safe=is_safe,
                violations=violations,
                confidence_score=confidence_score,
                explanation=explanation or "Async safety check completed",
            )
        except Exception as e:
            # Return safe by default on error, but log the issue
            return SafetyResult(
                is_safe=True,
                violations=[],
                explanation=f"Async safety check failed: {str(e)}",
            )
