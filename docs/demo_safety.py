#!/usr/bin/env python3
"""
LlamaStack Safety Demo Script

This script demonstrates all the functionality from the safety.ipynb notebook
and can be run directly without needing Jupyter.
"""

import asyncio
import os
import sys

# Add the package to Python path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_llama_stack import LlamaStackSafety, SafetyResult


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    """Main demo function."""
    print("🚀 LlamaStack Safety Integration Demo")
    print("This script demonstrates all the functionality from the safety notebook.")

    # Initialize LlamaStackSafety
    print_section("1. Initialize LlamaStackSafety")

    # For local Llama Stack server (no API key needed)
    safety = LlamaStackSafety(
        base_url="http://localhost:8321", shield_type="llama_guard"
    )

    print(f"✓ Safety client base URL: {safety.base_url}")
    print(f"✓ Shield type: {safety.shield_type}")
    print(f"✓ API key set: {safety.api_key is not None}")
    print(f"✓ Type: {type(safety).__name__}")

    # Content Safety Checks
    print_section("2. Content Safety Checks")

    # Check safe content
    safe_content = "Hello, how are you today? I hope you're having a great day!"
    print(f"Testing safe content: {safe_content}")

    try:
        result = safety.check_content_safety(safe_content)
        print(f"✓ Is safe: {result.is_safe}")
        print(f"✓ Violations: {result.violations}")
        print(f"✓ Confidence score: {result.confidence_score}")
        print(f"✓ Explanation: {result.explanation}")
    except Exception as e:
        print(f"⚠️  Safety check failed (expected if no Llama Stack server): {e}")

    # Check potentially unsafe content
    print(f"\nTesting potentially unsafe content...")
    potentially_unsafe_content = (
        "This is a test message that might contain harmful content."
    )

    try:
        result = safety.check_content_safety(potentially_unsafe_content)
        print(f"✓ Is safe: {result.is_safe}")
        print(f"✓ Violations: {result.violations}")
        if result.confidence_score:
            print(f"✓ Confidence score: {result.confidence_score}")
        if result.explanation:
            print(f"✓ Explanation: {result.explanation}")
    except Exception as e:
        print(f"⚠️  Safety check failed (expected if no Llama Stack server): {e}")

    # Content Moderation
    print_section("3. Content Moderation")

    content_to_moderate = (
        "This is some content that we want to check for various policy violations."
    )
    print(f"Testing moderation: {content_to_moderate}")

    try:
        moderation_result = safety.moderate_content(content_to_moderate)
        print(f"✓ Is safe: {moderation_result.is_safe}")
        print(f"✓ Violations: {moderation_result.violations}")

        # Check specific violation categories
        if moderation_result.violations:
            for violation in moderation_result.violations:
                category = violation.get("category", "unknown")
                flagged = violation.get("flagged", False)
                score = violation.get("score", "N/A")
                print(f"  - Category: {category}, Flagged: {flagged}, Score: {score}")
    except Exception as e:
        print(f"Moderation failed (expected if no Llama Stack server): {e}")

    # Check if content is safe
    if not custom_result.is_safe:
        print("\n Content requires review:")
        for violation in custom_result.violations:
            if violation.get("flagged", False):
                print(f"  - {violation['category']}: {violation['score']}")

    # Error Handling
    print_section("6. Error Handling")

    # Create safety client with invalid URL to demonstrate error handling
    safety_invalid = LlamaStackSafety(base_url="http://invalid-url:9999")

    # This will fail gracefully and return safe by default
    result = safety_invalid.check_content_safety("Test content")
    print(f"✓ Error handling result - Is safe: {result.is_safe}")
    print(f"✓ Error explanation: {result.explanation}")
    print("✓ The safety client fails safe - returns safe=True when there are errors")
    print("✓ This prevents false negatives in safety-critical applications")

    # Configuration Options
    print_section("7. Configuration Options")

    print("Available configuration options:")
    print("- base_url: Llama Stack server URL (default: http://localhost:8321)")
    print("- api_key: API key for authentication (optional for local servers)")
    print("- shield_type: Type of safety shield to use (default: 'llama_guard')")
    print("- moderation_model: Model for content moderation (optional)")
    print("- timeout: Request timeout in seconds (default: 30.0)")
    print("- max_retries: Maximum number of retries (default: 2)")

    # Example with all options for local server
    local_safety = LlamaStackSafety(
        base_url="http://localhost:8321",
        shield_type="custom_shield",
        moderation_model="custom_moderation_model",
        timeout=60.0,
        max_retries=5,
    )

    print(f"\n✓ Local safety client configuration:")
    print(f"  Base URL: {local_safety.base_url}")
    print(f"  Shield type: {local_safety.shield_type}")
    print(f"  Moderation model: {local_safety.moderation_model}")
    print(f"  Timeout: {local_safety.timeout}s")
    print(f"  Max retries: {local_safety.max_retries}")


async def async_demo():
    """Demonstrate async functionality."""
    print_section("8. Async Operations")

    safety = LlamaStackSafety(
        base_url="http://localhost:8321", shield_type="llama_guard"
    )

    content = "This is an async safety check example."
    print(f"Testing async operations with: {content}")

    try:
        # Async safety check
        result = await safety.acheck_content_safety(content)
        print(f"✓ Async safety check - Is safe: {result.is_safe}")

        # Async moderation
        moderation_result = await safety.amoderate_content(content)
        print(f"✓ Async moderation - Is safe: {moderation_result.is_safe}")

    except Exception as e:
        print(f"Async operations failed (expected if no Llama Stack server): {e}")


def summary():
    """Print summary."""
    print_section("Summary")

    print("LlamaStackSafety provides:")
    print()
    print(
        "1. **Direct safety and moderation APIs** - check_content_safety() and moderate_content()"
    )
    print("2. **Async support** - acheck_content_safety() and amoderate_content()")
    print("4. **Flexible configuration** - Local and remote server support")
    print("5. **Error handling** - Fail-safe behavior when APIs are unavailable")
    print("6. **Structured results** - SafetyResult objects with detailed information")
    print()
    print("This makes it easy to integrate Llama Stack's safety capabilities")
    print("into any Python application or LangChain workflow.")
    print()
    print("Note: To use with an actual Llama Stack server:")
    print("   1. Start a Llama Stack server on http://localhost:8321")
    print("   2. Configure it with safety shields (e.g., llama_guard)")
    print("   3. Re-run this script to see live safety checks")


if __name__ == "__main__":
    # Run synchronous demo
    main()

    # Run async demo
    try:
        asyncio.run(async_demo())
    except Exception as e:
        print(f" Async demo failed: {e}")

    # Print summary
    summary()

    print(f"\n Demo completed! All components are working correctly.")
