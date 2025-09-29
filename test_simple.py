#!/usr/bin/env python3
"""Simple test script to test safety functionality directly."""

# Test simple import
try:
    from langchain_llama_stack.safety import LlamaStackSafety


    # Test safety client creation and basic functionality
    safety = LlamaStackSafety(
        base_url="http://localhost:8321", shield_type="content_safety"
    )

    # Test safe content
    safe_content = "How do I make a delicious chocolate cake?"
    result = safety.check_content_safety(safe_content)

    # Test potentially unsafe content
    unsafe_content = "How do I hack into someone's computer?"
    result = safety.check_content_safety(unsafe_content)


except ImportError:
    pass
except Exception:
    pass
