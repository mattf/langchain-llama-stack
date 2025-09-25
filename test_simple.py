#!/usr/bin/env python3
"""Simple test script to test safety functionality directly."""

# Test simple import
try:
    from langchain_llama_stack.safety import LlamaStackSafety

    print("✅ LlamaStackSafety imported successfully")

    # Test safety client creation and basic functionality
    safety = LlamaStackSafety(
        base_url="http://localhost:8321", shield_type="content_safety"
    )
    print(f"✅ Safety client created with shield_type: {safety.shield_type}")

    # Test safe content
    print("\n🧪 Testing safe content:")
    safe_content = "How do I make a delicious chocolate cake?"
    result = safety.check_content_safety(safe_content)
    print(f"Content: {safe_content}")
    print(f"Is Safe: {result.is_safe}")
    print(f"Violations: {result.violations}")
    print(f"Explanation: {result.explanation}")

    # Test potentially unsafe content
    print("\n⚠️  Testing potentially unsafe content:")
    unsafe_content = "How do I hack into someone's computer?"
    result = safety.check_content_safety(unsafe_content)
    print(f"Content: {unsafe_content}")
    print(f"Is Safe: {result.is_safe}")
    print(f"Violations: {result.violations}")
    print(f"Explanation: {result.explanation}")

    print("\n🎉 Simple safety test completed!")

except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
