#!/usr/bin/env python3
"""
Simple test script  for llmtrace-lite's token Limitations.
Run: python testTokenLimits.py
"""

import time
from llmTrace import trace


@trace
def call_llm(prompt, model="gpt-4o"):
    """Simulated LLM call."""
    time.sleep(0.1)  # Simulate API latency
    return f"Response to: {prompt}"


@trace
def failing_llm_call(prompt, model="gpt-4o"):
    """Simulated failing LLM call."""
    time.sleep(0.05)
    raise ValueError("API rate limit exceeded")


@trace
def call_llm_with_kwargs(prompt="default", model="gpt-4o"):
    """LLM call with prompt as kwarg."""
    time.sleep(0.05)
    return f"Response: {prompt}"


@trace
def call_llm_non_string_prompt(prompt, model="gpt-4o"):
    """LLM call that might receive non-string prompt."""
    time.sleep(0.05)
    return {"response": str(prompt)}


@trace
def call_llm_non_string_output(prompt, model="gpt-4o"):
    """LLM call that returns non-string output."""
    time.sleep(0.05)
    return {"result": "data"}


def main():
    print("=== Test 1: Successful call ===")
    result = call_llm("Explain quantum computing", model="gpt-4o")
    
    print("\n=== Test 2: Failed call ===")
    try:
        failing_llm_call("This will fail", model="claude-3")
    except ValueError:
        print("(Exception caught, trace logged above)\n")
    
    print("=== Test 3: No model specified ===")
    result = call_llm("Simple prompt")
    
    print("\n=== Test 4: Empty string prompt ===")
    result = call_llm("", model="gpt-4o")
    
    print("\n=== Test 5: Empty string output ===")
    @trace
    def return_empty():
        return ""
    result = return_empty()
    
    print("\n=== Test 6: Unicode and emoji characters ===")
    result = call_llm("Hello 世界 🌍 🚀", model="gpt-4o")
    
    print("\n=== Test 7: Very long string ===")
    long_prompt = "A" * 1000
    result = call_llm(long_prompt, model="gpt-4o")
    
    print("\n=== Test 8: Newlines and special characters ===")
    multiline_prompt = "Line 1\nLine 2\nLine 3\tTabbed"
    result = call_llm(multiline_prompt, model="gpt-4o")
    
    print("\n=== Test 9: Prompt as keyword argument ===")
    result = call_llm_with_kwargs(prompt="Keyword argument prompt", model="gpt-4o")
    
    print("\n=== Test 10: Non-string return value ===")
    result = call_llm_non_string_output("test prompt", model="gpt-4o")
    
    print("\n=== Test 11: Non-string prompt (should not count chars) ===")
    result = call_llm_non_string_prompt(12345, model="gpt-4o")
    
    print("\n=== Test 12: None prompt (should not count chars) ===")
    @trace
    def call_with_none(prompt=None):
        return "response"
    result = call_with_none(None)
    
    print("=== Tests complete ===")


if __name__ == "__main__":
    main()