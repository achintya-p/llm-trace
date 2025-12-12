#!/usr/bin/env python3
"""
Simple test script for llmtrace-lite.
Run: python test_example.py
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
    
    print("=== Tests complete ===")


if __name__ == "__main__":
    main()