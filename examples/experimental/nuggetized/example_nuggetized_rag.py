#!/usr/bin/env python3
"""
Example usage of nuggetized RAG evaluation with TruLens.
Shows how to use nuggetized feedback for more granular evaluation.
"""

import os
import sys

# Add TruLens to path
sys.path.insert(0, "/Users/dcampos/trulens/src/core")
sys.path.insert(0, "/Users/dcampos/trulens/src/feedback")
sys.path.insert(0, "/Users/dcampos/trulens/src/providers/openai")

from trulens.core.feedback.nuggetized_feedback import (
    compare_traditional_vs_nuggetized,
)
from trulens.providers.openai import OpenAI as OpenAIProvider


def evaluate_rag_response():
    """Example of evaluating a RAG response with nuggetization."""

    # Sample RAG interaction
    question = "What are the key features of Python programming language?"

    context = """
    Python is a high-level, interpreted programming language known for its
    simplicity and readability. It was created by Guido van Rossum and first
    released in 1991. Python supports multiple programming paradigms including
    procedural, object-oriented, and functional programming. It has a
    comprehensive standard library and uses dynamic typing with automatic
    memory management through garbage collection. Python is widely used in
    web development, data science, artificial intelligence, and automation.
    """

    # Model's response
    answer = """
    Python's key features include its simple and readable syntax that emphasizes
    code clarity. It supports multiple programming paradigms like object-oriented
    and functional programming. Python has dynamic typing, automatic memory
    management, and a rich standard library. It's interpreted rather than
    compiled, making development faster. Python is extensively used in data
    science, AI, web development, and automation due to its versatility and
    large ecosystem of third-party packages.
    """

    # Initialize provider with OpenAI (gpt-4o-mini or any other model)
    # The provider now has built-in nugget extraction and classification methods
    provider = OpenAIProvider(model_engine="gpt-5-nano")

    print("=" * 70)
    print("RAG RESPONSE EVALUATION")
    print("=" * 70)
    print(f"\nQuestion: {question}")
    print(f"\nContext (first 150 chars): {context[:150]}...")
    print(f"\nAnswer (first 150 chars): {answer[:150]}...")

    # Run comparison - now uses provider's built-in nugget methods
    results = compare_traditional_vs_nuggetized(
        provider=provider, question=question, answer=answer, context=context
    )

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    # Traditional scores
    print("\n📊 TRADITIONAL EVALUATION:")
    print(f"   Groundedness: {results['traditional']['groundedness']:.1%}")
    print(f"   Relevance:    {results['traditional']['relevance']:.1%}")

    # Nuggetized scores
    print("\n🔬 NUGGETIZED EVALUATION:")
    print(f"   Groundedness: {results['nuggetized']['groundedness']:.1%}")
    print(f"   Relevance:    {results['nuggetized']['relevance']:.1%}")

    # Show nugget details
    nugget_g_meta = results["nuggetized"]["groundedness_metadata"]
    print(f"\n   Evaluated {nugget_g_meta['total_nuggets']} nuggets:")

    for i, eval_result in enumerate(nugget_g_meta["nugget_evaluations"][:3], 1):
        nugget = eval_result["nugget"]
        importance = eval_result["importance"]
        score = eval_result["score"]
        nugget_preview = nugget[:50] if len(nugget) > 50 else nugget
        print(f"   {i}. '{nugget_preview}...'")
        print(f"      Importance: {importance}, Score: {score:.1%}")

    if nugget_g_meta["total_nuggets"] > 3:
        print(f"   ... and {nugget_g_meta['total_nuggets'] - 3} more nuggets")

    # Comparison
    print("\n📈 COMPARISON:")
    print(
        f"   Groundedness difference: {results['comparison']['groundedness_diff']:+.1%}"
    )
    print(
        f"   Relevance difference:    {results['comparison']['relevance_diff']:+.1%}"
    )

    # Interpretation
    print("\n💡 INTERPRETATION:")
    if abs(results["comparison"]["groundedness_diff"]) > 0.1:
        direction = (
            "higher"
            if results["comparison"]["groundedness_diff"] > 0
            else "lower"
        )
        print(f"   Nuggetized evaluation shows {direction} groundedness,")
        print("   suggesting some facts have different support levels.")
    else:
        print("   Both methods show similar groundedness scores.")

    if abs(results["comparison"]["relevance_diff"]) > 0.1:
        direction = (
            "higher" if results["comparison"]["relevance_diff"] > 0 else "lower"
        )
        print(f"   Nuggetized evaluation shows {direction} relevance,")
        print("   indicating varied relevance across answer components.")
    else:
        print("   Both methods show similar relevance scores.")

    return results


def main():
    """Run the example."""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        print(
            "\nYou can get an API key from: https://platform.openai.com/api-keys"
        )
        return

    try:
        evaluate_rag_response()
        print("\n✅ Evaluation completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
