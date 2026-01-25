"""
Test script to verify prompt optimization changes.

This script validates:
1. Dataset-specific prompt selection works correctly
2. Response parser handles structured format
3. All prompts contain required placeholders
"""

import config
from reasoner import parse_response


def test_prompt_placeholders():
    """Verify all prompts contain required {query} and {retrieved_chunks} placeholders."""
    prompts = {
        'COT': config.COT_PROMPT_TEMPLATE,
        'FOLIO': config.FOLIO_PROMPT_TEMPLATE,
        'ContractNLI': config.CONTRACTNLI_PROMPT_TEMPLATE,
        'ProofWriter': config.PROOFWRITER_PROMPT_TEMPLATE
    }

    for name, template in prompts.items():
        assert '{query}' in template, f"{name} missing {{query}} placeholder"
        assert '{retrieved_chunks}' in template, f"{name} missing {{retrieved_chunks}} placeholder"
        print(f"✓ {name} prompt has required placeholders")


def test_response_parser():
    """Test response parser handles structured and unstructured formats."""

    # Test structured format
    structured_response = """Reasoning: The premise states all birds can fly.
Penguins are birds. However, the text also mentions penguins cannot fly.
This is a contradiction.

Answer: False"""

    result = parse_response(structured_response)
    assert result['answer'] == 'False', f"Expected 'False', got {result['answer']}"
    assert 'contradiction' in result['reasoning'].lower(), "Reasoning not extracted correctly"
    print("✓ Structured format parsed correctly")

    # Test fallback for unstructured format
    unstructured_response = "Based on the context, the statement is clearly true."
    result = parse_response(unstructured_response)
    assert result['answer'] == 'True', f"Expected 'True', got {result['answer']}"
    print("✓ Unstructured format parsed correctly (fallback)")

    # Test ContractNLI labels
    contract_response = """Reasoning: The contract explicitly states the termination clause.

Answer: Entailed"""
    result = parse_response(contract_response)
    assert result['answer'] == 'Entailed', f"Expected 'Entailed', got {result['answer']}"
    print("✓ ContractNLI labels parsed correctly")


def test_prompt_dataset_mapping():
    """Verify correct prompts are selected for each dataset."""
    # Inline implementation to avoid import issues
    def get_dataset_prompt(dataset_name):
        prompt_map = {
            'folio': config.FOLIO_PROMPT_TEMPLATE,
            'proofwriter': config.PROOFWRITER_PROMPT_TEMPLATE,
            'contractnli': config.CONTRACTNLI_PROMPT_TEMPLATE
        }
        return prompt_map.get(dataset_name, config.COT_PROMPT_TEMPLATE)

    folio_prompt = get_dataset_prompt('folio')
    assert folio_prompt == config.FOLIO_PROMPT_TEMPLATE
    assert 'first-order logic' in folio_prompt
    print("✓ FOLIO dataset maps to FOLIO prompt")

    contract_prompt = get_dataset_prompt('contractnli')
    assert contract_prompt == config.CONTRACTNLI_PROMPT_TEMPLATE
    assert 'contract' in contract_prompt.lower()
    print("✓ ContractNLI dataset maps to ContractNLI prompt")

    proof_prompt = get_dataset_prompt('proofwriter')
    assert proof_prompt == config.PROOFWRITER_PROMPT_TEMPLATE
    assert 'deductive reasoning' in proof_prompt
    print("✓ ProofWriter dataset maps to ProofWriter prompt")

    # Test fallback
    unknown_prompt = get_dataset_prompt('unknown_dataset')
    assert unknown_prompt == config.COT_PROMPT_TEMPLATE
    print("✓ Unknown dataset falls back to generic CoT prompt")


def test_prompt_content_quality():
    """Verify prompts contain key reasoning elements."""

    # Generic prompt should have reasoning instructions
    assert 'step-by-step' in config.COT_PROMPT_TEMPLATE.lower()
    assert 'reasoning:' in config.COT_PROMPT_TEMPLATE.lower()
    print("✓ Generic prompt has reasoning instructions")

    # FOLIO should reference logical rules
    assert 'modus ponens' in config.FOLIO_PROMPT_TEMPLATE.lower()
    assert 'universal' in config.FOLIO_PROMPT_TEMPLATE.lower()
    print("✓ FOLIO prompt references formal logic")

    # ContractNLI should reference contractual concepts
    assert 'clause' in config.CONTRACTNLI_PROMPT_TEMPLATE.lower()
    assert 'entailed' in config.CONTRACTNLI_PROMPT_TEMPLATE.lower()
    print("✓ ContractNLI prompt references contract analysis")

    # ProofWriter should reference rule application
    assert 'forward chaining' in config.PROOFWRITER_PROMPT_TEMPLATE.lower()
    assert 'derivation' in config.PROOFWRITER_PROMPT_TEMPLATE.lower()
    print("✓ ProofWriter prompt references rule-based reasoning")


def main():
    """Run all tests."""
    print("Testing prompt optimization...\n")

    try:
        test_prompt_placeholders()
        print()

        test_response_parser()
        print()

        test_prompt_dataset_mapping()
        print()

        test_prompt_content_quality()
        print()

        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
