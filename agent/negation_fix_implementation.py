"""
Negation Handling Fix Implementation for Logify

This module provides concrete implementations of Fix 1 (negation detection)
and Fix 2 (contradiction detection) from negation_fix_proposal.md.

Author: Alethea (AI Agent)
Date: 2026-01-30
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ==============================================================================
# FIX 1: NEGATION DETECTION IN QUERY TRANSLATION
# ==============================================================================

def detect_negation_in_hypothesis(hypothesis: str) -> bool:
    """
    Detect if hypothesis expresses a prohibition or negation.

    This function identifies negative constructions that should be
    reflected in the translated formula as negation operators.

    Args:
        hypothesis: Natural language hypothesis text

    Returns:
        True if hypothesis is negative/prohibition, False otherwise

    Examples:
        >>> detect_negation_in_hypothesis("Party shall not disclose information")
        True
        >>> detect_negation_in_hypothesis("Party shall disclose information")
        False
        >>> detect_negation_in_hypothesis("Information shall only include technical data")
        True
        >>> detect_negation_in_hypothesis("No party may create copies")
        True
    """
    negation_patterns = [
        # Modal prohibitions
        r'\bshall not\b',
        r'\bmust not\b',
        r'\bcannot\b',
        r'\bcan not\b',
        r'\bwill not\b',
        r'\bmay not\b',
        r'\bshould not\b',

        # Explicit prohibitions
        r'\bprohibit(ed|s)?\b',
        r'\bforbid(den|s)?\b',
        r'\bban(ned|s)?\b',

        # Negative quantifiers
        r'\bno\s+\w+\s+(shall|must|will|may|can)\b',  # "no party shall..."
        r'\bneither\b.*\bnor\b',

        # Restrictive constructions (implicit negation)
        r'\bonly\s+include\b',  # "shall only include X" implies "not include non-X"
        r'\bexclusively\b',
        r'\bsolely\b',

        # Negation in scope
        r'\bnot\s+\w+\s+(to|be|have)\b',  # "not required to", "not allowed to"
    ]

    for pattern in negation_patterns:
        if re.search(pattern, hypothesis, re.IGNORECASE):
            return True

    return False


def detect_negation_in_proposition(translation: str) -> bool:
    """
    Detect if a proposition translation already encodes negation.

    This checks if the proposition is phrased negatively, which affects
    how it should be used in query formulas.

    Args:
        translation: Natural language translation of proposition

    Returns:
        True if proposition is negative, False if affirmative

    Examples:
        >>> detect_negation_in_proposition("Party discloses information")
        False
        >>> detect_negation_in_proposition("Party does not disclose information")
        True
        >>> detect_negation_in_proposition("Information is not clearly musculoskeletal")
        True
    """
    negation_indicators = [
        r'\bnot\b',
        r'\bno\b',
        r'\bnever\b',
        r'\bcannot\b',
        r'\bwithout\b',
        r'\bexclude(s|d)?\b',
        r'\bprohibit(s|ed)?\b',
        r'\bforbid(s|den)?\b',
        r'\black(s|ing)?\b',
        r'\babsent\b',
        r'\bmissing\b',
    ]

    for pattern in negation_indicators:
        if re.search(pattern, translation, re.IGNORECASE):
            return True

    return False


def check_polarity_match(
    hypothesis: str,
    formula: str,
    retrieved_props: List[Dict]
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate that negation polarity is consistent between hypothesis and formula.

    This is the core validation that catches the bug where negative hypotheses
    are mapped to positive formulas without negation operators.

    Args:
        hypothesis: Natural language hypothesis
        formula: Translated propositional formula
        retrieved_props: List of propositions used, each with 'id' and 'translation'

    Returns:
        Tuple of:
        - is_valid: Whether polarity is consistent
        - explanation: Human-readable explanation
        - corrected_formula: Suggested correction if invalid, None otherwise

    Examples:
        >>> props = [{'id': 'P_1', 'translation': 'Party discloses information'}]
        >>> check_polarity_match(
        ...     "Party shall not disclose",
        ...     "P_1",
        ...     props
        ... )
        (False, 'POLARITY MISMATCH: ...', '¬P_1')
    """
    hypothesis_is_negative = detect_negation_in_hypothesis(hypothesis)
    formula_has_negation = bool(re.search(r'[¬~]', formula))

    # Check if retrieved propositions are negative
    props_are_negative = any(
        detect_negation_in_proposition(prop['translation'])
        for prop in retrieved_props
    )

    # Case 1: Negative hypothesis mapped to positive formula without negation
    if hypothesis_is_negative and not formula_has_negation and not props_are_negative:
        # Suggest correction: add negation to the formula
        corrected_formula = f"¬({formula})" if '∧' in formula or '∨' in formula else f"¬{formula}"

        explanation = (
            f"POLARITY MISMATCH detected:\n"
            f"  Hypothesis: '{hypothesis}' (NEGATIVE)\n"
            f"  Formula: '{formula}' (no negation operator)\n"
            f"  Retrieved props: {[p['translation'] for p in retrieved_props]} (AFFIRMATIVE)\n"
            f"  Suggested correction: '{corrected_formula}'\n"
            f"\n"
            f"The hypothesis expresses a prohibition or negation, but the formula does not "
            f"contain a negation operator (¬ or ~) and the retrieved propositions are affirmative. "
            f"This will cause the system to check if the positive action is entailed, rather than "
            f"checking if the prohibition holds."
        )

        return False, explanation, corrected_formula

    # Case 2: Positive hypothesis mapped to negative formula (less common)
    if not hypothesis_is_negative and formula_has_negation and not props_are_negative:
        # This might be intentional (checking absence), but flag for review
        explanation = (
            f"POTENTIAL POLARITY MISMATCH:\n"
            f"  Hypothesis: '{hypothesis}' (AFFIRMATIVE)\n"
            f"  Formula: '{formula}' (contains negation)\n"
            f"  This may be intentional (checking absence), but please verify."
        )
        return True, explanation, None  # Don't auto-correct, might be intentional

    # Case 3: Consistent polarity
    explanation = "Polarity is consistent between hypothesis and formula."
    return True, explanation, None


def apply_polarity_correction(
    formula: str,
    hypothesis: str,
    retrieved_props: List[Dict],
    auto_correct: bool = False
) -> Tuple[str, bool, str]:
    """
    Check and optionally auto-correct polarity mismatches.

    Args:
        formula: Original formula from LLM translation
        hypothesis: Natural language hypothesis
        retrieved_props: Retrieved propositions with translations
        auto_correct: If True, automatically apply suggested corrections

    Returns:
        Tuple of:
        - final_formula: Corrected formula (or original if valid/not auto-correcting)
        - was_corrected: Whether a correction was applied
        - explanation: Details about the check/correction

    Example:
        >>> props = [{'id': 'P_1', 'translation': 'Party discloses information'}]
        >>> apply_polarity_correction(
        ...     "P_1",
        ...     "Party shall not disclose",
        ...     props,
        ...     auto_correct=True
        ... )
        ('¬P_1', True, 'POLARITY MISMATCH detected: ... Applied auto-correction: ¬P_1')
    """
    is_valid, explanation, corrected_formula = check_polarity_match(
        hypothesis, formula, retrieved_props
    )

    if is_valid:
        return formula, False, explanation

    if auto_correct and corrected_formula:
        explanation += f"\n\nAuto-correction APPLIED: {formula} → {corrected_formula}"
        return corrected_formula, True, explanation
    else:
        explanation += f"\n\nAuto-correction DISABLED. Original formula retained: {formula}"
        return formula, False, explanation


# ==============================================================================
# FIX 2: CONTRADICTION DETECTION IN SOLVER
# ==============================================================================

@dataclass
class ContradictionCheckResult:
    """Result of checking if KB contradicts a query."""
    is_contradicted: bool
    confidence: float
    explanation: str
    entailment_result: Optional[Dict] = None  # Full entailment check result
    negation_entailment_result: Optional[Dict] = None  # Full ¬Q entailment result


def check_contradiction_via_negation_entailment(
    kb_entailment_checker,
    query_formula: str
) -> ContradictionCheckResult:
    """
    Check if KB contradicts query by checking if KB ⊨ ¬Q.

    A query Q is contradicted by KB if:
    1. KB ⊭ Q (query is not entailed)
    2. KB ⊨ ¬Q (negation of query is entailed)

    Args:
        kb_entailment_checker: MaxSATSolver instance with check_entailment method
        query_formula: Propositional formula to check

    Returns:
        ContradictionCheckResult with details

    Example:
        Suppose KB = {P_1, ¬P_2}
        Query: P_2

        Step 1: Check KB ⊨ P_2
          KB ∧ ¬P_2 = {P_1, ¬P_2, ¬P_2} = {P_1, ¬P_2}
          This is SAT, so KB ⊭ P_2 ✓

        Step 2: Check KB ⊨ ¬P_2
          KB ∧ ¬(¬P_2) = KB ∧ P_2 = {P_1, ¬P_2, P_2}
          This is UNSAT, so KB ⊨ ¬P_2 ✓

        Conclusion: P_2 is CONTRADICTED (return FALSE)
    """
    try:
        # Step 1: Check if KB ⊨ Q
        entailment_result = kb_entailment_checker.check_entailment(query_formula)

        if entailment_result.answer == "TRUE":
            # Query is entailed, cannot be contradicted
            return ContradictionCheckResult(
                is_contradicted=False,
                confidence=0.0,
                explanation=(
                    f"Query '{query_formula}' is ENTAILED by KB (not contradicted). "
                    f"KB ⊨ {query_formula}."
                ),
                entailment_result=entailment_result.__dict__,
                negation_entailment_result=None
            )

        # Step 2: Check if KB ⊨ ¬Q
        negated_formula = f"¬({query_formula})"
        negation_entailment = kb_entailment_checker.check_entailment(negated_formula)

        if negation_entailment.answer == "TRUE":
            # KB entails ¬Q, so Q is contradicted
            return ContradictionCheckResult(
                is_contradicted=True,
                confidence=negation_entailment.confidence,
                explanation=(
                    f"Query '{query_formula}' is CONTRADICTED by KB. "
                    f"KB ⊨ ¬({query_formula}) with confidence {negation_entailment.confidence:.2f}. "
                    f"The knowledge base entails the negation of the query."
                ),
                entailment_result=entailment_result.__dict__,
                negation_entailment_result=negation_entailment.__dict__
            )

        # Step 3: Neither Q nor ¬Q is entailed (uncertain)
        return ContradictionCheckResult(
            is_contradicted=False,
            confidence=0.5,
            explanation=(
                f"Query '{query_formula}' is UNCERTAIN. "
                f"Neither the query nor its negation is entailed by KB. "
                f"KB ⊭ {query_formula} and KB ⊭ ¬({query_formula})."
            ),
            entailment_result=entailment_result.__dict__,
            negation_entailment_result=negation_entailment.__dict__
        )

    except Exception as e:
        return ContradictionCheckResult(
            is_contradicted=False,
            confidence=0.0,
            explanation=f"ERROR during contradiction check: {str(e)}",
            entailment_result=None,
            negation_entailment_result=None
        )


def three_valued_query(
    kb_entailment_checker,
    query_formula: str
) -> Dict:
    """
    Perform three-valued query: TRUE (entailed), FALSE (contradicted), UNCERTAIN.

    This is the main interface that should replace the current two-valued
    entailment check.

    Args:
        kb_entailment_checker: MaxSATSolver instance
        query_formula: Propositional formula to check

    Returns:
        Dict with keys:
        - answer: "TRUE", "FALSE", or "UNCERTAIN"
        - confidence: float between 0 and 1
        - explanation: str describing the result
        - details: additional diagnostic information

    Example:
        >>> result = three_valued_query(solver, "P_1 ∧ P_2")
        >>> print(result['answer'])  # "TRUE", "FALSE", or "UNCERTAIN"
        >>> print(result['confidence'])  # 0.0 to 1.0
    """
    try:
        # Step 1: Check entailment (KB ⊨ Q?)
        entailment_result = kb_entailment_checker.check_entailment(query_formula)

        if entailment_result.answer == "TRUE":
            return {
                'answer': 'TRUE',
                'confidence': entailment_result.confidence,
                'explanation': (
                    f"Query is ENTAILED by knowledge base. "
                    f"KB ⊨ {query_formula}."
                ),
                'details': {
                    'entailment_check': entailment_result.__dict__,
                    'contradiction_check': None
                }
            }

        # Step 2: Check contradiction (KB ⊨ ¬Q?)
        contradiction_result = check_contradiction_via_negation_entailment(
            kb_entailment_checker,
            query_formula
        )

        if contradiction_result.is_contradicted:
            return {
                'answer': 'FALSE',
                'confidence': contradiction_result.confidence,
                'explanation': contradiction_result.explanation,
                'details': {
                    'entailment_check': entailment_result.__dict__,
                    'contradiction_check': contradiction_result.__dict__
                }
            }

        # Step 3: Uncertain (neither entailed nor contradicted)
        return {
            'answer': 'UNCERTAIN',
            'confidence': 0.5,
            'explanation': contradiction_result.explanation,
            'details': {
                'entailment_check': entailment_result.__dict__,
                'contradiction_check': contradiction_result.__dict__
            }
        }

    except Exception as e:
        return {
            'answer': 'ERROR',
            'confidence': 0.0,
            'explanation': f"Query processing failed: {str(e)}",
            'details': {'error': str(e)}
        }


# ==============================================================================
# INTEGRATION HELPERS
# ==============================================================================

def augment_translation_result(
    translation_result: Dict,
    hypothesis: str,
    retrieved_props: List[Dict],
    enable_auto_correction: bool = False
) -> Dict:
    """
    Augment LLM translation result with polarity checking and correction.

    This function wraps the existing translate_hypothesis() output and adds:
    - Negation detection
    - Polarity validation
    - Optional auto-correction
    - Detailed diagnostics

    Args:
        translation_result: Output from translate_hypothesis() with keys:
                           'formula', 'query_mode', 'translation', 'reasoning'
        hypothesis: Original natural language hypothesis
        retrieved_props: Propositions retrieved for this query
        enable_auto_correction: Whether to auto-correct polarity mismatches

    Returns:
        Augmented translation result with additional keys:
        - 'original_formula': Formula before any correction
        - 'final_formula': Formula after correction (if applied)
        - 'was_corrected': Boolean indicating if correction was applied
        - 'polarity_check': Details about polarity validation
        - 'hypothesis_is_negative': Boolean flag
    """
    original_formula = translation_result['formula']

    # Apply polarity checking and optional correction
    final_formula, was_corrected, polarity_explanation = apply_polarity_correction(
        formula=original_formula,
        hypothesis=hypothesis,
        retrieved_props=retrieved_props,
        auto_correct=enable_auto_correction
    )

    # Augment result
    augmented_result = {
        **translation_result,  # Original keys: formula, query_mode, translation, reasoning
        'original_formula': original_formula,
        'final_formula': final_formula,
        'was_corrected': was_corrected,
        'polarity_check': {
            'is_valid': not was_corrected or enable_auto_correction,
            'explanation': polarity_explanation,
            'hypothesis_is_negative': detect_negation_in_hypothesis(hypothesis)
        }
    }

    # Update the formula field to use corrected version
    if enable_auto_correction:
        augmented_result['formula'] = final_formula

    return augmented_result


# ==============================================================================
# TESTING AND VALIDATION
# ==============================================================================

def test_negation_detection():
    """Unit tests for negation detection."""
    print("Testing negation detection in hypotheses...")

    test_cases = [
        ("Party shall not disclose information", True),
        ("Party shall disclose information", False),
        ("Information shall only include technical data", True),
        ("No party may create copies", True),
        ("Receiving Party is prohibited from using confidential info", True),
        ("Agreement may grant rights to information", False),
        ("Party cannot reverse engineer products", True),
    ]

    passed = 0
    for hypothesis, expected in test_cases:
        result = detect_negation_in_hypothesis(hypothesis)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{hypothesis}' → {result} (expected {expected})")
        if result == expected:
            passed += 1

    print(f"\nPassed {passed}/{len(test_cases)} tests\n")


def test_proposition_negation():
    """Unit tests for proposition negation detection."""
    print("Testing negation detection in propositions...")

    test_cases = [
        ("Party discloses information", False),
        ("Party does not disclose information", True),
        ("Information is not clearly musculoskeletal", True),
        ("System is turned off", False),
        ("Researcher lacks protective equipment", True),
        ("Ventilation system cannot be disabled", True),
    ]

    passed = 0
    for translation, expected in test_cases:
        result = detect_negation_in_proposition(translation)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{translation}' → {result} (expected {expected})")
        if result == expected:
            passed += 1

    print(f"\nPassed {passed}/{len(test_cases)} tests\n")


def test_polarity_matching():
    """Unit tests for polarity matching."""
    print("Testing polarity matching...")

    test_cases = [
        # (hypothesis, formula, props, should_be_valid)
        (
            "Party shall not disclose information",
            "P_1",
            [{'id': 'P_1', 'translation': 'Party discloses information'}],
            False  # MISMATCH: negative hypothesis → positive formula
        ),
        (
            "Party shall not disclose information",
            "¬P_1",
            [{'id': 'P_1', 'translation': 'Party discloses information'}],
            True  # CORRECT: negative hypothesis → negated positive formula
        ),
        (
            "Party shall disclose information",
            "P_1",
            [{'id': 'P_1', 'translation': 'Party discloses information'}],
            True  # CORRECT: positive hypothesis → positive formula
        ),
        (
            "Information shall only include technical data",
            "P_2",
            [{'id': 'P_2', 'translation': 'Information includes all types of data'}],
            False  # MISMATCH: restrictive hypothesis → positive formula
        ),
    ]

    passed = 0
    for hypothesis, formula, props, expected_valid in test_cases:
        is_valid, explanation, correction = check_polarity_match(hypothesis, formula, props)
        status = "✓" if is_valid == expected_valid else "✗"
        print(f"  {status} H: '{hypothesis}' | F: '{formula}' → {is_valid} (expected {expected_valid})")
        if is_valid != expected_valid:
            print(f"     Explanation: {explanation[:100]}...")
        if is_valid == expected_valid:
            passed += 1

    print(f"\nPassed {passed}/{len(test_cases)} tests\n")


if __name__ == "__main__":
    print("="*80)
    print("NEGATION HANDLING FIX - UNIT TESTS")
    print("="*80)
    print()

    test_negation_detection()
    test_proposition_negation()
    test_polarity_matching()

    print("="*80)
    print("All tests completed. Review results above.")
    print("="*80)
