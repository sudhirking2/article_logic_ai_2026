# Integration Guide: Negation Handling Fix

This guide explains how to integrate the negation handling fixes into the Logify codebase.

## Files Created

1. **negation_fix_proposal.md** - Comprehensive analysis and design document
2. **negation_fix_implementation.py** - Concrete implementation with unit tests
3. **INTEGRATION_GUIDE.md** (this file) - Step-by-step integration instructions

## Quick Start

### Step 1: Copy the Implementation Module

```bash
cp agent/negation_fix_implementation.py code/logic_solver/negation_fixes.py
```

### Step 2: Modify translate.py

**File**: `code/interface_with_user/translate.py`

Add import at the top:

```python
from logic_solver.negation_fixes import (
    detect_negation_in_hypothesis,
    detect_negation_in_proposition,
    augment_translation_result
)
```

**Modify `translate_hypothesis()` function** (around line 180-230):

```python
def translate_hypothesis(
    hypothesis: str,
    logified: Dict,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    k: int = 20,
    debug: bool = False,
    enable_negation_fix: bool = True,  # NEW PARAMETER
    auto_correct_polarity: bool = False  # NEW PARAMETER
) -> Dict:
    """
    Translate natural language hypothesis to propositional formula.

    NEW: Includes optional negation detection and polarity correction.

    Args:
        ...existing args...
        enable_negation_fix: If True, perform negation detection and validation
        auto_correct_polarity: If True, automatically correct polarity mismatches
    """

    # ... existing code for retrieval ...

    # Call LLM to translate
    response = _call_llm_for_translation(...)  # existing code

    # NEW: Augment result with negation checking
    if enable_negation_fix:
        # Get retrieved proposition details
        retrieved_props = []
        for prop_id in retrieved_prop_ids:
            prop_data = next((p for p in logified['primitive_props'] if p['id'] == prop_id), None)
            if prop_data:
                retrieved_props.append({
                    'id': prop_data['id'],
                    'translation': prop_data['translation']
                })

        # Apply negation fix
        from logic_solver.negation_fixes import augment_translation_result
        response = augment_translation_result(
            translation_result=response,
            hypothesis=hypothesis,
            retrieved_props=retrieved_props,
            enable_auto_correction=auto_correct_polarity
        )

        # Log warnings if polarity mismatch detected
        if debug and not response['polarity_check']['is_valid']:
            print(f"\n⚠️  POLARITY WARNING:")
            print(response['polarity_check']['explanation'])
            if response['was_corrected']:
                print(f"✓ Auto-corrected: {response['original_formula']} → {response['final_formula']}")

    return response
```

### Step 3: Modify maxsat.py

**File**: `code/logic_solver/maxsat.py`

Add import at the top:

```python
from .negation_fixes import (
    check_contradiction_via_negation_entailment,
    three_valued_query
)
```

**Add new method to `MaxSATSolver` class** (after line 149):

```python
def check_contradiction(self, query_formula: str) -> 'SolverResult':
    """
    Check if query is explicitly contradicted by the knowledge base.

    A query Q is contradicted if:
    1. KB ⊭ Q (query is not entailed)
    2. KB ⊨ ¬Q (negation of query is entailed)

    Args:
        query_formula: Propositional formula

    Returns:
        SolverResult with answer FALSE if contradicted, UNCERTAIN otherwise
    """
    from .negation_fixes import check_contradiction_via_negation_entailment

    contradiction_result = check_contradiction_via_negation_entailment(
        kb_entailment_checker=self,
        query_formula=query_formula
    )

    if contradiction_result.is_contradicted:
        return SolverResult(
            answer="FALSE",
            confidence=contradiction_result.confidence,
            model=None,
            explanation=contradiction_result.explanation
        )
    else:
        return SolverResult(
            answer="UNCERTAIN",
            confidence=contradiction_result.confidence,
            model=None,
            explanation=contradiction_result.explanation
        )
```

**Modify `query()` method** (around line 203):

```python
def query(self, query_formula: str, enable_contradiction_check: bool = True) -> SolverResult:
    """
    Main query interface with three-valued logic: TRUE/FALSE/UNCERTAIN.

    NEW: Includes contradiction detection (returns FALSE when KB ⊨ ¬Q).

    Args:
        query_formula: Propositional formula
        enable_contradiction_check: If True, check for contradictions (NEW)

    Returns:
        SolverResult with answer in {TRUE, FALSE, UNCERTAIN}
    """
    try:
        # Step 1: Check entailment (KB ⊨ Q?)
        entailment_result = self.check_entailment(query_formula)

        if entailment_result.answer == "TRUE":
            return entailment_result

        # Step 2: Check contradiction (KB ⊨ ¬Q?) - NEW
        if enable_contradiction_check:
            contradiction_result = self.check_contradiction(query_formula)

            if contradiction_result.answer == "FALSE":
                return contradiction_result

        # Step 3: Neither entailed nor contradicted → UNCERTAIN
        return SolverResult(
            answer="UNCERTAIN",
            confidence=0.5,
            model=None,
            explanation="Query is neither entailed nor contradicted by KB"
        )

    except Exception as e:
        return SolverResult(
            answer="ERROR",
            confidence=0.0,
            model=None,
            explanation=f"Query processing failed: {str(e)}"
        )
```

### Step 4: Update Evaluation Scripts

**File**: `code/experiments/contractNLI/eval_logify.py` (or similar)

Add flags to enable the fixes:

```python
# At the top of main evaluation loop
ENABLE_NEGATION_FIX = True  # Set to True to enable negation detection
AUTO_CORRECT_POLARITY = False  # Set to True to auto-correct mismatches
ENABLE_CONTRADICTION_CHECK = True  # Set to True to detect FALSE cases

# When calling translate_hypothesis
translation_result = translate_hypothesis(
    hypothesis=hypothesis_text,
    logified=logified_data,
    k=20,
    debug=True,
    enable_negation_fix=ENABLE_NEGATION_FIX,  # NEW
    auto_correct_polarity=AUTO_CORRECT_POLARITY  # NEW
)

# When calling solver.query
result = solver.query(
    query_formula=translation_result['formula'],
    enable_contradiction_check=ENABLE_CONTRADICTION_CHECK  # NEW
)
```

## Testing the Fix

### Test 1: Unit Tests

```bash
cd agent
python negation_fix_implementation.py
```

Expected output:
```
Testing negation detection in hypotheses...
  ✓ All 7 tests pass

Testing negation detection in propositions...
  ✓ All 6 tests pass

Testing polarity matching...
  ✓ All 4 tests pass
```

### Test 2: Integration Test on ContractNLI FALSE Cases

Create `test_false_cases.py`:

```python
"""Test the negation fix on known FALSE cases from ContractNLI."""

from code.interface_with_user.translate import translate_hypothesis
from code.logic_solver.maxsat import MaxSATSolver

# Test case from paper: Doc 3, hypothesis nda-2
test_case = {
    'hypothesis': 'Confidential Information shall only include technical information.',
    'ground_truth': 'FALSE',
    'doc_id': 3
}

# Load logified document
logified_path = f"code/experiments/contractNLI/logified_docs/doc_{test_case['doc_id']}.json"
with open(logified_path) as f:
    logified = json.load(f)

# Translate hypothesis WITH negation fix
translation = translate_hypothesis(
    hypothesis=test_case['hypothesis'],
    logified=logified,
    enable_negation_fix=True,
    auto_correct_polarity=False,  # Don't auto-correct yet, just detect
    debug=True
)

print(f"Hypothesis: {test_case['hypothesis']}")
print(f"Formula: {translation['formula']}")
print(f"Ground truth: {test_case['ground_truth']}")

# Check for polarity warnings
if not translation['polarity_check']['is_valid']:
    print("\n⚠️  POLARITY WARNING DETECTED:")
    print(translation['polarity_check']['explanation'])

# Query with contradiction check
solver = MaxSATSolver(logified)
result = solver.query(
    query_formula=translation['formula'],
    enable_contradiction_check=True
)

print(f"\nPrediction: {result.answer} (confidence: {result.confidence:.2f})")
print(f"Expected: {test_case['ground_truth']}")
print(f"Match: {'✓' if result.answer == test_case['ground_truth'] else '✗'}")
```

Run:
```bash
python test_false_cases.py
```

### Test 3: Full ContractNLI Evaluation

```bash
cd code/experiments/contractNLI
python eval_logify.py --enable-negation-fix --enable-contradiction-check
```

Compare metrics:
- **Before fix**: FALSE recall = 0% (0/7)
- **After fix**: Expected FALSE recall = 70-80% (5-6/7)

## Configuration Options

### Conservative Mode (Recommended for Initial Deployment)

```python
ENABLE_NEGATION_FIX = True  # Detect issues, log warnings
AUTO_CORRECT_POLARITY = False  # Don't auto-correct, let LLM handle it
ENABLE_CONTRADICTION_CHECK = True  # Enable FALSE predictions
```

**Pros**: Safe, reversible, allows manual review
**Cons**: Won't fix existing polarity mismatches

### Aggressive Mode (For Maximum Performance)

```python
ENABLE_NEGATION_FIX = True
AUTO_CORRECT_POLARITY = True  # Automatically correct polarity mismatches
ENABLE_CONTRADICTION_CHECK = True
```

**Pros**: Maximum accuracy improvement
**Cons**: May over-correct in edge cases

## Expected Performance Impact

### Quantitative Improvements

| Metric | Before Fix | After Fix (Conservative) | After Fix (Aggressive) |
|--------|-----------|-------------------------|----------------------|
| FALSE recall | 0% (0/7) | 60-70% (4-5/7) | 70-80% (5-6/7) |
| UNCERTAIN false positive rate | 75% | 50-60% | 40-50% |
| Overall accuracy (7-doc subset) | 40% | 50-55% | 55-60% |

### Qualitative Improvements

- **Explicit FALSE predictions**: System can now distinguish "not mentioned" from "contradicted"
- **Polarity validation**: Warnings flag mismatches between hypothesis and formula
- **Better debugging**: Detailed explanations for contradiction detection
- **Consistent semantics**: Negations are explicitly represented in formulas

## Troubleshooting

### Issue: Too many false positives on FALSE

**Symptom**: System predicts FALSE too often (high false positive rate)

**Solution**: The contradiction check might be too aggressive. Check if KB has many negated propositions that cause spurious contradictions.

**Fix**: Add confidence threshold:

```python
contradiction_result = solver.check_contradiction(query_formula)
if contradiction_result.answer == "FALSE" and contradiction_result.confidence > 0.7:
    return contradiction_result  # Only return FALSE if highly confident
```

### Issue: Polarity warnings but no mismatches

**Symptom**: System logs polarity warnings for valid formulas

**Solution**: The negation detection patterns might be too broad.

**Fix**: Refine `detect_negation_in_hypothesis()` patterns to reduce false positives:

```python
# Add context-aware checks
if re.search(r'\bonly\s+include\b', hypothesis):
    # Check if this is truly restrictive or just descriptive
    # ...
```

### Issue: Auto-correction breaks complex formulas

**Symptom**: Auto-corrected formulas are semantically incorrect

**Solution**: Auto-correction is too simplistic for complex formulas.

**Fix**: Disable auto-correction for formulas with multiple operators:

```python
if auto_correct and '∧' not in formula and '∨' not in formula:
    # Only auto-correct simple atomic formulas
    return corrected_formula
else:
    # Log warning but don't correct
    return original_formula
```

## Rollback Plan

If the fix causes regressions:

1. **Disable auto-correction**: Set `auto_correct_polarity=False`
2. **Disable contradiction check**: Set `enable_contradiction_check=False`
3. **Disable entire fix**: Set `enable_negation_fix=False`

The codebase will revert to original behavior.

## Next Steps

After integrating this fix:

1. **Run full evaluation** on ContractNLI (all documents, not just 7-doc subset)
2. **Cross-dataset validation** on DocNLI (after fixing cache path bugs)
3. **Manual review** of FALSE predictions to validate correctness
4. **Tune confidence thresholds** based on precision/recall trade-offs
5. **Update prompt_logify** to guide better proposition extraction (Fix 3)
6. **Implement validation layer** for ongoing quality assurance (Fix 4)

## Contact

For questions or issues:
- Review `negation_fix_proposal.md` for design rationale
- Check `negation_fix_implementation.py` for implementation details
- See paper (agent/paper.tex) Section 4.2.2 for error pattern analysis
