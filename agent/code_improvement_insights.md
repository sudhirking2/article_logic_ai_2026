# Code Improvement Insights for Logify
## AI Agent Analysis Based on Error Pattern Investigation

**Date:** January 29, 2026
**Analyzed by:** Claude 3.5 Sonnet (AI Agent)
**Context:** Analysis of 7-document subset showing mixed performance (40% vs RAG 68%)

---

## Executive Summary

Three systematic error patterns emerged from the data analysis, each pointing to specific code improvements:

1. **100% failure on contradiction detection** (0/7 FALSE cases correct)
2. **75% over-confidence on uncertain evidence** (39/52 UNCERTAIN → TRUE)
3. **Lossy query translation** (84% collapse to single propositions, 8.4% contain negation)

---

## Error Pattern 1: Negation Handling Failures

### Observed Problem
- Only 8.4% of query formulas contain negation operators (`¬`)
- Many hypotheses are negative statements ("shall NOT reverse engineer")
- System never predicts FALSE (contradiction), always TRUE or UNCERTAIN
- Example: "Receiving Party may create copies" (FALSE) → predicted TRUE with confidence 0.5

### Root Cause Analysis

**File:** `code/interface_with_user/translate.py`

**Line 386:** The prompt example for negations is misleading:
```python
Example 2 - Negation:
Hypothesis: "The receiving party shall not reverse engineer any information"
If P_9 states "The Receiving Party shall not alter, modify, disassemble, reverse engineer..."
Output: {"formula": "P_9", "query_mode": "entailment", ...}
```

**Problem:** This tells the LLM that "shall NOT X" hypotheses should map to propositions that *already state the prohibition* (P_9 = "shall not reverse engineer"). But when checking if something is *contradicted*, the LLM needs to:
1. Find propositions about the action (P_X = "may copy")
2. Recognize contradiction with hypothesis ("may NOT copy")
3. Use negation: `¬P_X`

Instead, the LLM searches for a proposition that says "shall NOT copy" — but if the document says "may copy", no such proposition exists!

### Specific Code Improvements

#### 1. Update Query Translation Prompt (translate.py, line 418-426)

**Current guideline:**
```python
2. "Shall not" prohibitions → The proposition already captures the negation, use: P_i (mode: entailment)
```

**Improved guideline:**
```python
2. "Shall not" / "may not" prohibitions:
   - If a proposition states the prohibition (P_i = "shall not X"): Use P_i (mode: entailment)
   - If a proposition states the permitted action (P_j = "may X"): Use ¬P_j (mode: entailment)
   - Prefer explicit negation formulas when checking contradictions
```

#### 2. Add Contradiction-Specific Examples (translate.py, line 376-407)

**Add new example:**
```python
Example 7 - Detecting Contradictions:
Hypothesis: "Receiving Party may NOT create copies of Confidential Information"
If P_15 states "The Receiving Party may make copies..."
Output: {"formula": "¬P_15", "query_mode": "entailment",
         "translation": "It is NOT the case that copies may be made",
         "reasoning": "Hypothesis prohibits an action that P_15 permits - use negation to check contradiction"}
```

#### 3. Explicit Negation Detection in Query Preprocessing (translate.py, new function at line 170)

```python
def contains_negation(query: str) -> bool:
    """
    Detect if query contains negation keywords.

    Returns:
        True if query likely expresses negation
    """
    negation_patterns = [
        r'\b(not|no|never|neither|nor)\b',
        r'\b(shall\s+not|may\s+not|cannot|can\'t)\b',
        r'\b(without|exclude|prohibit|forbid)\b',
        r'\b(only\s+if)\b'  # "only technical info" = "NOT non-technical"
    ]

    import re
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in negation_patterns)
```

**Usage:** Pass this flag to prompt to emphasize negation handling:
```python
if contains_negation(query):
    prompt += "\n\nIMPORTANT: This hypothesis contains negation. Consider using ¬P_i if the document states the opposite."
```

---

## Error Pattern 2: Over-Confidence on UNCERTAIN Cases

### Observed Problem
- 75% of UNCERTAIN ground truth cases predicted as TRUE
- Same proposition (e.g., P_6) used for multiple unrelated hypotheses
- Confidence scores often default to 0.5 for incorrect predictions

### Root Cause Analysis

**File:** `code/interface_with_user/translate.py`

**Line 94-100:** Propositions encoded by translation text only:
```python
chunk = {
    'text': prop['translation'],  # SBERT only sees this
    'id': prop['id'],
    'translation': prop['translation'],
    'evidence': prop.get('evidence', ''),  # Not used in retrieval!
    ...
}
```

**Problem:** SBERT matches surface similarity between *query* and *proposition translation*, not semantic entailment.

Example:
- Query: "shall not solicit representatives"
- P_6 translation: "Receiving Party obligations regarding disclosure"
- SBERT: High similarity because both mention "Receiving Party"
- But P_6's *evidence* might be about disclosure, not solicitation!

### Specific Code Improvements

#### 1. Hybrid Retrieval: Translation + Evidence (translate.py, line 94)

**Current:**
```python
chunk = {
    'text': prop['translation'],  # Only this is embedded
    ...
}
```

**Improved:**
```python
chunk = {
    'text': f"{prop['translation']}. Context: {prop.get('evidence', '')[:200]}",  # Embed both
    'id': prop['id'],
    'translation': prop['translation'],
    'evidence': prop.get('evidence', ''),
    ...
}
```

**Rationale:** Evidence contains the actual document text supporting the proposition. Including it in the embedding ensures SBERT matches *semantic content*, not just proposition labels.

#### 2. Post-Retrieval Filtering by Evidence (translate.py, new function at line 143)

```python
def filter_by_evidence_relevance(
    query: str,
    retrieved_chunks: List[Dict],
    threshold: float = 0.3
) -> List[Dict]:
    """
    Re-rank retrieved propositions by evidence relevance.

    Args:
        query: User query
        retrieved_chunks: SBERT-retrieved propositions
        threshold: Minimum evidence similarity to keep

    Returns:
        Filtered list of propositions where evidence is relevant
    """
    from baseline_rag.retriever import encode_query, compute_cosine_similarity, load_sbert_model
    import numpy as np

    sbert_model = load_sbert_model("all-MiniLM-L6-v2")
    query_embedding = encode_query(query, sbert_model)

    filtered = []
    for chunk in retrieved_chunks:
        evidence = chunk.get('evidence', '')
        if not evidence:
            # No evidence available, keep with lower score
            chunk['evidence_relevance'] = 0.0
            filtered.append(chunk)
            continue

        # Compute evidence-query similarity
        evidence_embedding = encode_query(evidence[:500], sbert_model)  # Limit length
        evidence_sim = compute_cosine_similarity(query_embedding, evidence_embedding.reshape(1, -1))[0]

        chunk['evidence_relevance'] = float(evidence_sim)

        if evidence_sim >= threshold:
            filtered.append(chunk)

    return sorted(filtered, key=lambda x: x.get('evidence_relevance', 0), reverse=True)
```

**Usage in translate_query (line 706):**
```python
retrieved = retrieve_top_k_propositions(query, chunks, sbert_model, k=actual_k * 2)  # Retrieve 2x
retrieved = filter_by_evidence_relevance(query, retrieved, threshold=0.3)[:actual_k]  # Filter down
```

#### 3. Prompt Modification for Uncertain Cases (translate.py, line 426)

**Add to guidelines:**
```python
7. Absence of evidence → Return UNCERTAIN signal
   - If no proposition closely matches the hypothesis (all similarity < 0.4), add:
     "explanation": "No clear textual evidence found for this claim"
   - This helps the solver distinguish UNCERTAIN from weak entailment
```

---

## Error Pattern 3: Proposition Mapping Ambiguity

### Observed Problem
- Same proposition ID (P_15) maps to ground truths: {TRUE, FALSE, FALSE, UNCERTAIN} across different queries
- This is logically impossible if logification is semantically faithful

### Root Cause Analysis

**File:** `code/from_text_to_logic/logic_converter.py` (not shown in excerpts, but inferred)

**Problem:** Atomic propositions may be extracted at wrong granularity:
- Too coarse: "Receiving Party has obligations" (covers multiple distinct obligations)
- Too fine: "Receiving Party obligations about disclosure on Tuesday" (overly specific)

### Specific Code Improvements

#### 1. Add Proposition Granularity Guidance in Logification Prompt

**Current** (inferred from results): LLM extracts high-level propositions like "Receiving Party obligations"

**Improved:** Add granularity instruction to logify prompt:
```python
PROPOSITION EXTRACTION GUIDELINES:
1. Each proposition should express ONE atomic fact
2. Avoid vague terms like "obligations", "requirements", "conditions"
3. Make propositions specific enough to have unambiguous truth values
4. Good: "Receiving Party shall not disclose to competitors"
5. Bad: "Receiving Party has disclosure obligations" (which obligations?)
```

#### 2. Decompose Compound Propositions During Extraction

**Add validation step** in `logic_converter.py`:
```python
def validate_proposition_granularity(proposition: Dict) -> List[str]:
    """
    Check if proposition is too compound and suggest decomposition.

    Returns:
        List of warnings (empty if OK)
    """
    translation = proposition['translation'].lower()
    warnings = []

    # Check for conjunctions suggesting compound propositions
    if ' and ' in translation or ' or ' in translation:
        warnings.append(f"Proposition {proposition['id']} may be compound: contains 'and/or'")

    # Check for vague terms
    vague_terms = ['obligations', 'requirements', 'conditions', 'provisions', 'terms']
    if any(term in translation for term in vague_terms):
        warnings.append(f"Proposition {proposition['id']} uses vague term - may need decomposition")

    return warnings
```

---

## Error Pattern 4: Solver Contradiction Detection

### Observed Problem
- Solver never returns FALSE (contradiction)
- Only returns TRUE or UNCERTAIN

### Root Cause Analysis

**File:** `code/logic_solver/maxsat.py`

**Line 119-131:** Contradiction check depends on consistency check:
```python
consistency_result = self.check_consistency(query_formula)

if consistency_result.answer == "FALSE":
    # Q is inconsistent with KB, so ¬Q is entailed
    ...
    return SolverResult(answer="FALSE", ...)
```

**Problem:** This logic is correct, but `check_consistency` (line 151-194) only checks if `KB ∧ Q` is SAT. It doesn't distinguish:
- **Absent:** No proposition mentions Q → SAT (Q is consistent but not entailed)
- **Contradicted:** Proposition states ¬Q → UNSAT (Q is inconsistent)

If propositions don't capture negations (Error Pattern 1), then contradictions appear as *absence*, not inconsistency!

### Specific Code Improvements

#### 1. Explicit Negation Check in Solver (maxsat.py, line 63, before line 75)

```python
def check_entailment(self, query_formula: str) -> SolverResult:
    """Check if query is entailed by KB."""
    try:
        # FIRST: Check if ¬Q is explicitly entailed
        # If KB ⊨ ¬Q, then Q is contradicted
        negated_formula = f"¬({query_formula})"

        # Simplify double negation
        if query_formula.startswith("¬"):
            negated_formula = query_formula[1:].strip()

        # Check entailment of ¬Q
        negation_entailed = self._is_entailed_by_hard_constraints(negated_formula)

        if negation_entailed:
            return SolverResult(
                answer="FALSE",
                confidence=1.0,
                model=None,
                explanation="Query is contradicted: its negation is entailed by the KB"
            )

        # THEN: Proceed with normal entailment check...
        [rest of current code]
```

#### 2. Add Helper Method for Direct Entailment Check (maxsat.py, after line 200)

```python
def _is_entailed_by_hard_constraints(self, formula: str) -> bool:
    """
    Check if formula is entailed by hard constraints alone.

    Returns True if KB_hard ⊨ formula, False otherwise.
    """
    try:
        # Create WCNF with only hard constraints
        wcnf = self._copy_wcnf(self.base_wcnf)
        hard_only = self._extract_hard_clauses(wcnf)

        # Add ¬formula as hard clause
        negated_clauses = self.encoder.encode_query(formula, negate=True)
        for clause in negated_clauses:
            hard_only.append(clause)

        # Check if KB_hard ∧ ¬formula is UNSAT
        is_sat, _ = self._check_sat(hard_only)
        return not is_sat  # Entailed if UNSAT

    except Exception:
        return False
```

---

## Priority Ranking of Improvements

Based on impact on the three error patterns:

### High Priority (Immediate Impact)
1. **Add negation detection and explicit ¬P_i usage** in query translation (Pattern 1)
   - Files: `translate.py` lines 418-426, add example at line 407
   - Impact: Should fix 100% failure rate on FALSE cases

2. **Implement explicit negation entailment check** in solver (Pattern 1)
   - Files: `maxsat.py` line 63, add helper method
   - Impact: Enables contradiction detection even if query translation improves

3. **Hybrid retrieval with evidence** in query translation (Pattern 2)
   - Files: `translate.py` line 94, add evidence to SBERT embedding
   - Impact: Reduces spurious matches, lowers false positive rate on UNCERTAIN

### Medium Priority (Reduces Variability)
4. **Post-retrieval evidence filtering** (Pattern 2)
   - Files: `translate.py` new function at line 143
   - Impact: Further refines retrieval quality

5. **Proposition granularity validation** in logification (Pattern 3)
   - Files: `logic_converter.py` (add validation step)
   - Impact: Prevents ambiguous propositions from being extracted

### Low Priority (Robustness)
6. **Add UNCERTAIN signal in query translation** (Pattern 2)
   - Files: `translate.py` line 426
   - Impact: Helps distinguish "no evidence" from "weak evidence"

---

## Testing Recommendations

### Unit Tests Needed

**Test 1: Negation Handling**
```python
def test_negation_query_translation():
    """Verify negation formulas are generated for contradictions."""
    query = "Receiving Party may NOT share information"
    # Assume P_5 = "Receiving Party may share information"
    result = translate_query(query, json_path, api_key, k=5)

    # Expected: formula should contain negation
    assert "¬P_5" in result['formula'] or "~P_5" in result['formula'], \
        "Negation not detected in formula"
```

**Test 2: Contradiction Detection**
```python
def test_solver_contradiction_detection():
    """Verify solver returns FALSE for contradictions."""
    # Logified structure with: P_1 = "may copy"
    structure = {
        "primitive_props": [{"id": "P_1", "translation": "may make copies"}],
        "hard_constraints": [{"formula": "P_1"}],  # Hard constraint: may copy
        "soft_constraints": []
    }

    solver = LogicSolver(structure)
    result = solver.check_entailment("¬P_1")  # Query: may NOT copy

    assert result.answer == "FALSE", \
        f"Expected FALSE for contradiction, got {result.answer}"
```

**Test 3: Evidence Relevance**
```python
def test_evidence_based_retrieval():
    """Verify evidence is used in retrieval ranking."""
    chunks = [
        {"id": "P_1", "translation": "Receiving Party obligations",
         "evidence": "The Recipient shall notify the Discloser..."},
        {"id": "P_2", "translation": "Disclosure requirements",
         "evidence": "Information may not be shared with third parties..."}
    ]

    query = "Can information be shared with third parties?"

    # P_2's evidence should match better than P_1 despite translation similarity
    # (Requires updated hybrid retrieval)
```

### Integration Test on Failure Cases

Re-run on the 7 documents that showed worst performance:
- Doc 14 (-58.8%): Shortest document, should benefit from negation fixes
- Doc 3 (-35.3%): Cached but poor performance, suggests query translation issue
- Doc 9 (-29.4%): Check if contradiction detection improves

**Success criteria:**
- FALSE prediction rate > 0% (currently 0%)
- UNCERTAIN false positive rate < 50% (currently 75%)
- Overall accuracy improvement of +10-15% on these 7 docs

---

## Long-Term Architectural Considerations

### 1. Separate Negation Layer in Extraction
**Current:** Propositions capture positive or negative statements inconsistently

**Proposed:** Always extract positive propositions, mark negations explicitly:
```json
{
  "id": "P_5",
  "translation": "Receiving Party may share information",
  "polarity": "positive"
}
{
  "id": "P_6",
  "translation": "Receiving Party may share information",  // Same base
  "polarity": "negative",  // But this says "may NOT share"
  "negates": "P_5"
}
```

**Benefit:** Query translation can explicitly search for base proposition + check polarity

### 2. Entailment Confidence from Solver
**Current:** Confidence is 0.5 by default for UNCERTAIN

**Proposed:** Use MaxSAT cost differential:
```python
cost_Q = solve_maxsat(KB ∧ Q)  # Cost of satisfying Q
cost_neg_Q = solve_maxsat(KB ∧ ¬Q)  # Cost of satisfying ¬Q

confidence = cost_neg_Q / (cost_Q + cost_neg_Q)  # Higher if Q has lower cost
```

**Benefit:** Confidence reflects soft constraint support, not just binary SAT/UNSAT

### 3. Query Mode Auto-Detection
**Current:** LLM guesses "entailment" vs "consistency" mode

**Proposed:** Use NLI-trained classifier:
- Input: Query text
- Output: "entailment", "consistency", or "contradiction"
- Fine-tune on ContractNLI training set

**Benefit:** Removes ambiguity in query interpretation

---

## Conclusion

The three error patterns are **not fundamental flaws in the neuro-symbolic approach**, but rather implementation gaps in:
1. Negation preservation (extraction + query translation + solver logic)
2. Retrieval semantic grounding (SBERT matching + evidence filtering)
3. Proposition granularity (too coarse or ambiguous atomic statements)

All are **fixable with targeted code changes** outlined above. Priority should be negation handling (fixes Pattern 1 and enables Pattern 3), then retrieval improvements (fixes Pattern 2).

**Expected impact:** +15-20% accuracy improvement on current failure cases with High Priority fixes alone.

---

**Generated by:** Claude 3.5 Sonnet (AI Agent)
**For:** Logify ongoing research project
**Purpose:** Inform iterative system development based on error analysis
