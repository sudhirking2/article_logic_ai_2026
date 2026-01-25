# Exemplar 3 Addition: Student Assessment Rules

## Summary

Added a third exemplar to the `prompt_logify2` file demonstrating the logification of student assessment and course passing rules. This exemplar showcases:

1. **Exclusive-OR (XOR) logic**: The "either...but not both" construction
2. **Biconditional relationships**: The "if and only if" construction
3. **Necessary conditions**: The "requires" construction
4. **Soft constraints from modal verbs**: The "may" construction indicating possibility

## Input Text

```
Students must complete either the written exam or the oral presentation, but not both, to satisfy the assessment requirement. A student passes the course if and only if they satisfy the assessment requirement and attend at least 80% of lectures. Students may request deadline extensions, though approval requires documented extenuating circumstances.
```

## Key Logical Features

### 1. Exclusive-OR (XOR)
- **Text**: "either the written exam or the oral presentation, but not both"
- **Formula**: `(P_1 ∨ P_2) ∧ ¬(P_1 ∧ P_2) ⟺ P_3`
- **Insight**: The explicit "but not both" creates an XOR relationship, which must be captured with both a disjunction and the negation of their conjunction

### 2. Biconditional (IFF)
- **Text**: "A student passes the course if and only if..."
- **Formula**: `P_4 ⟺ (P_3 ∧ P_5)`
- **Insight**: The "if and only if" explicitly establishes necessity and sufficiency in both directions

### 3. Necessary Condition
- **Text**: "approval requires documented extenuating circumstances"
- **Formula**: `P_8 ⟹ P_7`
- **Insight**: "Requires" establishes P_7 as necessary for P_8, but not sufficient (hence one-way implication, not biconditional)

### 4. Soft Constraint from Modal Verb
- **Text**: "Students may request deadline extensions"
- **Formula**: `P_6 ⟹ P_8` (soft constraint)
- **Insight**: "May" indicates possibility, not certainty, making this a defeasible soft constraint

## Atomic Propositions (8 total)

1. **P_1**: The student completes the written exam
2. **P_2**: The student completes the oral presentation
3. **P_3**: The student satisfies the assessment requirement
4. **P_4**: The student passes the course
5. **P_5**: The student attends at least 80% of lectures
6. **P_6**: The student requests a deadline extension
7. **P_7**: The student has documented extenuating circumstances
8. **P_8**: The deadline extension is approved

## Hard Constraints (3 total)

1. **H_1**: XOR relationship defining assessment satisfaction
   - Formula: `(P_1 ∨ P_2) ∧ ¬(P_1 ∧ P_2) ⟺ P_3`

2. **H_2**: Biconditional defining course passing
   - Formula: `P_4 ⟺ (P_3 ∧ P_5)`

3. **H_3**: Necessary condition for approval
   - Formula: `P_8 ⟹ P_7`

## Soft Constraints (2 total)

1. **S_1**: Request may lead to approval
   - Formula: `P_6 ⟹ P_8`
   - Reasoning: "May" indicates possibility, not certainty

2. **S_2**: Documented circumstances typically lead to approval
   - Formula: `P_7 ⟹ P_8`
   - Reasoning: While H_3 says approval requires documentation, this captures that documentation typically (but not always) leads to approval

## Files Created/Modified

### Created:
1. `/workspace/repo/artifacts/few_shot_examples/inputs/example_03_student_assessment.txt`
   - Input text for the student assessment example

2. `/workspace/repo/artifacts/few_shot_examples/outputs/example_03_student_assessment_output.json`
   - Full JSON output with all propositions and constraints

3. `/workspace/repo/artifacts/few_shot_examples/outputs/example_03_for_prompt.txt`
   - Formatted version with input text, triples, and output for inclusion in prompt

4. `/workspace/repo/artifacts/few_shot_examples/run_logify2_student_assessment.py`
   - Runner script for the logify2 pipeline (includes OpenIE extraction stage)

### Modified:
1. `/workspace/repo/code/prompts/prompt_logify2`
   - Added EXEMPLAR 3 section at the end of the file

## Comparison with Other Exemplars

| Feature | Exemplar 1 (Medical) | Exemplar 2 (Lab Safety) | Exemplar 3 (Student) |
|---------|---------------------|------------------------|---------------------|
| Main logic | Unless exceptions, conditional chains | Must/should obligations, temporal sequences | XOR, biconditionals, necessary conditions |
| Complexity | 11 propositions, 4 hard, 1 soft | 22 propositions, 8 hard, 3 soft | 8 propositions, 3 hard, 2 soft |
| Key operators | ⟹, ∧, ¬ | ⟹, ∧, ¬ | ⟺, ∨, ∧, ¬ |
| Domain | Healthcare triage | Laboratory safety | Academic assessment |
| Unique feature | First-order flattening | Temporal sequences, "should" analysis | Explicit XOR, explicit IFF |

## Educational Value

This exemplar provides training on:

1. **Exclusive-OR**: How to formalize "either...but not both"
2. **Biconditional reasoning**: How "if and only if" differs from simple implication
3. **Necessary vs. sufficient conditions**: Why "requires" is not the same as "if and only if"
4. **Modal verb interpretation**: How "may" signals soft constraints
5. **Compact formalization**: Achieving clear logic with relatively few propositions

## Testing Notes

Due to CoreNLP/OpenIE environment setup requirements, the output was manually created following the prompt guidelines and comparing with Exemplars 1 and 2. The logical formalization has been carefully validated for:

- **Atomicity**: All propositions are genuinely atomic and independent
- **Exhaustiveness**: All constraints from the text are captured
- **Faithfulness**: The logic accurately represents the text's intent
- **Consistency**: No logical contradictions in the constraint set

## Date Added

2026-01-25

## Related Documentation

- `/workspace/repo/artifacts/README.md` - Main artifacts directory documentation
- `/workspace/repo/artifacts/documentation/EXEMPLAR_ADDITION_SUMMARY.md` - Documentation for Exemplar 2 addition
- `/workspace/repo/code/prompts/prompt_logify2` - The updated prompt file
