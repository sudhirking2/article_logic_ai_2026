# Example 03: Student Assessment Rules

## Overview

This is the third exemplar added to the `prompt_logify2` file, demonstrating the logification of student assessment and course passing rules.

## Files

### Input
- **Location**: `inputs/example_03_student_assessment.txt`
- **Content**: Student assessment rules with XOR logic, biconditionals, and necessary conditions

### Outputs
1. **example_03_student_assessment_output.json** - Full JSON output with propositions and constraints
2. **example_03_for_prompt.txt** - Formatted version with input text, triples, and output (matches prompt format)
3. **example_03_student_assessment_triples.json** - (Not generated yet - requires CoreNLP)
4. **example_03_student_assessment_llm_input.txt** - (Not generated yet - requires CoreNLP)

### Runner Script
- **run_logify2_student_assessment.py** - Script to run the full logify2 pipeline on this example

## Key Features

### Logical Constructs Demonstrated

1. **Exclusive-OR (XOR)**
   - Text: "either...but not both"
   - Formula: `(P_1 ∨ P_2) ∧ ¬(P_1 ∧ P_2) ⟺ P_3`

2. **Biconditional (IFF)**
   - Text: "if and only if"
   - Formula: `P_4 ⟺ (P_3 ∧ P_5)`

3. **Necessary Condition**
   - Text: "requires"
   - Formula: `P_8 ⟹ P_7`

4. **Modal Possibility**
   - Text: "may"
   - Soft constraint: `P_6 ⟹ P_8`

## Input Text

```
Students must complete either the written exam or the oral presentation, but not both, to satisfy the assessment requirement. A student passes the course if and only if they satisfy the assessment requirement and attend at least 80% of lectures. Students may request deadline extensions, though approval requires documented extenuating circumstances.
```

## Statistics

- **Atomic Propositions**: 8
- **Hard Constraints**: 3
- **Soft Constraints**: 2
- **Lines in Prompt**: ~110 (including input and output)

## Comparison with Other Examples

| Metric | Example 01 (Medical) | Example 02 (Lab Safety) | Example 03 (Student) |
|--------|---------------------|------------------------|---------------------|
| Propositions | 11 | 22 | 8 |
| Hard Constraints | 4 | 8 | 3 |
| Soft Constraints | 1 | 3 | 2 |
| Unique Features | Unless-clauses, doctor flattening | Temporal sequences, "should" analysis | XOR, IFF, necessary conditions |

## Usage

### To run the full pipeline (requires OpenAI API key and CoreNLP):

```bash
cd /workspace/repo/artifacts/few_shot_examples
export OPENAI_API_KEY=your_key_here
python run_logify2_student_assessment.py
```

### To use as a reference:

The formatted output in `example_03_for_prompt.txt` has been added to:
- `/workspace/repo/code/prompts/prompt_logify2` (as EXEMPLAR 3)

## Educational Value

This exemplar teaches:
1. How to formalize exclusive-OR relationships
2. How to distinguish biconditionals from simple implications
3. How to identify necessary vs. sufficient conditions
4. How to interpret modal verbs ("may", "must") in logic
5. How to create a clean, compact logical formalization

## Date Added

2026-01-25

## Related Documentation

- `/workspace/repo/artifacts/documentation/EXEMPLAR3_STUDENT_ASSESSMENT_ADDITION.md` - Detailed documentation
- `/workspace/repo/artifacts/README.md` - Main artifacts directory structure
