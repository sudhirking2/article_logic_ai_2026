# LogicBench Experiment with Logify

## Overview

This experiment evaluates the Logify neuro-symbolic pipeline on the LogicBench dataset (BQA task).

## Pipeline

For each LogicBench sample:
1. **Logify**: Convert context (premises) to weighted propositional logic
2. **Query**: For each QA pair, translate question and solve with MaxSAT
3. **Compare**: Match solver output (TRUE/FALSE/UNCERTAIN) against ground truth (yes/no)

## Dataset

- **Source**: LogicBench(Eval) / BQA
- **Logic types**: propositional_logic, first_order_logic, nm_logic
- **Total patterns**: 25 (8 + 9 + 8)

### Patterns by Logic Type

**Propositional Logic (8)**:
- bidirectional_dilemma, commutation, constructive_dilemma, destructive_dilemma
- disjunctive_syllogism, hypothetical_syllogism, material_implication, modus_tollens

**First-Order Logic (9)**:
- bidirectional_dilemma, constructive_dilemma, destructive_dilemma, disjunctive_syllogism
- existential_generalization, hypothetical_syllogism, modus_ponens, modus_tollens
- universal_instantiation

**Non-Monotonic Logic (8)**:
- default_reasoning_default, default_reasoning_irr, default_reasoning_open
- default_reasoning_several, reasoning_about_exceptions_1, reasoning_about_exceptions_2
- reasoning_about_exceptions_3, reasoning_about_priority

## Directory Structure

```
code/experiments/logicBench/
├── experiment_logify_logicBench.py      # Main experiment script
├── DESCRIPTION_EXPERIMENT_LOGICBENCH_LOGIFY.md  # This file
├── cache/                                # Cached logified structures
│   └── doc_{id}_weighted.json
└── results_logify_LOGICBENCH/           # Experiment outputs
    └── experiment_YYYYMMDD_HHMMSS.json
```

## Output Format

Each experiment produces a JSON file with an array of sample results:

```json
{
  "id": "1",
  "text": "Liam had finished his work...",
  "logic_type": "propositional_logic",
  "pattern": "modus_tollens",
  "logify_latency_sec": 12.3,
  "logify_cached": false,
  "logify_error": null,
  "questions": [
    {
      "query": "Did Liam finish his work early?",
      "predicted_answer": "TRUE",
      "confidence": 0.95,
      "ground_truth": "no",
      "query_latency_total_sec": 2.1,
      "query_error": null
    }
  ]
}
```

## Fields

### Sample-level
| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique sample identifier |
| text | string | Context/premises from LogicBench |
| logic_type | string | propositional_logic, first_order_logic, or nm_logic |
| pattern | string | Reasoning pattern name |
| logify_latency_sec | float | Time to run logify (0 if cached) |
| logify_cached | bool | Whether cached result was used |
| logify_error | string/null | Error message if logify failed |

### Question-level
| Field | Type | Description |
|-------|------|-------------|
| query | string | Question from LogicBench |
| predicted_answer | string | Solver output: TRUE, FALSE, or UNCERTAIN |
| confidence | float | Solver confidence [0, 1] |
| ground_truth | string | Expected answer: yes or no |
| query_latency_total_sec | float | Time to translate + solve |
| query_error | string/null | Error message if query failed |

## Usage

```bash
# Run on all patterns (default)
python experiment_logify_logicBench.py --api_key $OPENAI_API_KEY

# Run on specific logic type
python experiment_logify_logicBench.py --api_key $OPENAI_API_KEY --logic_type propositional_logic

# Run on specific patterns with sample limit
python experiment_logify_logicBench.py --api_key $OPENAI_API_KEY --patterns modus_tollens,disjunctive_syllogism --max_samples 10
```

## Metrics (for analysis)

- **Accuracy**: % of questions where predicted_answer matches ground_truth
- **Accuracy by logic type**: Breakdown by propositional/FOL/nm_logic
- **Accuracy by pattern**: Breakdown by 25 patterns
- **Error rate**: % of samples with logify_error or query_error
- **Latency**: Mean/median logify and query times
- **Confidence calibration**: Correlation between confidence and correctness

## Answer Mapping

| Solver Output | Ground Truth Match |
|---------------|-------------------|
| TRUE | yes |
| FALSE | no |
| UNCERTAIN | neither (always wrong) |
