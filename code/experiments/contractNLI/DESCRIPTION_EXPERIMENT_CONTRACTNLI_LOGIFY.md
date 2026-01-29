# Experiment: Logify on ContractNLI

## Objective

Evaluate whether Logify can correctly classify document-hypothesis pairs from the ContractNLI dataset.

## Dataset

**ContractNLI** (Koreeda & Manning, EMNLP Findings 2021)
- Source: https://stanfordnlp.github.io/contract-nli/
- 607 Non-Disclosure Agreements (NDAs)
- 17 fixed hypotheses per document
- Labels: `Entailment`, `Contradiction`, `NotMentioned`
- Evidence: Character-level spans (stored but not used in this experiment)

## Experimental Design

### Pipeline

```
For each document in ContractNLI (first 20 documents):
    1. Logify(document.text) → cached weighted JSON
    2. For each of 17 hypotheses:
        a. Translate(hypothesis) → (formula, query_mode)
        b. If query_mode == "consistency":
              Solve via check_consistency(formula)  # "May X?" queries
           Else:
              Solve via query(formula)              # "Shall X?" queries
        c. Extract ground truth and evidence count from annotations
        d. Store result
```

### Query Modes

The translator detects the semantic type of each hypothesis:

| Query Mode   | Hypothesis Keywords                          | Solver Method         | Semantics                    |
|--------------|----------------------------------------------|----------------------|------------------------------|
| `entailment` | "shall", "must", "is required", "shall not"  | `solver.query()`      | Is X necessarily true?       |
| `consistency`| "may", "can", "could", "is allowed"          | `solver.check_consistency()` | Is X possible/permitted? |

This distinction is critical: "may" questions ask about *permission* (is it allowed?), not *obligation* (is it required?). Using entailment for permission queries incorrectly returns UNCERTAIN.

### Label Mapping

| ContractNLI Label | Expected Logify Output | Confidence Expectation |
|-------------------|------------------------|------------------------|
| Entailment        | TRUE                   | High (→ 1.0)           |
| Contradiction     | FALSE                  | High (→ 1.0)           |
| NotMentioned      | UNCERTAIN              | Neutral (≈ 0.5)        |

### Output Schema

Each result record contains:

```json
{
  "doc_id": <int>,
  "hypothesis_key": "<string>",
  "hypothesis_text": "<string>",
  "prediction": "TRUE|FALSE|UNCERTAIN",
  "confidence": <float 0.0-1.0>,
  "ground_truth": "TRUE|FALSE|UNCERTAIN",
  "amount_evidence": <int>,
  "formula": "<propositional formula>",
  "query_mode": "entailment|consistency",
  "error": null | "<error message>"
}
```

### Metadata

```json
{
  "metadata": {
    "timestamp": "<ISO 8601>",
    "model": "gpt-5-nano",
    "num_documents": 20,
    "num_pairs": 340
  },
  "results": [...]
}
```

## Design Choices

### 1. No Train/Dev/Test Split

Logify is not a learned model—it's a reasoning pipeline. The splits in ContractNLI are irrelevant for this evaluation. We run on all documents (starting with 20 for validation).

### 2. Evidence Not Used

ContractNLI provides character-level evidence spans. We store `amount_evidence` (count of spans) for potential stratified analysis, but do not evaluate span prediction.

### 3. Intermediate Results

Results are saved incrementally (per document) to allow resumption if interrupted. Format: JSONL (one JSON object per line) for robustness.

### 4. Error Handling

Errors are logged inline with `"error": "<message>"` and `"prediction": null`. The experiment continues to the next hypothesis/document.

### 5. Token Usage and Latency Tracking

Per-document metrics:
- `logify_tokens`: Tokens used during logification
- `logify_latency_sec`: Time for logification
- `query_tokens_total`: Total tokens for all 17 hypothesis queries
- `query_latency_total_sec`: Total time for all queries

### 6. Caching

Logified documents are cached in `cache/doc_{id}_weighted.json`. If a cached file exists, logification is skipped and the cached version is loaded.

### 7. Default Model

`gpt-5-nano` — balances cost and capability for this evaluation.

## Evaluation Metrics (Post-Experiment)

1. **3-Class Accuracy**: % correct across all (TRUE, FALSE, UNCERTAIN)
2. **Binary Accuracy**: % correct on Entailment/Contradiction pairs only
3. **Confusion Matrix**: 3x3 matrix of predicted vs. ground truth
4. **Confidence Calibration**: Mean confidence per ground truth class
5. **Stratified Analysis**: Accuracy vs. `amount_evidence`

## File Structure

```
code/experiments/contractNLI/
├── experiment_logify_contract_NLI.py      # Main experiment script
├── DESCRIPTION_EXPERIMENT_CONTRACTNLI_LOGIFY.md  # This file
├── cache/                                  # Cached logified documents
│   └── doc_{id}_weighted.json
└── results_logify_contract_NLI/           # Experiment outputs
    └── experiment_YYYYMMDD_HHMMSS.json
```

## References

- Koreeda, Y., & Manning, C. D. (2021). ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts. *Findings of EMNLP 2021*.
- Dataset: https://stanfordnlp.github.io/contract-nli/
- Baseline code: https://github.com/stanfordnlp/contract-nli-bert
