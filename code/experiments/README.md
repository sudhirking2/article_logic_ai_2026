# Baseline Experiments

This directory contains the infrastructure for running baseline experiments on benchmark datasets.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

## Quick Start

### Test a Single Method on Small Sample

Test the Direct baseline on 10 FOLIO examples:
```bash
python run_experiments.py \
  --dataset folio \
  --methods direct \
  --limit 10
```

### Run All Baselines on One Dataset

Run all 4 baselines on FOLIO validation set:
```bash
python run_experiments.py \
  --dataset folio \
  --split validation
```

### Run Full Experiments (All Datasets)

Run all baselines on all datasets:
```bash
python run_experiments.py \
  --dataset all \
  --split test
```

**Warning**: This will make many API calls and may take several hours. Consider:
- Starting with `--limit 100` for testing
- Using `--methods direct cot` to test faster methods first
- Running datasets separately to checkpoint progress

### Generate Comparison Table

After running experiments, generate the comparison table:
```bash
python run_experiments.py --generate-table
```

## Baseline Methods

### 1. Direct
Standard GPT-4 prompting without chain-of-thought or retrieval.

```python
from baselines import DirectBaseline

baseline = DirectBaseline(api_key="...")
result = baseline.predict(text="...", question="...")
print(result.prediction)  # "True", "False", or "Unknown"
```

### 2. CoT (Chain-of-Thought)
GPT-4 with step-by-step reasoning prompts.

```python
from baselines import CoTBaseline

baseline = CoTBaseline(api_key="...")
result = baseline.predict(text="...", question="...")
print(result.prediction)
print(result.reasoning)  # Full reasoning chain
```

### 3. RAG (Retrieval-Augmented Generation)
Retrieves top-k relevant passages, then queries GPT-4 on retrieved context.

```python
from baselines import RAGBaseline

baseline = RAGBaseline(api_key="...", top_k=5)
result = baseline.predict(text="...", question="...")
```

### 4. Logic-LM
Per-query formalization: converts text and question to logic for each query.

```python
from baselines import LogicLMBaseline

baseline = LogicLMBaseline(api_key="...")
result = baseline.predict(text="...", question="...")
```

## Datasets

### FOLIO
- **Paper**: "FOLIO: Natural Language Reasoning with First-Order Logic"
- **Size**: 1,430 examples
- **Format**: Premises + Conclusion → True/False/Unknown
- **Download**: Automatic from HuggingFace

### ProofWriter
- **Paper**: "ProofWriter: Generating Implications via Iterative Forward Reasoning"
- **Size**: ~600 examples (depth-5 subset)
- **Format**: Theory + Question → True/False
- **Download**: Automatic from HuggingFace

### ContractNLI
- **Paper**: "ContractNLI: A Dataset for Document-level Natural Language Inference"
- **Size**: 607 NDAs with 17 hypothesis types
- **Format**: Contract + Hypothesis → Entailment/Contradiction/NotMentioned
- **Download**: Automatic from HuggingFace

## Output Structure

Results are saved in `results/baselines/` with the following structure:

```
results/baselines/
├── folio/
│   └── test/
│       ├── direct_results.json
│       ├── cot_results.json
│       ├── rag_results.json
│       ├── logic-lm_results.json
│       └── summary.json
├── proofwriter/
│   └── test/
│       └── ...
├── contractnli/
│   └── test/
│       └── ...
└── comparison_table.md
```

### Result Format

Each method's results file contains:

```json
{
  "method": "direct",
  "model": "gpt-4",
  "total_examples": 100,
  "correct": 85,
  "accuracy": 85.0,
  "results": [
    {
      "example_id": "folio_001",
      "prediction": "True",
      "ground_truth": "True",
      "correct": true,
      "execution_time": 1.23,
      "metadata": {...}
    },
    ...
  ]
}
```

## Advanced Usage

### Custom Model

Use a different model (e.g., GPT-3.5):
```bash
python run_experiments.py \
  --dataset folio \
  --model gpt-3.5-turbo
```

### Specific Methods Only

Run only Direct and CoT:
```bash
python run_experiments.py \
  --dataset all \
  --methods direct cot
```

### Different Split

Use validation split instead of test:
```bash
python run_experiments.py \
  --dataset folio \
  --split validation
```

### Custom Output Directory

```bash
python run_experiments.py \
  --dataset folio \
  --output-dir my_results/
```

## Testing Dataset Loading

Test dataset loading without running experiments:

```bash
# Test FOLIO
python datasets.py --dataset folio --split validation

# Test all datasets
python datasets.py --dataset all
```

## Cost Estimation

Rough API cost estimates (using GPT-4 at $0.03/1K input + $0.06/1K output):

| Dataset | Examples | Avg Length | Cost per Method |
|---------|----------|------------|-----------------|
| FOLIO | 1,430 | 200 tokens | ~$15-20 |
| ProofWriter | 600 | 300 tokens | ~$10-15 |
| ContractNLI | 607 | 2,000 tokens | ~$40-60 |

**Total cost for all 4 baselines on all 3 datasets**: ~$260-380

### Cost Reduction Strategies

1. **Use validation splits** (smaller):
   ```bash
   --split validation
   ```

2. **Test with limits first**:
   ```bash
   --limit 50  # Test on 50 examples
   ```

3. **Use GPT-3.5** (10x cheaper):
   ```bash
   --model gpt-3.5-turbo
   ```

4. **Run subsets**:
   ```bash
   --methods direct cot  # Skip RAG and Logic-LM
   ```

## Troubleshooting

### Rate Limits

If you hit rate limits, the script has built-in 0.5s delays. For stricter limits:

```python
# In run_experiments.py, increase sleep time:
time.sleep(1.0)  # Instead of 0.5
```

### Dataset Download Failures

If automatic download fails:

1. Install datasets library: `pip install datasets`
2. Download manually:
   ```python
   from datasets import load_dataset
   load_dataset("yale-nlp/folio")
   ```

### API Errors

Common issues:
- **Invalid API key**: Check `OPENAI_API_KEY` environment variable
- **Model not available**: Use `--model gpt-3.5-turbo` if GPT-4 access is limited
- **Context length exceeded**: Some ContractNLI documents are very long; this is expected

## Next Steps

After running baseline experiments:

1. **Implement Logify system**: Complete the core modules in `/code/from_text_to_logic/` and `/code/logic_solver/`

2. **Add Logify to experiments**: Create `logify_baseline.py` that uses your system

3. **Compare results**: Run comparison table to see Logify vs. baselines

4. **Generate paper tables**: Use results to populate Tables 1-7 in your paper

5. **Run ablation studies**: Implement depth analysis, length analysis, etc.

## Paper Tables Mapping

The experiments in this directory generate data for:

- **Table 1** (Main Results): Direct comparison of accuracy across datasets
- **Table 6** (Execution Rate): Track parsing/execution success rate
- **Table 7** (Incremental Updates): Timing comparisons

Additional tables require:
- **Table 2** (Depth Analysis): Filter ProofWriter by depth metadata
- **Table 3** (Document Length): Filter ContractNLI by `doc_length` metadata
- **Table 4** (Soft Constraints): Requires MedGuide dataset (not included here)
- **Table 5** (Ablations): Requires Logify system variants
