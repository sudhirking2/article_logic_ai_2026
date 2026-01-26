# Structural Issues and Fixes for baseline_logiclm_plus

## Executive Summary

The code in `/code/baseline_logiclm_plus/` was written by parallel agents without coordination, resulting in **9 critical structural incompatibilities**. This document catalogs each issue and provides the required fix.

---

## Issue 1: OpenAI API Version Mismatch (CRITICAL)

**Severity:** CRITICAL - Will cause immediate runtime crashes

**Location:** `refiner.py` lines 79, 124, 195, 246

**Problem:**
```python
# refiner.py uses OLD OpenAI API (v0.x):
import openai
response = openai.ChatCompletion.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    temperature=temperature,
    max_tokens=MAX_TOKENS
)
```

**But `formalizer.py` uses NEW API (v1.x):**
```python
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(...)
```

**Error that will occur:**
```
AttributeError: module 'openai' has no attribute 'ChatCompletion'
```

**Fix Required:**
Update `refiner.py` to use OpenAI v1.x API consistently:

```python
# At top of refiner.py (line 79)
from openai import OpenAI

# In generate_refinements() (replace line 124):
def generate_refinements(...):
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=MAX_TOKENS
    )
    raw_response = response.choices[0].message.content

# In pairwise_compare() (replace line 195):
def pairwise_compare(...):
    client = OpenAI()
    response = client.chat.completions.create(...)

# In backtracking_decision() (replace line 246):
def backtracking_decision(...):
    client = OpenAI()
    response = client.chat.completions.create(...)
```

---

## Issue 2: Model Name Format Inconsistency

**Severity:** HIGH - API calls will fail

**Location:** `config.py` line 19

**Problem:**
```python
MODEL_NAME = "openai/gpt-4"  # With "openai/" prefix
```

**OpenAI API expects:**
```python
model="gpt-4"  # Without prefix
```

**Error that will occur:**
```
openai.NotFoundError: The model 'openai/gpt-4' does not exist
```

**Fix Required:**
```python
# config.py line 19
MODEL_NAME = "gpt-4"  # Remove "openai/" prefix

# OR strip prefix in formalizer.py and refiner.py:
model_name_clean = model_name.replace("openai/", "")
response = client.chat.completions.create(model=model_name_clean, ...)
```

**Recommended:** Change `config.py` to `MODEL_NAME = "gpt-4"`

---

## Issue 3: Function Signature Mismatch - `config` Dict Not Supported

**Severity:** HIGH - Documentation examples won't work

**Location:** `main.py` lines 129-134

**Problem:**

**Documentation shows (how_to_use_logiclm_plus.md line 139-149):**
```python
result = run_logiclm_plus(
    text=text,
    query=query,
    model_name=MODEL_NAME,
    config={
        'max_iterations': MAX_REFINEMENT_ITERATIONS,
        'num_candidates': 2,
        'max_consecutive_backtracks': 2
    },
    ground_truth=ground_truth
)
```

**Actual implementation:**
```python
def run_logiclm_plus(text, query, model_name=MODEL_NAME, ground_truth=None,
                     max_iterations=MAX_REFINEMENT_ITERATIONS,
                     solver='z3', solver_timeout=SOLVER_TIMEOUT,
                     temperature=TEMPERATURE,
                     num_candidates=NUM_REFINEMENT_CANDIDATES,
                     max_consecutive_backtracks=MAX_CONSECUTIVE_BACKTRACKS):
```

**Fix Required:**

Option A: Update implementation to accept `config` dict:
```python
def run_logiclm_plus(text, query, model_name=MODEL_NAME, ground_truth=None,
                     config=None, **kwargs):
    """
    Args:
        config: Optional dict with keys: max_iterations, num_candidates,
                max_consecutive_backtracks, solver, solver_timeout, temperature
        **kwargs: Individual parameters (override config if both provided)
    """
    # Merge config dict with defaults
    if config is None:
        config = {}

    max_iterations = kwargs.get('max_iterations',
                                config.get('max_iterations', MAX_REFINEMENT_ITERATIONS))
    num_candidates = kwargs.get('num_candidates',
                                config.get('num_candidates', NUM_REFINEMENT_CANDIDATES))
    max_consecutive_backtracks = kwargs.get('max_consecutive_backtracks',
                                           config.get('max_consecutive_backtracks',
                                                     MAX_CONSECUTIVE_BACKTRACKS))
    solver = kwargs.get('solver', config.get('solver', 'z3'))
    solver_timeout = kwargs.get('solver_timeout',
                               config.get('solver_timeout', SOLVER_TIMEOUT))
    temperature = kwargs.get('temperature',
                           config.get('temperature', TEMPERATURE))

    # Rest of function unchanged
```

Option B: Update documentation to match implementation (remove `config` dict examples)

**Recommended:** Option A (accept config dict for API consistency)

---

## Issue 4: `run_batch()` Parameter Mismatch

**Severity:** HIGH - Documentation examples won't work

**Location:** `main.py` line 260

**Problem:**

**Documentation shows:**
```python
run_batch(
    examples=subset,
    model_name='openai/gpt-4',
    config={'max_iterations': 4, 'num_candidates': 2, ...},
    output_dir='test_output'
)
```

**Implementation:**
```python
def run_batch(examples, model_name=MODEL_NAME, max_iterations=MAX_REFINEMENT_ITERATIONS,
              solver='z3', output_dir=None, save_interval=10):
```

**Fix Required:**
Same as Issue 3 - add `config` dict support:

```python
def run_batch(examples, model_name=MODEL_NAME, config=None,
              output_dir=None, save_interval=10, **kwargs):
    """
    Args:
        config: Optional dict with refinement parameters
        **kwargs: Override individual parameters
    """
    if config is None:
        config = {}

    max_iterations = kwargs.get('max_iterations',
                                config.get('max_iterations', MAX_REFINEMENT_ITERATIONS))
    solver = kwargs.get('solver', config.get('solver', 'z3'))

    # Pass to run_logiclm_plus
    for example in examples:
        result = run_logiclm_plus(
            text=example['text'],
            query=example['query'],
            model_name=model_name,
            ground_truth=example.get('ground_truth'),
            config=config,
            **kwargs
        )
```

---

## Issue 5: CLI Parameter Name Mismatch

**Severity:** MEDIUM - Command-line examples won't work

**Location:** `main.py` line 482

**Problem:**

**Documentation shows (line 163):**
```bash
python main.py --dataset folio --output results_folio.json
```

**Implementation (line 482):**
```python
parser.add_argument('--output-dir', type=str, help='Output directory for results')
```

**Fix Required:**

Option A: Add both `--output` and `--output-dir` (for compatibility):
```python
parser.add_argument('--output', type=str, help='Output file path (deprecated, use --output-dir)')
parser.add_argument('--output-dir', type=str, help='Output directory for results')
```

Option B: Update documentation to use `--output-dir`

**Recommended:** Option B (documentation matches implementation)

---

## Issue 6: Dataset Loading - HuggingFace Not Integrated

**Severity:** HIGH - No datasets available

**Location:** `main.py` lines 331-371

**Problem:**

**Documentation claims:**
> Datasets come from HuggingFace (yale-nlp/FOLIO, allenai/proofwriter, allenai/ar-lsat)

**Implementation:**
```python
def load_dataset(dataset_name, data_dir='data'):
    file_path = os.path.join(data_dir, 'folio_test.json')  # Local files only
    with open(file_path, 'r') as f:
        data = json.load(f)
```

**Error that will occur:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/folio_test.json'
```

**Fix Required:**

Integrate HuggingFace `datasets` library:

```python
def load_dataset(dataset_name, data_dir='data', use_huggingface=True):
    """
    Load dataset from HuggingFace or local files.

    Args:
        dataset_name: 'folio', 'proofwriter', or 'ar-lsat'
        data_dir: Local data directory (fallback if use_huggingface=False)
        use_huggingface: If True, load from HuggingFace Hub
    """
    dataset_name = dataset_name.lower()

    if use_huggingface:
        from datasets import load_dataset as hf_load_dataset

        # Map to HuggingFace dataset names
        hf_names = {
            'folio': 'yale-nlp/FOLIO',
            'proofwriter': 'allenai/proofwriter',
            'ar-lsat': 'allenai/ar-lsat'
        }

        if dataset_name not in hf_names:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Load from HuggingFace
        dataset = hf_load_dataset(hf_names[dataset_name], split='test')

        # Normalize format
        examples = []
        for item in dataset:
            example = {
                'text': item.get('premises', item.get('context', '')),
                'query': item.get('conclusion', item.get('question', '')),
                'ground_truth': item.get('label', item.get('answer'))
            }
            examples.append(example)

        return examples

    else:
        # Original local file loading
        file_paths = {
            'folio': os.path.join(data_dir, 'folio_test.json'),
            'proofwriter': os.path.join(data_dir, 'proofwriter_owa_5hop.json'),
            'ar-lsat': os.path.join(data_dir, 'ar_lsat.json')
        }

        # ... rest of original implementation
```

---

## Issue 7: Missing Evaluator Tests

**Severity:** MEDIUM - Incomplete test coverage

**Location:** `test_logiclm.py`

**Problem:**

**Documentation promises (lines 49-55):**
```
Evaluation tests (Tables 1-2, Figure 4):
- test_accuracy_metrics(): Standard classification metrics (Table 1)
- test_execution_rate_Er(): % formulations that execute
- test_execution_accuracy_Ea(): % correct among executed (NOT among all)
- test_backtracking_stats(): Figure 4 metrics (corrected per iteration)
- test_efficiency_metrics(): Time, LLM calls tracking
- test_comparison_to_logic_lm(): Verify improvement over baseline
```

**Implementation:**
None of these tests exist. Only basic tests for solver, formalizer, and integration.

**Fix Required:**

Add evaluator tests to `test_logiclm.py`:

```python
from evaluator import (
    evaluate_predictions,
    compute_logiclm_metrics,
    compute_backtracking_stats,
    compute_efficiency_metrics,
    generate_report
)

def test_accuracy_metrics():
    """Standard classification metrics (Table 1)."""
    print("Running test_accuracy_metrics...")

    predictions = ['True', 'False', 'True', 'Unknown']
    ground_truth = ['True', 'True', 'True', 'Unknown']

    result = evaluate_predictions(predictions, ground_truth)

    assert 'overall_accuracy' in result
    assert 0.0 <= result['overall_accuracy'] <= 1.0
    assert 'per_class' in result
    print(f"  Overall accuracy: {result['overall_accuracy']:.2f}")
    print("  ✓ test_accuracy_metrics passed")


def test_execution_rate_Er():
    """% formulations that execute."""
    print("Running test_execution_rate_Er...")

    results = [
        {'execution_success': True, 'correct': True},
        {'execution_success': True, 'correct': False},
        {'execution_success': False, 'correct': None},
        {'execution_success': True, 'correct': True}
    ]

    metrics = compute_logiclm_metrics(results)

    assert 'execution_rate_Er' in metrics
    assert metrics['execution_rate_Er'] == 0.75  # 3/4
    print(f"  Execution rate: {metrics['execution_rate_Er']:.2f}")
    print("  ✓ test_execution_rate_Er passed")


def test_execution_accuracy_Ea():
    """% correct among executed (NOT among all)."""
    print("Running test_execution_accuracy_Ea...")

    results = [
        {'execution_success': True, 'correct': True},
        {'execution_success': True, 'correct': False},
        {'execution_success': False, 'correct': None},  # Not counted
        {'execution_success': True, 'correct': True}
    ]

    metrics = compute_logiclm_metrics(results)

    assert 'execution_accuracy_Ea' in metrics
    # 2 correct out of 3 executed (NOT 4 total)
    assert abs(metrics['execution_accuracy_Ea'] - 0.6667) < 0.01
    print(f"  Execution accuracy: {metrics['execution_accuracy_Ea']:.2f}")
    print("  ✓ test_execution_accuracy_Ea passed")


# Add these to run_all_tests():
def run_all_tests():
    # ... existing tests ...

    print("\n--- EVALUATOR TESTS ---")
    test_accuracy_metrics()
    test_execution_rate_Er()
    test_execution_accuracy_Ea()
```

---

## Issue 8: Unused Imports

**Severity:** LOW - Code smell, no runtime impact

**Location:** `evaluator.py` line 108

**Problem:**
```python
from collections import defaultdict  # Imported but never used
```

**Fix Required:**
Remove unused import:
```python
import statistics
# from collections import defaultdict  # Remove - not used
```

---

## Issue 9: Config Import Circular Dependency Risk

**Severity:** LOW - Potential issue

**Location:** Multiple files

**Problem:**

`evaluator.py` line 223:
```python
from config import MAX_REFINEMENT_ITERATIONS
```

`evaluator.py` line 249:
```python
from config import MAX_REFINEMENT_ITERATIONS
```

Importing inside functions can cause issues if `config` is modified at runtime.

**Fix Required:**

Import at module level:
```python
# At top of evaluator.py
import statistics
from config import MAX_REFINEMENT_ITERATIONS

# Remove internal imports in functions
```

---

## Summary of Fixes Priority

| Priority | Issue | Files to Change | Complexity |
|----------|-------|-----------------|------------|
| **P0 (CRITICAL)** | #1: OpenAI API mismatch | `refiner.py` | Medium |
| **P0 (CRITICAL)** | #2: Model name format | `config.py` or all files | Low |
| **P1 (HIGH)** | #3: config dict in run_logiclm_plus | `main.py` | Medium |
| **P1 (HIGH)** | #4: config dict in run_batch | `main.py` | Medium |
| **P1 (HIGH)** | #6: HuggingFace datasets | `main.py` | High |
| **P2 (MEDIUM)** | #5: CLI parameter names | `main.py` or docs | Low |
| **P2 (MEDIUM)** | #7: Evaluator tests | `test_logiclm.py` | Medium |
| **P3 (LOW)** | #8: Unused imports | `evaluator.py` | Trivial |
| **P3 (LOW)** | #9: Import location | `evaluator.py` | Trivial |

---

## Recommended Fix Order

1. **Fix #2 first** (model name) - simplest, affects all API calls
2. **Fix #1** (OpenAI API) - critical runtime crash
3. **Fix #3 and #4** (config dict) - API consistency
4. **Fix #6** (datasets) - enables actual usage
5. **Fix #7** (tests) - complete test coverage
6. **Fix #5, #8, #9** (cleanup) - polish

---

## Testing After Fixes

After implementing fixes, verify with:

```bash
# 1. Test imports
python -c "from refiner import refine_loop; from formalizer import formalize_to_fol"

# 2. Test config dict support
python -c "from main import run_logiclm_plus; print(run_logiclm_plus.__doc__)"

# 3. Run unit tests
python test_logiclm.py

# 4. Test HuggingFace loading
python -c "from main import load_dataset; examples = load_dataset('folio'); print(len(examples))"

# 5. Test CLI
python main.py --help
```

---

## Root Cause Analysis

**Why did this happen?**

1. **Parallel development** without API contracts
2. **No integration testing** between modules
3. **Documentation written before implementation**
4. **Mixed OpenAI SDK versions** (v0.x vs v1.x knowledge)
5. **No central design authority** for API decisions

**Lessons learned:**

- Define API contracts before parallel work
- Use integration tests, not just unit tests
- Version lock external dependencies (OpenAI SDK)
- Documentation should match implementation (or vice versa)
- Have one person own API consistency

---

## Compatibility Matrix

| Module | OpenAI API | Config Dict | Model Name | Datasets |
|--------|------------|-------------|------------|----------|
| `config.py` | - | - | ❌ v0.x format | - |
| `formalizer.py` | ✅ v1.x | ✅ Individual params | ✅ Accepts both | - |
| `refiner.py` | ❌ v0.x | ✅ Individual params | ✅ Accepts both | - |
| `solver_interface.py` | - | ✅ Individual params | - | - |
| `main.py` | - | ❌ No config dict | ✅ Accepts both | ❌ Local only |
| `evaluator.py` | - | - | - | - |
| `test_logiclm.py` | - | - | - | - |

✅ = Working correctly
❌ = Needs fix
- = Not applicable
