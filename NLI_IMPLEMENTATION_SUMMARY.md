# NLI Cross-Encoder Implementation Summary

## What Was Implemented

Three new/modified files to add NLI-based semantic filtering.

### 1. `/code/config/retrieval_config.py` (NEW)
Centralized configuration for all retrieval parameters.

Key settings:
- `SBERT_TOP_K = 50` (increased from 20)
- `SBERT_MIN_SIMILARITY = 0.3`
- `NLI_MODEL = "cross-encoder/nli-deberta-v3-large"`
- `ENABLE_NLI_FILTERING = True` (feature flag)

### 2. `/code/baseline_rag/nli_reranker.py` (NEW)
NLI cross-encoder filtering functions.

Functions:
- `load_nli_model()` - Load cross-encoder
- `score_nli_pairs()` - Batch score pairs
- `filter_propositions_by_nli()` - Filter by scores

### 3. `/code/interface_with_user/translate.py` (MODIFIED)
Integrated two-stage retrieval.

Changes:
- Hybrid embedding (translation + evidence)
- Two-stage pipeline (SBERT + NLI)
- Handles empty retrieval → UNCERTAIN

## Two-Stage Pipeline

```
Stage 1: SBERT (fast, broad recall)
   ↓ Retrieve top-50 by cosine similarity
   ↓ Filter by min_similarity > 0.3
   ↓
Stage 2: NLI Cross-Encoder (semantic precision)
   ↓ Score (evidence, hypothesis) pairs
   ↓ Keep only entailing/contradicting
   ↓
Result: Grounded propositions or empty → UNCERTAIN
```

## Impact

**Problem**: 75% false positive rate on UNCERTAIN cases

**Solution**: NLI filters out neutral propositions (document silent)

**Expected**: Reduces 75% FP to 15-20%

## Dependencies

```bash
pip install sentence-transformers
```

Models: `all-MiniLM-L6-v2` (~80MB), `cross-encoder/nli-deberta-v3-large` (~1.5GB)
