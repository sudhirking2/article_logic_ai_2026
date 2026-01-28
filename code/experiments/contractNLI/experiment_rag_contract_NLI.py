#!/usr/bin/env python3
"""
experiment_rag_contract_NLI.py

Experiment: Evaluate RAG baseline on ContractNLI dataset.

This script runs the Reasoning LLM + RAG baseline on ContractNLI,
using the same data source, evaluation structure, and output format
as the Logify experiment for fair comparison.

Pipeline:
    For each document in ContractNLI:
        1. Chunk document into overlapping segments
        2. Encode chunks using SBERT
        3. For each of 17 hypotheses:
            a. Retrieve top-k relevant chunks
            b. Perform Chain-of-Thought reasoning with LLM
            c. Parse response to extract prediction and confidence
            d. Store result
        4. Save intermediate results

Output format matches experiment_logify_contract_NLI.py exactly.

Usage:
    python experiment_rag_contract_NLI.py --dataset-path contract-nli/dev.json
    python experiment_rag_contract_NLI.py --dataset-path contract-nli/dev.json --num-docs 5
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add code directory to Python path
_script_dir = Path(__file__).resolve().parent
_code_dir = _script_dir.parent.parent
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))

# Import baseline_rag modules
from baseline_rag.chunker import chunk_document
from baseline_rag.retriever import (
    load_sbert_model,
    encode_chunks,
    encode_query,
    retrieve
)
from baseline_rag import config as rag_config

# Results directory
RESULTS_DIR = _script_dir / "results_rag_contract_NLI"


# =============================================================================
# ContractNLI-specific Chain-of-Thought Prompt Template
# =============================================================================

CONTRACTNLI_COT_PROMPT = """You are a legal contract analyst. Given excerpts from a contract and a hypothesis, determine whether the hypothesis is supported by the contract.

**Contract Excerpts:**
{context}

**Hypothesis:** {hypothesis}

**Instructions:**
1. Carefully read the contract excerpts provided above
2. Determine if the hypothesis is:
   - TRUE: The contract clearly states or logically entails this hypothesis
   - FALSE: The contract clearly contradicts this hypothesis
   - UNCERTAIN: The contract does not mention or address this hypothesis
3. Provide your confidence level as a number between 0.0 and 1.0:
   - 1.0 = Absolutely certain (explicit statement in contract)
   - 0.7-0.9 = High confidence (strong implication)
   - 0.4-0.6 = Moderate confidence (some evidence but not conclusive)
   - 0.1-0.3 = Low confidence (weak or indirect evidence)
   - 0.0 = No support for this conclusion

**Format your response exactly as follows:**
**Reasoning:** [Your step-by-step analysis of the contract excerpts]
**Answer:** [TRUE or FALSE or UNCERTAIN]
**Confidence:** [A number between 0.0 and 1.0]

Begin your analysis:"""


# =============================================================================
# Data Loading (same as Logify experiment)
# =============================================================================

def load_contractnli_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Load ContractNLI dataset from local JSON file.

    Args:
        dataset_path: Path to ContractNLI JSON file (e.g., dev.json)

    Returns:
        Dictionary containing 'documents' and 'labels' keys
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_ground_truth_label(choice: str) -> str:
    """
    Map ContractNLI label to experiment output format.

    Mapping:
        Entailment -> TRUE
        Contradiction -> FALSE
        NotMentioned -> UNCERTAIN

    Args:
        choice: ContractNLI label string

    Returns:
        Mapped label (TRUE, FALSE, or UNCERTAIN)
    """
    mapping = {
        "Entailment": "TRUE",
        "Contradiction": "FALSE",
        "NotMentioned": "UNCERTAIN"
    }
    return mapping.get(choice, "UNCERTAIN")


# =============================================================================
# Response Parsing
# =============================================================================

def parse_rag_response(response: str) -> Dict[str, Any]:
    """
    Parse LLM response to extract answer and confidence.

    Extracts:
        - Answer: TRUE, FALSE, or UNCERTAIN
        - Confidence: float between 0 and 1 (default 0.5 if not found)

    Args:
        response: Raw LLM response string

    Returns:
        Dictionary with 'answer', 'confidence', and 'reasoning' keys
    """
    import re

    answer = None
    confidence = 0.5  # Default confidence
    reasoning = response

    # Extract answer from **Answer:** section
    answer_match = re.search(r'\*\*Answer:\*\*\s*(TRUE|FALSE|UNCERTAIN)', response, re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).upper()
    else:
        # Fallback: search for keywords in the response
        response_upper = response.upper()
        if 'UNCERTAIN' in response_upper:
            answer = 'UNCERTAIN'
        elif 'TRUE' in response_upper:
            answer = 'TRUE'
        elif 'FALSE' in response_upper:
            answer = 'FALSE'
        else:
            answer = 'UNCERTAIN'  # Default if nothing found

    # Extract confidence from **Confidence:** section
    confidence_match = re.search(r'\*\*Confidence:\*\*\s*([\d.]+)', response)
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
            # Clamp to [0, 1]
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            confidence = 0.5
    else:
        # Fallback: search for any decimal number after "confidence"
        fallback_match = re.search(r'confidence[:\s]+(\d*\.?\d+)', response, re.IGNORECASE)
        if fallback_match:
            try:
                confidence = float(fallback_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5

    # Extract reasoning from **Reasoning:** section
    reasoning_match = re.search(r'\*\*Reasoning:\*\*\s*(.*?)(?=\*\*Answer:|\Z)', response, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    return {
        'answer': answer,
        'confidence': confidence,
        'reasoning': reasoning
    }


# =============================================================================
# LLM Interaction
# =============================================================================

def call_llm(prompt: str, model_name: str, temperature: float = 0) -> str:
    """
    Call the language model API with the constructed prompt.

    Uses OpenRouter API with OPENROUTER_API_KEY environment variable.

    Args:
        prompt: Complete prompt string
        model_name: Name of the model (e.g., "openai/gpt-5-nano")
        temperature: Sampling temperature (0 for deterministic)

    Returns:
        Raw string response from the LLM
    """
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    return response.choices[0].message.content


def construct_prompt(hypothesis: str, retrieved_chunks: List[Dict]) -> str:
    """
    Construct the full prompt from template, hypothesis, and retrieved chunks.

    Args:
        hypothesis: The hypothesis text to evaluate
        retrieved_chunks: List of retrieved chunk dictionaries with 'text' field

    Returns:
        Formatted prompt string ready for LLM
    """
    # Format chunks with separators
    formatted_chunks = []
    for i, chunk in enumerate(retrieved_chunks):
        formatted_chunks.append(f"[Excerpt {i+1}]\n{chunk['text']}")

    context = "\n\n".join(formatted_chunks)

    # Fill in the template
    prompt = CONTRACTNLI_COT_PROMPT.format(
        context=context,
        hypothesis=hypothesis
    )

    return prompt


# =============================================================================
# Single Query Processing
# =============================================================================

def process_single_hypothesis(
    hypothesis_text: str,
    chunk_embeddings,
    chunks: List[Dict],
    sbert_model,
    model_name: str,
    temperature: float
) -> Dict[str, Any]:
    """
    Process a single hypothesis against pre-computed document chunks.

    Pipeline:
        1. Encode hypothesis as query
        2. Retrieve top-k relevant chunks
        3. Construct prompt with retrieved context
        4. Call LLM for reasoning
        5. Parse response for answer and confidence

    Args:
        hypothesis_text: The hypothesis to evaluate
        chunk_embeddings: Pre-computed chunk embeddings (numpy array)
        chunks: List of chunk dictionaries from chunker
        sbert_model: Loaded SBERT model
        model_name: LLM model name
        temperature: Sampling temperature

    Returns:
        Dictionary with 'prediction', 'confidence', 'latency_sec', 'error'
    """
    start_time = time.time()

    try:
        # Step 1: Encode hypothesis as query
        query_embedding = encode_query(hypothesis_text, sbert_model)

        # Step 2: Retrieve top-k relevant chunks
        retrieved_chunks = retrieve(
            query_embedding,
            chunk_embeddings,
            chunks,
            k=rag_config.TOP_K
        )

        # Step 3: Construct prompt with retrieved context
        prompt = construct_prompt(hypothesis_text, retrieved_chunks)

        # Step 4: Call LLM for reasoning
        raw_response = call_llm(prompt, model_name, temperature)

        # Step 5: Parse response for answer and confidence
        parsed = parse_rag_response(raw_response)

        return {
            "prediction": parsed['answer'],
            "confidence": parsed['confidence'],
            "query_latency_sec": time.time() - start_time,
            "error": None
        }

    except Exception as e:
        return {
            "prediction": None,
            "confidence": None,
            "query_latency_sec": time.time() - start_time,
            "error": str(e)
        }


# =============================================================================
# Document Processing
# =============================================================================

def process_document(
    doc_text: str,
    sbert_model
) -> tuple:
    """
    Chunk and encode a document for retrieval.

    Args:
        doc_text: Full document text
        sbert_model: Loaded SBERT model

    Returns:
        Tuple of (chunks, chunk_embeddings)
    """
    # Chunk the document using config settings
    chunks = chunk_document(
        doc_text,
        chunk_size=rag_config.CHUNK_SIZE,
        overlap=rag_config.OVERLAP
    )

    # Encode chunks
    chunk_embeddings = encode_chunks(chunks, sbert_model)

    return chunks, chunk_embeddings


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    dataset_path: str,
    model_name: str = None,
    temperature: float = 0,
    num_docs: int = 20
) -> Dict[str, Any]:
    """
    Run the ContractNLI RAG baseline experiment.

    Args:
        dataset_path: Path to ContractNLI JSON file
        model_name: LLM model name (default: from rag_config)
        temperature: Sampling temperature (default: 0)
        num_docs: Number of documents to process (default: 20)

    Returns:
        Experiment results dictionary matching Logify output format
    """
    pass


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """
    Command-line entry point.

    Parses arguments and runs the experiment.
    """
    pass


if __name__ == "__main__":
    sys.exit(main())
