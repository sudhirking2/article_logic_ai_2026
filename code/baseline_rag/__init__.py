"""
baseline_rag - RAG (Retrieval-Augmented Generation) baseline implementation.

This package provides chunking and retrieval utilities for document processing.

Modules:
    chunker: Document chunking with overlapping windows
    retriever: SBERT-based semantic retrieval
    config: Configuration settings
    evaluator: Evaluation metrics
    reasoner: LLM reasoning module
"""

from baseline_rag.chunker import chunk_document, tokenize, detokenize
from baseline_rag.retriever import (
    load_sbert_model,
    encode_chunks,
    encode_query,
    retrieve,
    compute_cosine_similarity
)

__all__ = [
    'chunk_document',
    'tokenize',
    'detokenize',
    'load_sbert_model',
    'encode_chunks',
    'encode_query',
    'retrieve',
    'compute_cosine_similarity',
]
