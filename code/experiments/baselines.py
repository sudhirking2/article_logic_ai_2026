#!/usr/bin/env python3
"""
baselines.py - Baseline method implementations for benchmark experiments

Implements:
1. Direct: GPT-4 with standard prompting
2. CoT: GPT-4 with chain-of-thought prompting
3. RAG: Retrieval-augmented generation
4. Logic-LM: Neuro-symbolic baseline (per-query formalization)
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
from openai import OpenAI


@dataclass
class BaselineResult:
    """Result from a baseline method."""
    prediction: str  # "True", "False", or "Unknown"
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DirectBaseline:
    """
    Direct prompting baseline.

    Simply asks GPT-4 to answer the question given the text.
    No chain-of-thought, no retrieval, no symbolic reasoning.
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def predict(self, text: str, question: str) -> BaselineResult:
        """Make a prediction using direct prompting."""
        start_time = time.time()

        prompt = f"""Given the following text, answer the question.

TEXT:
{text}

QUESTION: {question}

Respond with ONLY one of: True, False, or Unknown.
Do not provide any explanation.

ANSWER:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a logical reasoning assistant. Answer questions based strictly on the given text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )

            prediction = response.choices[0].message.content.strip()

            # Normalize answer
            prediction = self._normalize_answer(prediction)

            execution_time = time.time() - start_time

            return BaselineResult(
                prediction=prediction,
                execution_time=execution_time,
                metadata={"method": "direct"}
            )

        except Exception as e:
            return BaselineResult(
                prediction="Unknown",
                execution_time=time.time() - start_time,
                metadata={"method": "direct", "error": str(e)}
            )

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer to standard format."""
        answer = answer.strip().lower()

        if "true" in answer:
            return "True"
        elif "false" in answer:
            return "False"
        else:
            return "Unknown"


class CoTBaseline:
    """
    Chain-of-Thought prompting baseline.

    Asks GPT-4 to reason step-by-step before answering.
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def predict(self, text: str, question: str) -> BaselineResult:
        """Make a prediction using chain-of-thought prompting."""
        start_time = time.time()

        prompt = f"""Given the following text, answer the question using step-by-step reasoning.

TEXT:
{text}

QUESTION: {question}

Let's think step by step:
1. First, identify the relevant information from the text
2. Then, reason through the logical connections
3. Finally, determine if the question is True, False, or Unknown

Provide your step-by-step reasoning, then end with:
FINAL ANSWER: [True/False/Unknown]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a logical reasoning assistant. Use chain-of-thought reasoning to answer questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )

            response_text = response.choices[0].message.content.strip()

            # Extract final answer
            prediction = self._extract_final_answer(response_text)

            execution_time = time.time() - start_time

            return BaselineResult(
                prediction=prediction,
                reasoning=response_text,
                execution_time=execution_time,
                metadata={"method": "cot"}
            )

        except Exception as e:
            return BaselineResult(
                prediction="Unknown",
                execution_time=time.time() - start_time,
                metadata={"method": "cot", "error": str(e)}
            )

    def _extract_final_answer(self, text: str) -> str:
        """Extract final answer from CoT response."""
        text = text.lower()

        # Look for "final answer:" pattern
        if "final answer:" in text:
            answer_part = text.split("final answer:")[-1].strip()
            if "true" in answer_part[:20]:
                return "True"
            elif "false" in answer_part[:20]:
                return "False"
            else:
                return "Unknown"

        # Fallback: look for answer patterns
        if "true" in text:
            return "True"
        elif "false" in text:
            return "False"
        else:
            return "Unknown"


class RAGBaseline:
    """
    Retrieval-Augmented Generation baseline.

    Retrieves relevant passages using semantic similarity,
    then uses GPT-4 to answer based on retrieved context.
    """

    def __init__(self, api_key: str, model: str = "gpt-4", top_k: int = 5):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.top_k = top_k

    def predict(self, text: str, question: str) -> BaselineResult:
        """Make a prediction using RAG."""
        start_time = time.time()

        # Step 1: Chunk the document
        chunks = self._chunk_text(text, chunk_size=200, overlap=50)

        # Step 2: Retrieve top-k relevant chunks
        retrieved_chunks = self._retrieve_chunks(question, chunks)

        # Step 3: Generate answer from retrieved context
        context = "\n\n".join([f"[Passage {i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks)])

        prompt = f"""Based on the following retrieved passages, answer the question.

RETRIEVED CONTEXT:
{context}

QUESTION: {question}

Respond with ONLY one of: True, False, or Unknown.
Do not provide any explanation.

ANSWER:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a logical reasoning assistant. Answer questions based on the retrieved context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )

            prediction = response.choices[0].message.content.strip()
            prediction = self._normalize_answer(prediction)

            execution_time = time.time() - start_time

            return BaselineResult(
                prediction=prediction,
                execution_time=execution_time,
                metadata={
                    "method": "rag",
                    "num_chunks": len(chunks),
                    "retrieved_chunks": len(retrieved_chunks)
                }
            )

        except Exception as e:
            return BaselineResult(
                prediction="Unknown",
                execution_time=time.time() - start_time,
                metadata={"method": "rag", "error": str(e)}
            )

    def _chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    def _retrieve_chunks(self, query: str, chunks: List[str]) -> List[str]:
        """
        Retrieve top-k most relevant chunks.

        For now, uses simple heuristic (word overlap).
        TODO: Replace with proper embedding-based retrieval.
        """
        query_words = set(query.lower().split())

        # Score chunks by word overlap
        scored_chunks = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words & chunk_words)
            scored_chunks.append((overlap, chunk))

        # Sort by score and take top-k
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[:self.top_k]]

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer to standard format."""
        answer = answer.strip().lower()
        if "true" in answer:
            return "True"
        elif "false" in answer:
            return "False"
        else:
            return "Unknown"


class LogicLMBaseline:
    """
    Logic-LM baseline (simplified version).

    Per-query formalization: For each query, asks LLM to convert
    both the text and question into formal logic, then reasons symbolically.

    Note: This is a simplified implementation. Full Logic-LM uses
    multiple solvers (Prover9, Z3, etc.) with fallback strategies.
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def predict(self, text: str, question: str) -> BaselineResult:
        """Make a prediction using per-query formalization."""
        start_time = time.time()

        # Step 1: Formalize text and question into logic
        formalization_prompt = f"""Convert the following text and question into propositional logic.

TEXT:
{text}

QUESTION: {question}

Provide:
1. Atomic propositions (P1, P2, ...)
2. Logical constraints from the text
3. The question expressed as a logical formula

Format as JSON:
{{
  "propositions": {{"P1": "description", ...}},
  "constraints": ["P1 -> P2", ...],
  "query": "P3"
}}"""

        try:
            # Formalize
            formalization_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a logic formalization expert."},
                    {"role": "user", "content": formalization_prompt}
                ],
                temperature=0.0,
                max_tokens=1500
            )

            formalization_text = formalization_response.choices[0].message.content.strip()

            # Parse JSON
            try:
                # Extract JSON if wrapped in markdown
                if "```json" in formalization_text:
                    json_start = formalization_text.find("```json") + 7
                    json_end = formalization_text.find("```", json_start)
                    formalization_text = formalization_text[json_start:json_end].strip()
                elif "```" in formalization_text:
                    json_start = formalization_text.find("```") + 3
                    json_end = formalization_text.find("```", json_start)
                    formalization_text = formalization_text[json_start:json_end].strip()

                logic_structure = json.loads(formalization_text)
            except json.JSONDecodeError:
                # Fallback: if formalization fails, use direct reasoning
                return BaselineResult(
                    prediction="Unknown",
                    execution_time=time.time() - start_time,
                    metadata={"method": "logic-lm", "error": "formalization_failed"}
                )

            # Step 2: Reason over the logic (simplified - no actual SAT solver)
            reasoning_prompt = f"""Given the following logical formalization, determine if the query is True, False, or Unknown.

PROPOSITIONS:
{json.dumps(logic_structure.get('propositions', {}), indent=2)}

CONSTRAINTS:
{json.dumps(logic_structure.get('constraints', []), indent=2)}

QUERY: {logic_structure.get('query', '')}

Use logical reasoning to determine if the query:
- Must be True (entailed by constraints)
- Must be False (contradicts constraints)
- Unknown (underdetermined by constraints)

Respond with ONLY: True, False, or Unknown

ANSWER:"""

            reasoning_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a logical reasoning expert."},
                    {"role": "user", "content": reasoning_prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )

            prediction = reasoning_response.choices[0].message.content.strip()
            prediction = self._normalize_answer(prediction)

            execution_time = time.time() - start_time

            return BaselineResult(
                prediction=prediction,
                execution_time=execution_time,
                metadata={
                    "method": "logic-lm",
                    "num_propositions": len(logic_structure.get('propositions', {})),
                    "num_constraints": len(logic_structure.get('constraints', []))
                }
            )

        except Exception as e:
            return BaselineResult(
                prediction="Unknown",
                execution_time=time.time() - start_time,
                metadata={"method": "logic-lm", "error": str(e)}
            )

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer to standard format."""
        answer = answer.strip().lower()
        if "true" in answer:
            return "True"
        elif "false" in answer:
            return "False"
        else:
            return "Unknown"


def get_baseline(method: str, api_key: str, **kwargs):
    """
    Factory function to get baseline method.

    Args:
        method: One of "direct", "cot", "rag", "logic-lm"
        api_key: OpenAI API key
        **kwargs: Additional arguments for the baseline

    Returns:
        Baseline instance
    """
    baselines = {
        'direct': DirectBaseline,
        'cot': CoTBaseline,
        'rag': RAGBaseline,
        'logic-lm': LogicLMBaseline,
        'logiclm': LogicLMBaseline  # alias
    }

    method_lower = method.lower().replace('_', '-')

    if method_lower not in baselines:
        raise ValueError(f"Unknown baseline: {method}. Choose from {list(baselines.keys())}")

    baseline_class = baselines[method_lower]
    return baseline_class(api_key=api_key, **kwargs)


if __name__ == "__main__":
    """Test baseline methods."""
    import os

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        exit(1)

    # Test text
    text = """
    All students who attend lectures pass the exam.
    Alice attends lectures.
    Bob does not attend lectures.
    """

    questions = [
        ("Does Alice pass the exam?", "True"),
        ("Does Bob pass the exam?", "Unknown"),
        ("Do all students pass the exam?", "False")
    ]

    for method in ['direct', 'cot', 'rag', 'logic-lm']:
        print(f"\n{'='*60}")
        print(f"Testing {method.upper()}")
        print(f"{'='*60}")

        baseline = get_baseline(method, api_key)

        for question, expected in questions:
            result = baseline.predict(text, question)
            status = "✓" if result.prediction == expected else "✗"
            print(f"{status} Q: {question}")
            print(f"  Predicted: {result.prediction} (expected: {expected})")
            print(f"  Time: {result.execution_time:.2f}s")
