"""
Configuration module for Reasoning LLM + RAG baseline.

This module stores all hyperparameters and configuration settings for the
RAG baseline experiment. All parameters are fixed to ensure reproducibility
and fair comparison with the Logify system.

Key Parameters:
- SBERT model for retrieval
- Chunk size and overlap for document segmentation
- Number of retrieved chunks (k)
- LLM model name (configurable)
- Temperature for deterministic generation
- Fixed Chain-of-Thought prompt template
"""


# SBERT Configuration
SBERT_MODEL = "all-MiniLM-L6-v2"

# Retrieval Configuration
TOP_K = 5

# Chunking Configuration
CHUNK_SIZE = 512
OVERLAP = 50

# LLM Configuration
DEFAULT_MODEL = "gpt-4"
TEMPERATURE = 0

# Chain-of-Thought Prompt Template (Optimized)
COT_PROMPT_TEMPLATE = """You are a precise logical reasoning assistant. Your task is to determine whether a statement is True, False, or Unknown based solely on the provided context.

RETRIEVED CONTEXT:
{retrieved_chunks}

QUERY: {query}

REASONING INSTRUCTIONS:
1. Extract all relevant facts from the context that relate to the query
2. Identify any logical relationships, constraints, or implications
3. Apply deductive reasoning step-by-step, showing each inference
4. Consider both supporting and contradicting evidence
5. Check for logical consistency across all facts
6. If information is insufficient or contradictory, acknowledge uncertainty

IMPORTANT:
- Base your answer ONLY on the provided context
- Do not introduce external knowledge or assumptions
- For True: the query must be definitively supported by the context
- For False: the query must be definitively contradicted by the context
- For Unknown: insufficient information or neither confirmed nor contradicted

RESPONSE FORMAT:
Reasoning: [Explain your step-by-step logical analysis]
Answer: [True/False/Unknown]
"""

# Dataset-specific prompt templates for optimized reasoning
FOLIO_PROMPT_TEMPLATE = """You are an expert in first-order logic and formal reasoning. Analyze whether the conclusion logically follows from the premises.

PREMISES:
{retrieved_chunks}

CONCLUSION: {query}

REASONING INSTRUCTIONS:
1. List all premises explicitly
2. Identify universal statements (all, every, no) and existential statements (some, there exists)
3. Apply formal logical rules: modus ponens, modus tollens, universal instantiation, etc.
4. Check if the conclusion can be derived through valid logical inference
5. Look for counterexamples that would make the conclusion false
6. Distinguish between "definitely follows" vs "possibly true but not proven"

CRITICAL DISTINCTIONS:
- True: conclusion is a logical consequence of premises
- False: conclusion contradicts the premises or counterexample exists
- Unknown: conclusion is consistent with premises but not derivable

Reasoning: [Show formal logical steps]
Answer: [True/False/Unknown]
"""

CONTRACTNLI_PROMPT_TEMPLATE = """You are a contract analysis expert. Determine whether a contractual hypothesis is Entailed, Contradicted, or NotMentioned by the contract document.

CONTRACT EXCERPTS:
{retrieved_chunks}

HYPOTHESIS: {query}

ANALYSIS INSTRUCTIONS:
1. Identify all relevant clauses, obligations, and conditions in the contract
2. Parse conditional statements (if-then), exceptions, and qualifications
3. Check for explicit statements supporting or contradicting the hypothesis
4. Consider implied obligations from standard contractual language
5. Look for limiting conditions, carve-outs, or exceptions that affect the hypothesis
6. Distinguish between absence of mention vs explicit contradiction

CLASSIFICATION RULES:
- Entailed: hypothesis directly stated or necessarily implied by contract terms
- Contradicted: hypothesis explicitly denied or incompatible with contract provisions
- NotMentioned: hypothesis neither supported nor contradicted by available text

Reasoning: [Detailed contract analysis with clause references]
Answer: [Entailed/Contradicted/NotMentioned]
"""

PROOFWRITER_PROMPT_TEMPLATE = """You are a formal deductive reasoning system. Determine if the query statement is True, False, or Unknown given the theory.

THEORY (Rules and Facts):
{retrieved_chunks}

QUERY: {query}

REASONING INSTRUCTIONS:
1. Separate facts (direct assertions) from rules (conditional statements)
2. Apply rules systematically through forward chaining
3. Derive all implied facts through transitive closure
4. Check if query matches derived facts or their negations
5. Track the derivation chain for provenance
6. If query cannot be derived or refuted, return Unknown

LOGICAL OPERATIONS:
- Apply implications: If A and (A → B), then B
- Apply contrapositive: If (A → B) and ¬B, then ¬A
- Check for direct contradictions: If both P and ¬P derivable, flag inconsistency

Reasoning: [Show derivation steps with rule applications]
Answer: [True/False/Unknown]
"""
