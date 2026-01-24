#!/usr/bin/env python3
"""
datasets.py - Dataset loading utilities for benchmark experiments

Supports:
- FOLIO: First-order logic reasoning
- ProofWriter: Synthetic deductive reasoning (depth-5 subset)
- ContractNLI: Long document NLI with contracts
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import requests
from tqdm import tqdm


@dataclass
class Example:
    """Base class for benchmark examples."""
    id: str
    text: str
    question: str
    answer: str  # "True", "False", or "Unknown"
    metadata: Dict[str, Any]


class FOLIODataset:
    """
    FOLIO dataset loader.

    Paper: "FOLIO: Natural Language Reasoning with First-Order Logic"
    Dataset: 1,430 examples with FOL annotations
    """

    def __init__(self, data_dir: str = "data/folio"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download(self):
        """Download FOLIO dataset from HuggingFace."""
        print("Downloading FOLIO dataset...")

        # FOLIO is available on HuggingFace datasets
        try:
            from datasets import load_dataset
            dataset = load_dataset("yale-nlp/folio")

            # Save to local JSON
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    output_file = self.data_dir / f"{split}.jsonl"
                    with open(output_file, 'w') as f:
                        for example in dataset[split]:
                            f.write(json.dumps(example) + '\n')
                    print(f"  Saved {split} split: {len(dataset[split])} examples")

            return True

        except ImportError:
            print("ERROR: 'datasets' library not installed.")
            print("Install with: pip install datasets")
            return False
        except Exception as e:
            print(f"ERROR downloading FOLIO: {e}")
            return False

    def load(self, split: str = "validation") -> List[Example]:
        """Load FOLIO examples from disk."""
        file_path = self.data_dir / f"{split}.jsonl"

        if not file_path.exists():
            print(f"FOLIO {split} split not found. Attempting download...")
            if not self.download():
                raise FileNotFoundError(f"Could not load FOLIO {split} split")

        examples = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)

                # FOLIO format: premises, conclusion, label
                premises = data.get('premises', '')
                conclusion = data.get('conclusion', '')
                label = data.get('label', 'Unknown')

                # Map label to standard format
                label_map = {
                    'True': 'True',
                    'False': 'False',
                    'Unknown': 'Unknown',
                    'Uncertain': 'Unknown'
                }

                example = Example(
                    id=data.get('id', f"folio_{len(examples)}"),
                    text=premises,
                    question=conclusion,
                    answer=label_map.get(label, 'Unknown'),
                    metadata={'split': split, 'dataset': 'folio'}
                )
                examples.append(example)

        print(f"Loaded {len(examples)} FOLIO examples from {split} split")
        return examples


class ProofWriterDataset:
    """
    ProofWriter dataset loader (depth-5 subset).

    Paper: "ProofWriter: Generating Implications via Iterative Forward Reasoning"
    Dataset: Using depth-5 subset (~600 examples) for experiments
    """

    def __init__(self, data_dir: str = "data/proofwriter"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download(self):
        """Download ProofWriter dataset."""
        print("Downloading ProofWriter dataset...")

        try:
            from datasets import load_dataset
            # ProofWriter is also on HuggingFace
            dataset = load_dataset("allenai/proofwriter")

            # Filter for depth-5 and save
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    # Filter for depth-5 (adjust based on actual dataset structure)
                    filtered = [ex for ex in dataset[split]]

                    output_file = self.data_dir / f"{split}.jsonl"
                    with open(output_file, 'w') as f:
                        for example in filtered:
                            f.write(json.dumps(example) + '\n')
                    print(f"  Saved {split} split: {len(filtered)} examples")

            return True

        except ImportError:
            print("ERROR: 'datasets' library not installed.")
            return False
        except Exception as e:
            print(f"ERROR downloading ProofWriter: {e}")
            return False

    def load(self, split: str = "validation", max_depth: int = 5) -> List[Example]:
        """Load ProofWriter examples."""
        file_path = self.data_dir / f"{split}.jsonl"

        if not file_path.exists():
            print(f"ProofWriter {split} split not found. Attempting download...")
            if not self.download():
                raise FileNotFoundError(f"Could not load ProofWriter {split} split")

        examples = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)

                # ProofWriter format varies, adapt as needed
                theory = data.get('theory', data.get('context', ''))
                question = data.get('question', '')
                answer = data.get('answer', 'Unknown')
                depth = data.get('depth', 0)

                # Filter by depth if specified
                if depth > max_depth:
                    continue

                # Standardize answer
                if isinstance(answer, bool):
                    answer = 'True' if answer else 'False'

                example = Example(
                    id=data.get('id', f"pw_{len(examples)}"),
                    text=theory,
                    question=question,
                    answer=str(answer),
                    metadata={
                        'split': split,
                        'dataset': 'proofwriter',
                        'depth': depth
                    }
                )
                examples.append(example)

        print(f"Loaded {len(examples)} ProofWriter examples (depth ≤ {max_depth})")
        return examples


class ContractNLIDataset:
    """
    ContractNLI dataset loader.

    Paper: "ContractNLI: A Dataset for Document-level Natural Language Inference"
    Dataset: 607 non-disclosure agreements with 17 hypothesis types
    """

    def __init__(self, data_dir: str = "data/contractnli"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download(self):
        """Download ContractNLI dataset."""
        print("Downloading ContractNLI dataset...")

        try:
            from datasets import load_dataset
            dataset = load_dataset("contract-nli/contractnli")

            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    output_file = self.data_dir / f"{split}.jsonl"
                    with open(output_file, 'w') as f:
                        for example in dataset[split]:
                            f.write(json.dumps(example) + '\n')
                    print(f"  Saved {split} split: {len(dataset[split])} examples")

            return True

        except ImportError:
            print("ERROR: 'datasets' library not installed.")
            return False
        except Exception as e:
            print(f"ERROR downloading ContractNLI: {e}")
            return False

    def load(self, split: str = "test") -> List[Example]:
        """Load ContractNLI examples."""
        file_path = self.data_dir / f"{split}.jsonl"

        if not file_path.exists():
            print(f"ContractNLI {split} split not found. Attempting download...")
            if not self.download():
                raise FileNotFoundError(f"Could not load ContractNLI {split} split")

        examples = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)

                # ContractNLI format
                document = data.get('document', data.get('context', ''))
                hypothesis = data.get('hypothesis', '')
                label = data.get('label', 'NotMentioned')

                # Map labels
                label_map = {
                    'Entailment': 'True',
                    'Contradiction': 'False',
                    'NotMentioned': 'Unknown',
                    'Neutral': 'Unknown'
                }

                example = Example(
                    id=data.get('id', f"cnli_{len(examples)}"),
                    text=document,
                    question=hypothesis,
                    answer=label_map.get(label, 'Unknown'),
                    metadata={
                        'split': split,
                        'dataset': 'contractnli',
                        'doc_length': len(document.split())
                    }
                )
                examples.append(example)

        print(f"Loaded {len(examples)} ContractNLI examples")
        return examples


def load_dataset(dataset_name: str, split: str = "test") -> List[Example]:
    """
    Unified interface to load any dataset.

    Args:
        dataset_name: One of "folio", "proofwriter", "contractnli"
        split: Dataset split to load (train/validation/test)

    Returns:
        List of Example objects
    """
    loaders = {
        'folio': FOLIODataset,
        'proofwriter': ProofWriterDataset,
        'contractnli': ContractNLIDataset
    }

    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(loaders.keys())}")

    loader_class = loaders[dataset_name.lower()]
    loader = loader_class()

    return loader.load(split=split)


if __name__ == "__main__":
    """Test dataset loading."""
    import argparse

    parser = argparse.ArgumentParser(description="Download and test benchmark datasets")
    parser.add_argument("--dataset", choices=["folio", "proofwriter", "contractnli", "all"],
                       default="all", help="Dataset to download")
    parser.add_argument("--split", default="validation", help="Split to load")

    args = parser.parse_args()

    datasets_to_test = ['folio', 'proofwriter', 'contractnli'] if args.dataset == 'all' else [args.dataset]

    for dataset_name in datasets_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {dataset_name.upper()}")
        print(f"{'='*60}")

        try:
            examples = load_dataset(dataset_name, split=args.split)
            print(f"\n✓ Successfully loaded {len(examples)} examples")

            if examples:
                print("\nSample example:")
                ex = examples[0]
                print(f"  ID: {ex.id}")
                print(f"  Text (first 200 chars): {ex.text[:200]}...")
                print(f"  Question: {ex.question}")
                print(f"  Answer: {ex.answer}")
                print(f"  Metadata: {ex.metadata}")

        except Exception as e:
            print(f"\n✗ Error loading {dataset_name}: {e}")
