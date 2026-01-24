#!/usr/bin/env python3
"""
run_experiments.py - Main experiment runner for baseline comparisons

Runs baseline methods on benchmark datasets and saves results.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import time

from datasets import load_dataset, Example
from baselines import get_baseline, BaselineResult


def evaluate_method(
    method: str,
    examples: List[Example],
    api_key: str,
    output_dir: str,
    limit: int = None,
    model: str = "gpt-4"
) -> Dict[str, Any]:
    """
    Evaluate a baseline method on a set of examples.

    Args:
        method: Baseline method name
        examples: List of examples to evaluate
        api_key: OpenAI API key
        output_dir: Directory to save results
        limit: Limit number of examples (for testing)
        model: Model to use

    Returns:
        Evaluation results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {method.upper()}")
    print(f"{'='*60}")

    # Initialize baseline
    baseline = get_baseline(method, api_key=api_key, model=model)

    # Limit examples if specified
    if limit:
        examples = examples[:limit]

    # Run predictions
    results = []
    correct = 0
    total = 0

    for example in tqdm(examples, desc=f"{method}"):
        # Make prediction
        result = baseline.predict(example.text, example.question)

        # Check correctness
        is_correct = (result.prediction == example.answer)
        if is_correct:
            correct += 1
        total += 1

        # Store result
        results.append({
            'example_id': example.id,
            'prediction': result.prediction,
            'ground_truth': example.answer,
            'correct': is_correct,
            'execution_time': result.execution_time,
            'reasoning': result.reasoning,
            'metadata': {**example.metadata, **result.metadata}
        })

        # Rate limiting: sleep briefly between API calls
        time.sleep(0.5)

    # Calculate metrics
    accuracy = 100 * correct / total if total > 0 else 0

    evaluation_results = {
        'method': method,
        'model': model,
        'total_examples': total,
        'correct': correct,
        'accuracy': accuracy,
        'results': results
    }

    # Save results
    output_path = Path(output_dir) / f"{method}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"\n✓ {method.upper()} Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Results saved to: {output_path}")

    return evaluation_results


def run_all_baselines(
    dataset_name: str,
    split: str,
    api_key: str,
    output_dir: str,
    methods: List[str] = None,
    limit: int = None,
    model: str = "gpt-4"
) -> Dict[str, Dict[str, Any]]:
    """
    Run all baseline methods on a dataset.

    Args:
        dataset_name: Name of dataset (folio, proofwriter, contractnli)
        split: Dataset split
        api_key: OpenAI API key
        output_dir: Directory to save results
        methods: List of methods to run (if None, runs all)
        limit: Limit number of examples per method
        model: Model to use

    Returns:
        Dictionary mapping method names to their results
    """
    # Load dataset
    print(f"Loading {dataset_name} dataset ({split} split)...")
    examples = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(examples)} examples")

    # Default methods if not specified
    if methods is None:
        methods = ['direct', 'cot', 'rag', 'logic-lm']

    # Create output directory for this dataset
    dataset_output_dir = Path(output_dir) / dataset_name / split

    # Run each method
    all_results = {}
    for method in methods:
        try:
            results = evaluate_method(
                method=method,
                examples=examples,
                api_key=api_key,
                output_dir=str(dataset_output_dir),
                limit=limit,
                model=model
            )
            all_results[method] = results

        except Exception as e:
            print(f"\n✗ Error evaluating {method}: {e}")
            all_results[method] = {
                'method': method,
                'error': str(e),
                'accuracy': 0.0
            }

    # Save summary
    summary_path = dataset_output_dir / "summary.json"
    summary = {
        'dataset': dataset_name,
        'split': split,
        'num_examples': len(examples),
        'methods': {
            method: {
                'accuracy': results.get('accuracy', 0.0),
                'correct': results.get('correct', 0),
                'total': results.get('total_examples', 0)
            }
            for method, results in all_results.items()
        }
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name} ({split})")
    print(f"Examples: {len(examples)}")
    print("\nAccuracy by method:")
    for method, results in all_results.items():
        acc = results.get('accuracy', 0.0)
        print(f"  {method:15s}: {acc:6.2f}%")
    print(f"\nSummary saved to: {summary_path}")

    return all_results


def generate_comparison_table(results_dir: str, output_file: str = None):
    """
    Generate comparison table from saved results.

    Args:
        results_dir: Directory containing experiment results
        output_file: Optional output file for table
    """
    results_path = Path(results_dir)

    # Collect all summaries
    datasets = ['folio', 'proofwriter', 'contractnli']
    methods = ['direct', 'cot', 'rag', 'logic-lm']

    table_data = {}

    for dataset in datasets:
        summary_file = results_path / dataset / "test" / "summary.json"

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                table_data[dataset] = summary['methods']

    # Print table
    print("\n" + "="*80)
    print("ACCURACY COMPARISON TABLE")
    print("="*80)
    print(f"{'Method':<15} {'FOLIO':>10} {'ProofWriter':>15} {'ContractNLI':>15}")
    print("-"*80)

    for method in methods:
        folio_acc = table_data.get('folio', {}).get(method, {}).get('accuracy', 0.0)
        pw_acc = table_data.get('proofwriter', {}).get(method, {}).get('accuracy', 0.0)
        cnli_acc = table_data.get('contractnli', {}).get(method, {}).get('accuracy', 0.0)

        print(f"{method.capitalize():<15} {folio_acc:>9.1f}% {pw_acc:>14.1f}% {cnli_acc:>14.1f}%")

    print("="*80)

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write("# Baseline Comparison Results\n\n")
            f.write("| Method | FOLIO | ProofWriter | ContractNLI |\n")
            f.write("|--------|-------|-------------|-------------|\n")
            for method in methods:
                folio_acc = table_data.get('folio', {}).get(method, {}).get('accuracy', 0.0)
                pw_acc = table_data.get('proofwriter', {}).get(method, {}).get('accuracy', 0.0)
                cnli_acc = table_data.get('contractnli', {}).get(method, {}).get('accuracy', 0.0)
                f.write(f"| {method.capitalize()} | {folio_acc:.1f}% | {pw_acc:.1f}% | {cnli_acc:.1f}% |\n")

        print(f"\nTable saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments")

    parser.add_argument('--dataset', choices=['folio', 'proofwriter', 'contractnli', 'all'],
                       default='all', help='Dataset to evaluate')
    parser.add_argument('--split', default='test', help='Dataset split')
    parser.add_argument('--methods', nargs='+', default=None,
                       help='Methods to run (default: all)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of examples (for testing)')
    parser.add_argument('--output-dir', default='results/baselines',
                       help='Output directory for results')
    parser.add_argument('--api-key', default=None,
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--model', default='gpt-4',
                       help='Model to use (default: gpt-4)')
    parser.add_argument('--generate-table', action='store_true',
                       help='Generate comparison table from existing results')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key and not args.generate_table:
        print("ERROR: OpenAI API key required. Set --api-key or OPENAI_API_KEY env var")
        return 1

    # Generate table mode
    if args.generate_table:
        generate_comparison_table(args.output_dir)
        return 0

    # Determine datasets to run
    if args.dataset == 'all':
        datasets = ['folio', 'proofwriter', 'contractnli']
    else:
        datasets = [args.dataset]

    # Run experiments
    for dataset_name in datasets:
        print(f"\n{'#'*80}")
        print(f"# RUNNING EXPERIMENTS ON {dataset_name.upper()}")
        print(f"{'#'*80}")

        try:
            run_all_baselines(
                dataset_name=dataset_name,
                split=args.split,
                api_key=api_key,
                output_dir=args.output_dir,
                methods=args.methods,
                limit=args.limit,
                model=args.model
            )
        except Exception as e:
            print(f"\n✗ Error running experiments on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Generate final table
    print(f"\n{'#'*80}")
    print("# GENERATING COMPARISON TABLE")
    print(f"{'#'*80}")
    generate_comparison_table(
        args.output_dir,
        output_file=f"{args.output_dir}/comparison_table.md"
    )

    return 0


if __name__ == "__main__":
    exit(main())
