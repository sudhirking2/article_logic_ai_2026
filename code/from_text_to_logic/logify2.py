#!/usr/bin/env python3
"""
logify2.py - Enhanced Text to Logic Converter with OpenIE

This module converts natural language text to structured propositional logic
using OpenIE preprocessing followed by an LLM call with a specialized system prompt.
"""

import json
import os
import argparse
import re
import time
from typing import Dict, Any, List, Tuple, Optional
from openai import OpenAI
from openie import StanfordOpenIE


class LogifyConverter2:
    """Enhanced converter that uses OpenIE preprocessing before LLM conversion."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize the converter with API key and model.

        Args:
            api_key (str): OpenAI API key
            model (str): Model to use (default: gpt-4)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = self._load_system_prompt()

        # Initialize Stanford OpenIE
        print("Initializing Stanford OpenIE...")
        try:
            self.openie = StanfordOpenIE()
            print("Stanford OpenIE initialization complete.")
        except Exception as e:
            print(f"Error initializing Stanford OpenIE: {e}")
            raise RuntimeError(f"Failed to initialize Stanford OpenIE: {e}")

    def _load_system_prompt(self) -> str:
        """Load the system prompt from the prompt file."""
        prompt_path = "/workspace/repo/code/prompts/prompt_logify2"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract only the SYSTEM part (remove the INPUT FORMAT section)
                if "INPUT FORMAT" in content:
                    content = content.split("INPUT FORMAT")[0].strip()
                # Remove the "SYSTEM" header if present
                if content.startswith("SYSTEM"):
                    content = content[6:].strip()
                return content
        except FileNotFoundError:
            raise FileNotFoundError(f"System prompt file not found at {prompt_path}")

    def extract_openie_triples(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract OpenIE relation triples from the input text using Stanford OpenIE.

        Args:
            text (str): Input text to extract relations from

        Returns:
            List[Dict[str, Any]]: List of relation triples with confidence scores
        """
        print("Extracting relation triples using Stanford OpenIE...")
        try:
            # Use Stanford OpenIE to extract triples
            raw_triples = self.openie.annotate(text)

            triples = []
            for triple_data in raw_triples:
                # Stanford OpenIE returns different formats, handle them
                if isinstance(triple_data, dict):
                    # Handle dictionary format
                    subject = triple_data.get('subject', '').strip()
                    predicate = triple_data.get('relation', '').strip()
                    obj = triple_data.get('object', '').strip()
                    confidence = float(triple_data.get('confidence', 1.0))
                elif isinstance(triple_data, (list, tuple)) and len(triple_data) >= 3:
                    # Handle tuple/list format (subject, relation, object)
                    subject = str(triple_data[0]).strip()
                    predicate = str(triple_data[1]).strip()
                    obj = str(triple_data[2]).strip()
                    confidence = float(triple_data[3]) if len(triple_data) > 3 else 1.0
                else:
                    # Handle string format (tab-separated)
                    parts = str(triple_data).strip().split('\t')
                    if len(parts) >= 3:
                        subject = parts[0].strip()
                        predicate = parts[1].strip()
                        obj = parts[2].strip()
                        confidence = float(parts[3]) if len(parts) > 3 and parts[3].replace('.','').isdigit() else 1.0
                    else:
                        continue

                # Filter out empty or very short components
                if len(subject) > 0 and len(predicate) > 0 and len(obj) > 0:
                    triples.append({
                        'subject': subject,
                        'predicate': predicate,
                        'object': obj,
                        'confidence': confidence
                    })

            print(f"Extracted {len(triples)} relation triples from Stanford OpenIE")

            # Log some examples for debugging
            if triples:
                print("Sample triples:")
                for i, triple in enumerate(triples[:3]):
                    print(f"  {i+1}. ({triple['subject']}; {triple['predicate']}; {triple['object']}) [conf: {triple['confidence']:.3f}]")

            return triples

        except Exception as e:
            print(f"Warning: Stanford OpenIE extraction failed: {e}")
            print("Continuing without OpenIE preprocessing...")
            import traceback
            traceback.print_exc()
            return []


    def format_triples_for_prompt(self, triples: List[Dict[str, Any]]) -> str:
        """
        Format OpenIE triples for inclusion in the LLM prompt.

        Args:
            triples (List[Dict[str, Any]]): List of relation triples

        Returns:
            str: Formatted string of triples
        """
        if not triples:
            return "No OpenIE triples extracted."

        formatted_lines = []
        for triple in triples:
            line = f"{triple['subject']}\t{triple['predicate']}\t{triple['object']}\t{triple['confidence']:.4f}"
            formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def convert_text_to_logic(self, text: str) -> Dict[str, Any]:
        """
        Convert input text to structured logic using OpenIE + LLM.

        Args:
            text (str): Input text to convert

        Returns:
            Dict[str, Any]: JSON structure with primitive props, hard/soft constraints
        """
        try:
            # Step 1: Extract OpenIE triples
            openie_triples = self.extract_openie_triples(text)
            formatted_triples = self.format_triples_for_prompt(openie_triples)

            # Step 2: Format the combined input for the LLM
            combined_input = f"""ORIGINAL TEXT:
<<<
{text}
>>>

OPENIE TRIPLES:
<<<
{formatted_triples}
>>>"""

            print("Sending to LLM for logical structure extraction...")

            # Step 3: Send to LLM with the enhanced prompt
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": combined_input}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=4000   # Sufficient for complex logic structures
            )

            response_text = response.choices[0].message.content.strip()

            # Parse the JSON response
            try:
                logic_structure = json.loads(response_text)
                return logic_structure
            except json.JSONDecodeError as e:
                # If JSON parsing fails, try to extract JSON from response
                if "{" in response_text and "}" in response_text:
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    json_text = response_text[json_start:json_end]
                    logic_structure = json.loads(json_text)
                    return logic_structure
                else:
                    raise ValueError(f"Failed to parse JSON response: {e}")

        except Exception as e:
            raise RuntimeError(f"Error in LLM conversion: {e}")

    def save_output(self, logic_structure: Dict[str, Any], output_path: str = "logified2.JSON"):
        """
        Save the logic structure to a JSON file.

        Args:
            logic_structure (Dict[str, Any]): The converted logic structure
            output_path (str): Path to save the JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(logic_structure, f, indent=2, ensure_ascii=False)
        print(f"Output saved to {output_path}")

    def close(self):
        """Manually close Stanford OpenIE resources."""
        if hasattr(self, 'openie'):
            try:
                # The Stanford OpenIE wrapper handles server cleanup automatically
                # No explicit close method needed
                pass
            except:
                pass

    def __del__(self):
        """Clean up Stanford OpenIE resources."""
        self.close()


def main():
    """Main function to handle command line usage."""
    parser = argparse.ArgumentParser(description="Convert text to structured propositional logic using OpenIE + LLM")
    parser.add_argument("input_text", help="Input text to convert (or path to text file)")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4", help="Model to use (default: gpt-4)")
    parser.add_argument("--output", default="logified2.JSON", help="Output JSON file path")
    parser.add_argument("--file", action="store_true", help="Treat input_text as file path")

    args = parser.parse_args()

    # Get input text
    if args.file:
        if not os.path.exists(args.input_text):
            print(f"Error: Input file '{args.input_text}' not found")
            return 1
        with open(args.input_text, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = args.input_text

    try:
        # Initialize converter
        converter = LogifyConverter2(api_key=args.api_key, model=args.model)

        # Convert text to logic
        print(f"Converting text using model: {args.model}")
        logic_structure = converter.convert_text_to_logic(text)

        # Save output
        converter.save_output(logic_structure, args.output)

        print("Conversion completed successfully!")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())