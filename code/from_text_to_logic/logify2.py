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
import spacy


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

        # Initialize spaCy for simple OpenIE-style extraction
        print("Initializing spaCy for relation extraction...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("spaCy initialization complete.")
        except OSError:
            print("Warning: spaCy model not found. Downloading...")
            os.system("python3 -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

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
        Extract OpenIE-style relation triples from the input text using spaCy.

        Args:
            text (str): Input text to extract relations from

        Returns:
            List[Dict[str, Any]]: List of relation triples with confidence scores
        """
        print("Extracting relation triples using spaCy...")
        try:
            doc = self.nlp(text)
            triples = []

            # Build coreference resolution map (simple approach)
            coref_map = self._build_simple_coref_map(doc)

            for sent in doc.sents:
                # Extract triples based on dependency parsing
                for token in sent:
                    # Look for verbs as predicates (including auxiliary and copula)
                    if token.pos_ in ["VERB", "AUX"] and token.dep_ in ["ROOT", "ccomp", "xcomp", "conj"]:
                        predicate = token.lemma_

                        # Handle copula (is, am, are, was, were)
                        if token.lemma_ in ["be", "become", "seem", "appear"]:
                            # Find the predicate complement
                            for child in token.children:
                                if child.dep_ in ["attr", "acomp"]:
                                    predicate = f"is {child.text}"
                                    break

                        # Find subject
                        subject = None
                        subject_token = None
                        for child in token.children:
                            if child.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                                subject_token = child
                                subject = self._get_noun_phrase(child)
                                # Apply coreference resolution
                                subject = coref_map.get(subject.lower(), subject)
                                break

                        # Find direct object
                        obj = None
                        obj_found = False
                        for child in token.children:
                            if child.dep_ in ["dobj", "attr", "acomp"]:
                                obj = self._get_noun_phrase(child)
                                obj_found = True
                                break

                        # Find prepositional objects
                        if not obj_found:
                            for child in token.children:
                                if child.dep_ == "prep" and child.pos_ == "ADP":
                                    prep_text = child.text
                                    for prep_child in child.children:
                                        if prep_child.dep_ == "pobj":
                                            obj = self._get_noun_phrase(prep_child)
                                            predicate = f"{predicate} {prep_text}"
                                            break
                                    if obj:
                                        break

                        # If we found a complete triple, add it
                        if subject and obj and predicate:
                            triples.append({
                                'subject': subject.strip(),
                                'predicate': predicate.strip(),
                                'object': obj.strip(),
                                'confidence': 0.8
                            })

                        # Handle complement clauses for conditional statements
                        for child in token.children:
                            if child.dep_ in ["ccomp", "xcomp"] and child.pos_ == "VERB":
                                comp_pred = child.lemma_
                                comp_subj = subject if subject else "it"

                                # Find object of complement
                                comp_obj = None
                                for comp_child in child.children:
                                    if comp_child.dep_ in ["dobj", "attr", "acomp"]:
                                        comp_obj = self._get_noun_phrase(comp_child)
                                        break

                                # Look for prepositional complements
                                if not comp_obj:
                                    for comp_child in child.children:
                                        if comp_child.dep_ == "prep":
                                            for prep_child in comp_child.children:
                                                if prep_child.dep_ == "pobj":
                                                    comp_obj = self._get_noun_phrase(prep_child)
                                                    comp_pred = f"{comp_pred} {comp_child.text}"
                                                    break
                                            if comp_obj:
                                                break

                                if comp_obj:
                                    triples.append({
                                        'subject': comp_subj.strip(),
                                        'predicate': comp_pred.strip(),
                                        'object': comp_obj.strip(),
                                        'confidence': 0.7
                                    })

            # Additional pattern-based extraction for common constructions
            self._extract_additional_patterns(doc, triples, coref_map)

            print(f"Extracted {len(triples)} relation triples")
            return triples

        except Exception as e:
            print(f"Warning: Relation extraction failed: {e}")
            print("Continuing without OpenIE preprocessing...")
            return []

    def _build_simple_coref_map(self, doc):
        """Build a simple coreference resolution map."""
        coref_map = {}

        # Find proper nouns (potential entities)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                entities.append(ent.text)

        # Also look for capitalized words that might be names
        for token in doc:
            if token.text[0].isupper() and token.pos_ in ["PROPN", "NOUN"] and len(token.text) > 1:
                if token.text not in entities:
                    entities.append(token.text)

        # Map pronouns to most recent appropriate entity
        current_person = None
        for token in doc:
            if token.text in entities:
                if any(char.isupper() for char in token.text):  # Likely a proper noun
                    current_person = token.text
            elif token.text.lower() in ["she", "he", "they"] and current_person:
                coref_map[token.text.lower()] = current_person

        return coref_map

    def _extract_additional_patterns(self, doc, triples, coref_map):
        """Extract additional patterns like relative clauses and appositives."""

        # Pattern: "X who/that Y" -> Extract Y relationship
        for token in doc:
            if token.text.lower() in ["who", "that"] and token.dep_ == "nsubj":
                # Find the head of the relative clause
                rel_head = token.head
                if rel_head.pos_ == "VERB":
                    # Find the antecedent (what "who/that" refers to)
                    antecedent = None
                    for ancestor in token.ancestors:
                        if ancestor.dep_ in ["nsubj", "dobj", "pobj"]:
                            antecedent = self._get_noun_phrase(ancestor)
                            break

                    if antecedent:
                        # Find the object of the relative clause verb
                        rel_obj = None
                        for child in rel_head.children:
                            if child.dep_ in ["dobj", "attr", "acomp"]:
                                rel_obj = self._get_noun_phrase(child)
                                break

                        if rel_obj:
                            antecedent = coref_map.get(antecedent.lower(), antecedent)
                            triples.append({
                                'subject': antecedent.strip(),
                                'predicate': rel_head.lemma_.strip(),
                                'object': rel_obj.strip(),
                                'confidence': 0.75
                            })

        # Pattern: "X is a Y" -> Extract type relationship
        for token in doc:
            if token.lemma_ == "be" and token.pos_ == "AUX":
                subj = None
                obj = None

                for child in token.children:
                    if child.dep_ == "nsubj":
                        subj = self._get_noun_phrase(child)
                        subj = coref_map.get(subj.lower(), subj)
                    elif child.dep_ in ["attr", "acomp"]:
                        obj = self._get_noun_phrase(child)

                if subj and obj:
                    triples.append({
                        'subject': subj.strip(),
                        'predicate': 'is',
                        'object': obj.strip(),
                        'confidence': 0.85
                    })

    def _get_noun_phrase(self, token):
        """Get the full noun phrase for a token."""
        # Start with the token itself
        phrase_tokens = [token]

        # Add determiners, adjectives, and compounds to the left
        for child in token.children:
            if child.dep_ in ["det", "amod", "compound", "nummod"] and child.i < token.i:
                phrase_tokens.insert(0, child)

        # Add prepositional phrases and relative clauses to the right
        for child in token.children:
            if child.dep_ in ["prep", "relcl"] and child.i > token.i:
                phrase_tokens.append(child)
                # Add children of prepositions
                if child.dep_ == "prep":
                    for prep_child in child.children:
                        if prep_child.dep_ == "pobj":
                            phrase_tokens.append(prep_child)

        # Sort by position in sentence and join
        phrase_tokens.sort(key=lambda x: x.i)
        return " ".join([t.text for t in phrase_tokens])

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

    def __del__(self):
        """Clean up resources."""
        # spaCy doesn't require explicit cleanup
        pass


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