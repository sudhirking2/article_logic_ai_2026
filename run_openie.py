#!/usr/bin/env python3
"""
Script to run OpenIE extractor on the provided input text.
"""

import sys
import os

# Add the code directory to the path
sys.path.insert(0, '/workspace/repo/code/from_text_to_logic')

from openie_extractor import OpenIEExtractor

# Input text
INPUT_TEXT = """The hospital's emergency triage protocol requires immediate attention for patients presenting with chest pain,
unless the pain is clearly musculoskeletal in origin and the patient is under 40 years old.
Dr. Martinez, who has been working double shifts this week, believes that patients over 65 should always receive an ECG regardless of symptoms, althought Dr. Yang only sometimes believes this.
The official guidelines only mandate this when cardiac history is documented."""

def main():
    print("=" * 80)
    print("OpenIE Extractor - Running on Input Text")
    print("=" * 80)
    print("\nINPUT TEXT:")
    print("-" * 80)
    print(INPUT_TEXT)
    print("-" * 80)
    print()

    # Initialize the extractor
    try:
        extractor = OpenIEExtractor(
            memory='4G',
            timeout=60000,
            enable_coref=True,
            use_depparse_fallback=True,
            port=9000,
            language='en',
            download_models=False  # Assume models are already downloaded
        )

        # Extract triples
        print("\n" + "=" * 80)
        print("EXTRACTION PROCESS")
        print("=" * 80)
        triples = extractor.extract_triples(INPUT_TEXT)

        # Display results
        print("\n" + "=" * 80)
        print("RESULTS - VERBOSE FORMAT")
        print("=" * 80)
        verbose_output = extractor.format_triples_verbose(triples)
        print(verbose_output)

        print("\n" + "=" * 80)
        print("RESULTS - TAB-SEPARATED FORMAT")
        print("=" * 80)
        tsv_output = extractor.format_triples(triples)
        print(tsv_output)

        print("\n" + "=" * 80)
        print(f"SUMMARY: Extracted {len(triples)} relation triples")
        print("=" * 80)

        # Clean up
        extractor.close()

    except Exception as e:
        print(f"\n[ERROR] Failed to run OpenIE extractor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
