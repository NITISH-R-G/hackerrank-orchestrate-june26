import argparse
import sys
from pathlib import Path

def parse_args(args):
    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review System")
    parser.add_argument("--input", type=str, default="dataset/claims.csv", help="Path to input claims CSV file")
    parser.add_argument("--output", type=str, default="output.csv", help="Path to output CSV file")
    parser.add_argument("--sample", action="store_true", help="Run on sample_claims.csv instead")
    return parser.parse_args(args)

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    try:
        parsed_args = parse_args(args)
    except SystemExit as e:
        return e.code

    input_file = "dataset/sample_claims.csv" if parsed_args.sample else parsed_args.input
    output_file = parsed_args.output

    print(f"Starting evidence review pipeline...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    # TODO: Initialize and run the pipeline

    print("Pipeline completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
