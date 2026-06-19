"""Main entry point for the claim verification pipeline.

Parses arguments and runs the multi-modal evidence review system over the
input dataset to produce output predictions.
"""
import argparse
import sys
import logging

# Adjust imports according to actual available modules (like config)
try:
    from config import CLAIMS_CSV, OUTPUT_CSV, USER_HISTORY_CSV, EVIDENCE_CSV
except ImportError:
    # Fallback paths for testing if config is not available or main.py is tested in isolation
    CLAIMS_CSV = "dataset/claims.csv"
    OUTPUT_CSV = "output.csv"
    USER_HISTORY_CSV = "dataset/user_history.csv"
    EVIDENCE_CSV = "dataset/evidence_requirements.csv"


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_pipeline(claims_path: str, output_path: str) -> None:
    """Execute the main processing pipeline."""
    logging.info(f"Starting pipeline. Input: {claims_path}, Output: {output_path}")
    # Placeholder for actual pipeline execution logic
    logging.info("Pipeline execution completed.")


def main(args: list[str] | None = None) -> int:
    """Parse arguments and start the pipeline.

    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description="Verify damage claims using multi-modal evidence."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(CLAIMS_CSV),
        help="Path to input claims CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_CSV),
        help="Path to write output predictions CSV"
    )
    parser.add_argument(
        "--user-history",
        type=str,
        default=str(USER_HISTORY_CSV),
        help="Path to user history CSV file"
    )
    parser.add_argument(
        "--evidence-reqs",
        type=str,
        default=str(EVIDENCE_CSV),
        help="Path to evidence requirements CSV file"
    )

    try:
        parsed_args = parser.parse_args(args)
        run_pipeline(
            claims_path=parsed_args.input,
            output_path=parsed_args.output
        )
        return 0
    except SystemExit as e:
        # argparse throws SystemExit on --help or bad args
        return e.code
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
