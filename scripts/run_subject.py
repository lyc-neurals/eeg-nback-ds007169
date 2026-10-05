#!/usr/bin/env python3
"""Run a subject from a ds007169 checkout (or a flat legacy subject folder)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from eeg_pipeline import run_subject  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="sub-001", help="BIDS ID, e.g. sub-002")
    parser.add_argument("--data-root", type=Path, default=ROOT/"data"/"ds007169",
                        help="Root of downloaded ds007169 tree")
    parser.add_argument("--output-root", type=Path, default=ROOT/"results")
    parser.add_argument("--config", type=Path, default=ROOT/"config"/"default.yaml")
    parser.add_argument("--no-figures", action="store_true", help="Only generate CSV and JSON")
    args = parser.parse_args(argv)
    result = run_subject(args.subject, args.data_root, args.output_root,
                         args.config, figures=not args.no_figures)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"Outputs: {result['output_dir']}")


if __name__ == "__main__":
    main()
