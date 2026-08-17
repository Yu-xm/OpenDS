from __future__ import annotations

import argparse

from .config import load_config
from .runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OmniBrowse with controlled Serper tools")
    parser.add_argument("--config", required=True, help="Path to a JSON run config")
    parser.add_argument("--resume", action="store_true", help="Skip samples that already have a trace")
    parser.add_argument("--limit", type=int, help="Run only the first N samples")
    args = parser.parse_args()
    run_dir = run(load_config(args.config), resume=args.resume, limit=args.limit)
    print(f"Done: {run_dir}")


if __name__ == "__main__":
    main()

