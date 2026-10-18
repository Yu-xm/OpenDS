from __future__ import annotations

import argparse

from .config import load_config
from .runner import execution_statuses, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OpenDS-World with controlled Serper tools")
    parser.add_argument("--config", required=True, help="Path to a YAML or JSON run config")
    parser.add_argument(
        "--model",
        "--model-name",
        dest="model_name",
        help="Temporarily override model.name for this run without editing the config file",
    )
    parser.add_argument("--resume", action="store_true", help="Skip samples that already have a trace")
    parser.add_argument("--limit", type=int, help="Run only the first N samples")
    parser.add_argument("--index", type=int, help="Run one CSV data row (1-based, not sample ID), in an isolated output directory")
    parser.add_argument("--mineru-service", choices=("start", "stop"), help="Explicitly manage the one shared MinerU container; do not evaluate")
    return parser


def apply_model_override(config: dict, model_name: str | None) -> dict:
    if model_name is None:
        return config
    model_name = model_name.strip()
    if not model_name:
        raise ValueError("--model must be a non-empty string")
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("config.model must be an object")
    model["name"] = model_name
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = apply_model_override(load_config(args.config), args.model_name)
    if args.mineru_service:
        from pathlib import Path
        from toolbox.file_services import mineru_service
        readers = config.get("toolbox", {}).get("file_readers", {})
        if not (readers.get("mineru", {}).get("docker") or {}).get("enabled", False):
            print("MinerU uses an external or disabled service; no container changed.")
            return
        mineru_service(readers.get("mineru", {}), Path(readers.get("cache_dir", ".cache/toolbox/files")), stop=args.mineru_service == "stop")
        print(f"MinerU service: {args.mineru_service}")
        return
    run_dir = run(config, resume=args.resume, limit=args.limit, index=args.index)
    print(f"Done: {run_dir}")
    statuses = execution_statuses(run_dir)
    print("Statuses: " + ", ".join(f"{key}={value}" for key, value in sorted(statuses.items())))
    if statuses.get("infrastructure_error"):
        print(
            "Evaluation contains infrastructure_error rows; do not score them as model failures."
        )
        raise SystemExit(2)
    if any(status != "success" and count for status, count in statuses.items()):
        print("Evaluation contains incomplete or failed cases; inspect their finalization status and use explicit trace recovery.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
