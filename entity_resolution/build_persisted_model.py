from __future__ import annotations

from pathlib import Path

from src.lightgbm_pipeline import main


if __name__ == "__main__":
    import sys

    argv = ["lightgbm_pipeline.py", "--mode", "train"]
    if len(sys.argv) > 1:
        argv.extend(sys.argv[1:])
    import argparse

    parser = argparse.ArgumentParser(description="Train the LightGBM-only challenge model.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv[1:])

    from src.lightgbm_pipeline import train_lightgbm_only_experiment

    result = train_lightgbm_only_experiment(
        data_root=args.data_root,
        config_path=args.config,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        artifact_dir=args.artifact_dir,
        limit=args.limit,
    )
    print(result)
