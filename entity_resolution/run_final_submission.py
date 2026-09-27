from __future__ import annotations

import argparse

from src.lightgbm_pipeline import run_lightgbm_inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate final LightGBM challenge submission files.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    result = run_lightgbm_inference(
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        model_dir=args.model_dir,
        artifact_dir=args.artifact_dir,
        config_path=args.config,
    )
    print(result)
