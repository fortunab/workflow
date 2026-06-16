import argparse
import json
import yaml
from pathlib import Path
from app.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--yolo", default=None)
    args = parser.parse_args()

    config_path = args.config

    if args.yolo is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cfg["yolo_weights"] = args.yolo
        config_path = "configs/_tmp_pipeline.yaml"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    result = run_pipeline(args.image, config_path)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
