from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import run_co3d_reprojection_metrics


def main():
    config_path = PROJECT_ROOT / "configs" / "metrics.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_co3d_reprojection_metrics(config, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()