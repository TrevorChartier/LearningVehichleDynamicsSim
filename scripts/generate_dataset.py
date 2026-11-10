import argparse
import json
import sys
from pathlib import Path

# Dynamically add the root project directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.pipelines.dataset_generator import DatasetGenerator

def main():
    parser = argparse.ArgumentParser(
        description="Run the dataset generation pipeline to sample environments and collect rollouts.",
        epilog="Example: python scripts/generate_dataset.py --config configs/dataset_config.json"
    )
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=True, 
        help="Path to the dataset configuration JSON file."
    )
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    print(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
            
    try:
        generator = DatasetGenerator(config=config)
        generator.run_pipeline()
    except FileExistsError as e:
        print(f"\n[Pipeline Aborted] {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"\n[Config Error] Missing required key in configuration file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()