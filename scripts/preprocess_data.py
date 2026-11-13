import argparse
import json
import sys
from pathlib import Path

from data.preprocessing.preprocessor import DatasetPreprocessor

def main():
    parser = argparse.ArgumentParser(
        description="Run the dataset preprocessing pipeline to calculate splits and OOD holdouts.",
        epilog="Example: python scripts/preprocess_data.py --config configs/preprocessing/default.json"
    )
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=True, 
        help="Path to the preprocessing configuration JSON file."
    )
    
    # Force help menu if run blindly
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    print(f"Loading preprocessing configuration from {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
            
    try:
        preprocessor = DatasetPreprocessor(config=config)
        preprocessor.run()
    except FileNotFoundError as e:
        print(f"\n[I/O Error] {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"\n[Config Error] Missing required key in configuration file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()