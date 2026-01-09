import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["prepare", "train", "evaluate"],
        help="Pipeline stage to run"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model", default="models/final_model")
    args = parser.parse_args()

    if args.stage == "prepare":
        from stages import prepare
        prepare.main()
        
    elif args.stage == "train":
        from stages import train
        train.main(args.config, args.verbose)
        
    elif args.stage == "evaluate":
        from stages import evaluate
        evaluate.main(args.config, args.model)

if __name__ == "__main__":
    main()