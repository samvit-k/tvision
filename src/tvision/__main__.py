import argparse
import sys

def main() -> int:
    parser = argparse.ArgumentParser(prog="tvision")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("explore", help="Validate and summarize the raw NuCLS dataset")
    p.add_argument("--overlays", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    sub.add_parser("splits", help="Assign leakage-safe train/val/test splits (configs/splits.yaml)")
    sub.add_parser("convert", help="Convert NuCLS to YOLO labels (configs/convert.yaml)")
    args = parser.parse_args()

    if args.command == "explore":
        from tvision.explore import explore
        return 0 if explore(args.overlays, args.seed) else 1
    if args.command == "splits":
        from tvision.splits import make_splits
        return 0 if make_splits() else 1
    if args.command == "convert":
        from tvision.convert import convert
        return 0 if convert() else 1
    return 2

if __name__ == "__main__":
    sys.exit(main())