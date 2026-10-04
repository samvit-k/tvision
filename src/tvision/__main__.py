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
    p = sub.add_parser("verify", help="Verify the YOLO conversion independently from the raw data")
    p.add_argument("--overlays", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("train", help="Fine-tune the pretrained detector (configs/train.yaml)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Apply the smoke overrides from configs/train.yaml")
    mode.add_argument("--check", action="store_true", help="Run only the verify gate and Ultralytics loader check")
    sub.add_parser("baseline-manifest", help="Hash every frozen baseline artifact once (Step 5A anchor)")
    sub.add_parser("two-class", help="Derive the two-class train/val dataset (configs/two_class.yaml)")
    sub.add_parser("comparability", help="Baseline best.pt on val vs. the ambiguous-removed 3-class view")
    p = sub.add_parser("aug-preview", help="Render augmented training samples of an experiment for review")
    p.add_argument("name")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("experiment", help="Train one Step 5A experiment (configs/experiments.yaml)")
    p.add_argument("name")
    p.add_argument("--seed", type=int, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Apply the smoke overrides from configs/experiments.yaml")
    mode.add_argument("--check", action="store_true", help="Run only the gates and Ultralytics loader check")
    sub.add_parser("compare", help="Compare Step 5A runs on val")
    sub.add_parser("select", help="Apply the validation-only selection rule and freeze the choice")
    p = sub.add_parser("evaluate", help="Evaluate the frozen baseline and selected model on the test split (once)")
    p.add_argument("--check", action="store_true", help="Run only the gates and the val phase; never touches test")
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
    if args.command == "verify":
        from tvision.verify import verify
        return 0 if verify(args.overlays, args.seed) else 1
    if args.command == "train":
        from tvision.train import train
        return 0 if train(args.smoke, args.check) else 1
    if args.command == "baseline-manifest":
        from tvision.experiments import baseline_manifest
        return 0 if baseline_manifest() else 1
    if args.command == "two-class":
        from tvision.two_class import two_class
        return 0 if two_class() else 1
    if args.command == "comparability":
        from tvision.experiments import Context, comparability
        return 0 if comparability(Context.load()) else 1
    if args.command == "aug-preview":
        from tvision.experiments import Context, aug_preview
        aug_preview(Context.load(), args.name, args.n, args.variants, args.seed)
        return 0
    if args.command == "experiment":
        from tvision.experiments import experiment
        return 0 if experiment(args.name, args.seed, args.smoke, args.check) else 1
    if args.command == "compare":
        from tvision.selection import compare_runs
        return 0 if compare_runs() else 1
    if args.command == "select":
        from tvision.selection import select_final
        return 0 if select_final() else 1
    if args.command == "evaluate":
        from tvision.evaluate import evaluate
        return 0 if evaluate(args.check) else 1
    return 2

if __name__ == "__main__":
    sys.exit(main())