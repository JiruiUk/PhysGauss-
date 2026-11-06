import argparse
from .config import TrainConfig
from .data import SceneDataset, make_synthetic
from .engine.trainer import train
from .engine.evaluate import evaluate, export_state


def main(argv=None):
    parser = argparse.ArgumentParser(prog="physgauss")
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("make-synthetic", help="Generate a tiny self-consistency scene")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--frames", type=int, default=9)
    synthetic.add_argument("--size", type=int, default=24)
    synthetic.add_argument("--particles", type=int, default=12)
    validate = commands.add_parser("validate-data", help="Validate a scene manifest and array contracts")
    validate.add_argument("--manifest", required=True)
    training = commands.add_parser("train")
    training.add_argument("--manifest", required=True)
    training.add_argument("--config", default="configs/smoke.json")
    training.add_argument("--output", required=True)
    training.add_argument("--steps", type=int)
    training.add_argument("--device")
    training.add_argument("--resume")
    evaluation = commands.add_parser("evaluate")
    export = commands.add_parser("export-state")
    for command in (evaluation, export):
        command.add_argument("--manifest", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--device", default="cpu")
    export.add_argument("--time", required=True, type=float)
    args = parser.parse_args(argv)
    if args.command == "make-synthetic":
        print(make_synthetic(args.output, args.frames, args.size, args.particles))
    elif args.command == "validate-data":
        dataset = SceneDataset(args.manifest)
        print(f"valid: {dataset.metadata['protocol']}; frames={dataset.count}; gaussians={len(dataset.xyz)}; "
              f"train={len(dataset.train_indices)}; test={len(dataset.test_indices)}")
    elif args.command == "train":
        config = TrainConfig.load(args.config)
        if args.steps is not None:
            config.steps = args.steps
        if args.device:
            config.device = args.device
        config.validate()
        dataset = SceneDataset(args.manifest, config.device)
        if not float(dataset.arrays["times"].min()) <= config.canonical_time <= float(dataset.arrays["times"].max()):
            raise ValueError("canonical_time must lie within the observed interval")
        print(train(dataset, config, args.output, args.resume))
    elif args.command == "evaluate":
        evaluate(SceneDataset(args.manifest, args.device), args.checkpoint, args.output, args.device)
    elif args.command == "export-state":
        print(export_state(SceneDataset(args.manifest, args.device), args.checkpoint,
                           args.time, args.output, args.device))
