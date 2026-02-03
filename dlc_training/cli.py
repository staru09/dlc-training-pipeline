"""Command-line interface for dlc_training."""

import argparse
from pathlib import Path

from . import project, dataset, trainer, evaluator, analyzer, visualizer


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="dlc_training",
        description="DeepLabCut Training Pipeline - Modular training utilities",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # create-project
    p = subparsers.add_parser("create-project", help="Create new DLC project")
    p.add_argument("--name", required=True, help="Project name")
    p.add_argument("--experimenter", required=True, help="Experimenter/scorer name")
    p.add_argument("--videos", nargs="*", default=[], help="Video paths")
    p.add_argument("--working-dir", help="Working directory")
    p.add_argument("--multianimal", action="store_true", help="Multi-animal project")
    
    # create-dataset
    p = subparsers.add_parser("create-dataset", help="Create training dataset")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--net-type", default="resnet_50", help="Network type")
    p.add_argument("--num-shuffles", type=int, default=1, help="Number of shuffles")
    p.add_argument("--detector-type", help="Detector type for top-down models")
    
    # train
    p = subparsers.add_parser("train", help="Train network")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--shuffle", type=int, default=1, help="Shuffle index")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size")
    p.add_argument("--snapshot", help="Resume from snapshot")
    
    # evaluate
    p = subparsers.add_parser("evaluate", help="Evaluate network")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--shuffles", nargs="+", type=int, default=[1], help="Shuffles")
    p.add_argument("--snapshots", nargs="*", help="Specific snapshots")
    
    # analyze
    p = subparsers.add_parser("analyze", help="Analyze videos")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--videos", nargs="+", required=True, help="Video paths")
    p.add_argument("--shuffle", type=int, default=1, help="Shuffle index")
    p.add_argument("--save-csv", action="store_true", help="Save as CSV")
    
    # create-video
    p = subparsers.add_parser("create-video", help="Create labeled video")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--videos", nargs="+", required=True, help="Video paths")
    p.add_argument("--shuffle", type=int, default=1, help="Shuffle index")
    p.add_argument("--pcutoff", type=float, default=0.6, help="Confidence threshold")
    p.add_argument("--skeleton", action="store_true", help="Draw skeleton")
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
        
    if args.command == "create-project":
        config_path = project.create_project(
            project_name=args.name,
            experimenter=args.experimenter,
            videos=args.videos,
            working_directory=args.working_dir,
            multianimal=args.multianimal,
        )
        print(f"Created project: {config_path}")
        
    elif args.command == "create-dataset":
        dataset.create_training_dataset(
            config_path=args.config,
            net_type=args.net_type,
            num_shuffles=args.num_shuffles,
            detector_type=args.detector_type,
        )
        print("Training dataset created")
        
    elif args.command == "train":
        trainer.train_network(
            config_path=args.config,
            shuffle=args.shuffle,
            epochs=args.epochs,
            batch_size=args.batch_size,
            snapshot_path=args.snapshot,
        )
        
    elif args.command == "evaluate":
        evaluator.evaluate_network(
            config_path=args.config,
            shuffles=args.shuffles,
            snapshots_to_evaluate=args.snapshots,
        )
        
    elif args.command == "analyze":
        analyzer.analyze_videos(
            config_path=args.config,
            videos=args.videos,
            shuffle=args.shuffle,
            save_as_csv=args.save_csv,
        )
        
    elif args.command == "create-video":
        visualizer.create_labeled_video(
            config_path=args.config,
            videos=args.videos,
            shuffle=args.shuffle,
            pcutoff=args.pcutoff,
            draw_skeleton=args.skeleton,
        )


if __name__ == "__main__":
    main()
