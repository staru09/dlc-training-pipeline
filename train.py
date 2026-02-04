#!/usr/bin/env python3
"""
Train a DeepLabCut model.

Usage:
    python train.py --config path/to/config.yaml
    python train.py --config path/to/config.yaml --superanimal
    python train.py --config path/to/config.yaml --epochs 100 --batch-size 4
"""

import argparse
from pathlib import Path
import sys


def train_model(
    config_path: str,
    epochs: int = 200,
    batch_size: int = 8,
    use_superanimal: bool = False,
    shuffle: int = 1
):
    """Train a DLC model with progress logging."""
    
    print("=" * 60)
    print("  DLC TRAINING")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"SuperAnimal: {use_superanimal}")
    print(f"Shuffle: {shuffle}")
    
    # Import DLC
    print("\n→ Loading DeepLabCut...")
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions
    
    print(f"  DLC version: {deeplabcut.__version__}")
    
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)
    
    cfg = auxiliaryfunctions.read_config(str(config_path))
    net_type = cfg.get("default_net_type", "resnet_50")
    print(f"  Network: {net_type}")
    
    # Step 1: Create training dataset
    print("\n" + "=" * 60)
    print("  STEP 1: Creating Training Dataset")
    print("=" * 60)
    
    try:
        deeplabcut.create_training_dataset(
            str(config_path),
            num_shuffles=1,
            net_type=net_type,
            augmenter_type="default",
        )
        print("✓ Training dataset created")
    except Exception as e:
        print(f"⚠ Dataset creation warning: {e}")
        print("  Continuing with existing dataset...")
    
    # Step 2: Train network
    print("\n" + "=" * 60)
    print("  STEP 2: Training Network")
    print("=" * 60)
    
    # Calculate max iterations
    maxiters = epochs * 1000  # DLC uses iterations, roughly 1000 per epoch
    
    train_kwargs = {
        "shuffle": shuffle,
        "max_snapshots_to_keep": 5,
        "maxiters": maxiters,
        "displayiters": 500,
        "saveiters": 5000,
    }
    
    # SuperAnimal weight initialization (DLC 3.0+)
    if use_superanimal:
        print("→ Attempting SuperAnimal transfer learning...")
        try:
            # Try different import paths for WeightInitialization
            WeightInitialization = None
            try:
                from deeplabcut.pose_estimation_pytorch.config import WeightInitialization
            except ImportError:
                try:
                    from deeplabcut.pose_estimation_pytorch.runners.train import WeightInitialization
                except ImportError:
                    try:
                        from deeplabcut.pose_estimation_pytorch import WeightInitialization
                    except ImportError:
                        pass
            
            if WeightInitialization:
                weight_init = WeightInitialization(
                    dataset="superanimal_quadruped",
                    with_decoder=False
                )
                train_kwargs["weight_init"] = weight_init
                print("✓ SuperAnimal weights configured")
            else:
                print("⚠ WeightInitialization not available in this DLC version")
                print("  Training without pretrained weights...")
        except Exception as e:
            print(f"⚠ SuperAnimal setup failed: {e}")
            print("  Training without pretrained weights...")
    
    print(f"\n→ Starting training...")
    print(f"  Max iterations: {maxiters}")
    print(f"  Save every: {train_kwargs['saveiters']} iters")
    print("-" * 50)
    
    try:
        deeplabcut.train_network(str(config_path), **train_kwargs)
        print("\n✓ Training complete!")
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        print("  Snapshots saved - you can resume training later")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise
    
    # Step 3: Evaluate
    print("\n" + "=" * 60)
    print("  STEP 3: Evaluating Network")
    print("=" * 60)
    
    try:
        deeplabcut.evaluate_network(
            str(config_path),
            Shuffles=[shuffle],
            plotting=True
        )
        print("✓ Evaluation complete!")
        print(f"  Results saved in: {Path(config_path).parent / 'evaluation-results'}")
    except Exception as e:
        print(f"⚠ Evaluation warning: {e}")
    
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Check evaluation: {Path(config_path).parent / 'evaluation-results'}")
    print(f"  2. Run inference: python inference.py --config {config_path} --video your_video.mp4")


def main():
    parser = argparse.ArgumentParser(
        description="Train a DeepLabCut model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --config project/config.yaml
  python train.py --config project/config.yaml --superanimal
  python train.py --config project/config.yaml --epochs 100 --batch-size 4
        """
    )
    parser.add_argument("--config", "-c", required=True, help="Path to config.yaml")
    parser.add_argument("--epochs", "-e", type=int, default=200, help="Training epochs (default: 200)")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--superanimal", "-s", action="store_true", help="Use SuperAnimal transfer learning")
    parser.add_argument("--shuffle", type=int, default=1, help="Shuffle index (default: 1)")
    
    args = parser.parse_args()
    
    train_model(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_superanimal=args.superanimal,
        shuffle=args.shuffle
    )


if __name__ == "__main__":
    main()
