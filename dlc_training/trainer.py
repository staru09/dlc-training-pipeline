"""Network training functions."""

from pathlib import Path
from typing import Optional, Dict, Any, Union


import deeplabcut


def train_network(
    config_path: Union[str, Path],
    shuffle: int = 1,
    epochs: int = 100,
    batch_size: int = 8,
    snapshot_path: Optional[Union[str, Path]] = None,
    detector_epochs: Optional[int] = None,
    detector_batch_size: Optional[int] = None,
    detector_path: Optional[Union[str, Path]] = None,
    save_epochs: int = 10,
    pytorch_cfg_updates: Optional[Dict[str, Any]] = None,
) -> None:
    """Train a DeepLabCut network.
    
    Args:
        config_path: Path to project config.yaml
        shuffle: Training shuffle index
        epochs: Number of epochs to train
        batch_size: Training batch size
        snapshot_path: Resume pose model from this snapshot
        detector_epochs: Epochs for detector training (top-down models)
        detector_batch_size: Batch size for detector
        detector_path: Resume detector from this snapshot
        save_epochs: Save model every N epochs
        pytorch_cfg_updates: Additional PyTorch config updates
        
    Example:
        >>> # Basic training
        >>> train_network(
        ...     config_path="./config.yaml",
        ...     epochs=200,
        ...     batch_size=8
        ... )
        
        >>> # Resume training from checkpoint
        >>> train_network(
        ...     config_path="./config.yaml",
        ...     epochs=50,
        ...     snapshot_path="./dlc-models/train/snapshot-best-100.pt"
        ... )
    """
    kwargs = {
        "config": str(config_path),
        "shuffle": shuffle,
        "epochs": epochs,
        "batch_size": batch_size,
        "save_epochs": save_epochs,
    }
    
    if snapshot_path:
        kwargs["snapshot_path"] = str(snapshot_path)
    if detector_epochs:
        kwargs["detector_epochs"] = detector_epochs
    if detector_batch_size:
        kwargs["detector_batch_size"] = detector_batch_size
    if detector_path:
        kwargs["detector_path"] = str(detector_path)
    if pytorch_cfg_updates:
        kwargs["pytorch_cfg_updates"] = pytorch_cfg_updates
        
    deeplabcut.train_network(**kwargs)


def finetune_network(
    config_path: Union[str, Path],
    snapshot_path: Union[str, Path],
    shuffle: int = 1,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate_factor: float = 0.1,
) -> None:
    """Fine-tune a trained network with lower learning rate.
    
    Typically used after initial training to refine the model.
    Note: You may need to manually adjust lr milestones in pytorch_config.yaml.
    
    Args:
        config_path: Path to project config.yaml
        snapshot_path: Path to trained snapshot to fine-tune
        shuffle: Training shuffle index
        epochs: Additional epochs for fine-tuning
        batch_size: Training batch size
        learning_rate_factor: Factor to reduce learning rate by
        
    Example:
        >>> finetune_network(
        ...     config_path="./config.yaml",
        ...     snapshot_path="./train/snapshot-best-150.pt",
        ...     epochs=50
        ... )
    """
    # Note: Learning rate adjustment must be done in pytorch_config.yaml
    # This wrapper just resumes training with the existing config
    print(f"Fine-tuning from: {snapshot_path}")
    print(f"Note: Adjust learning rate milestones in pytorch_config.yaml")
    print(f"Recommended: Reduce lr by factor of {learning_rate_factor}")
    
    train_network(
        config_path=config_path,
        shuffle=shuffle,
        epochs=epochs,
        batch_size=batch_size,
        snapshot_path=snapshot_path,
    )
