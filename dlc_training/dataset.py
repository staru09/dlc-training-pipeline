"""Training dataset creation."""

from pathlib import Path
from typing import List, Optional, Union

import deeplabcut
from deeplabcut.core.engine import Engine


def create_training_dataset(
    config_path: Union[str, Path],
    num_shuffles: int = 1,
    net_type: str = "resnet_50",
    augmenter_type: str = "albumentations",
    userfeedback: bool = False,
    detector_type: Optional[str] = None,
    weight_init = None,
) -> None:
    """Create training dataset for DLC model.
    
    Args:
        config_path: Path to project config.yaml
        num_shuffles: Number of training shuffles to create
        net_type: Network architecture type. Options:
            - Bottom-up: "resnet_50", "resnet_101"
            - Top-down: "top_down_hrnet_w32", "top_down_hrnet_w48"
            - CTD: "ctd_prenet_rtmpose_x"
        augmenter_type: Data augmentation type
        userfeedback: Prompt for user confirmation
        detector_type: Detector for top-down models (e.g., "fasterrcnn_resnet50_fpn_v2")
        weight_init: WeightInitialization object for transfer learning
        
    Example:
        >>> # Create dataset for top-down model with detector
        >>> create_training_dataset(
        ...     config_path="./project/config.yaml",
        ...     net_type="ctd_prenet_rtmpose_x",
        ...     detector_type="fasterrcnn_resnet50_fpn_v2"
        ... )
    """
    kwargs = {
        "config": str(config_path),
        "num_shuffles": num_shuffles,
        "net_type": net_type,
        "augmenter_type": augmenter_type,
        "userfeedback": userfeedback,
    }
    
    if detector_type:
        kwargs["detector_type"] = detector_type
    if weight_init:
        kwargs["weight_init"] = weight_init
        
    deeplabcut.create_training_dataset(**kwargs)


def create_dataset_from_existing_split(
    config_path: Union[str, Path],
    from_shuffle: int,
    shuffles: List[int],
    net_type: str = "resnet_50",
    engine = None,
    weight_init = None,
    userfeedback: bool = False,
) -> None:
    """Create new training dataset from existing shuffle's train/test split.
    
    Useful for model comparison or transfer learning with same data split.
    
    Args:
        config_path: Path to project config.yaml
        from_shuffle: Source shuffle to copy split from
        shuffles: List of new shuffle indices to create
        net_type: Network architecture for new shuffles
        engine: DLC engine (PYTORCH or TF)
        weight_init: WeightInitialization for transfer learning
        userfeedback: Prompt for user confirmation
        
    Example:
        >>> # Create shuffle 2 with ResNet from shuffle 1's split
        >>> create_dataset_from_existing_split(
        ...     config_path="./config.yaml",
        ...     from_shuffle=1,
        ...     shuffles=[2],
        ...     net_type="resnet_50",
        ...     engine=Engine.PYTORCH
        ... )
    """
    if engine is None:
        engine = Engine.PYTORCH
        
    deeplabcut.create_training_dataset_from_existing_split(
        config=str(config_path),
        from_shuffle=from_shuffle,
        shuffles=shuffles,
        net_type=net_type,
        engine=engine,
        weight_init=weight_init,
        userfeedback=userfeedback,
    )


def create_training_comparison(
    config_path: Union[str, Path],
    net_types: List[str] = None,
    num_shuffles: int = 1,
) -> None:
    """Create training datasets for model comparison.
    
    Creates multiple shuffles with different architectures for benchmarking.
    
    Args:
        config_path: Path to project config.yaml
        net_types: List of network types to compare
        num_shuffles: Shuffles per network type
    """
    if net_types is None:
        net_types = ["resnet_50", "top_down_hrnet_w32"]
        
    deeplabcut.create_training_model_comparison(
        config=str(config_path),
        net_types=net_types,
        num_shuffles=num_shuffles,
    )
