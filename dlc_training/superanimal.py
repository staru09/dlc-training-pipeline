"""SuperAnimal transfer learning utilities."""

from pathlib import Path
from typing import Dict, Optional, List, Union

import numpy as np

import deeplabcut
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.core.engine import Engine


# Default keypoint mappings for common projects
QUADRUPED_MAPPINGS = {
    # Front legs
    "L_F_Paw": "front_left_paw",
    "L_F_Knee": "front_left_knee",
    "L_F_Elbow": "front_left_thai",
    "R_F_Paw": "front_right_paw",
    "R_F_Knee": "front_right_knee",
    "R_F_Elbow": "front_right_thai",
    # Back legs
    "L_B_Paw": "back_left_paw",
    "L_B_Knee": "back_left_knee",
    "L_B_Elbow": "back_left_thai",
    "R_B_Paw": "back_right_paw",
    "R_B_Knee": "back_right_knee",
    "R_B_Elbow": "back_right_thai",
    # Tail
    "TailBase": "tail_base",
    "TailEnd": "tail_end",
    # Ears
    "L_EarBase": "left_earbase",
    "R_EarBase": "right_earbase",
    "L_Ear_Tip": "left_earend",
    "R_Ear_Tip": "right_earend",
    # Eyes
    "L_Eye": "left_eye",
    "R_Eye": "right_eye",
    # Face
    "Nose": "nose",
    "Chin": "throat_end",
    # Torso
    "Withers": "neck_base",
    "Throat": "throat_base",
}


def create_keypoint_mapping(
    config_path: Union[str, Path],
    superanimal_name: str = "superanimal_quadruped",
    mapping: Optional[Dict[str, str]] = None,
) -> None:
    """Create conversion table mapping project keypoints to SuperAnimal.
    
    Args:
        config_path: Path to project config.yaml
        superanimal_name: Target SuperAnimal model
        mapping: Dict mapping {project_keypoint: superanimal_keypoint}
                Uses QUADRUPED_MAPPINGS if not provided
                
    Example:
        >>> create_keypoint_mapping(
        ...     config_path="./config.yaml",
        ...     mapping={"my_nose": "nose", "my_tail": "tail_base"}
        ... )
    """
    if mapping is None:
        mapping = QUADRUPED_MAPPINGS
        
    deeplabcut.modelzoo.create_conversion_table(
        config=str(config_path),
        super_animal=superanimal_name,
        project_to_super_animal=mapping,
    )


def setup_weight_init(
    superanimal_name: str = "superanimal_quadruped",
    model_name: str = "hrnet_w32",
    with_decoder: bool = False,
    snapshot_path: Optional[str] = None,
    detector_name: Optional[str] = None,
    detector_snapshot_path: Optional[str] = None,
    memory_replay: bool = False,
    conversion_array: Optional[np.ndarray] = None,
) -> WeightInitialization:
    """Create weight initialization for transfer learning.
    
    Args:
        superanimal_name: SuperAnimal model name
        model_name: Pose model architecture
        with_decoder: Include decoder weights
        snapshot_path: Custom pose model weights path
        detector_name: Detector architecture (for top-down)
        detector_snapshot_path: Custom detector weights path
        memory_replay: Enable memory replay training
        conversion_array: Keypoint index mapping array
        
    Returns:
        WeightInitialization object for training
        
    Example:
        >>> weight_init = setup_weight_init(
        ...     superanimal_name="superanimal_quadruped",
        ...     model_name="hrnet_w32",
        ...     with_decoder=True,
        ...     memory_replay=True
        ... )
    """
    if memory_replay:
        from deeplabcut.modelzoo import build_weight_init
        return build_weight_init(
            cfg=None,  # Will be set during training
            super_animal=superanimal_name,
            model_name=model_name,
            detector_name=detector_name or "fasterrcnn_resnet50_fpn_v2",
            with_decoder=True,
            memory_replay=True,
        )
    
    kwargs = {
        "dataset": superanimal_name,
        "with_decoder": with_decoder,
    }
    
    if snapshot_path:
        kwargs["snapshot_path"] = snapshot_path
    if detector_snapshot_path:
        kwargs["detector_snapshot_path"] = detector_snapshot_path
    if conversion_array is not None:
        kwargs["conversion_array"] = conversion_array
        
    return WeightInitialization(**kwargs)


def train_with_transfer_learning(
    config_path: Union[str, Path],
    from_shuffle: int,
    target_shuffle: int,
    superanimal_name: str = "superanimal_quadruped",
    model_name: str = "hrnet_w32",
    epochs: int = 50,
    batch_size: int = 8,
    save_epochs: int = 10,
    with_decoder: bool = False,
) -> None:
    """Train network using SuperAnimal pretrained weights.
    
    Args:
        config_path: Path to project config.yaml
        from_shuffle: Source shuffle for data split
        target_shuffle: New shuffle to create
        superanimal_name: SuperAnimal model to use
        model_name: Pose model architecture
        epochs: Training epochs
        batch_size: Training batch size
        save_epochs: Save every N epochs
        with_decoder: Include decoder weights
        
    Example:
        >>> train_with_transfer_learning(
        ...     config_path="./config.yaml",
        ...     from_shuffle=1,
        ...     target_shuffle=2,
        ...     epochs=100
        ... )
    """
    # Create weight initialization
    weight_init = setup_weight_init(
        superanimal_name=superanimal_name,
        model_name=model_name,
        with_decoder=with_decoder,
    )
    
    # Map model name to net_type
    net_type = f"top_down_{model_name}"
    
    # Create dataset with transfer learning
    deeplabcut.create_training_dataset_from_existing_split(
        config=str(config_path),
        from_shuffle=from_shuffle,
        shuffles=[target_shuffle],
        engine=Engine.PYTORCH,
        net_type=net_type,
        weight_init=weight_init,
        userfeedback=False,
    )
    
    # Train
    deeplabcut.train_network(
        config=str(config_path),
        shuffle=target_shuffle,
        epochs=epochs,
        batch_size=batch_size,
        save_epochs=save_epochs,
    )


def train_with_memory_replay(
    config_path: Union[str, Path],
    from_shuffle: int,
    target_shuffle: int,
    superanimal_name: str = "superanimal_quadruped",
    model_name: str = "hrnet_w32",
    detector_name: str = "fasterrcnn_resnet50_fpn_v2",
    epochs: int = 50,
    batch_size: int = 8,
) -> None:
    """Train with memory replay to prevent catastrophic forgetting.
    
    Memory replay mixes SuperAnimal data with your project data during training.
    
    Args:
        config_path: Path to project config.yaml
        from_shuffle: Source shuffle for data split
        target_shuffle: New shuffle to create
        superanimal_name: SuperAnimal model
        model_name: Pose model architecture
        detector_name: Detector architecture
        epochs: Training epochs
        batch_size: Training batch size
    """
    from deeplabcut.modelzoo import build_weight_init
    
    weight_init = build_weight_init(
        cfg=str(config_path),
        super_animal=superanimal_name,
        model_name=model_name,
        detector_name=detector_name,
        with_decoder=True,
        memory_replay=True,
    )
    
    print(f"Pose checkpoint: {weight_init.snapshot_path}")
    print(f"Detector checkpoint: {weight_init.detector_snapshot_path}")
    
    deeplabcut.create_training_dataset_from_existing_split(
        config=str(config_path),
        from_shuffle=from_shuffle,
        shuffles=[target_shuffle],
        weight_init=weight_init,
        engine=Engine.PYTORCH,
    )
    
    deeplabcut.train_network(
        config=str(config_path),
        shuffle=target_shuffle,
        epochs=epochs,
        batch_size=batch_size,
    )
