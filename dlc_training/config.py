"""Training configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class TrainingConfig:
    """Configuration for DLC training pipeline.
    
    Attributes:
        config_path: Path to DLC project config.yaml
        shuffle: Training shuffle index (default: 1)
        batch_size: Training batch size (default: 8)
        epochs: Number of training epochs (default: 100)
        detector_epochs: Detector training epochs for top-down models
        detector_batch_size: Detector batch size
        snapshot_path: Resume training from this snapshot
        detector_path: Resume detector training from this path
        net_type: Network architecture type
        augmenter_type: Data augmentation type (default: "albumentations")
        save_epochs: Save model every N epochs
        superanimal_name: SuperAnimal model name for transfer learning
    """
    config_path: Path
    shuffle: int = 1
    batch_size: int = 8
    epochs: int = 100
    detector_epochs: Optional[int] = None
    detector_batch_size: Optional[int] = None
    snapshot_path: Optional[Path] = None
    detector_path: Optional[Path] = None
    net_type: str = "resnet_50"
    augmenter_type: str = "albumentations"
    save_epochs: int = 10
    superanimal_name: Optional[str] = None
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.config_path, str):
            self.config_path = Path(self.config_path)
        if isinstance(self.snapshot_path, str):
            self.snapshot_path = Path(self.snapshot_path)
        if isinstance(self.detector_path, str):
            self.detector_path = Path(self.detector_path)


# Common network architectures
NET_TYPES = {
    "bottom_up": [
        "resnet_50",
        "resnet_101", 
    ],
    "top_down": [
        "top_down_hrnet_w32",
        "top_down_hrnet_w48",
        "ctd_prenet_rtmpose_x",
    ],
}

# Common SuperAnimal models
SUPERANIMAL_MODELS = [
    "superanimal_quadruped",
    "superanimal_topviewmouse",
]
