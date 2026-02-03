"""Model evaluation functions."""

from pathlib import Path
from typing import List, Optional, Union

import deeplabcut


def evaluate_network(
    config_path: Union[str, Path],
    shuffles: List[int] = None,
    snapshots_to_evaluate: Optional[List[str]] = None,
    plotting: bool = True,
    show_errors: bool = True,
) -> None:
    """Evaluate a trained DeepLabCut network.
    
    Results are stored as CSV in evaluation-results-pytorch subdirectory.
    
    Args:
        config_path: Path to project config.yaml
        shuffles: List of shuffle indices to evaluate
        snapshots_to_evaluate: Specific snapshots (e.g., ["snapshot-best-100"])
        plotting: Generate evaluation plots
        show_errors: Display error metrics
        
    Example:
        >>> evaluate_network(
        ...     config_path="./config.yaml",
        ...     shuffles=[1],
        ...     snapshots_to_evaluate=["snapshot-best-150"]
        ... )
    """
    if shuffles is None:
        shuffles = [1]
        
    kwargs = {
        "config": str(config_path),
        "Shuffles": shuffles,
        "plotting": plotting,
        "show_errors": show_errors,
    }
    
    if snapshots_to_evaluate:
        kwargs["snapshots_to_evaluate"] = snapshots_to_evaluate
        
    deeplabcut.evaluate_network(**kwargs)


def evaluate_snapshots(
    config_path: Union[str, Path],
    shuffle: int,
    snapshots: List[str],
) -> None:
    """Evaluate multiple snapshots for comparison.
    
    Useful for finding the best training epoch.
    
    Args:
        config_path: Path to project config.yaml
        shuffle: Training shuffle index
        snapshots: List of snapshot names to evaluate
        
    Example:
        >>> evaluate_snapshots(
        ...     config_path="./config.yaml",
        ...     shuffle=1,
        ...     snapshots=["snapshot-050", "snapshot-100", "snapshot-best-150"]
        ... )
    """
    for snapshot in snapshots:
        print(f"Evaluating: {snapshot}")
        evaluate_network(
            config_path=config_path,
            shuffles=[shuffle],
            snapshots_to_evaluate=[snapshot],
        )
