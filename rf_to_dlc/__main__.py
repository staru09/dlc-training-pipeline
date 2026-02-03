"""
Entry point for running rf_to_dlc as a module.

Usage:
    python -m rf_to_dlc <command> [options]
    
Examples:
    python -m rf_to_dlc --help
    python -m rf_to_dlc convert --input-dir ./dataset --output ./CollectedData.csv
    python -m rf_to_dlc full-pipeline --output-dir ./dataset --project-dir ./dlc_project
"""

from .cli import main

if __name__ == "__main__":
    main()
