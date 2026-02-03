"""
Command-line interface for rf_to_dlc package.
"""

import argparse
from pathlib import Path

from . import (
    Config,
    copy_images_to_project,
    create_all_videos,
    create_frame_order_csv,
    create_multi_animal_headers,
    create_single_animal_headers,
    download_and_extract,
    find_missing_keypoints,
    format_dataframe,
    get_image_names_from_df,
    parse_all_annotations,
    remove_duplicates,
    save_collected_data,
    convert_to_hdf5,
    verify_keypoint_mapping,
)


def cmd_download(args):
    """Download and extract Roboflow dataset."""
    download_and_extract(
        url=args.url,
        output_dir=Path(args.output_dir),
        keep_zip=args.keep_zip,
    )


def cmd_create_videos(args):
    """Create videos from image sequences."""
    create_all_videos(
        base_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


def cmd_convert(args):
    """Convert COCO annotations to DLC format."""
    base_dir = Path(args.input_dir)
    output_path = Path(args.output)

    # Parse annotations
    print(f"📂 Parsing annotations from: {base_dir}")
    df, keypoints = parse_all_annotations(
        base_dir,
        image_prefix=args.image_prefix,
    )
    print(f"📊 Found {len(df)} annotations with {len(keypoints)} keypoints")

    # Create headers
    if args.multi_animal:
        individuals = args.individuals.split(",") if args.individuals else ["individual1"]
        headers = create_multi_animal_headers(args.scorer, individuals, keypoints)
        header_rows = 4
    else:
        headers = create_single_animal_headers(args.scorer, keypoints)
        header_rows = 3

    # Format DataFrame
    df_formatted = format_dataframe(df, headers)

    # Remove duplicates
    df_unique = remove_duplicates(df_formatted, header_rows=header_rows)
    print(f"📋 Unique images: {len(df_unique) - header_rows}")

    # Save
    save_collected_data(df_unique, output_path)

    # Optionally find missing keypoints
    if args.report_missing:
        missing_df = find_missing_keypoints(base_dir)
        if not missing_df.empty:
            missing_path = output_path.with_name(
                output_path.stem + "_missing_keypoints.csv"
            )
            missing_df.to_csv(missing_path, index=False)
            print(f"⚠️ Found {len(missing_df)} annotations with missing keypoints")
            print(f"   Report saved to: {missing_path}")


def cmd_copy_images(args):
    """Copy images to DLC project folder."""
    import pandas as pd

    csv_path = Path(args.csv)
    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)

    # Load CSV
    df = pd.read_csv(csv_path, header=None)
    header_rows = 4 if args.multi_animal else 3

    # Get image names
    image_names = get_image_names_from_df(df, header_rows=header_rows)

    # Copy
    copy_images_to_project(image_names, source_dir, dest_dir)


def cmd_visualize(args):
    """Visualize keypoint annotations on sample images."""
    verify_keypoint_mapping(
        csv_path=Path(args.csv),
        image_dir=Path(args.image_dir),
        num_samples=args.samples,
        header_rows=4 if args.multi_animal else 3,
        save_dir=Path(args.save_dir) if args.save_dir else None,
    )


def cmd_frame_order(args):
    """Create frame order CSV for the dataset."""
    create_frame_order_csv(
        base_dir=Path(args.input_dir),
        output_path=Path(args.output),
    )


def cmd_to_hdf5(args):
    """Convert CollectedData CSV to HDF5 format."""
    convert_to_hdf5(
        csv_path=Path(args.csv),
        output_path=Path(args.output) if args.output else None,
        header_rows=4 if args.multi_animal else 3,
    )


def cmd_full_pipeline(args):
    """Run the complete conversion pipeline."""
    base_dir = Path(args.output_dir)
    project_dir = Path(args.project_dir)

    # Step 1: Download
    if args.url:
        print("\n" + "=" * 50)
        print("STEP 1: Downloading dataset")
        print("=" * 50)
        download_and_extract(args.url, base_dir)

    # Step 2: Create videos (optional)
    if args.create_videos:
        print("\n" + "=" * 50)
        print("STEP 2: Creating videos")
        print("=" * 50)
        create_all_videos(base_dir)

    # Step 3: Parse and convert annotations
    print("\n" + "=" * 50)
    print("STEP 3: Converting annotations")
    print("=" * 50)
    df, keypoints = parse_all_annotations(base_dir, image_prefix=args.image_prefix)
    print(f"📊 Found {len(df)} annotations with {len(keypoints)} keypoints")

    # Create headers
    if args.multi_animal:
        individuals = args.individuals.split(",") if args.individuals else ["individual1"]
        headers = create_multi_animal_headers(args.scorer, individuals, keypoints)
        header_rows = 4
    else:
        headers = create_single_animal_headers(args.scorer, keypoints)
        header_rows = 3

    df_formatted = format_dataframe(df, headers)
    df_unique = remove_duplicates(df_formatted, header_rows=header_rows)

    # Step 4: Save to project
    print("\n" + "=" * 50)
    print("STEP 4: Saving to DLC project")
    print("=" * 50)
    labeled_data_dir = project_dir / "labeled-data" / args.video_name
    csv_path = labeled_data_dir / f"CollectedData_{args.scorer}.csv"

    save_collected_data(df_unique, csv_path)

    # Step 5: Copy images
    print("\n" + "=" * 50)
    print("STEP 5: Copying images")
    print("=" * 50)
    image_names = get_image_names_from_df(df_unique, header_rows=header_rows)
    copy_images_to_project(image_names, base_dir, labeled_data_dir)

    print("\n" + "=" * 50)
    print("✅ Pipeline complete!")
    print("=" * 50)
    print(f"Project: {project_dir}")
    print(f"Labeled data: {labeled_data_dir}")
    print(f"\nNext steps:")
    print(f"  1. Open your DLC project notebook")
    print(f"  2. Convert CSV to H5 using deeplabcut.convertcsv2h5()")
    print(f"  3. Create training dataset and train!")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="rf_to_dlc",
        description="Convert Roboflow annotated datasets to DeepLabCut format",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    p_download = subparsers.add_parser("download", help="Download Roboflow dataset")
    p_download.add_argument("--url", required=True, help="Roboflow download URL")
    p_download.add_argument("--output-dir", required=True, help="Output directory")
    p_download.add_argument("--keep-zip", action="store_true", help="Keep zip file")
    p_download.set_defaults(func=cmd_download)

    # Create videos command
    p_videos = subparsers.add_parser("create-videos", help="Create videos from frames")
    p_videos.add_argument("--input-dir", required=True, help="Directory with images")
    p_videos.add_argument("--output-dir", help="Output directory for videos")
    p_videos.set_defaults(func=cmd_create_videos)

    # Convert command
    p_convert = subparsers.add_parser("convert", help="Convert COCO to DLC format")
    p_convert.add_argument("--input-dir", required=True, help="Dataset directory")
    p_convert.add_argument("--output", required=True, help="Output CSV path")
    p_convert.add_argument("--scorer", default="j", help="Scorer name")
    p_convert.add_argument("--multi-animal", action="store_true", help="Multi-animal format")
    p_convert.add_argument("--individuals", help="Comma-separated individual names")
    p_convert.add_argument("--image-prefix", default="labeled-data/video", help="Image path prefix")
    p_convert.add_argument("--report-missing", action="store_true", help="Report missing keypoints")
    p_convert.set_defaults(func=cmd_convert)

    # Copy images command
    p_copy = subparsers.add_parser("copy-images", help="Copy images to DLC project")
    p_copy.add_argument("--csv", required=True, help="CollectedData CSV path")
    p_copy.add_argument("--source-dir", required=True, help="Source image directory")
    p_copy.add_argument("--dest-dir", required=True, help="Destination directory")
    p_copy.add_argument("--multi-animal", action="store_true", help="Multi-animal format")
    p_copy.set_defaults(func=cmd_copy_images)

    # Visualize command
    p_viz = subparsers.add_parser("visualize", help="Visualize annotations")
    p_viz.add_argument("--csv", required=True, help="CollectedData CSV path")
    p_viz.add_argument("--image-dir", required=True, help="Image directory")
    p_viz.add_argument("--samples", type=int, default=5, help="Number of samples")
    p_viz.add_argument("--multi-animal", action="store_true", help="Multi-animal format")
    p_viz.add_argument("--save-dir", help="Directory to save visualizations")
    p_viz.set_defaults(func=cmd_visualize)

    # Frame order command
    p_frames = subparsers.add_parser("frame-order", help="Create frame order CSV")
    p_frames.add_argument("--input-dir", required=True, help="Dataset directory")
    p_frames.add_argument("--output", required=True, help="Output CSV path")
    p_frames.set_defaults(func=cmd_frame_order)

    # To HDF5 command
    p_hdf5 = subparsers.add_parser("to-hdf5", help="Convert CSV to HDF5")
    p_hdf5.add_argument("--csv", required=True, help="CollectedData CSV path")
    p_hdf5.add_argument("--output", help="Output H5 path")
    p_hdf5.add_argument("--multi-animal", action="store_true", help="Multi-animal format")
    p_hdf5.set_defaults(func=cmd_to_hdf5)

    # Full pipeline command
    p_full = subparsers.add_parser("full-pipeline", help="Run complete pipeline")
    p_full.add_argument("--url", help="Roboflow download URL (optional if already downloaded)")
    p_full.add_argument("--output-dir", required=True, help="Dataset output directory")
    p_full.add_argument("--project-dir", required=True, help="DLC project directory")
    p_full.add_argument("--video-name", default="video", help="Video/folder name in labeled-data")
    p_full.add_argument("--scorer", default="j", help="Scorer name")
    p_full.add_argument("--multi-animal", action="store_true", help="Multi-animal format")
    p_full.add_argument("--individuals", help="Comma-separated individual names")
    p_full.add_argument("--image-prefix", default="labeled-data/video", help="Image path prefix")
    p_full.add_argument("--create-videos", action="store_true", help="Also create videos")
    p_full.set_defaults(func=cmd_full_pipeline)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
