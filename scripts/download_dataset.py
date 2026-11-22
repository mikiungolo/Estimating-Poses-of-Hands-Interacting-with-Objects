#!/usr/bin/env python3
"""
Download DexYCB subset (10%) from Google Drive.
Uses DEXYCB_DRIVE_FILE_ID from environment/config.
Usage: python scripts/download_dataset.py --output /path/to/dataset/raw
"""
import argparse
import os
import sys
import zipfile

import gdown
from omegaconf import OmegaConf


def get_drive_url(config_path: str = "configs/config.yaml") -> str:
    """Get Drive URL from config or env."""
    config = OmegaConf.load(config_path)
    file_id = config.get("dexycb_drive_file_id", os.environ.get("DEXYCB_DRIVE_FILE_ID"))
    if not file_id:
        raise ValueError("DEXYCB_DRIVE_FILE_ID not found in config or env")
    return f"https://drive.google.com/uc?id={file_id}"


def is_dataset_extracted(output_dir: str) -> bool:
    """Check if dataset appears to be already extracted."""
    if not os.path.isdir(output_dir):
        return False
    try:
        # Check if directory has any files (not just specific extensions)
        files = os.listdir(output_dir)
        # Ignore the zip file itself
        files = [f for f in files if not f.endswith('.zip')]
        return len(files) > 0
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Download DexYCB subset from Google Drive")
    parser.add_argument("--output", default="./dataset/raw", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Re-download if exists")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "dexycb_subset.zip")

    # Check if already extracted
    if not args.force and is_dataset_extracted(args.output):
        print(f"Dataset already extracted in {args.output}")
        return

    # Check if zip already downloaded
    if os.path.exists(output_path) and not args.force:
        print(f"Zip already exists: {output_path}, extracting...")
    else:
        drive_url = get_drive_url(args.config)
        print("Downloading from Google Drive...")
        print(f"URL: {drive_url}")
        print(f"Destination: {output_path}")

        try:
            gdown.download(drive_url, output_path, quiet=False, fuzzy=True)
        except Exception as e:
            print(f"Download error: {e}", file=sys.stderr)
            sys.exit(1)

    # Extract if zip
    if output_path.endswith(".zip"):
        print("Extracting...")
        try:
            with zipfile.ZipFile(output_path, 'r') as z:
                z.extractall(args.output)
            os.remove(output_path)
            print(f"Extracted to: {args.output}")
        except Exception as e:
            print(f"Extract error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Not a zip file, no extraction needed.")

    print("Download complete.")


if __name__ == "__main__":
    main()