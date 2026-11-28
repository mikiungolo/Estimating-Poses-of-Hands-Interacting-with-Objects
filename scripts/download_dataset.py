#!/usr/bin/env python3
"""
Prepare the DexYCB data for a Colab session.

Every team member keeps their OWN copy of the archives in their Google Drive
(config.drive_dataset_dir). The script:

1. Downloads the missing archives from the official DexYCB Google Drive files
   into your Drive (only the first time: ~13.9 GB in total).
2. Copies each archive to the Colab local disk and extracts it into
   config.data_root (every session: Colab wipes its disk when it disconnects).
3. Checks that the MANO model is in config.mano_dir.

Usage (from the repository root):
    python scripts/download_dataset.py --config configs/config.yaml
    python scripts/download_dataset.py --download-only   # only fill your Drive
    python scripts/download_dataset.py --skip-download   # archives already on Drive

Resulting layout:
    <data_root>/20200709-subject-01/<sequence>/meta.yml
    <data_root>/20200709-subject-01/<sequence>/<camera serial>/{color_*.jpg, aligned_depth_to_color_*.png, labels_*.npz}
    <data_root>/calibration/...
    <data_root>/models/...
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

DRIVE_URL = "https://drive.google.com/file/d/{}"


def load_config(path: str):
    config = OmegaConf.load(path)
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))


def download_to_drive(name: str, drive_id: str, drive_dir: Path) -> Path:
    """Download one archive into your Drive, unless it is already there."""
    target = drive_dir / name
    if target.exists() and target.stat().st_size > 0:
        print(f"[drive] {name}: already in your Drive ({target.stat().st_size / 1e9:.2f} GB)")
        return target

    import gdown

    drive_dir.mkdir(parents=True, exist_ok=True)
    partial = drive_dir / f"{name}.part"
    print(f"[drive] {name}: downloading into {drive_dir} ...")
    try:
        result = gdown.download(id=drive_id, output=str(partial), quiet=False)
    except Exception as e:  # gdown raises on quota / permission errors
        result = None
        print(f"[drive] download error: {e}", file=sys.stderr)

    if not result or not partial.exists():
        print(
            f"\n[drive] Could not download {name} automatically.\n"
            f"  Do it by hand: open {DRIVE_URL.format(drive_id)}\n"
            f"  -> 'Make a copy' (or download and upload it) into {drive_dir}\n"
            f"  -> rename the copy to '{name}', then run this script again.",
            file=sys.stderr,
        )
        sys.exit(1)

    partial.replace(target)
    return target


def extract_locally(archive: Path, data_root: Path, tmp_dir: Path) -> None:
    """Copy an archive from Drive to the local disk and extract it into data_root."""
    marker = data_root / f".extracted_{archive.name}"
    if marker.exists():
        print(f"[local] {archive.name}: already extracted in {data_root}")
        return

    data_root.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_copy = tmp_dir / archive.name

    start = time.time()
    print(f"[local] {archive.name}: copying from Drive ...")
    shutil.copyfile(archive, local_copy)
    print(f"[local] {archive.name}: extracting into {data_root} ...")
    subprocess.run(["tar", "-xzf", str(local_copy), "-C", str(data_root)], check=True)
    local_copy.unlink()
    marker.touch()
    print(f"[local] {archive.name}: done in {(time.time() - start) / 60:.1f} min")


def check_mano(mano_dir: Path) -> bool:
    """MANO cannot be downloaded automatically: it requires a personal licence."""
    right = mano_dir / "MANO_RIGHT.pkl"
    if right.exists():
        print(f"[mano] OK: {right}")
        return True
    print(
        f"\n[mano] MANO_RIGHT.pkl not found in {mano_dir}.\n"
        "  1. Register at https://mano.is.tue.mpg.de and download 'Models & Code'.\n"
        "  2. Upload mano_v1_2/models/MANO_RIGHT.pkl (and MANO_LEFT.pkl) into that Drive folder.\n"
        "  Needed for hand vertices (MPVPE, PAV, ...) and mesh rendering.",
        file=sys.stderr,
    )
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DexYCB on Drive and on the Colab disk")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--download-only", action="store_true", help="Only fill your Drive")
    parser.add_argument("--skip-download", action="store_true", help="Do not download, only extract")
    parser.add_argument("--tmp-dir", default="/content/_archives", help="Local copy of the archives")
    args = parser.parse_args()

    config = load_config(args.config)
    drive_dir = Path(config.drive_dataset_dir)
    data_root = Path(config.data_root)

    archives = []
    for key, f in config.dexycb.files.items():
        if args.skip_download:
            path = drive_dir / f.name
            if not path.exists():
                sys.exit(f"[drive] {f.name} ({key}) not found in {drive_dir}")
        else:
            path = download_to_drive(f.name, f.drive_id, drive_dir)
        archives.append(path)

    if not args.download_only:
        for archive in archives:
            extract_locally(archive, data_root, Path(args.tmp_dir))
        print("\n[local] contents of", data_root)
        for p in sorted(data_root.iterdir()):
            if not p.name.startswith("."):
                print("   ", p.name)

    check_mano(Path(config.mano_dir))
    print("\nData ready.")


if __name__ == "__main__":
    main()
