"""
Dataset Downloader & Verifier for Sharma et al. (2024) Replication
Downloads:
1. NSL-KDD: KDDTrain+.txt and KDDTest+.txt
2. UNSW-NB15: UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv

Supports running locally or in Google Colab.
"""

import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional


DATA_SOURCES = {
    "nsl_kdd": {
        "train": {
            "filename": "KDDTrain+.txt",
            "urls": [
                "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
                "https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/nsl-kdd/KDDTrain+.txt"
            ]
        },
        "test": {
            "filename": "KDDTest+.txt",
            "urls": [
                "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
                "https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/nsl-kdd/KDDTest+.txt"
            ]
        }
    },
    "unsw_nb15": {
        "train": {
            "filename": "UNSW_NB15_training-set.csv",
            "urls": [
                "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/train.csv"
            ]
        },
        "test": {
            "filename": "UNSW_NB15_testing-set.csv",
            "urls": [
                "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/test.csv"
            ]
        }
    }
}


def download_file(urls: list, dest_path: Path, min_bytes: int = 1000) -> bool:
    """Download file from a list of mirror URLs with fallback."""
    if dest_path.exists() and dest_path.stat().st_size >= min_bytes:
        print(f"[CACHE] {dest_path.name} already exists ({dest_path.stat().st_size / (1024*1024):.2f} MB). Skipping download.")
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    for url in urls:
        print(f"[DOWNLOAD] Attempting: {url} -> {dest_path.name} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as out_file:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                block_size = 64 * 1024
                start_time = time.time()
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        elapsed = time.time() - start_time
                        speed = (downloaded / (1024 * 1024)) / max(elapsed, 0.001)
                        sys.stdout.write(f"\r  Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%) at {speed:.2f} MB/s")
                        sys.stdout.flush()
                print()

            if temp_path.stat().st_size >= min_bytes:
                temp_path.replace(dest_path)
                print(f"[SUCCESS] Downloaded {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB).")
                return True
            else:
                print(f"[WARN] Downloaded file too small ({temp_path.stat().st_size} bytes). Trying next mirror...")
                if temp_path.exists():
                    temp_path.unlink()
        except Exception as e:
            print(f"[ERROR] Failed to download from {url}: {e}")
            if temp_path.exists():
                temp_path.unlink()

    return False


def download_all_datasets(base_dir: Optional[Path] = None) -> Dict[str, Dict[str, Path]]:
    """Downloads all raw datasets into data/raw."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent / "data" / "raw"

    paths = {}
    print("=" * 60)
    print("STARTING DATASET ACQUISITION FOR SHARMA ET AL. (2024)")
    print(f"Destination: {base_dir}")
    print("=" * 60)

    for dataset_name, parts in DATA_SOURCES.items():
        dataset_dir = base_dir / dataset_name
        paths[dataset_name] = {}
        print(f"\n--- Checking {dataset_name.upper()} ---")
        for split, info in parts.items():
            dest = dataset_dir / info["filename"]
            success = download_file(info["urls"], dest)
            if not success:
                raise RuntimeError(f"Could not download {info['filename']} for {dataset_name} from any mirror!")
            paths[dataset_name][split] = dest

    print("\n[COMPLETE] All raw datasets verified successfully.")
    return paths


if __name__ == "__main__":
    download_all_datasets()
