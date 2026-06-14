"""Dataset downloader for the video generation training pipeline.

Supported datasets
------------------
  ucf101   → UCF-101 Action Recognition (~13 GB, academic license)
             http://crcv.ucf.edu/data/UCF101.php
  msvd     → Microsoft Video Description (MSVD) (~2 GB, academic license)
             https://www.cs.utexas.edu/users/ml/clamp/videoDescription/
  kinetics → Kinetics-700-2020 (~450 GB, academic license)
             https://www.deepmind.com/open-source/kinetics
  pexels   → Pexels stock video (CC0, requires free API key)
             https://www.pexels.com/api/

Usage
-----
    # Download UCF-101 and MSVD (recommended starter):
    python scripts/download_datasets.py --datasets ucf101 msvd --out data/raw

    # Also download N Pexels videos per query term:
    python scripts/download_datasets.py --datasets pexels --pexels-key YOUR_KEY \
        --pexels-queries "nature landscape" "city street" --pexels-per-query 100 \
        --out data/raw

    # List available datasets only:
    python scripts/download_datasets.py --list
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "ucf101": {
        "description": "UCF-101 Action Recognition: 13 320 clips, 101 classes, ~13 GB",
        "url": "https://crcv.ucf.edu/data/UCF101/UCF101.rar",
        "filename": "UCF101.rar",
        "size_gb": 6.5,
        "license": "Academic (research use only)",
        "note": "Contains action clips; no paired text captions — use prepare_data.py for auto-captioning",
    },
    "msvd": {
        "description": "Microsoft Video Description: 1 970 clips, 80 000 descriptions, ~2 GB",
        "url": "https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar",
        "filename": "YouTubeClips.tar",
        "size_gb": 2.0,
        "license": "Academic (research use only)",
        "note": "Video-text pairs ideal for text-conditioned training",
    },
    "kinetics": {
        "description": "Kinetics-700-2020: 650 000+ clips, 700 classes, ~450 GB",
        "url": None,  # must use official downloader
        "size_gb": 450,
        "license": "Creative Commons Attribution 4.0",
        "note": "Use the official Kinetics downloader: pip install kinetics-dataset",
    },
    "pexels": {
        "description": "Pexels stock video (CC0): requires free API key at pexels.com/api",
        "url": None,  # API-driven
        "size_gb": None,
        "license": "Pexels License (free for commercial use)",
        "note": "Best for diverse, high-quality training footage",
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download public video datasets for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS) + ["all"],
                   default=["ucf101"], help="Datasets to download")
    p.add_argument("--out", default="data/raw", help="Output root directory")
    p.add_argument("--list", action="store_true", help="List available datasets and exit")
    # Pexels-specific
    p.add_argument("--pexels-key", default=os.environ.get("PEXELS_API_KEY"),
                   help="Pexels API key (or set PEXELS_API_KEY env var)")
    p.add_argument("--pexels-queries", nargs="+",
                   default=["nature", "city", "people walking", "ocean waves",
                            "mountains", "animals", "technology", "food"],
                   help="Search terms for Pexels")
    p.add_argument("--pexels-per-query", type=int, default=50,
                   help="Number of videos per Pexels search query")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("\nAvailable datasets:\n")
        for name, meta in DATASETS.items():
            size = f"{meta['size_gb']} GB" if meta["size_gb"] else "variable"
            print(f"  {name:<12} {size:<12} {meta['description']}")
            print(f"             License: {meta['license']}")
            if meta.get("note"):
                print(f"             Note: {meta['note']}")
            print()
        return

    selected = list(DATASETS.keys()) if "all" in args.datasets else args.datasets
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for ds in selected:
        log.info("=== %s ===", ds.upper())
        if ds == "ucf101":
            _download_ucf101(out)
        elif ds == "msvd":
            _download_msvd(out)
        elif ds == "kinetics":
            _print_kinetics_instructions(out)
        elif ds == "pexels":
            if not args.pexels_key:
                log.error("Pexels requires --pexels-key or PEXELS_API_KEY env var")
            else:
                _download_pexels(out, args.pexels_key, args.pexels_queries,
                                 args.pexels_per_query)

    log.info("Done. Raw data saved to: %s", out.resolve())
    log.info("Next step: python scripts/prepare_data.py --input %s --output data/processed", out)


# ---------------------------------------------------------------------------
# Per-dataset downloaders
# ---------------------------------------------------------------------------

def _download_ucf101(out: Path):
    """Download and extract UCF-101 RAR archive."""
    dest = out / "ucf101"
    dest.mkdir(exist_ok=True)
    rar_path = dest / "UCF101.rar"

    url = DATASETS["ucf101"]["url"]
    if not rar_path.exists():
        log.info("Downloading UCF-101 (~6.5 GB) from %s", url)
        _download_url(url, rar_path)
    else:
        log.info("UCF-101 archive already exists at %s", rar_path)

    # Extract — requires `unrar` or `7z` on the system
    clips_dir = dest / "UCF-101"
    if not clips_dir.exists():
        log.info("Extracting UCF-101 …")
        extracted = _extract_archive(rar_path, dest)
        if not extracted:
            log.warning(
                "Could not auto-extract. Please extract %s manually:\n"
                "  unrar x %s %s", rar_path, rar_path, dest
            )
    else:
        log.info("UCF-101 already extracted at %s", clips_dir)

    _write_info(dest, DATASETS["ucf101"])


def _download_msvd(out: Path):
    """Download and extract the MSVD YouTubeClips tar archive."""
    dest = out / "msvd"
    dest.mkdir(exist_ok=True)
    tar_path = dest / "YouTubeClips.tar"

    url = DATASETS["msvd"]["url"]
    if not tar_path.exists():
        log.info("Downloading MSVD (~2 GB) from %s", url)
        _download_url(url, tar_path)
    else:
        log.info("MSVD archive already exists at %s", tar_path)

    clips_dir = dest / "YouTubeClips"
    if not clips_dir.exists():
        log.info("Extracting MSVD …")
        import tarfile
        with tarfile.open(tar_path) as tf:
            tf.extractall(dest)
    else:
        log.info("MSVD already extracted at %s", clips_dir)

    # Download caption annotations
    captions_url = (
        "https://raw.githubusercontent.com/xudejing/video-clip-order-prediction"
        "/master/data/msvd/train_list.txt"
    )
    cap_path = dest / "train_list.txt"
    if not cap_path.exists():
        try:
            _download_url(captions_url, cap_path)
        except Exception as exc:
            log.warning("Could not download MSVD captions: %s", exc)

    _write_info(dest, DATASETS["msvd"])


def _print_kinetics_instructions(out: Path):
    log.info(
        "Kinetics-700 requires the official downloader:\n"
        "  pip install kinetics-dataset\n"
        "  python -m kinetics_dataset download --version 700_2020 --split train "
        "--out_dir %s/kinetics\n"
        "This will download ~450 GB. Run on a machine with ample storage and bandwidth.",
        out,
    )


def _download_pexels(
    out: Path, api_key: str, queries: list[str], per_query: int
):
    """Download videos from the Pexels API (CC0 license)."""
    try:
        import urllib.request
        import json
    except ImportError:
        log.error("urllib not available")
        return

    dest = out / "pexels"
    dest.mkdir(exist_ok=True)

    total = 0
    for query in queries:
        page = 1
        downloaded = 0
        while downloaded < per_query:
            per_page = min(80, per_query - downloaded)
            url = (
                f"https://api.pexels.com/videos/search"
                f"?query={urllib.parse.quote(query)}&per_page={per_page}&page={page}"
            )
            req = urllib.request.Request(url, headers={"Authorization": api_key})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
            except Exception as exc:
                log.error("Pexels API error for query %r: %s", query, exc)
                break

            videos = data.get("videos", [])
            if not videos:
                break

            for v in videos:
                # Pick the smallest HD file (≤ 720p for storage efficiency)
                files = sorted(
                    [f for f in v.get("video_files", []) if f.get("height", 0) <= 720],
                    key=lambda f: f.get("height", 0),
                    reverse=True,
                )
                if not files:
                    files = v.get("video_files", [])[:1]
                if not files:
                    continue

                file_url = files[0]["link"]
                vid_id = v["id"]
                ext = "mp4"
                save_path = dest / f"pexels_{vid_id}.{ext}"
                meta_path = dest / f"pexels_{vid_id}.json"

                if save_path.exists():
                    downloaded += 1
                    continue

                try:
                    _download_url(file_url, save_path, show_progress=False)
                    with open(meta_path, "w") as f:
                        import json as _json
                        _json.dump({
                            "id": vid_id, "url": v.get("url"),
                            "duration": v.get("duration"),
                            "photographer": v.get("user", {}).get("name"),
                            "query": query,
                        }, f)
                    downloaded += 1
                    total += 1
                    if total % 10 == 0:
                        log.info("  %d Pexels videos downloaded so far …", total)
                except Exception as exc:
                    log.warning("Failed to download Pexels video %d: %s", vid_id, exc)

            page += 1

    log.info("Pexels: %d videos saved to %s", total, dest)
    _write_info(dest, DATASETS["pexels"])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _download_url(url: str, dest: Path, show_progress: bool = True, retries: int = 5):
    """Download url to dest with resume support and automatic retries."""
    dest = Path(dest)
    tmp = dest.with_suffix(".part")

    try:
        import requests
    except ImportError:
        requests = None  # type: ignore[assignment]

    for attempt in range(1, retries + 1):
        resume_pos = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}

        try:
            if requests is not None:
                with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                    if r.status_code == 416:
                        # Server says range not satisfiable — file already complete
                        tmp.rename(dest)
                        return
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0)) + resume_pos
                    done = resume_pos
                    with open(tmp, "ab") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            f.write(chunk)
                            done += len(chunk)
                            if show_progress and total > 0:
                                log.info("  %.1f%%  (%d / %d MB)", done / total * 100,
                                         done >> 20, total >> 20)
            else:
                # Fallback: no resume, plain urlretrieve
                def _reporthook(block, block_size, total):
                    if not show_progress or total <= 0:
                        return
                    done = block * block_size
                    if block % 100 == 0:
                        log.info("  %.1f%%  (%d / %d MB)", min(done / total * 100, 100),
                                 done >> 20, total >> 20)
                urllib.request.urlretrieve(url, tmp, reporthook=_reporthook)

            tmp.rename(dest)
            return

        except Exception as exc:
            if attempt == retries:
                if tmp.exists():
                    tmp.unlink()
                raise
            wait = 2 ** attempt
            log.warning("Download error (attempt %d/%d): %s — retrying in %ds", attempt, retries, exc, wait)
            time.sleep(wait)


def _extract_archive(path: Path, dest: Path) -> bool:
    """Try to extract a RAR/ZIP/TAR archive. Returns True on success."""
    if path.suffix.lower() == ".rar":
        for tool in (["unrar", "x", str(path), str(dest) + os.sep],
                     ["7z", "x", str(path), f"-o{dest}"]):
            if shutil.which(tool[0]):
                import subprocess
                r = subprocess.run(tool, capture_output=True)
                return r.returncode == 0
        return False
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
        return True
    if path.suffix.lower() in (".tar", ".gz", ".bz2", ".xz"):
        import tarfile
        with tarfile.open(path) as tf:
            tf.extractall(dest)
        return True
    return False


def _write_info(dest: Path, meta: dict):
    info_path = dest / "DATASET_INFO.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")


# ---------------------------------------------------------------------------

import urllib.parse  # noqa: E402  (needed by pexels downloader, imported at module level)

if __name__ == "__main__":
    main()
