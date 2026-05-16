"""
download_quickdraw.py

Downloads the Google QuickDraw NDJSON dataset for the 4 classes needed:
  apple, circle, star, triangle.

The files are hosted publicly by Google at:
  https://storage.googleapis.com/quickdraw_dataset/full/simplified/<class>.ndjson

Each file is 30-60 MB. Total download: ~170 MB.

Usage:
    python download_quickdraw.py

Output:
    data/apple.ndjson
    data/circle.ndjson
    data/star.ndjson
    data/triangle.ndjson
"""

import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

CLASSES = ["apple", "circle", "star", "triangle"]
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified"
DATA_DIR = Path("./data")
CHUNK_SIZE = 1024 * 64  # 64 KB


def human_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def download_class(class_name, out_path, max_retries=3):
    """Download one NDJSON file, with retry on failure."""
    url = f"{BASE_URL}/{class_name}.ndjson"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [attempt {attempt}/{max_retries}] {url}")
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 quickdraw-downloader"})
            with urlopen(req, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                start = time.time()

                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            elapsed = time.time() - start
                            speed = downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
                            print(f"    {pct:5.1f}% | {human_size(downloaded)}/{human_size(total)} | {speed:.1f} MB/s", end="\r")
                        else:
                            print(f"    {human_size(downloaded)} downloaded", end="\r")

                print()  # newline after progress

            # Atomic move from .tmp to final path
            tmp_path.replace(out_path)
            print(f"  Saved: {out_path} ({human_size(out_path.stat().st_size)})")
            return True

        except (URLError, HTTPError, TimeoutError) as e:
            print(f"  Attempt {attempt} failed: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  FAILED after {max_retries} attempts.")
                return False
        except Exception as e:
            print(f"  Unexpected error: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            return False

    return False


def verify_ndjson(path):
    """Quick sanity check: count lines, ensure it's not truncated."""
    try:
        line_count = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_count += 1
                if line_count > 100000:  # arbitrary safe upper bound
                    break
        return line_count
    except Exception as e:
        return -1


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(CLASSES)} QuickDraw classes to {DATA_DIR.resolve()}")
    print(f"Each file is 30-60 MB. Total: ~170 MB.\n")

    results = {}
    for cls in CLASSES:
        out_path = DATA_DIR / f"{cls}.ndjson"
        if out_path.exists() and out_path.stat().st_size > 1_000_000:
            line_count = verify_ndjson(out_path)
            print(f"[{cls}] Already exists ({human_size(out_path.stat().st_size)}, ~{line_count} drawings). Skipping.")
            results[cls] = True
            continue

        print(f"\n[{cls}] Downloading...")
        success = download_class(cls, out_path)
        results[cls] = success

        if success:
            line_count = verify_ndjson(out_path)
            print(f"  Verified: {line_count} drawings found")

    print("\n" + "=" * 50)
    print("Download summary:")
    for cls, ok in results.items():
        status = "OK" if ok else "FAILED"
        if ok:
            path = DATA_DIR / f"{cls}.ndjson"
            size = human_size(path.stat().st_size)
            print(f"  {cls:<10s} {status}  ({size})")
        else:
            print(f"  {cls:<10s} {status}")

    failed = [c for c, ok in results.items() if not ok]
    if failed:
        print(f"\nFAILED classes: {failed}")
        print("Try running this script again, or download manually from:")
        for cls in failed:
            print(f"  {BASE_URL}/{cls}.ndjson")
        sys.exit(1)
    else:
        print("\nAll classes downloaded successfully.")


if __name__ == "__main__":
    main()
