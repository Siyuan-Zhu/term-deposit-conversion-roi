"""Download and verify the UCI Bank Marketing dataset (id 222, bank-additional-full.csv).

The file is fetched at runtime into data/raw/ (gitignored). Its SHA-256 and row
count are checked before anything reads it.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "bank-additional-full.csv"

UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
EXPECTED_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"
EXPECTED_ROWS = 41188
INNER_ZIP = "bank-additional.zip"
INNER_CSV = "bank-additional/bank-additional-full.csv"


class ChecksumError(RuntimeError):
    """Raised when the downloaded file does not match the pinned checksum."""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_raw_csv(dest: Path = CSV_PATH, url: str = UCI_URL) -> Path:
    """Fetch the archive, extract the CSV, and verify its checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https URL
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(INNER_ZIP)))
    dest.write_bytes(inner.read(INNER_CSV))
    verify_checksum(dest)
    return dest


def verify_checksum(path: Path, expected: str = EXPECTED_SHA256) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise ChecksumError(f"{path.name}: expected sha256 {expected}, got {actual}")


def load_dataset(path: Path = CSV_PATH) -> pd.DataFrame:
    """Return the verified dataset, downloading it first if it is not on disk."""
    if not path.exists():
        download_raw_csv(path)
    else:
        verify_checksum(path)
    df = pd.read_csv(path, sep=";")
    if len(df) != EXPECTED_ROWS:
        raise ChecksumError(f"expected {EXPECTED_ROWS} rows, found {len(df)}")
    df["subscribed"] = (df["y"] == "yes").astype(int)
    return df
