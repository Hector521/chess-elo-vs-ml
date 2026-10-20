"""download_data.py - Download one month of rated Lichess games (run once).

Source: https://database.lichess.org (free public database of every rated Lichess game)
The file is PGN (standard chess game text format), compressed with Zstandard (.zst).
"""
import io
import urllib.request
from pathlib import Path

import zstandard

MONTH = "2013-01"  # 121,332 rated games: small enough for a laptop
URL = f"https://database.lichess.org/standard/lichess_db_standard_rated_{MONTH}.pgn.zst"
DATA_DIR = Path("data")
OUT_FILE = DATA_DIR / f"lichess_{MONTH}.pgn.zst"


def download() -> None:
    """Download the file once; skip if it's already there."""
    DATA_DIR.mkdir(exist_ok=True)
    if OUT_FILE.exists():
        print(f"Already downloaded: {OUT_FILE}")
        return
    print(f"Downloading {URL} ...")
    partial = OUT_FILE.with_suffix(".part")  # temporary name until the download finishes
    urllib.request.urlretrieve(URL, partial)
    partial.rename(OUT_FILE)


def peek(path: Path, n_lines: int = 25) -> None:
    """Print the first lines of the compressed file without unpacking all of it."""
    with open(path, "rb") as fh:
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
        text = io.TextIOWrapper(reader, encoding="utf-8")
        for _ in range(n_lines):
            print(text.readline().rstrip())


if __name__ == "__main__":
    download()
    print(f"File: {OUT_FILE} ({OUT_FILE.stat().st_size / 1_000_000:.1f} MB compressed)\n")
    print("First game in the file:")
    peek(OUT_FILE)