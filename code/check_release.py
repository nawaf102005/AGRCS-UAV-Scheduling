"""Validate every file declared in this distribution's SHA-256 manifest."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "SHA256SUMS.json").read_text())
    failures = []
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(f"Missing or invalid path: {name}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            failures.append(f"Changed file: {name}")
    print(f"Checked {len(manifest)} files; {len(failures)} failures.")
    for failure in failures:
        print(failure)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
