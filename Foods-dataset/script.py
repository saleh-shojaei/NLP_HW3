#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List
from collections import OrderedDict


def maybe_unstringify(val: Any) -> Any:
    """If val is a JSON-looking string (starts with { or [), try to parse it."""
    if isinstance(val, str):
        s = val.strip()
        if s and s[0] in "[{":
            try:
                return json.loads(s)
            except Exception:
                pass
    return val


def load_records_from_file(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize: accept either a single object or a list of objects.
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Top-level JSON in {path} must be an object or array.")

    # Clean up any JSON-encoded string fields.
    cleaned: List[dict] = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        cleaned.append({k: maybe_unstringify(v) for k, v in obj.items()})
    return cleaned


def iter_json_files(directory: Path) -> Iterable[Path]:
    # Sort for deterministic order
    yield from sorted(directory.glob("*.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate all JSON files in a directory into one list and add incremental ids (id first)."
    )
    parser.add_argument(
        "-i", "--input-dir", default=".", help="Directory containing *.json files (default: current dir)"
    )
    parser.add_argument(
        "-o", "--output", default="aggregated.json", help="Output file path (default: aggregated.json)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    all_records: List[dict] = []
    total_files = 0
    for path in iter_json_files(input_dir):
        try:
            records = load_records_from_file(path)
            all_records.extend(records)
            total_files += 1
        except Exception as e:
            print(f"Skipping {path.name}: {e}")

    # Add incremental ids, ensuring 'id' is the first key
    ordered_records: List[OrderedDict] = []
    for idx, rec in enumerate(all_records, start=1):
        od = OrderedDict()
        od["id"] = idx
        for k, v in rec.items():
            if k == "id":
                continue  # ensure our new id wins and stays first
            od[k] = v
        ordered_records.append(od)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(ordered_records, f, ensure_ascii=False, indent=2)

    print(f"Aggregated {len(ordered_records)} records from {total_files} files into {output_path}")


if __name__ == "__main__":
    main()
