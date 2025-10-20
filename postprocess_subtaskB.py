from __future__ import annotations

import csv
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import argparse


SplitName = str
RowKey = Tuple[str, ...]


def _read_rows(path: Path, expected_fields: List[str]) -> Tuple[List[str], Iterable[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"No headers found in {path}")
        if expected_fields and fieldnames != expected_fields:
            raise ValueError(f"Header mismatch in {path}: expected {expected_fields}, found {fieldnames}")
        return fieldnames, list(reader)


def _filter_rows(rows: Iterable[Dict[str, str]], filter_in_use: bool) -> Iterable[Dict[str, str]]:
    if not filter_in_use:
        return rows
    return (row for row in rows if row.get("In Use", "1") != "0")


def postprocess_subtask_b(
    input_dir: str,
    output_dir: str,
    filter_in_use: bool,
) -> None:
    """Combine subtask B CSVs into deduplicated global splits."""
    base_dir = Path(input_dir).expanduser().resolve()
    if not base_dir.is_dir():
        raise ValueError(f"Input directory not found: {base_dir}")

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_filenames: Dict[SplitName, str] = {
        "train": "raw_train_all_onecol.csv",
        "validation": "raw_val_all_onecol.csv",
        "test": "raw_test_all_onecol.csv",
    }

    aggregated_rows: Dict[SplitName, OrderedDict[RowKey, Dict[str, str]]] = {
        split: OrderedDict() for split in split_filenames
    }
    split_fieldnames: Dict[SplitName, List[str]] = {split: [] for split in split_filenames}
    duplicate_counts: Dict[SplitName, int] = defaultdict(int)
    domain_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir())

    if not domain_dirs:
        raise RuntimeError(f"No domain directories found in {base_dir}")

    for domain_dir in domain_dirs:
        for split, filename in split_filenames.items():
            csv_path = domain_dir / filename
            if not csv_path.is_file():
                raise FileNotFoundError(f"Missing {filename} in {domain_dir}")

            expected_fields = split_fieldnames[split]
            fieldnames, rows = _read_rows(csv_path, expected_fields)
            if not expected_fields:
                split_fieldnames[split] = fieldnames
            rows = list(_filter_rows(rows, filter_in_use if split != "test" else False))

            split_store = aggregated_rows[split]
            for row in rows:
                key = tuple(row[field] for field in fieldnames)
                if key in split_store:
                    duplicate_counts[split] += 1
                    continue
                split_store[key] = row

    summary_lines = []
    for split, rows in aggregated_rows.items():
        output_path = out_dir / f"{split}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=split_fieldnames[split])
            writer.writeheader()
            writer.writerows(rows.values())

        summary_lines.append(
            f"{split}: wrote {len(rows)} unique rows "
            f"(duplicates skipped: {duplicate_counts[split]}) -> {output_path}"
        )

    print("\n".join(summary_lines))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine subtask B CSVs into deduplicated global splits.",
    )
    parser.add_argument(
        "--input-dir",
        default="data/subtaskB",
        help="Directory containing per-domain subtask B CSV folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/subtaskB_combined",
        help="Directory to write deduplicated CSV files.",
    )
    parser.add_argument(
        "--no-filter-in-use",
        action="store_true",
        help="Disable filtering rows where 'In Use' == 0 for train/validation splits.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    postprocess_subtask_b(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        filter_in_use=not args.no_filter_in_use,
    )


if __name__ == "__main__":
    main()
