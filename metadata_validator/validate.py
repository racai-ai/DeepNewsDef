#!/usr/bin/env python3
"""CLI entry point for metadata validation using Pydantic models."""

import argparse
import json
import os
import sys
from typing import List, NamedTuple, Tuple

from pydantic import ValidationError

from metadata_validator.image_schema import ImageMetadata
from metadata_validator.text_schema import TextMetadata

MODELS = {
    "image": ImageMetadata,
    "text": TextMetadata,
}


class FileResult(NamedTuple):
    file_path: str
    valid: bool
    errors: List[Tuple[str, str]]
    warnings: List[str]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate JSON metadata files against DeepNewsDef schemas.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a single JSON file to validate.")
    group.add_argument(
        "--folder", help="Path to a folder of JSON files to validate."
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=["text", "image"],
        help="Metadata type to validate against.",
    )
    parser.add_argument(
        "--report",
        default="validation_report.json",
        help="Path for the detailed JSON report (default: validation_report.json).",
    )
    return parser


def validate_file(file_path: str, metadata_type: str) -> FileResult:
    """Validate a single JSON file against the specified Pydantic model."""
    model_cls = MODELS[metadata_type]

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        instance = model_cls(**data)
        warnings = getattr(instance, "warnings_", [])
        return FileResult(
            file_path=file_path,
            valid=True,
            errors=[],
            warnings=warnings,
        )
    except ValidationError as exc:
        error_messages = []
        for err in exc.errors():
            loc = ".".join(str(part) for part in err["loc"])
            error_messages.append((loc, err["msg"]))
        return FileResult(
            file_path=file_path,
            valid=False,
            errors=error_messages,
            warnings=[],
        )


def result_to_dict(result: FileResult) -> dict:
    """Convert a FileResult to a JSON-serializable dict with grouped errors."""
    grouped_errors: dict = {}
    for loc, msg in result.errors:
        parts = loc.rsplit(".", 1)
        if len(parts) == 2:
            parent, field = parts
        else:
            parent, field = "(root)", parts[0]
        grouped_errors.setdefault(parent, []).append({"field": field, "message": msg})

    return {
        "file": result.file_path,
        "valid": result.valid,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "errors": grouped_errors,
        "warnings": result.warnings,
    }


def main(argv=None) -> int:
    """Main entry point. Returns exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code != 0 else 0

    metadata_type = args.type

    # Collect file paths
    file_paths: List[str] = []
    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            return 3
        file_paths.append(args.file)
    elif args.folder:
        if not os.path.isdir(args.folder):
            print(f"Error: Folder not found: {args.folder}", file=sys.stderr)
            return 3
        for name in sorted(os.listdir(args.folder)):
            if name.endswith(".json"):
                file_paths.append(os.path.join(args.folder, name))
        if not file_paths:
            print(
                f"Error: No JSON files found in {args.folder}", file=sys.stderr
            )
            return 3

    # Validate each file
    results: List[FileResult] = []
    for fp in file_paths:
        try:
            result = validate_file(fp, metadata_type)
            results.append(result)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Error: Could not read {fp}: {exc}", file=sys.stderr)
            return 3

    # Stdout: PASS/FAIL per file (no details)
    for result in results:
        status = "PASS" if result.valid else "FAIL"
        extra = ""
        if result.valid and result.warnings:
            extra = f"  ({len(result.warnings)} warning(s))"
        elif not result.valid:
            extra = f"  ({len(result.errors)} error(s))"
        print(f"  {status}: {os.path.basename(result.file_path)}{extra}")

    # Statistics
    valid_count = sum(1 for r in results if r.valid)
    invalid_count = len(results) - valid_count
    warning_count = sum(1 for r in results if r.valid and r.warnings)

    print()
    print(f"Summary: {valid_count}/{len(results)} files valid", end="")
    if invalid_count:
        print(f", {invalid_count} invalid", end="")
    if warning_count:
        print(f", {warning_count} with warnings", end="")
    print()

    if invalid_count:
        error_counts: dict = {}
        for r in results:
            for loc, msg in r.errors:
                parts = loc.rsplit(".", 1)
                field = parts[-1]
                error_counts[field] = error_counts.get(field, 0) + 1

        by_freq = sorted(error_counts.items(), key=lambda x: -x[1])
        print(f"\nMost common missing/invalid fields:")
        for field, count in by_freq[:20]:
            pct = count * 100 // len(results)
            bar = "#" * (pct // 5)
            print(f"  {field:30s} {count:4d}/{len(results)} ({pct:3d}%) {bar}")

    # Write detailed JSON report
    report_path = args.report
    report = {
        "summary": {
            "total": len(results),
            "valid": valid_count,
            "invalid": invalid_count,
            "with_warnings": warning_count,
        },
        "files": [result_to_dict(r) for r in results],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report written to: {report_path}")

    # Determine exit code
    has_errors = any(not r.valid for r in results)
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
