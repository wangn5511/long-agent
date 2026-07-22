#!/usr/bin/env python3
"""Inspect compact gesture annotation CSV files for common data issues."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


KNOWN_LABEL_FIXES = {
    "NOND1": "NONE1",
    "NONE22": "NONE2",
    "NONE33": "NONE3",
    "NONE77": "NONE7",
    "WAVE-OUT": "WAVE_OUT",
    "PUSH-HOME": "PUSH_HOME",
}
LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass(frozen=True)
class RowIssue:
    row_idx: int
    issue: str
    row: list[str]


@dataclass(frozen=True)
class Segment:
    row_idx: int
    video: str
    start: int
    end: int
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect compact gesture annotation CSV.")
    parser.add_argument("--csv", required=True, help="CSV path.")
    parser.add_argument(
        "--expected_labels",
        nargs="*",
        default=None,
        help="Optional whitelist. Labels outside this set are reported.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full machine-readable report as JSON.",
    )
    return parser.parse_args()


def _is_header(row: Sequence[str]) -> bool:
    lowered = [cell.strip().lower() for cell in row]
    header_words = {"video", "video_name", "start", "start_frame", "end", "end_frame", "label"}
    return any(cell in header_words for cell in lowered)


def load_segments(path: Path) -> tuple[list[Segment], list[RowIssue]]:
    segments: list[Segment] = []
    issues: list[RowIssue] = []
    cur_video = ""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row_idx, raw_row in enumerate(reader, start=1):
            if not raw_row or all(not cell.strip() for cell in raw_row):
                continue
            row = [cell.strip() for cell in raw_row]
            if row_idx == 1 and _is_header(row):
                continue
            if len(row) < 4:
                issues.append(RowIssue(row_idx, "too_few_columns", row))
                continue

            if row[0]:
                cur_video = Path(row[0]).stem
            elif not cur_video:
                issues.append(RowIssue(row_idx, "continuation_without_video", row))
                continue

            try:
                start = int(float(row[1]))
                end = int(float(row[2]))
            except ValueError:
                issues.append(RowIssue(row_idx, "non_numeric_frame", row))
                continue
            if start <= 0:
                issues.append(RowIssue(row_idx, "non_positive_start_frame", row))
            if end < start:
                issues.append(RowIssue(row_idx, "end_before_start", row))

            label = row[3].strip().upper()
            if not label:
                issues.append(RowIssue(row_idx, "empty_label", row))
                continue
            segments.append(Segment(row_idx, cur_video, start, end, label))
    return segments, issues


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    segments, row_issues = load_segments(csv_path)
    expected_labels = set(args.expected_labels or [])

    label_counts = Counter(seg.label for seg in segments)
    known_fix_counts = {label: label_counts[label] for label in KNOWN_LABEL_FIXES if label_counts[label]}
    bad_format_labels = sorted(label for label in label_counts if not LABEL_RE.match(label))
    outside_whitelist = sorted(label for label in label_counts if expected_labels and label not in expected_labels)

    by_key: dict[tuple[str, int, int, str], list[Segment]] = defaultdict(list)
    by_span: dict[tuple[str, int, int], list[Segment]] = defaultdict(list)
    by_video: dict[str, list[Segment]] = defaultdict(list)
    for seg in segments:
        by_key[(seg.video, seg.start, seg.end, seg.label)].append(seg)
        by_span[(seg.video, seg.start, seg.end)].append(seg)
        by_video[seg.video].append(seg)

    duplicate_exact = [
        {
            "video": key[0],
            "start": key[1],
            "end": key[2],
            "label": key[3],
            "rows": [seg.row_idx for seg in vals],
        }
        for key, vals in by_key.items()
        if len(vals) > 1
    ]
    same_span_different_labels = [
        {
            "video": key[0],
            "start": key[1],
            "end": key[2],
            "labels": sorted({seg.label for seg in vals}),
            "rows": [seg.row_idx for seg in vals],
        }
        for key, vals in by_span.items()
        if len({seg.label for seg in vals}) > 1
    ]

    overlaps = []
    for video, items in by_video.items():
        ordered = sorted(items, key=lambda seg: (seg.start, seg.end, seg.row_idx))
        for prev, cur in zip(ordered, ordered[1:]):
            if cur.start <= prev.end:
                overlaps.append(
                    {
                        "video": video,
                        "prev": {"row": prev.row_idx, "start": prev.start, "end": prev.end, "label": prev.label},
                        "cur": {"row": cur.row_idx, "start": cur.start, "end": cur.end, "label": cur.label},
                    }
                )

    report = {
        "csv": str(csv_path),
        "segments": len(segments),
        "videos": len(by_video),
        "label_counts": dict(sorted(label_counts.items())),
        "known_label_fixes": {
            label: {"suggested": KNOWN_LABEL_FIXES[label], "count": count}
            for label, count in sorted(known_fix_counts.items())
        },
        "bad_format_labels": bad_format_labels,
        "outside_whitelist": outside_whitelist,
        "row_issues": [
            {"row_idx": issue.row_idx, "issue": issue.issue, "row": issue.row}
            for issue in row_issues
        ],
        "duplicate_exact": duplicate_exact,
        "same_span_different_labels": same_span_different_labels,
        "overlaps": overlaps,
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"csv: {csv_path}")
    print(f"segments: {report['segments']}")
    print(f"videos: {report['videos']}")
    print("label_counts:")
    for label, count in report["label_counts"].items():
        print(f"  {label}: {count}")
    print(f"known_label_fixes: {len(report['known_label_fixes'])}")
    for label, item in report["known_label_fixes"].items():
        print(f"  {label} -> {item['suggested']}: {item['count']}")
    print(f"bad_format_labels: {report['bad_format_labels']}")
    print(f"outside_whitelist: {report['outside_whitelist']}")
    print(f"row_issues: {len(report['row_issues'])}")
    print(f"duplicate_exact: {len(report['duplicate_exact'])}")
    print(f"same_span_different_labels: {len(report['same_span_different_labels'])}")
    print(f"overlaps: {len(report['overlaps'])}")
    for section in ("row_issues", "same_span_different_labels", "overlaps"):
        items = report[section]
        if items:
            print(f"{section}_preview:")
            for item in items[:20]:
                print(f"  {item}")


if __name__ == "__main__":
    main()
