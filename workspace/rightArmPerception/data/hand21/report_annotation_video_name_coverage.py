#!/usr/bin/env python3
"""Report whether compact annotation video names exist under a video root.

The expected annotation layout is the same compact 4-column format used by
``prepare_hand21_from_excel_segments.py``:

1. video name (only filled when a new source video begins)
2. start frame
3. end frame
4. label

Rows with an empty first column inherit the most recent non-empty video name.
The script recursively scans ``--video-root`` for ``.mp4`` files and checks
whether each referenced video name exists by final filename or stem. It can
also export the inverse list: videos present on disk but absent from the
annotation file.
"""

from __future__ import annotations

import argparse
import csv
import zipfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class VideoStats:
    video_name: str
    segment_count: int = 0
    block_count: int = 0
    block_start_rows: List[int] = field(default_factory=list)
    first_segment_row: int = 0
    first_start_frame: str = ""
    last_end_frame: str = ""
    labels: "OrderedDict[str, None]" = field(default_factory=OrderedDict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether compact annotation video names exist under a video root."
    )
    parser.add_argument("--annotation-file", type=Path, required=True, help="Compact 4-col CSV/XLSX annotation file.")
    parser.add_argument("--video-root", type=Path, required=True, help="Root directory containing mp4 files.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Detailed per-video report CSV.")
    parser.add_argument("--summary-txt", type=Path, required=True, help="Human-readable summary file.")
    parser.add_argument(
        "--annotated-missing-txt",
        type=Path,
        default=None,
        help="Optional text file. Writes video names that appear in the annotation but are missing on disk.",
    )
    parser.add_argument(
        "--unannotated-videos-txt",
        type=Path,
        default=None,
        help="Optional text file. Writes on-disk video stems that never appear in the annotation file.",
    )
    parser.add_argument("--sheet-name", type=str, default=None, help="Excel sheet name when the input is XLSX.")
    return parser.parse_args()


def _read_xlsx_with_stdlib(path: Path, sheet_name: Optional[str]) -> List[List[object]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}

    with zipfile.ZipFile(path) as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared_strings.append("".join((t.text or "") for t in si.iterfind(".//a:t", ns)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets = []
        for sheet in workbook.find("a:sheets", ns):
            sheets.append(
                (
                    sheet.attrib.get("name"),
                    sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"),
                )
            )
        if not sheets:
            return []

        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("r:Relationship", rel_ns)
        }

        if sheet_name is None:
            target_rid = sheets[0][1]
        else:
            matched = [rid for name, rid in sheets if name == sheet_name]
            if not matched:
                raise KeyError(f"Sheet {sheet_name!r} not found in {path}")
            target_rid = matched[0]

        target = rel_map[target_rid]
        if not target.startswith("xl/"):
            target = f"xl/{target}"

        root = ET.fromstring(zf.read(target))
        rows: List[List[object]] = []
        for row in root.findall(".//a:sheetData/a:row", ns):
            current: List[object] = []
            next_col = 1
            for cell in row.findall("a:c", ns):
                ref = cell.attrib.get("r", "")
                col_letters = "".join(ch for ch in ref if ch.isalpha())
                col_idx = 0
                for ch in col_letters:
                    col_idx = col_idx * 26 + (ord(ch.upper()) - ord("A") + 1)
                while next_col < col_idx:
                    current.append(None)
                    next_col += 1

                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", ns)
                value: object = None
                if value_node is not None:
                    raw = value_node.text
                    if cell_type == "s" and raw is not None:
                        value = shared_strings[int(raw)]
                    else:
                        value = raw
                current.append(value)
                next_col += 1
            rows.append(current)
    return rows


def load_compact_rows(path: Path, sheet_name: Optional[str]) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            raw_rows = [row for row in csv.reader(f)]
    elif suffix == ".xlsx":
        raw_rows = _read_xlsx_with_stdlib(path, sheet_name)
    else:
        raise ValueError(f"Unsupported annotation format: {path.suffix}")

    output = []
    current_video = None
    for row_idx, row in enumerate(raw_rows, start=1):
        if not row:
            continue
        padded = list(row) + [None] * (4 - len(row))
        video_value, start_value, end_value, label_value = padded[:4]
        if video_value not in (None, ""):
            current_video = str(video_value).strip()
        if current_video is None:
            raise ValueError(f"Row {row_idx} is missing a video name before segment data.")
        if start_value in (None, "") and end_value in (None, "") and label_value in (None, ""):
            continue
        output.append(
            {
                "row_idx": row_idx,
                "video_name": current_video,
                "start_frame": "" if start_value is None else str(start_value),
                "end_frame": "" if end_value is None else str(end_value),
                "label": "" if label_value is None else str(label_value),
                "is_new_video_row": video_value not in (None, ""),
            }
        )
    return output


def build_video_index(video_root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = defaultdict(list)
    for video_path in sorted(video_root.rglob("*.mp4")):
        candidates = {video_path.name, video_path.stem}
        for key in candidates:
            index[key].append(video_path)
    return index


def list_video_paths(video_root: Path) -> List[Path]:
    return sorted(video_root.rglob("*.mp4"))


def summarize_rows(rows: Iterable[dict]) -> "OrderedDict[str, VideoStats]":
    videos: "OrderedDict[str, VideoStats]" = OrderedDict()
    for row in rows:
        video_name = row["video_name"]
        row_idx = int(row["row_idx"])
        item = videos.get(video_name)
        if item is None:
            item = VideoStats(
                video_name=video_name,
                first_segment_row=row_idx,
                first_start_frame=row["start_frame"],
            )
            videos[video_name] = item
        if row["is_new_video_row"]:
            item.block_count += 1
            item.block_start_rows.append(row_idx)
        item.segment_count += 1
        item.last_end_frame = row["end_frame"]
        if row["label"] and row["label"] not in item.labels:
            item.labels[row["label"]] = None
    return videos


def key_candidates(video_name: str) -> List[str]:
    raw = video_name.strip()
    path_name = Path(raw).name
    stem = Path(path_name).stem
    candidates = [
        raw,
        path_name,
        stem,
        raw.removesuffix(".mp4"),
        f"{raw}.mp4" if not raw.endswith(".mp4") else raw,
    ]
    ordered: List[str] = []
    seen = set()
    for key in candidates:
        if key and key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def find_matches(video_name: str, index: Dict[str, List[Path]]) -> List[Path]:
    matches: List[Path] = []
    seen = set()
    for key in key_candidates(video_name):
        for path in index.get(key, []):
            if path not in seen:
                matches.append(path)
                seen.add(path)
    return matches


def _write_name_list(output_path: Optional[Path], names: List[str]) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def write_report(
    output_csv: Path,
    summary_txt: Path,
    annotated_missing_txt: Optional[Path],
    unannotated_videos_txt: Optional[Path],
    annotation_file: Path,
    video_root: Path,
    rows: List[dict],
    videos: "OrderedDict[str, VideoStats]",
    index: Dict[str, List[Path]],
    all_video_paths: List[Path],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.parent.mkdir(parents=True, exist_ok=True)

    total_mp4_files = len(all_video_paths)

    present = 0
    missing = 0
    duplicate = 0
    repeated_blocks = 0
    missing_names: List[str] = []
    matched_video_paths = set()

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "video_name",
                "status",
                "matched_count",
                "matched_paths",
                "segment_count",
                "block_count",
                "block_start_rows",
                "first_segment_row",
                "first_start_frame",
                "last_end_frame",
                "labels",
            ]
        )
        for video_name, stats in videos.items():
            matches = find_matches(video_name, index)
            if not matches:
                status = "missing"
                missing += 1
                missing_names.append(video_name)
            elif len(matches) == 1:
                status = "present"
                present += 1
            else:
                status = "duplicate_name"
                duplicate += 1
            matched_video_paths.update(matches)
            if stats.block_count > 1:
                repeated_blocks += 1
            writer.writerow(
                [
                    video_name,
                    status,
                    len(matches),
                    "|".join(str(path) for path in matches),
                    stats.segment_count,
                    stats.block_count,
                    "|".join(str(v) for v in stats.block_start_rows),
                    stats.first_segment_row,
                    stats.first_start_frame,
                    stats.last_end_frame,
                    "|".join(stats.labels.keys()),
                ]
            )

    unannotated_video_names = [video_path.stem for video_path in all_video_paths if video_path not in matched_video_paths]
    _write_name_list(annotated_missing_txt, missing_names)
    _write_name_list(unannotated_videos_txt, unannotated_video_names)

    summary_lines = [
        f"annotation_file: {annotation_file}",
        f"video_root: {video_root}",
        f"report_csv: {output_csv}",
        f"total_annotation_rows: {len(rows)}",
        f"unique_annotated_videos: {len(videos)}",
        f"total_mp4_files_scanned: {total_mp4_files}",
        f"present_videos: {present}",
        f"missing_videos: {missing}",
        f"unannotated_videos: {len(unannotated_video_names)}",
        f"duplicate_name_matches: {duplicate}",
        f"repeated_video_name_blocks: {repeated_blocks}",
    ]
    if annotated_missing_txt is not None:
        summary_lines.append(f"annotated_missing_txt: {annotated_missing_txt}")
    if unannotated_videos_txt is not None:
        summary_lines.append(f"unannotated_videos_txt: {unannotated_videos_txt}")
    if missing_names:
        summary_lines.append("missing_video_names:")
        summary_lines.extend(f"- {name}" for name in missing_names)

    summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = load_compact_rows(args.annotation_file, args.sheet_name)
    videos = summarize_rows(rows)
    index = build_video_index(args.video_root)
    all_video_paths = list_video_paths(args.video_root)
    write_report(
        output_csv=args.output_csv,
        summary_txt=args.summary_txt,
        annotated_missing_txt=args.annotated_missing_txt,
        unannotated_videos_txt=args.unannotated_videos_txt,
        annotation_file=args.annotation_file,
        video_root=args.video_root,
        rows=rows,
        videos=videos,
        index=index,
        all_video_paths=all_video_paths,
    )
    print(f"Wrote detailed report to {args.output_csv}")
    print(f"Wrote summary to {args.summary_txt}")
    if args.annotated_missing_txt is not None:
        print(f"Wrote annotation-missing video names to {args.annotated_missing_txt}")
    if args.unannotated_videos_txt is not None:
        print(f"Wrote unannotated video names to {args.unannotated_videos_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
