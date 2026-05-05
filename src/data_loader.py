"""Data loading and guide-rule extraction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import pandas as pd

from .utils import (
    DATAS_DIR,
    GUIDES_DIR,
    coerce_numeric_columns,
    normalize_columns,
)


SUBMISSION_CSV = "video_engagement_fraud_dataset.csv"
GRAPH_CSV = "video_engagement_graph_timeseries.csv"
SUBMISSION_XLSX = "video_engagement_fraud_dataset.xlsx"
GENERATED_SUBMISSION_CSV = "video_engagement_fraud_dataset_generated_10000.csv"
GENERATED_GRAPH_CSV = "video_engagement_graph_timeseries_generated_10000.csv"

NUMERIC_COLUMNS = {
    "days_live",
    "creator_followers",
    "avg_views_30d",
    "video_length_sec",
    "views",
    "likes",
    "comments",
    "shares",
    "saves",
    "reach",
    "avg_watch_time_sec",
    "watch_time",
    "completion_rate",
    "tier1_audience_pct",
    "profile_traffic_pct",
    "search_traffic_pct",
    "fyp_reels_explore_pct",
    "max_view_jump_pct",
    "flatline_hours_before_jump",
    "suspicious_comment_pct",
    "repeated_comment_pct",
    "bot_like_profile_pct",
    "views_to_avg_multiplier",
    "like_rate",
    "comment_rate",
    "share_rate",
    "engagement_rate",
    "total_engagement_rate",
    "reach_view_ratio",
    "fraud_risk_score",
    "hour_since_post",
    "views_cumulative",
    "likes_cumulative",
    "comments_cumulative",
    "shares_cumulative",
}

BOOLEAN_COLUMNS = {
    "like_freeze_flag",
    "late_like_spike_flag",
    "no_matching_engagement_flag",
}


FALLBACK_RULES = [
    {
        "rule_id": "R001",
        "category": "Core review principle",
        "rule": "Never rely on one signal only; cross-check views, likes, comments, shares, account history, and graph behavior.",
        "source_file": "embedded fallback",
    },
    {
        "rule_id": "R002",
        "category": "High views / low engagement",
        "rule": "High views with almost no likes or comments is suspicious; short-form views and engagement should usually move together.",
        "source_file": "embedded fallback",
    },
    {
        "rule_id": "R003",
        "category": "Graph red flags",
        "rule": "Vertical view spikes, flatline then explosion, likes freeze, late like spike, and step-like growth raise fraud risk.",
        "source_file": "embedded fallback",
    },
    {
        "rule_id": "R004",
        "category": "Reviewer action",
        "rule": "If clearly botted reject; if suspicious but incomplete ask for analytics in ticket; if clean approve but monitor later.",
        "source_file": "embedded fallback",
    },
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    return coerce_numeric_columns(df, NUMERIC_COLUMNS, BOOLEAN_COLUMNS)


def load_csv(path: Path) -> pd.DataFrame:
    return clean_dataframe(pd.read_csv(path))


def read_xlsx_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read one xlsx sheet with pandas when available, else with stdlib XML parsing."""
    try:
        return clean_dataframe(pd.read_excel(path, sheet_name=sheet_name))
    except Exception:
        pass

    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
    r_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    def normalize_target(target: str) -> str:
        target = target.lstrip("/")
        return target if target.startswith("xl/") else f"xl/{target}"

    def cell_text(cell: ET.Element, shared: list[str]) -> str:
        cell_type = cell.attrib.get("t")
        if cell_type == "inlineStr":
            return "".join(t.text or "" for t in cell.findall(".//a:t", ns))
        value_node = cell.find("a:v", ns)
        value = "" if value_node is None else value_node.text or ""
        if cell_type == "s" and value.isdigit() and int(value) < len(shared):
            return shared[int(value)]
        return value

    with ZipFile(path) as archive:
        names = set(archive.namelist())
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared_strings.append("".join(t.text or "" for t in item.findall(".//a:t", ns)))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("rel:Relationship", rel_ns)}
        target_path = None
        for sheet in workbook.findall(".//a:sheet", ns):
            if sheet.attrib.get("name") == sheet_name:
                relationship_id = sheet.attrib.get(f"{{{r_ns}}}id")
                target_path = normalize_target(rel_map[relationship_id])
                break
        if not target_path:
            raise ValueError(f"Sheet not found: {sheet_name}")

        sheet_root = ET.fromstring(archive.read(target_path))
        rows: list[list[str]] = []
        for row in sheet_root.findall(".//a:sheetData/a:row", ns):
            rows.append([cell_text(cell, shared_strings) for cell in row.findall("a:c", ns)])
        if not rows:
            return pd.DataFrame()
        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        header = rows[0]
        data = rows[1:]
        return clean_dataframe(pd.DataFrame(data, columns=header))


def load_submission_data(include_generated: bool = True, data_dir: Path = DATAS_DIR) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    csv_path = data_dir / SUBMISSION_CSV
    if csv_path.exists():
        frames.append(load_csv(csv_path))
    else:
        xlsx_path = data_dir / SUBMISSION_XLSX
        if xlsx_path.exists():
            frames.append(read_xlsx_sheet(xlsx_path, "Submission_Dataset"))

    generated_path = data_dir / GENERATED_SUBMISSION_CSV
    if include_generated and generated_path.exists():
        frames.append(load_csv(generated_path))

    if not frames:
        raise FileNotFoundError(f"No submission dataset found in {data_dir}")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_graph_timeseries(include_generated: bool = True, data_dir: Path = DATAS_DIR) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    csv_path = data_dir / GRAPH_CSV
    if csv_path.exists():
        frames.append(load_csv(csv_path))
    else:
        xlsx_path = data_dir / SUBMISSION_XLSX
        if xlsx_path.exists():
            frames.append(read_xlsx_sheet(xlsx_path, "Graph_TimeSeries"))

    generated_path = data_dir / GENERATED_GRAPH_CSV
    if include_generated and generated_path.exists():
        frames.append(load_csv(generated_path))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def merge_graph_summary(submissions: pd.DataFrame, graph: pd.DataFrame) -> pd.DataFrame:
    if graph.empty or "submission_id" not in submissions.columns or "submission_id" not in graph.columns:
        return submissions.copy()
    summary = (
        graph.sort_values(["submission_id", "hour_since_post"])
        .groupby("submission_id")
        .agg(
            graph_points=("submission_id", "size"),
            graph_final_hour=("hour_since_post", "max"),
            graph_final_views=("views_cumulative", "max"),
            graph_final_likes=("likes_cumulative", "max"),
            graph_final_comments=("comments_cumulative", "max"),
            graph_final_shares=("shares_cumulative", "max"),
        )
        .reset_index()
    )
    return submissions.merge(summary, on="submission_id", how="left")


def extract_pdf_text(path: Path, max_pages: int | None = None) -> str:
    """Best-effort PDF extraction. Empty text is acceptable for image PDFs."""
    try:
        import pdfplumber

        text_parts: list[str] = []
        with pdfplumber.open(path) as pdf:
            pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
            for page in pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts).strip()
    except Exception:
        pass

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = reader.pages if max_pages is None else reader.pages[:max_pages]
        return "\n".join(page.extract_text() or "" for page in pages).strip()
    except Exception:
        return ""


def load_review_rules(data_dir: Path = DATAS_DIR, guides_dir: Path = GUIDES_DIR) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []

    for pdf_path in sorted(guides_dir.glob("*.pdf")):
        text = extract_pdf_text(pdf_path, max_pages=3)
        if text:
            rules.append(
                {
                    "rule_id": f"PDF-{len(rules) + 1:03d}",
                    "category": "PDF extracted guidance",
                    "rule": text[:3000],
                    "source_file": pdf_path.name,
                }
            )

    workbook_path = data_dir / SUBMISSION_XLSX
    if workbook_path.exists():
        try:
            workbook_rules = read_xlsx_sheet(workbook_path, "Labeling_Rules")
            for record in workbook_rules.to_dict(orient="records"):
                if record.get("rule"):
                    rules.append(record)
        except Exception:
            pass

    return rules or FALLBACK_RULES.copy()


def load_all(include_generated: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    submissions = load_submission_data(include_generated=include_generated)
    graph = load_graph_timeseries(include_generated=include_generated)
    rules = load_review_rules()
    return submissions, graph, rules

