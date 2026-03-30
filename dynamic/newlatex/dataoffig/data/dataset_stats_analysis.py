#!/usr/bin/env python3
"""
Compute raw-data statistics for the 8 benchmark subsets.

Inputs (per subset directory, e.g., oa_1):
  - oa_1_covert.csv                      # yearly panel: year x node publication counts
  - oa_1_coo/*.xlsx                      # yearly co-occurrence matrix files

Outputs (written to --output-dir, default: script directory):
  - raw_data_stats_summary.csv
  - raw_data_stats_report.md
  - raw_data_stats_latex_rows.txt

No third-party dependencies are required.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET


OA_ORDER = [
    "oa_1",
    "oa_2",
    "oa_3",
    "oa_4",
    "oa_1702",
    "oa_1705",
    "oa_1707",
    "oa_1710",
]

DISPLAY_NAME = {
    "oa_1": "domain 1",
    "oa_2": "domain 2",
    "oa_3": "domain 3",
    "oa_4": "domain 4",
    "oa_1702": "subfile 1702",
    "oa_1705": "subfile 1705",
    "oa_1707": "subfile 1707",
    "oa_1710": "subfile 1710",
}

XLSX_YEAR_RE = re.compile(r"(\d{4})\.xlsx$", re.IGNORECASE)
CELL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")
XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


@dataclass
class PanelStats:
    start_year: int
    end_year: int
    T: int
    N: int
    total_start: float
    total_end: float
    growth_ratio: Optional[float]
    sparsity_pct: float


@dataclass
class GraphStats:
    graph_year_count: int
    graph_start_year: Optional[int]
    graph_end_year: Optional[int]
    n_from_graph: Optional[int]
    avg_density_pct: Optional[float]
    avg_turnover_pct: Optional[float]


def col_letters_to_idx(col_letters: str) -> int:
    """Excel column letters to 1-based index. A -> 1, Z -> 26, AA -> 27."""
    idx = 0
    for ch in col_letters.upper():
        if "A" <= ch <= "Z":
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx


def safe_float(text: str) -> float:
    if text is None:
        return 0.0
    t = text.strip()
    if not t:
        return 0.0
    return float(t)


def parse_panel_csv(csv_path: Path) -> PanelStats:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValueError(f"Invalid panel csv header: {csv_path}")

        node_count = len(header) - 1
        years: List[int] = []
        yearly_totals: List[float] = []
        zero_count = 0
        total_cells = 0

        for row in reader:
            if not row:
                continue
            year = int(row[0])
            vals = [safe_float(x) for x in row[1 : node_count + 1]]
            if len(vals) < node_count:
                vals.extend([0.0] * (node_count - len(vals)))

            years.append(year)
            yearly_totals.append(sum(vals))
            zero_count += sum(1 for v in vals if v == 0.0)
            total_cells += len(vals)

    if not years:
        raise ValueError(f"No data rows in panel csv: {csv_path}")

    start_idx = min(range(len(years)), key=lambda i: years[i])
    end_idx = max(range(len(years)), key=lambda i: years[i])
    total_start = yearly_totals[start_idx]
    total_end = yearly_totals[end_idx]
    growth_ratio = None if total_start == 0 else (total_end / total_start)
    sparsity_pct = 100.0 * zero_count / total_cells if total_cells else 0.0

    return PanelStats(
        start_year=min(years),
        end_year=max(years),
        T=len(years),
        N=node_count,
        total_start=total_start,
        total_end=total_end,
        growth_ratio=growth_ratio,
        sparsity_pct=sparsity_pct,
    )


def parse_graph_edges_from_xlsx(xlsx_path: Path) -> Tuple[int, Set[Tuple[int, int]]]:
    """
    Parse one yearly co-occurrence matrix from xlsx and return:
      - N (matrix node count)
      - undirected edge set on index pairs (i, j), i < j, where weight > 0
    """
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        with zf.open("xl/worksheets/sheet1.xml") as fp:
            root = ET.parse(fp).getroot()

    sheet_data = root.find(f"{XML_NS}sheetData")
    if sheet_data is None:
        raise ValueError(f"sheetData not found in {xlsx_path}")

    rows = list(sheet_data.findall(f"{XML_NS}row"))
    if not rows:
        raise ValueError(f"No rows in sheet1.xml for {xlsx_path}")

    # Header row: columns B.. correspond to node IDs, so N = number of header cells with col >= 2.
    header_cells = rows[0].findall(f"{XML_NS}c")
    n = 0
    for c in header_cells:
        ref = c.attrib.get("r", "")
        m = CELL_RE.match(ref)
        if not m:
            continue
        col_idx = col_letters_to_idx(m.group(1))
        if col_idx >= 2:
            n += 1

    if n <= 0:
        # Fallback: infer from number of data rows.
        n = max(0, len(rows) - 1)

    edges: Set[Tuple[int, int]] = set()
    # Data rows correspond to i=1..N.
    for i, row in enumerate(rows[1:], start=1):
        for c in row.findall(f"{XML_NS}c"):
            ref = c.attrib.get("r", "")
            m = CELL_RE.match(ref)
            if not m:
                continue

            col_idx = col_letters_to_idx(m.group(1))
            # col A is row label; col B corresponds to j=1
            if col_idx <= 1:
                continue

            j = col_idx - 1
            if j <= i:
                # Keep upper triangle only (undirected, no diagonal).
                continue

            v = c.find(f"{XML_NS}v")
            val = safe_float(v.text if v is not None else "0")
            if val > 0:
                edges.add((i, j))

    return n, edges


def parse_graph_stats(coo_dir: Path) -> GraphStats:
    xlsx_files = sorted(
        [p for p in coo_dir.glob("*.xlsx") if XLSX_YEAR_RE.search(p.name)],
        key=lambda p: int(XLSX_YEAR_RE.search(p.name).group(1)),
    )

    if not xlsx_files:
        return GraphStats(
            graph_year_count=0,
            graph_start_year=None,
            graph_end_year=None,
            n_from_graph=None,
            avg_density_pct=None,
            avg_turnover_pct=None,
        )

    densities: List[float] = []
    turnovers: List[float] = []
    prev_edges: Optional[Set[Tuple[int, int]]] = None
    n_values: List[int] = []
    years: List[int] = []

    for fp in xlsx_files:
        year = int(XLSX_YEAR_RE.search(fp.name).group(1))
        n, edges = parse_graph_edges_from_xlsx(fp)
        years.append(year)
        n_values.append(n)

        denom = n * (n - 1) / 2.0
        density = (len(edges) / denom) if denom > 0 else 0.0
        densities.append(density)

        if prev_edges is not None:
            union = prev_edges | edges
            if len(union) == 0:
                turnover = 0.0
            else:
                inter = prev_edges & edges
                turnover = 1.0 - (len(inter) / len(union))
            turnovers.append(turnover)
        prev_edges = edges

    # Most robust N selection in case of minor inconsistency across files.
    n_from_graph = int(statistics.mode(n_values)) if n_values else None

    return GraphStats(
        graph_year_count=len(xlsx_files),
        graph_start_year=min(years),
        graph_end_year=max(years),
        n_from_graph=n_from_graph,
        avg_density_pct=100.0 * statistics.fmean(densities) if densities else None,
        avg_turnover_pct=100.0 * statistics.fmean(turnovers) if turnovers else None,
    )


def fmt_num(v: Optional[float], ndigits: int = 2) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "NA"
    return f"{v:.{ndigits}f}"


def build_rows(data_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for subset in OA_ORDER:
        subset_dir = data_root / subset
        if not subset_dir.exists():
            continue

        panel_csv = subset_dir / f"{subset}_covert.csv"
        coo_dir = subset_dir / f"{subset}_coo"

        panel = parse_panel_csv(panel_csv)
        graph = parse_graph_stats(coo_dir)

        years_text = f"{panel.start_year}--{panel.end_year}"
        n_match = (
            "yes"
            if (graph.n_from_graph is not None and graph.n_from_graph == panel.N)
            else ("no" if graph.n_from_graph is not None else "NA")
        )

        rows.append(
            {
                "subset_key": subset,
                "subset_name": DISPLAY_NAME.get(subset, subset),
                "years": years_text,
                "T": str(panel.T),
                "N": str(panel.N),
                "total_vol_start": fmt_num(panel.total_start, 0),
                "total_vol_end": fmt_num(panel.total_end, 0),
                "growth_ratio": fmt_num(panel.growth_ratio, 4),
                "panel_sparsity_pct": fmt_num(panel.sparsity_pct, 2),
                "graph_year_count": str(graph.graph_year_count),
                "graph_years": (
                    f"{graph.graph_start_year}--{graph.graph_end_year}"
                    if graph.graph_start_year is not None and graph.graph_end_year is not None
                    else "NA"
                ),
                "n_from_graph": str(graph.n_from_graph) if graph.n_from_graph is not None else "NA",
                "n_match_panel": n_match,
                "avg_graph_density_pct": fmt_num(graph.avg_density_pct, 3),
                "avg_edge_turnover_pct": fmt_num(graph.avg_turnover_pct, 3),
            }
        )
    return rows


def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "subset_key",
        "subset_name",
        "years",
        "T",
        "N",
        "total_vol_start",
        "total_vol_end",
        "growth_ratio",
        "panel_sparsity_pct",
        "graph_year_count",
        "graph_years",
        "n_from_graph",
        "n_match_panel",
        "avg_graph_density_pct",
        "avg_edge_turnover_pct",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, str]], out_md: Path) -> None:
    lines = [
        "# Raw Data Statistics Summary",
        "",
        "| Subset | Years | T | N | Total vol. (start) | Total vol. (end) | Growth ratio | Panel sparsity (%) | Avg. graph density (%) | Avg. edge turnover (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {subset_name} | {years} | {T} | {N} | {total_vol_start} | {total_vol_end} | {growth_ratio} | {panel_sparsity_pct} | {avg_graph_density_pct} | {avg_edge_turnover_pct} |".format(
                **r
            )
        )

    lines += [
        "",
        "## Consistency Checks",
        "",
        "| Subset | Graph files | Graph years | N from graph | N match panel |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {subset_name} | {graph_year_count} | {graph_years} | {n_from_graph} | {n_match_panel} |".format(
                **r
            )
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_rows(rows: List[Dict[str, str]], out_tex_rows: Path) -> None:
    """
    Write rows that can be pasted directly into the placeholder table:
      Subset & Years & T & N & Total vol. (start) & Total vol. (end) & Growth ratio &
      Panel sparsity (%) & Avg. graph density (%) & Avg. edge turnover (%) \\
    """
    lines: List[str] = []
    for r in rows:
        lines.append(
            "{subset_name} & {years} & {T} & {N} & {total_vol_start} & {total_vol_end} & {growth_ratio} & {panel_sparsity_pct} & {avg_graph_density_pct} & {avg_edge_turnover_pct} \\\\".format(
                **r
            )
        )
    out_tex_rows.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dataset statistics for OpenAlex subsets.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root folder containing oa_* subset directories (default: script directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to write outputs (default: script directory).",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(data_root)
    if not rows:
        raise RuntimeError(f"No subset rows parsed from data root: {data_root}")

    out_csv = output_dir / "raw_data_stats_summary.csv"
    out_md = output_dir / "raw_data_stats_report.md"
    out_tex_rows = output_dir / "raw_data_stats_latex_rows.txt"

    write_csv(rows, out_csv)
    write_markdown(rows, out_md)
    write_latex_rows(rows, out_tex_rows)

    print(f"[OK] Parsed subsets: {len(rows)}")
    print(f"[OK] CSV: {out_csv}")
    print(f"[OK] Markdown: {out_md}")
    print(f"[OK] LaTeX rows: {out_tex_rows}")


if __name__ == "__main__":
    main()

