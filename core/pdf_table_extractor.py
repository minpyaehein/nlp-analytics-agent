"""Extract structured tables from searchable or OCR-processed PDF files.

The extractor uses two strategies in order:
1. PyMuPDF's native table detector for vector/searchable PDFs.
2. Word-coordinate reconstruction for OCR text layers.

The output includes DataFrames, extraction evidence, confidence, and warnings.
"""

from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import pandas as pd
import pymupdf


DEFAULT_EXPECTED_HEADERS = [
    "order_id",
    "order_date",
    "product",
    "category",
    "region",
    "quantity",
    "unit_price",
    "unit_cost",
]

HEADER_ALIASES = {
    "order_id": {"order_id", "orderid", "order no", "order number", "id"},
    "order_date": {"order_date", "orderdate", "order date", "date"},
    "product": {"product", "item", "product name"},
    "category": {"category", "type", "product category"},
    "region": {"region", "location", "area", "state", "division"},
    "quantity": {"quantity", "qty", "units", "unit count"},
    "unit_price": {"unit_price", "unitprice", "unit price", "price"},
    "unit_cost": {"unit_cost", "unitcost", "unit cost", "cost"},
}

NUMERIC_HEADERS = {
    "order_id",
    "quantity",
    "unit_price",
    "unit_cost",
}

DATE_HEADERS = {"order_date", "date"}


@dataclass
class ExtractedTable:
    """One extracted PDF table with provenance and validation evidence."""

    page_number: int
    table_number: int
    strategy: str
    dataframe: pd.DataFrame
    confidence: float
    header_matches: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        """Return JSON-friendly metadata without embedding the DataFrame."""

        return {
            "page_number": self.page_number,
            "table_number": self.table_number,
            "strategy": self.strategy,
            "confidence": self.confidence,
            "row_count": int(len(self.dataframe)),
            "column_count": int(len(self.dataframe.columns)),
            "columns": [str(column) for column in self.dataframe.columns],
            "header_matches": list(self.header_matches),
            "warnings": list(self.warnings),
            "validation": dict(self.validation),
        }


@dataclass
class PdfTableExtractionResult:
    """Complete result for one PDF table-extraction operation."""

    success: bool
    filename: str
    page_count: int
    tables: list[ExtractedTable]
    warnings: list[str]
    validation: dict[str, Any]

    def metadata(self) -> dict[str, Any]:
        """Return JSON-friendly extraction metadata."""

        return {
            "success": self.success,
            "filename": self.filename,
            "page_count": self.page_count,
            "table_count": len(self.tables),
            "tables": [table.metadata() for table in self.tables],
            "warnings": list(self.warnings),
            "validation": dict(self.validation),
        }

    def best_table(self) -> ExtractedTable:
        """Return the highest-confidence extracted table."""

        if not self.tables:
            raise ValueError("No tables were extracted from the PDF.")

        return max(
            self.tables,
            key=lambda item: (
                item.confidence,
                len(item.dataframe),
                len(item.dataframe.columns),
            ),
        )


def normalize_header(value: Any) -> str:
    """Normalize OCR or PDF header text to a comparable identifier."""

    text = "" if value is None else str(value)
    text = text.strip().casefold()
    text = text.replace("-", "_")
    text = re.sub(r"[\s/]+", "_", text)
    text = re.sub(r"[^0-9a-z_]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def canonical_header(
    value: Any,
    expected_headers: Iterable[str],
) -> str | None:
    """Map a detected header to an expected canonical field name."""

    normalized = normalize_header(value)

    if not normalized:
        return None

    for header in expected_headers:
        candidates = {
            normalize_header(header),
            *(normalize_header(alias) for alias in HEADER_ALIASES.get(header, set())),
        }

        if normalized in candidates:
            return header

    return None


def _read_pdf_bytes(
    uploaded_file: BinaryIO | bytes | bytearray | str | Path,
) -> bytes:
    """Read PDF bytes from a Streamlit upload, path, or bytes object."""

    if isinstance(uploaded_file, (bytes, bytearray)):
        raw_bytes = bytes(uploaded_file)
    elif isinstance(uploaded_file, (str, Path)):
        raw_bytes = Path(uploaded_file).read_bytes()
    else:
        try:
            uploaded_file.seek(0)
            raw_bytes = uploaded_file.read()
        except (AttributeError, OSError) as error:
            raise ValueError("The uploaded PDF stream is not readable.") from error

    if not raw_bytes:
        raise ValueError("The uploaded PDF is empty.")

    if not raw_bytes.startswith(b"%PDF"):
        raise ValueError("The uploaded file is not a valid PDF.")

    return raw_bytes


def _make_unique_headers(headers: list[str]) -> list[str]:
    """Make duplicate and empty column names deterministic."""

    counts: dict[str, int] = {}
    output: list[str] = []

    for index, header in enumerate(headers, start=1):
        base = normalize_header(header) or f"column_{index}"
        counts[base] = counts.get(base, 0) + 1
        output.append(base if counts[base] == 1 else f"{base}_{counts[base]}")

    return output


def _coerce_dataframe_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Trim values and convert known numeric/date fields safely."""

    result = dataframe.copy()
    result.columns = [normalize_header(column) for column in result.columns]

    for column in result.columns:
        result[column] = result[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        result[column] = result[column].replace({"": pd.NA, "None": pd.NA})

        if column in NUMERIC_HEADERS:
            result[column] = pd.to_numeric(result[column], errors="coerce")

        if column in DATE_HEADERS:
            parsed = pd.to_datetime(result[column], errors="coerce")
            result[column] = parsed.dt.strftime("%Y-%m-%d")

    result = result.dropna(how="all").reset_index(drop=True)
    return result


def validate_table(
    dataframe: pd.DataFrame,
    expected_headers: list[str],
) -> dict[str, Any]:
    """Validate extracted structure and return deterministic evidence."""

    actual_headers = [normalize_header(column) for column in dataframe.columns]
    matched_headers = [
        header for header in expected_headers if header in actual_headers
    ]
    missing_headers = [
        header for header in expected_headers if header not in actual_headers
    ]
    nonempty_rows = int(len(dataframe.dropna(how="all")))
    header_ratio = (
        len(matched_headers) / len(expected_headers)
        if expected_headers
        else 1.0
    )

    status = "passed"

    if nonempty_rows == 0 or header_ratio < 0.5:
        status = "failed"
    elif missing_headers:
        status = "warning"

    return {
        "status": status,
        "row_count": nonempty_rows,
        "column_count": int(len(dataframe.columns)),
        "matched_headers": matched_headers,
        "missing_headers": missing_headers,
        "header_match_ratio": round(header_ratio, 4),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "total_missing_values": int(dataframe.isna().sum().sum()),
    }


def _confidence_from_validation(validation: dict[str, Any]) -> float:
    """Calculate extraction confidence from objective validation evidence."""

    header_score = float(validation.get("header_match_ratio", 0.0))
    row_score = min(float(validation.get("row_count", 0)) / 10.0, 1.0)
    status_bonus = {
        "passed": 0.10,
        "warning": 0.04,
        "failed": 0.0,
    }.get(str(validation.get("status")), 0.0)

    return round(
        min(1.0, 0.72 * header_score + 0.18 * row_score + status_bonus),
        4,
    )


def _dataframe_from_native_table(
    extracted_rows: list[list[Any]],
    expected_headers: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Convert PyMuPDF native-table output into a normalized DataFrame."""

    warnings: list[str] = []

    if not extracted_rows:
        return pd.DataFrame(), ["The native table contained no rows."]

    matrix = [
        ["" if value is None else str(value).strip() for value in row]
        for row in extracted_rows
    ]

    first_row = matrix[0]
    canonical = [
        canonical_header(value, expected_headers) for value in first_row
    ]
    header_matches = sum(value is not None for value in canonical)

    if header_matches >= max(2, len(expected_headers) // 2):
        headers = [
            detected or normalize_header(original) or f"column_{index + 1}"
            for index, (detected, original) in enumerate(zip(canonical, first_row))
        ]
        data_rows = matrix[1:]
    else:
        headers = [
            expected_headers[index]
            if index < len(expected_headers)
            else f"column_{index + 1}"
            for index in range(len(first_row))
        ]
        data_rows = matrix
        warnings.append(
            "The native table header was not confidently detected; "
            "positional expected headers were applied."
        )

    headers = _make_unique_headers(headers)
    width = len(headers)
    normalized_rows = [
        (row + [""] * width)[:width]
        for row in data_rows
    ]

    dataframe = pd.DataFrame(normalized_rows, columns=headers)
    return _coerce_dataframe_types(dataframe), warnings


def _extract_native_tables(
    page: pymupdf.Page,
    page_number: int,
    expected_headers: list[str],
) -> list[ExtractedTable]:
    """Extract vector/searchable tables using PyMuPDF's table detector."""

    output: list[ExtractedTable] = []

    try:
        finder = page.find_tables()
    except Exception:
        return output

    for table_number, table in enumerate(finder.tables, start=1):
        try:
            extracted_rows = table.extract()
            dataframe, warnings = _dataframe_from_native_table(
                extracted_rows,
                expected_headers,
            )
        except Exception as error:
            output.append(
                ExtractedTable(
                    page_number=page_number,
                    table_number=table_number,
                    strategy="native",
                    dataframe=pd.DataFrame(),
                    confidence=0.0,
                    warnings=[f"Native table extraction failed: {error}"],
                    validation={"status": "failed"},
                )
            )
            continue

        validation = validate_table(dataframe, expected_headers)
        output.append(
            ExtractedTable(
                page_number=page_number,
                table_number=table_number,
                strategy="native",
                dataframe=dataframe,
                confidence=_confidence_from_validation(validation),
                header_matches=validation["matched_headers"],
                warnings=warnings,
                validation=validation,
            )
        )

    return output


def _cluster_words_by_line(
    words: list[tuple[Any, ...]],
    tolerance: float = 4.0,
) -> list[list[tuple[Any, ...]]]:
    """Cluster PyMuPDF words into visual rows using vertical centers."""

    sorted_words = sorted(
        words,
        key=lambda word: (((float(word[1]) + float(word[3])) / 2), float(word[0])),
    )
    lines: list[list[tuple[Any, ...]]] = []
    line_centers: list[float] = []

    for word in sorted_words:
        center_y = (float(word[1]) + float(word[3])) / 2
        assigned = False

        for index, existing_center in enumerate(line_centers):
            if abs(center_y - existing_center) <= tolerance:
                lines[index].append(word)
                count = len(lines[index])
                line_centers[index] = (
                    existing_center * (count - 1) + center_y
                ) / count
                assigned = True
                break

        if not assigned:
            lines.append([word])
            line_centers.append(center_y)

    paired = sorted(zip(line_centers, lines), key=lambda item: item[0])
    return [sorted(line, key=lambda word: float(word[0])) for _, line in paired]


def _line_text(line: list[tuple[Any, ...]]) -> str:
    return " ".join(str(word[4]).strip() for word in line if str(word[4]).strip())


def _find_header_line(
    lines: list[list[tuple[Any, ...]]],
    expected_headers: list[str],
) -> tuple[int, list[str]] | None:
    """Find the visual line most likely to contain the table header."""

    best: tuple[int, list[str]] | None = None

    for index, line in enumerate(lines):
        line_text = _line_text(line)
        normalized_line = normalize_header(line_text)
        matched = []

        for header in expected_headers:
            candidates = {
                normalize_header(header),
                *(normalize_header(alias) for alias in HEADER_ALIASES.get(header, set())),
            }

            if any(candidate and candidate in normalized_line for candidate in candidates):
                matched.append(header)

        if best is None or len(matched) > len(best[1]):
            best = (index, matched)

    if best is None or len(best[1]) < 2:
        return None

    return best


def _column_centers_from_header(
    header_line: list[tuple[Any, ...]],
    expected_headers: list[str],
    page_width: float,
) -> tuple[list[float], list[str]]:
    """Estimate column centers from OCR header words and expected order."""

    detected: dict[str, float] = {}

    for word in header_line:
        token = str(word[4]).strip()
        canonical = canonical_header(token, expected_headers)

        if canonical is not None and canonical not in detected:
            detected[canonical] = (float(word[0]) + float(word[2])) / 2

    known_indices = [
        index for index, header in enumerate(expected_headers) if header in detected
    ]

    if len(known_indices) >= 2:
        centers: list[float | None] = [
            detected.get(header) for header in expected_headers
        ]

        for index, value in enumerate(centers):
            if value is not None:
                continue

            left = max((i for i in known_indices if i < index), default=None)
            right = min((i for i in known_indices if i > index), default=None)

            if left is not None and right is not None:
                fraction = (index - left) / (right - left)
                centers[index] = float(centers[left]) + fraction * (
                    float(centers[right]) - float(centers[left])
                )
            elif left is not None:
                step = page_width / (len(expected_headers) + 1)
                centers[index] = float(centers[left]) + step * (index - left)
            elif right is not None:
                step = page_width / (len(expected_headers) + 1)
                centers[index] = float(centers[right]) - step * (right - index)

        return [float(value) for value in centers], list(detected)

    step = page_width / len(expected_headers)
    fallback_centers = [step * (index + 0.5) for index in range(len(expected_headers))]
    return fallback_centers, list(detected)


def _assign_line_to_columns(
    line: list[tuple[Any, ...]],
    centers: list[float],
) -> list[str]:
    """Assign each OCR word to the nearest expected column center."""

    columns: list[list[tuple[float, str]]] = [[] for _ in centers]

    for word in line:
        text = str(word[4]).strip()

        if not text:
            continue

        center_x = (float(word[0]) + float(word[2])) / 2
        index = min(
            range(len(centers)),
            key=lambda item: abs(center_x - centers[item]),
        )
        columns[index].append((float(word[0]), text))

    return [
        " ".join(text for _, text in sorted(items)).strip()
        for items in columns
    ]


def _looks_like_data_row(values: list[str]) -> bool:
    """Reject headings and retain rows containing date or numeric evidence."""

    joined = " ".join(values)
    has_date = bool(re.search(r"\b20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b", joined))
    numeric_cells = sum(
        bool(re.fullmatch(r"[\d,]+(?:\.\d+)?", value.strip()))
        for value in values
        if value.strip()
    )
    return has_date or numeric_cells >= 3


def _extract_coordinate_table(
    page: pymupdf.Page,
    page_number: int,
    expected_headers: list[str],
) -> ExtractedTable | None:
    """Reconstruct an OCR table from word coordinates."""

    words = page.get_text("words", sort=True)

    if not words:
        return None

    lines = _cluster_words_by_line(words)
    header_result = _find_header_line(lines, expected_headers)

    if header_result is None:
        return None

    header_index, header_matches = header_result
    centers, individually_detected = _column_centers_from_header(
        lines[header_index],
        expected_headers,
        float(page.rect.width),
    )

    reconstructed_rows: list[list[str]] = []
    empty_run = 0

    for line in lines[header_index + 1 :]:
        values = _assign_line_to_columns(line, centers)

        if not any(values):
            empty_run += 1
            if empty_run >= 2 and reconstructed_rows:
                break
            continue

        if _looks_like_data_row(values):
            reconstructed_rows.append(values)
            empty_run = 0
        elif reconstructed_rows:
            # Stop when narrative text starts after the table.
            if len(_line_text(line)) > 25:
                break

    if not reconstructed_rows:
        return None

    dataframe = pd.DataFrame(reconstructed_rows, columns=expected_headers)
    dataframe = _coerce_dataframe_types(dataframe)
    validation = validate_table(dataframe, expected_headers)
    warnings: list[str] = []

    if len(individually_detected) < len(expected_headers):
        warnings.append(
            "Some column positions were interpolated because OCR did not "
            "recognize every header individually."
        )

    if validation["total_missing_values"]:
        warnings.append(
            "The OCR table contains missing cells; review low-confidence rows."
        )

    confidence = _confidence_from_validation(validation)
    confidence *= min(1.0, 0.65 + 0.35 * len(individually_detected) / len(expected_headers))

    return ExtractedTable(
        page_number=page_number,
        table_number=1,
        strategy="word_coordinates",
        dataframe=dataframe,
        confidence=round(confidence, 4),
        header_matches=header_matches,
        warnings=warnings,
        validation=validation,
    )


def extract_pdf_tables(
    uploaded_file: BinaryIO | bytes | bytearray | str | Path,
    filename: str | None = None,
    expected_headers: list[str] | None = None,
) -> PdfTableExtractionResult:
    """Extract validated tables from a searchable or OCR-processed PDF.

    Args:
        uploaded_file: PDF path, bytes, or seekable binary upload.
        filename: Original filename for provenance.
        expected_headers: Optional expected schema in canonical order.

    Returns:
        PdfTableExtractionResult containing all viable candidates.
    """

    headers = list(expected_headers or DEFAULT_EXPECTED_HEADERS)
    raw_bytes = _read_pdf_bytes(uploaded_file)
    resolved_filename = filename or (
        Path(uploaded_file).name
        if isinstance(uploaded_file, (str, Path))
        else "uploaded.pdf"
    )

    try:
        document = pymupdf.open(stream=raw_bytes, filetype="pdf")
    except Exception as error:
        raise ValueError(f"Unable to open the PDF: {error}") from error

    candidates: list[ExtractedTable] = []
    warnings: list[str] = []

    try:
        for page_index, page in enumerate(document, start=1):
            native_tables = _extract_native_tables(page, page_index, headers)
            viable_native = [
                table
                for table in native_tables
                if not table.dataframe.empty
                and table.validation.get("status") != "failed"
            ]
            candidates.extend(viable_native)

            if not viable_native:
                coordinate_table = _extract_coordinate_table(
                    page,
                    page_index,
                    headers,
                )

                if coordinate_table is not None:
                    candidates.append(coordinate_table)
                else:
                    warnings.append(
                        f"No table could be reconstructed from page {page_index}."
                    )

        page_count = int(document.page_count)
    finally:
        document.close()

    candidates.sort(
        key=lambda item: (
            item.confidence,
            len(item.dataframe),
        ),
        reverse=True,
    )

    successful = bool(candidates)
    validation = {
        "status": "passed" if successful else "failed",
        "page_count": page_count,
        "table_count": len(candidates),
        "highest_confidence": (
            candidates[0].confidence if candidates else 0.0
        ),
        "strategies_used": sorted({table.strategy for table in candidates}),
    }

    return PdfTableExtractionResult(
        success=successful,
        filename=resolved_filename,
        page_count=page_count,
        tables=candidates,
        warnings=warnings,
        validation=validation,
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Extract tables from a searchable PDF."
    )
    parser.add_argument("pdf_path")
    parser.add_argument("--csv", default=None)
    arguments = parser.parse_args()

    result = extract_pdf_tables(arguments.pdf_path)
    print(json.dumps(result.metadata(), indent=2, ensure_ascii=False))

    if result.success:
        best = result.best_table()
        print()
        print(best.dataframe.to_string(index=False))

        if arguments.csv:
            best.dataframe.to_csv(
                arguments.csv,
                index=False,
                encoding="utf-8-sig",
            )
            print(f"Saved CSV: {arguments.csv}")
