"""Searchable PDF text and table extraction for InsightFlow AI.

The loader uses PyMuPDF for text blocks and document metadata, and
pdfplumber for heuristic table extraction. Scanned/image-only PDFs are
detected and rejected with a clear OCR recommendation.
"""

from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import pymupdf
import pandas as pd
import pdfplumber


@dataclass
class PdfTextBlock:
    block_id: str
    page_number: int
    block_index: int
    text: str
    block_type: str
    bbox: list[float]
    source_reference: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PdfTableData:
    table_index: int
    table_id: str
    page_number: int
    page_table_index: int
    title: str
    row_count: int
    column_count: int
    headers: list[str]
    dataframe: pd.DataFrame
    source_reference: str
    bbox: list[float] | None = None
    warnings: list[str] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("dataframe", None)
        return data


@dataclass
class PdfMetadata:
    filename: str
    format: str
    page_count: int
    searchable_page_count: int
    image_only_page_count: int
    text_block_count: int
    text_character_count: int
    extracted_table_count: int
    title: str | None
    author: str | None
    subject: str | None
    creator: str | None
    producer: str | None
    creation_date: str | None
    modification_date: str | None
    is_searchable: bool
    needs_ocr: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedPdf:
    metadata: PdfMetadata
    text_blocks: list[PdfTextBlock]
    tables: list[PdfTableData]

    def text_records(self) -> list[dict[str, Any]]:
        return [block.to_dict() for block in self.text_blocks]

    def table_summaries(self) -> list[dict[str, Any]]:
        return [table.metadata() for table in self.tables]

    def combined_text(self) -> str:
        return "\n\n".join(block.text for block in self.text_blocks)


@dataclass
class SelectedPdfTable:
    dataframe: pd.DataFrame
    table_metadata: dict[str, Any]
    document_metadata: dict[str, Any]


SUPPORTED_EXTENSION = ".pdf"
MIN_SEARCHABLE_CHARACTERS = 20


def extract_pdf(
    uploaded_file: BinaryIO,
    filename: str,
    extract_tables: bool = True,
) -> ExtractedPdf:
    """Extract searchable text, metadata, and heuristic tables from a PDF."""

    cleaned_filename = _validate_filename(filename)
    raw_bytes = _read_bytes(uploaded_file)

    text_blocks, page_text_counts, fitz_metadata, page_count = _extract_text(
        raw_bytes
    )
    tables = _extract_tables(raw_bytes) if extract_tables else []

    searchable_pages = sum(
        count >= MIN_SEARCHABLE_CHARACTERS
        for count in page_text_counts
    )
    image_only_pages = page_count - searchable_pages
    total_characters = sum(len(block.text) for block in text_blocks)
    is_searchable = total_characters >= MIN_SEARCHABLE_CHARACTERS
    needs_ocr = not is_searchable or image_only_pages == page_count
    warnings: list[str] = []

    if not is_searchable:
        warnings.append(
            "The PDF contains insufficient searchable text. Run OCR before "
            "narrative or table analysis."
        )
    elif image_only_pages > 0:
        warnings.append(
            f"{image_only_pages} page(s) contain little or no searchable text; "
            "OCR may be required for complete extraction."
        )
    if extract_tables and not tables:
        warnings.append(
            "No structured tables were detected. PDF table extraction is "
            "heuristic and works best with ruled or consistently aligned tables."
        )

    metadata = PdfMetadata(
        filename=cleaned_filename,
        format="pdf",
        page_count=page_count,
        searchable_page_count=int(searchable_pages),
        image_only_page_count=int(image_only_pages),
        text_block_count=len(text_blocks),
        text_character_count=total_characters,
        extracted_table_count=len(tables),
        title=_clean_optional(fitz_metadata.get("title")),
        author=_clean_optional(fitz_metadata.get("author")),
        subject=_clean_optional(fitz_metadata.get("subject")),
        creator=_clean_optional(fitz_metadata.get("creator")),
        producer=_clean_optional(fitz_metadata.get("producer")),
        creation_date=_clean_optional(fitz_metadata.get("creationDate")),
        modification_date=_clean_optional(fitz_metadata.get("modDate")),
        is_searchable=is_searchable,
        needs_ocr=needs_ocr,
        warnings=warnings,
    )

    return ExtractedPdf(
        metadata=metadata,
        text_blocks=text_blocks,
        tables=tables,
    )


def list_pdf_tables(
    uploaded_file: BinaryIO,
    filename: str,
) -> list[dict[str, Any]]:
    """Return detected table summaries."""

    return extract_pdf(uploaded_file, filename).table_summaries()


def select_pdf_table(
    uploaded_file: BinaryIO,
    filename: str,
    table_index: int = 0,
) -> SelectedPdfTable:
    """Select one detected PDF table for Pandas analytics."""

    extracted = extract_pdf(uploaded_file, filename)

    if not extracted.tables:
        if extracted.metadata.needs_ocr:
            raise ValueError(
                "The PDF has no usable searchable table and appears to need OCR."
            )
        raise ValueError("The PDF contains no detected analytical tables.")

    if table_index < 0 or table_index >= len(extracted.tables):
        raise ValueError(
            f"Table index {table_index} is outside the valid range 0 to "
            f"{len(extracted.tables) - 1}."
        )

    selected = extracted.tables[table_index]
    return SelectedPdfTable(
        dataframe=selected.dataframe.copy(),
        table_metadata=selected.metadata(),
        document_metadata=extracted.metadata.to_dict(),
    )


def _validate_filename(filename: str) -> str:
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    cleaned = filename.strip()
    if not cleaned:
        raise ValueError("The uploaded filename is empty.")
    extension = Path(cleaned).suffix.casefold()
    if extension != SUPPORTED_EXTENSION:
        raise ValueError(
            f"Unsupported document type '{extension or 'unknown'}'. "
            "This loader supports PDF files only."
        )
    return cleaned


def _read_bytes(uploaded_file: BinaryIO) -> bytes:
    if uploaded_file is None:
        raise ValueError("No PDF file was provided.")
    try:
        uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)
    except (AttributeError, OSError) as error:
        raise ValueError("The PDF stream is not readable or seekable.") from error
    if not isinstance(raw_bytes, bytes):
        raise ValueError("The PDF upload did not return binary content.")
    if not raw_bytes:
        raise ValueError("The uploaded PDF is empty.")
    return raw_bytes


def _extract_text(
    raw_bytes: bytes,
) -> tuple[list[PdfTextBlock], list[int], dict[str, Any], int]:
    """Extract ordered text blocks and bounding boxes with PyMuPDF."""

    try:
        document = pymupdf.open(stream=raw_bytes, filetype="pdf")
    except Exception as error:
        raise ValueError(
            f"Unable to open the PDF. Technical details: {error}"
        ) from error

    blocks: list[PdfTextBlock] = []
    page_text_counts: list[int] = []

    try:
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            raw_blocks = page.get_text("blocks", sort=True)
            page_character_count = 0

            for block_index, block in enumerate(raw_blocks):
                x0, y0, x1, y1, text = block[:5]
                cleaned = _clean_text(text)
                if not cleaned:
                    continue

                page_character_count += len(cleaned)
                blocks.append(
                    PdfTextBlock(
                        block_id=f"page-{page_number}-block-{block_index + 1}",
                        page_number=page_number,
                        block_index=block_index,
                        text=cleaned,
                        block_type=_classify_text_block(cleaned),
                        bbox=[
                            round(float(x0), 2),
                            round(float(y0), 2),
                            round(float(x1), 2),
                            round(float(y1), 2),
                        ],
                        source_reference=(
                            f"pdf://page/{page_number}/block/{block_index + 1}"
                        ),
                    )
                )

            page_text_counts.append(page_character_count)

        metadata = dict(document.metadata or {})
        page_count = int(document.page_count)
    finally:
        document.close()

    return blocks, page_text_counts, metadata, page_count


def _extract_tables(raw_bytes: bytes) -> list[PdfTableData]:
    """Extract tables with pdfplumber using line and text strategies."""

    extracted: list[PdfTableData] = []

    try:
        pdf = pdfplumber.open(io.BytesIO(raw_bytes))
    except Exception as error:
        raise ValueError(
            f"Unable to inspect PDF tables. Technical details: {error}"
        ) from error

    with pdf:
        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1
            page_tables = _find_page_tables(page)

            for page_table_index, table in enumerate(page_tables):
                raw_rows = table.extract()
                rows = _clean_table_rows(raw_rows)
                if len(rows) < 2:
                    continue

                dataframe, headers, warnings = _rows_to_dataframe(rows)
                if dataframe.empty and len(dataframe.columns) == 0:
                    continue

                overall_index = len(extracted)
                bbox = [round(float(value), 2) for value in table.bbox]
                extracted.append(
                    PdfTableData(
                        table_index=overall_index,
                        table_id=f"page-{page_number}-table-{page_table_index + 1}",
                        page_number=page_number,
                        page_table_index=page_table_index,
                        title=f"Page {page_number} - Table {page_table_index + 1}",
                        row_count=int(len(dataframe)),
                        column_count=int(len(dataframe.columns)),
                        headers=headers,
                        dataframe=dataframe,
                        source_reference=(
                            f"pdf://page/{page_number}/table/{page_table_index + 1}"
                        ),
                        bbox=bbox,
                        warnings=warnings,
                    )
                )

    return extracted


def _find_page_tables(page: Any) -> list[Any]:
    """Try ruled-table extraction first, then aligned-text extraction."""

    line_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "intersection_tolerance": 5,
    }
    text_settings = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "min_words_vertical": 3,
        "min_words_horizontal": 1,
        "intersection_tolerance": 5,
    }

    try:
        tables = page.find_tables(table_settings=line_settings)
    except Exception:
        tables = []

    if tables:
        return tables

    try:
        return page.find_tables(table_settings=text_settings)
    except Exception:
        return []


def _clean_table_rows(raw_rows: Any) -> list[list[str]]:
    """Clean PDF table cells while removing empty rows."""

    if not raw_rows:
        return []

    rows: list[list[str]] = []
    for raw_row in raw_rows:
        if raw_row is None:
            continue
        row = [_clean_text(cell or "") for cell in raw_row]
        if any(row):
            rows.append(row)
    return rows


def _rows_to_dataframe(
    rows: list[list[str]],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Convert extracted PDF rows to a typed DataFrame."""

    warnings: list[str] = []
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    raw_headers = normalized[0]
    data_rows = normalized[1:]
    headers, header_warnings = _unique_headers(raw_headers)
    warnings.extend(header_warnings)

    dataframe = pd.DataFrame(data_rows, columns=headers)
    dataframe = dataframe.replace(r"^\s*$", pd.NA, regex=True)
    dataframe = dataframe.dropna(how="all").dropna(axis=1, how="all")
    dataframe = dataframe.reset_index(drop=True)

    for column in dataframe.columns:
        dataframe[column] = _infer_series_type(dataframe[column])

    if any(header.startswith("column_") for header in headers):
        warnings.append(
            "One or more PDF table headers were empty or unreliable and "
            "were replaced with generated names."
        )

    return dataframe, list(dataframe.columns), warnings


def _unique_headers(raw_headers: list[str]) -> tuple[list[str], list[str]]:
    headers: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}

    for index, raw in enumerate(raw_headers):
        base = _normalize_header(raw)
        if not base:
            base = f"column_{index + 1}"
            warnings.append(
                f"Empty header {index + 1} was renamed to '{base}'."
            )
        count = counts.get(base, 0) + 1
        counts[base] = count
        header = base if count == 1 else f"{base}_{count}"
        if count > 1:
            warnings.append(
                f"Duplicate header '{base}' was renamed to '{header}'."
            )
        headers.append(header)

    return headers, warnings


def _normalize_header(value: str) -> str:
    normalized = _clean_text(value).casefold()
    normalized = re.sub(r"[^0-9a-z\u1000-\u109f]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def _infer_series_type(series: pd.Series) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        return series

    cleaned = non_null.astype(str).str.replace(",", "", regex=False)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        full = pd.to_numeric(
            series.astype("string").str.replace(",", "", regex=False),
            errors="coerce",
        )
        if full.dropna().map(lambda value: float(value).is_integer()).all():
            return full.astype("Int64")
        return full.astype("Float64")

    if _looks_like_date_series(non_null):
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().sum() >= max(1, int(len(non_null) * 0.95)):
            return parsed

    return series.astype("string").str.strip()


def _looks_like_date_series(series: pd.Series) -> bool:
    samples = series.astype(str).head(20)
    pattern = re.compile(
        r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
        r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$"
    )
    return samples.map(lambda value: bool(pattern.match(value.strip()))).mean() >= 0.8


def _classify_text_block(text: str) -> str:
    """Classify a text block conservatively for evidence display."""

    if len(text) <= 120 and not text.endswith((".", "á‹", ":", ";")):
        if text.isupper() or re.match(r"^(?:\d+[.)]|[á-á‰á€]+[á‹.)])", text):
            return "heading"
    if re.match(r"^(?:[-*â€¢]|\d+[.)])\s+", text):
        return "list_item"
    return "paragraph"


def _clean_text(value: str) -> str:
    text = str(value or "").replace("\xa0", " ")
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None

