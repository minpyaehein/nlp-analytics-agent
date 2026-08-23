"""Unified tabular upload pipeline for CSV, XLSX, and PDF files.

PDF workflow:
- Detect whether the PDF already contains searchable text.
- Run bilingual OCR only when needed.
- Extract and validate the highest-confidence table.
- Report revenue and profit readiness without inventing missing values.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd
import pymupdf

from core.analytics_quality_gate import (
    validate_profit_readiness,
    validate_revenue_readiness,
)
from core.file_loader import load_uploaded_file
from core.pdf_ocr_service import check_ocr_dependencies, ocr_pdf
from core.pdf_table_extractor import extract_pdf_tables


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".pdf"}


@dataclass
class UnifiedUploadResult:
    """Normalized upload result consumed by the Streamlit interface."""

    success: bool
    filename: str
    source_type: str
    dataframe: pd.DataFrame
    processing_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    extraction_metadata: dict[str, Any] = field(default_factory=dict)
    revenue_readiness: dict[str, Any] = field(default_factory=dict)
    profit_readiness: dict[str, Any] = field(default_factory=dict)
    searchable_pdf_bytes: bytes | None = None

    def metadata(self) -> dict[str, Any]:
        """Return JSON-friendly metadata without embedding binary data."""

        return {
            "success": self.success,
            "filename": self.filename,
            "source_type": self.source_type,
            "row_count": int(len(self.dataframe)),
            "column_count": int(len(self.dataframe.columns)),
            "columns": [str(column) for column in self.dataframe.columns],
            "processing_steps": list(self.processing_steps),
            "warnings": list(self.warnings),
            "source_metadata": dict(self.source_metadata),
            "extraction_metadata": dict(self.extraction_metadata),
            "revenue_readiness": dict(self.revenue_readiness),
            "profit_readiness": dict(self.profit_readiness),
            "has_searchable_pdf": self.searchable_pdf_bytes is not None,
            "searchable_pdf_size": (
                len(self.searchable_pdf_bytes)
                if self.searchable_pdf_bytes is not None
                else 0
            ),
        }


def _read_upload_bytes(uploaded_file: BinaryIO | bytes | bytearray) -> bytes:
    """Read a binary upload safely and rewind seekable streams."""

    if isinstance(uploaded_file, (bytes, bytearray)):
        raw_bytes = bytes(uploaded_file)
    else:
        try:
            uploaded_file.seek(0)
            raw_bytes = uploaded_file.read()
            uploaded_file.seek(0)
        except (AttributeError, OSError) as error:
            raise ValueError("The uploaded file stream is not readable.") from error

    if not raw_bytes:
        raise ValueError("The uploaded file is empty.")

    return raw_bytes


def inspect_pdf_searchability(raw_bytes: bytes) -> dict[str, Any]:
    """Inspect PDF text coverage without modifying the document."""

    try:
        document = pymupdf.open(stream=raw_bytes, filetype="pdf")
    except Exception as error:
        raise ValueError(f"Unable to open the PDF: {error}") from error

    try:
        page_character_counts = [
            len(page.get_text().strip())
            for page in document
        ]
        page_count = int(document.page_count)
    finally:
        document.close()

    text_character_count = int(sum(page_character_counts))
    pages_with_text = int(
        sum(count >= 20 for count in page_character_counts)
    )
    searchable = (
        page_count > 0
        and pages_with_text == page_count
        and text_character_count >= 20
    )

    return {
        "page_count": page_count,
        "text_character_count": text_character_count,
        "page_character_counts": page_character_counts,
        "pages_with_text": pages_with_text,
        "is_searchable": searchable,
        "needs_ocr": not searchable,
    }


def _readiness_metadata(dataframe: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate but do not enforce deterministic metric readiness."""

    revenue = validate_revenue_readiness(dataframe).to_dict()
    profit = validate_profit_readiness(dataframe).to_dict()
    return revenue, profit


def _load_tabular_upload(
    uploaded_file: BinaryIO | bytes | bytearray,
    filename: str,
    extension: str,
) -> UnifiedUploadResult:
    """Load CSV or XLSX through the existing safe file loader."""

    raw_bytes = _read_upload_bytes(uploaded_file)
    dataframe = load_uploaded_file(
        uploaded_file=io.BytesIO(raw_bytes),
        filename=filename,
    )
    revenue, profit = _readiness_metadata(dataframe)

    return UnifiedUploadResult(
        success=True,
        filename=filename,
        source_type=extension.lstrip("."),
        dataframe=dataframe,
        processing_steps=[
            f"Loaded {extension.upper().lstrip('.')} as a Pandas DataFrame.",
            "Evaluated deterministic revenue and profit readiness.",
        ],
        revenue_readiness=revenue,
        profit_readiness=profit,
    )


def _load_pdf_upload(
    uploaded_file: BinaryIO | bytes | bytearray,
    filename: str,
    language: str,
    timeout_seconds: int,
) -> UnifiedUploadResult:
    """OCR when required, extract the best PDF table, and assess readiness."""

    raw_bytes = _read_upload_bytes(uploaded_file)
    source_metadata = inspect_pdf_searchability(raw_bytes)
    processing_steps = ["Inspected the PDF text layer and page count."]
    warnings: list[str] = []
    searchable_pdf_bytes = raw_bytes
    ocr_metadata: dict[str, Any] = {}

    if source_metadata["needs_ocr"]:
        dependency_status = check_ocr_dependencies(language)

        if not dependency_status.ready:
            raise RuntimeError(
                "The PDF requires OCR, but OCR dependencies are incomplete: "
                f"{dependency_status.missing_requirements}"
            )

        ocr_result = ocr_pdf(
            uploaded_file=io.BytesIO(raw_bytes),
            filename=filename,
            language=language,
            force=False,
            deskew=True,
            rotate_pages=True,
            timeout_seconds=timeout_seconds,
        )
        ocr_metadata = ocr_result.metadata()

        if not ocr_result.success or not ocr_result.output_pdf_bytes:
            raise RuntimeError(
                "OCR did not produce a validated searchable PDF. "
                f"Validation: {ocr_result.validation}"
            )

        searchable_pdf_bytes = ocr_result.output_pdf_bytes
        processing_steps.append(
            f"Applied OCRmyPDF using language pack '{language}'."
        )
    else:
        processing_steps.append(
            "Skipped OCR because every PDF page already contained searchable text."
        )

    extraction = extract_pdf_tables(
        searchable_pdf_bytes,
        filename=filename,
    )

    if not extraction.success:
        raise ValueError(
            "No usable table could be extracted from the PDF. "
            f"Warnings: {extraction.warnings}"
        )

    best_table = extraction.best_table()
    dataframe = best_table.dataframe.copy()
    warnings.extend(extraction.warnings)
    warnings.extend(best_table.warnings)
    processing_steps.extend(
        [
            f"Extracted the best PDF table using '{best_table.strategy}'.",
            "Normalized table headers and converted numeric/date fields.",
            "Evaluated deterministic revenue and profit readiness.",
        ]
    )

    revenue, profit = _readiness_metadata(dataframe)

    if not revenue.get("ready", False):
        warnings.append(
            "The extracted table is not ready for revenue calculation. "
            "Review the revenue quality-gate evidence."
        )

    if not profit.get("ready", False):
        warnings.append(
            "The extracted table is not ready for profit calculation. "
            "Review the profit quality-gate evidence."
        )

    combined_source_metadata = dict(source_metadata)
    combined_source_metadata["ocr"] = ocr_metadata

    return UnifiedUploadResult(
        success=True,
        filename=filename,
        source_type="pdf",
        dataframe=dataframe,
        processing_steps=processing_steps,
        warnings=warnings,
        source_metadata=combined_source_metadata,
        extraction_metadata=best_table.metadata(),
        revenue_readiness=revenue,
        profit_readiness=profit,
        searchable_pdf_bytes=searchable_pdf_bytes,
    )


def process_uploaded_file(
    uploaded_file: BinaryIO | bytes | bytearray,
    filename: str,
    *,
    ocr_language: str = "eng+mya",
    timeout_seconds: int = 600,
) -> UnifiedUploadResult:
    """Process CSV, XLSX, or PDF into a validated tabular upload result."""

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("A valid filename is required.")

    cleaned_filename = filename.strip()
    extension = Path(cleaned_filename).suffix.casefold()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{extension or 'unknown'}'. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if extension in {".csv", ".xlsx"}:
        return _load_tabular_upload(
            uploaded_file,
            cleaned_filename,
            extension,
        )

    return _load_pdf_upload(
        uploaded_file,
        cleaned_filename,
        ocr_language,
        timeout_seconds,
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Process CSV, XLSX, or PDF into a validated DataFrame."
    )
    parser.add_argument("input_path")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--ocr-language", default="eng+mya")
    arguments = parser.parse_args()

    input_path = Path(arguments.input_path)

    with input_path.open("rb") as binary_file:
        result = process_uploaded_file(
            binary_file,
            input_path.name,
            ocr_language=arguments.ocr_language,
        )

    print(json.dumps(result.metadata(), indent=2, ensure_ascii=False, default=str))
    print()
    print(result.dataframe.to_string(index=False))

    if arguments.csv:
        result.dataframe.to_csv(
            arguments.csv,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Saved CSV: {arguments.csv}")
