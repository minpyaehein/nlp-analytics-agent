"""End-to-end validation for the InsightFlow AI PDF OCR workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The downloaded test file may initially be placed outside the project.
# If this file is located at <project>/scripts/test_pdf_ocr.py, parents[1]
# correctly resolves to the project root. If it is run from another location,
# the current working directory is used when it contains the core package.
if not (PROJECT_ROOT / "core").exists():
    current_directory = Path.cwd().resolve()

    if (current_directory / "core").exists():
        PROJECT_ROOT = current_directory

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.pdf_loader import extract_pdf
from core.pdf_ocr_service import (
    check_ocr_dependencies,
    ocr_pdf,
)


INPUT_PATH = (
    PROJECT_ROOT
    / "sample_data"
    / "scanned_sales_report.pdf"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "sample_data"
    / "scanned_sales_report_searchable.pdf"
)

OCR_LANGUAGE = "eng+mya"
OCR_TIMEOUT_SECONDS = 600


def print_json(
    heading: str,
    value: dict[str, Any],
) -> None:
    """Print a labeled dictionary as readable JSON."""

    print()
    print("=" * 80)
    print(heading)
    print("=" * 80)
    print(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


def validate_input_file() -> None:
    """Ensure the scanned input PDF exists and is not empty."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Scanned PDF does not exist: {INPUT_PATH}"
        )

    if not INPUT_PATH.is_file():
        raise ValueError(
            f"OCR input is not a regular file: {INPUT_PATH}"
        )

    if INPUT_PATH.stat().st_size == 0:
        raise ValueError(
            f"Scanned PDF is empty: {INPUT_PATH}"
        )


def validate_dependencies() -> None:
    """Validate OCR software and English/Myanmar language packs."""

    status = check_ocr_dependencies(OCR_LANGUAGE)

    print_json(
        "OCR Dependency Status",
        status.to_dict(),
    )

    if not status.ready:
        missing = ", ".join(
            status.missing_requirements
        )

        raise RuntimeError(
            "OCR dependencies are incomplete. "
            f"Missing: {missing}"
        )

    print(
        "OCRmyPDF, Tesseract, Ghostscript, English, "
        "and Myanmar OCR are ready."
    )


def inspect_source_pdf() -> dict[str, Any]:
    """Inspect the source and confirm whether OCR is required."""

    with INPUT_PATH.open("rb") as source_file:
        extracted = extract_pdf(
            uploaded_file=source_file,
            filename=INPUT_PATH.name,
            extract_tables=False,
        )

    metadata = extracted.metadata.to_dict()

    print_json(
        "Source PDF Metadata",
        metadata,
    )

    if extracted.metadata.is_searchable:
        print(
            "Warning: The source PDF already contains searchable text. "
            "The OCR service will skip processing unless force=True."
        )
    elif extracted.metadata.needs_ocr:
        print(
            "The source PDF is image-only or contains insufficient "
            "searchable text. OCR is required."
        )
    else:
        print(
            "The source PDF is not clearly searchable, but the loader "
            "did not mark OCR as required."
        )

    return metadata


def run_ocr() -> dict[str, Any]:
    """Run bilingual OCR and save the validated searchable PDF."""

    with INPUT_PATH.open("rb") as source_file:
        result = ocr_pdf(
            uploaded_file=source_file,
            filename=INPUT_PATH.name,
            language=OCR_LANGUAGE,
            force=False,
            deskew=True,
            rotate_pages=True,
            timeout_seconds=OCR_TIMEOUT_SECONDS,
        )

    metadata = result.metadata()

    print_json(
        "OCR Result Metadata",
        metadata,
    )

    if not result.success:
        raise RuntimeError(
            "OCR validation failed: "
            + json.dumps(
                result.validation,
                ensure_ascii=False,
                default=str,
            )
        )

    if not result.output_pdf_bytes:
        raise RuntimeError(
            "OCR service returned no output PDF bytes."
        )

    if not result.output_pdf_bytes.startswith(b"%PDF"):
        raise RuntimeError(
            "OCR output does not begin with a valid PDF header."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_bytes(
        result.output_pdf_bytes
    )

    print()
    print("OCR completed successfully.")
    print(f"Saved: {OUTPUT_PATH}")
    print(
        "Output size: "
        f"{OUTPUT_PATH.stat().st_size:,} bytes"
    )

    return metadata


def inspect_output_pdf() -> dict[str, Any]:
    """Verify the output PDF contains a valid searchable text layer."""

    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"OCR output does not exist: {OUTPUT_PATH}"
        )

    with OUTPUT_PATH.open("rb") as output_file:
        extracted = extract_pdf(
            uploaded_file=output_file,
            filename=OUTPUT_PATH.name,
            extract_tables=False,
        )

    metadata = extracted.metadata.to_dict()

    print_json(
        "OCR Output PDF Metadata",
        metadata,
    )

    if not extracted.metadata.is_searchable:
        raise RuntimeError(
            "OCR output is not searchable."
        )

    if extracted.metadata.needs_ocr:
        raise RuntimeError(
            "OCR output is still marked as requiring OCR."
        )

    if extracted.metadata.text_character_count <= 0:
        raise RuntimeError(
            "OCR output contains no extractable text."
        )

    extracted_text = extracted.combined_text()

    print()
    print("Extracted text preview")
    print("-" * 80)
    print(extracted_text[:3000])
    print("-" * 80)

    return metadata


def test_searchable_pdf_skip() -> dict[str, Any]:
    """Confirm that a searchable PDF is not OCRed a second time."""

    with OUTPUT_PATH.open("rb") as output_file:
        result = ocr_pdf(
            uploaded_file=output_file,
            filename=OUTPUT_PATH.name,
            language=OCR_LANGUAGE,
            force=False,
        )

    validation = result.validation

    print_json(
        "Searchable PDF Skip Validation",
        validation,
    )

    if validation.get("ocr_executed") is not False:
        raise RuntimeError(
            "The searchable PDF was unexpectedly OCRed again."
        )

    if (
        validation.get("reason")
        != "source_already_searchable"
    ):
        raise RuntimeError(
            "The searchable-PDF skip reason is incorrect."
        )

    if validation.get("status") != "passed":
        raise RuntimeError(
            "Searchable-PDF skip validation did not pass."
        )

    print("Searchable PDF skip behavior passed.")

    return validation


def main() -> None:
    """Run the complete scanned-PDF OCR validation workflow."""

    print("=" * 80)
    print("InsightFlow AI PDF OCR Integration Test")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input PDF: {INPUT_PATH}")
    print(f"Output PDF: {OUTPUT_PATH}")
    print(f"OCR language: {OCR_LANGUAGE}")
    print(f"OCR timeout: {OCR_TIMEOUT_SECONDS} seconds")

    validate_input_file()
    validate_dependencies()
    source_metadata = inspect_source_pdf()
    ocr_metadata = run_ocr()
    output_metadata = inspect_output_pdf()
    skip_validation = test_searchable_pdf_skip()

    summary = {
        "input_pdf": str(INPUT_PATH),
        "output_pdf": str(OUTPUT_PATH),
        "ocr_language": OCR_LANGUAGE,
        "source_is_searchable": source_metadata.get(
            "is_searchable"
        ),
        "source_needs_ocr": source_metadata.get(
            "needs_ocr"
        ),
        "ocr_success": ocr_metadata.get("success"),
        "output_is_searchable": output_metadata.get(
            "is_searchable"
        ),
        "output_needs_ocr": output_metadata.get(
            "needs_ocr"
        ),
        "output_text_character_count": output_metadata.get(
            "text_character_count"
        ),
        "searchable_skip_passed": (
            skip_validation.get("status") == "passed"
        ),
    }

    print_json(
        "Final OCR Test Summary",
        summary,
    )

    print()
    print("=" * 80)
    print("ALL PDF OCR TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
 