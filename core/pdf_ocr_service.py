"""Validated scanned-PDF OCR service for InsightFlow AI.

The service uses OCRmyPDF to create a searchable PDF, validates the output
with core.pdf_loader, preserves the original upload, and removes all temporary
files automatically.
"""

from __future__ import annotations

import importlib.util
import io
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal

from core.pdf_loader import extract_pdf


OcrLanguage = Literal["eng", "mya", "eng+mya"]
SUPPORTED_LANGUAGES = {"eng", "mya", "eng+mya"}


@dataclass
class OcrDependencyStatus:
    """Availability of software and language packs required for OCR."""

    ocrmypdf_python: bool
    ocrmypdf_command: bool
    tesseract_command: bool
    ghostscript_command: bool
    tesseract_languages: list[str] = field(default_factory=list)
    missing_requirements: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dependency metadata."""

        return asdict(self)


@dataclass
class OcrResult:
    """Validated OCR output together with audit metadata."""

    success: bool
    filename: str
    output_filename: str
    language: str
    source_page_count: int
    source_is_searchable: bool
    source_needs_ocr: bool
    output_page_count: int
    output_is_searchable: bool
    output_needs_ocr: bool
    output_text_character_count: int
    output_pdf_bytes: bytes
    command: list[str]
    warnings: list[str]
    validation: dict[str, Any]

    def metadata(self) -> dict[str, Any]:
        """Return result metadata without embedding PDF bytes."""

        data = asdict(self)
        data.pop("output_pdf_bytes", None)
        return data


def check_ocr_dependencies(
    language: OcrLanguage = "eng+mya",
) -> OcrDependencyStatus:
    """Check OCRmyPDF, Tesseract, Ghostscript, and language packs."""

    _validate_language(language)

    ocrmypdf_python = importlib.util.find_spec("ocrmypdf") is not None
    ocrmypdf_command = shutil.which("ocrmypdf") is not None
    tesseract_command = shutil.which("tesseract") is not None
    ghostscript_command = any(
        shutil.which(command) is not None
        for command in (
            "gs",
            "gswin64c",
            "gswin32c",
        )
    )

    tesseract_languages = (
        _get_tesseract_languages()
        if tesseract_command
        else []
    )

    missing_requirements: list[str] = []

    if not (ocrmypdf_python or ocrmypdf_command):
        missing_requirements.append("ocrmypdf")

    if not tesseract_command:
        missing_requirements.append("tesseract")

    if not ghostscript_command:
        missing_requirements.append("ghostscript")

    for required_language in language.split("+"):
        if (
            tesseract_command
            and required_language not in tesseract_languages
        ):
            missing_requirements.append(
                f"tesseract-language-{required_language}"
            )

    return OcrDependencyStatus(
        ocrmypdf_python=ocrmypdf_python,
        ocrmypdf_command=ocrmypdf_command,
        tesseract_command=tesseract_command,
        ghostscript_command=ghostscript_command,
        tesseract_languages=tesseract_languages,
        missing_requirements=missing_requirements,
        ready=not missing_requirements,
    )


def ocr_pdf(
    uploaded_file: BinaryIO,
    filename: str,
    language: OcrLanguage = "eng+mya",
    force: bool = False,
    deskew: bool = True,
    rotate_pages: bool = True,
    timeout_seconds: int = 600,
) -> OcrResult:
    """Convert a scanned PDF into a validated searchable PDF.

    A searchable source is returned unchanged unless ``force=True``. A scanned
    source is written to a temporary directory, processed with OCRmyPDF, read
    back into memory, validated, and removed from temporary storage.
    """

    cleaned_filename = _validate_filename(filename)
    _validate_language(language)
    _validate_timeout(timeout_seconds)
    source_bytes = _read_bytes(uploaded_file)

    source = extract_pdf(
        io.BytesIO(source_bytes),
        cleaned_filename,
        extract_tables=False,
    )

    warnings: list[str] = []
    output_filename = _output_filename(cleaned_filename)

    if source.metadata.is_searchable and not force:
        warnings.append(
            "The source PDF is already searchable; OCR was skipped."
        )

        validation = {
            "status": "passed",
            "ocr_executed": False,
            "reason": "source_already_searchable",
            "source_page_count": source.metadata.page_count,
            "output_page_count": source.metadata.page_count,
            "page_count_preserved": True,
            "source_text_character_count": (
                source.metadata.text_character_count
            ),
            "output_text_character_count": (
                source.metadata.text_character_count
            ),
            "output_is_searchable": source.metadata.is_searchable,
            "output_needs_ocr": source.metadata.needs_ocr,
            "language": language,
        }

        return OcrResult(
            success=True,
            filename=cleaned_filename,
            output_filename=output_filename,
            language=language,
            source_page_count=source.metadata.page_count,
            source_is_searchable=source.metadata.is_searchable,
            source_needs_ocr=source.metadata.needs_ocr,
            output_page_count=source.metadata.page_count,
            output_is_searchable=source.metadata.is_searchable,
            output_needs_ocr=source.metadata.needs_ocr,
            output_text_character_count=(
                source.metadata.text_character_count
            ),
            output_pdf_bytes=source_bytes,
            command=[],
            warnings=warnings,
            validation=validation,
        )

    dependencies = check_ocr_dependencies(language)

    if not dependencies.ready:
        raise RuntimeError(
            "OCR dependencies are incomplete. Missing: "
            + ", ".join(dependencies.missing_requirements)
            + ". Install OCRmyPDF, Tesseract, Ghostscript, and the "
            "requested Tesseract language packs."
        )

    with tempfile.TemporaryDirectory(
        prefix="insightflow_ocr_"
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        input_path = temporary_path / "input.pdf"
        output_path = temporary_path / "output_searchable.pdf"

        input_path.write_bytes(source_bytes)

        command = _build_ocr_command(
            input_path=input_path,
            output_path=output_path,
            language=language,
            force=force,
            deskew=deskew,
            rotate_pages=rotate_pages,
        )

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "OCRmyPDF exceeded the configured timeout of "
                f"{timeout_seconds} seconds."
            ) from error
        except OSError as error:
            raise RuntimeError(
                f"OCRmyPDF could not be started: {error}"
            ) from error

        if completed.returncode != 0:
            details = (
                completed.stderr
                or completed.stdout
                or "No diagnostic output was returned."
            ).strip()

            raise RuntimeError(
                "OCRmyPDF failed with exit code "
                f"{completed.returncode}. Details: {details[-4000:]}"
            )

        if not output_path.exists():
            raise RuntimeError(
                "OCRmyPDF completed without creating an output PDF."
            )

        if output_path.stat().st_size == 0:
            raise RuntimeError(
                "OCRmyPDF created an empty output PDF."
            )

        output_bytes = output_path.read_bytes()

    if not output_bytes.startswith(b"%PDF"):
        raise RuntimeError(
            "OCRmyPDF output does not contain a valid PDF header."
        )

    output = extract_pdf(
        io.BytesIO(output_bytes),
        output_filename,
        extract_tables=False,
    )

    page_count_preserved = (
        source.metadata.page_count
        == output.metadata.page_count
    )
    output_has_text = (
        output.metadata.is_searchable
        and output.metadata.text_character_count > 0
    )

    validation_status = (
        "passed"
        if page_count_preserved and output_has_text
        else "failed"
    )

    if not page_count_preserved:
        warnings.append(
            "OCR output page count differs from the source PDF."
        )

    if not output_has_text:
        warnings.append(
            "OCR output still contains insufficient searchable text."
        )

    validation = {
        "status": validation_status,
        "ocr_executed": True,
        "source_page_count": source.metadata.page_count,
        "output_page_count": output.metadata.page_count,
        "page_count_preserved": page_count_preserved,
        "source_text_character_count": (
            source.metadata.text_character_count
        ),
        "output_text_character_count": (
            output.metadata.text_character_count
        ),
        "output_is_searchable": output.metadata.is_searchable,
        "output_needs_ocr": output.metadata.needs_ocr,
        "language": language,
    }

    return OcrResult(
        success=validation_status == "passed",
        filename=cleaned_filename,
        output_filename=output_filename,
        language=language,
        source_page_count=source.metadata.page_count,
        source_is_searchable=source.metadata.is_searchable,
        source_needs_ocr=source.metadata.needs_ocr,
        output_page_count=output.metadata.page_count,
        output_is_searchable=output.metadata.is_searchable,
        output_needs_ocr=output.metadata.needs_ocr,
        output_text_character_count=(
            output.metadata.text_character_count
        ),
        output_pdf_bytes=output_bytes,
        command=_redact_command_paths(command),
        warnings=warnings,
        validation=validation,
    )


def _build_ocr_command(
    input_path: Path,
    output_path: Path,
    language: OcrLanguage,
    force: bool,
    deskew: bool,
    rotate_pages: bool,
) -> list[str]:
    """Build an OCRmyPDF 17.x-compatible command argument list."""

    if shutil.which("ocrmypdf"):
        command = ["ocrmypdf"]
    else:
        command = [
            sys.executable,
            "-m",
            "ocrmypdf",
        ]

    command.extend(
        [
            "--language",
            language,
            "--optimize",
            "1",
            "--output-type",
            "pdf",
        ]
    )

    if force:
        command.append("--force-ocr")
    else:
        command.append("--skip-text")

    if deskew:
        command.append("--deskew")

    if rotate_pages:
        command.append("--rotate-pages")

    command.extend(
        [
            str(input_path),
            str(output_path),
        ]
    )

    return command


def _get_tesseract_languages() -> list[str]:
    """Return installed Tesseract language codes."""

    try:
        completed = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (
        OSError,
        subprocess.SubprocessError,
    ):
        return []

    if completed.returncode != 0:
        return []

    lines = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]

    languages = [
        line
        for line in lines
        if not line.casefold().startswith(
            "list of available"
        )
    ]

    return sorted(set(languages))


def _redact_command_paths(
    command: list[str],
) -> list[str]:
    """Preserve OCR settings while removing temporary file paths."""

    redacted: list[str] = []

    for item in command:
        normalized = item.replace("\\", "/")

        if normalized.endswith("/input.pdf"):
            redacted.append("<input.pdf>")
        elif normalized.endswith(
            "/output_searchable.pdf"
        ):
            redacted.append(
                "<output_searchable.pdf>"
            )
        else:
            redacted.append(item)

    return redacted


def _validate_filename(filename: str) -> str:
    """Validate and normalize a PDF filename."""

    if not isinstance(filename, str):
        raise TypeError("filename must be a string")

    cleaned = filename.strip()

    if not cleaned:
        raise ValueError("The uploaded filename is empty.")

    if Path(cleaned).suffix.casefold() != ".pdf":
        raise ValueError(
            "OCR is supported only for PDF files."
        )

    return cleaned


def _validate_language(language: str) -> None:
    """Validate the requested OCR language combination."""

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            "language must be 'eng', 'mya', or 'eng+mya'"
        )


def _validate_timeout(timeout_seconds: int) -> None:
    """Validate the OCR subprocess timeout."""

    if not isinstance(timeout_seconds, int):
        raise TypeError(
            "timeout_seconds must be an integer"
        )

    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds must be greater than zero"
        )


def _read_bytes(uploaded_file: BinaryIO) -> bytes:
    """Read PDF bytes while preserving a reusable seekable stream."""

    if uploaded_file is None:
        raise ValueError("No PDF file was provided.")

    try:
        uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)
    except (AttributeError, OSError) as error:
        raise ValueError(
            "The PDF stream is not readable or seekable."
        ) from error

    if not isinstance(raw_bytes, bytes):
        raise ValueError(
            "The PDF upload did not return binary content."
        )

    if not raw_bytes:
        raise ValueError("The uploaded PDF is empty.")

    return raw_bytes


def _output_filename(filename: str) -> str:
    """Create a non-destructive searchable-PDF filename."""

    return f"{Path(filename).stem}_searchable.pdf"
