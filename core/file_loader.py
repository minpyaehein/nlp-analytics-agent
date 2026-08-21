"""Validated tabular file loading for InsightFlow AI.

Supported formats:
- CSV
- TSV
- TXT containing delimited tabular data
- JSON arrays, record dictionaries, and nested record containers
- XLSX workbooks, including explicit worksheet selection
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
    ".json",
    ".xlsx",
}

TEXT_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "latin-1",
)

DELIMITER_CANDIDATES = (",", "\t", ";", "|")

COMMON_RECORD_KEYS = (
    "data",
    "records",
    "rows",
    "items",
    "results",
    "values",
)


@dataclass
class FileLoadMetadata:
    """Audit metadata describing how a file was loaded."""

    filename: str
    extension: str
    format: str
    row_count: int
    column_count: int
    encoding: str | None = None
    delimiter: str | None = None
    selected_sheet: str | None = None
    available_sheets: list[str] | None = None
    json_record_path: str | None = None
    warnings: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly metadata."""

        return asdict(self)


@dataclass
class LoadedDataset:
    """A loaded DataFrame and its file-loading metadata."""

    dataframe: pd.DataFrame
    metadata: FileLoadMetadata


def load_uploaded_file(
    uploaded_file: BinaryIO,
    filename: str,
    sheet_name: str | int | None = None,
) -> pd.DataFrame:
    """Load a supported uploaded file and return only its DataFrame.

    This function preserves compatibility with the current Streamlit app.
    Use ``load_uploaded_file_with_metadata`` when audit metadata is needed.
    """

    loaded = load_uploaded_file_with_metadata(
        uploaded_file=uploaded_file,
        filename=filename,
        sheet_name=sheet_name,
    )
    return loaded.dataframe


def load_uploaded_file_with_metadata(
    uploaded_file: BinaryIO,
    filename: str,
    sheet_name: str | int | None = None,
) -> LoadedDataset:
    """Load CSV, TSV, TXT, JSON, or XLSX data with audit metadata."""

    cleaned_filename, extension = _validate_filename(filename)
    raw_bytes = _read_uploaded_bytes(uploaded_file)

    if extension == ".csv":
        dataframe, encoding, delimiter = _load_delimited_text(
            raw_bytes=raw_bytes,
            preferred_delimiter=",",
            require_multiple_columns=False,
        )
        metadata = FileLoadMetadata(
            filename=cleaned_filename,
            extension=extension,
            format="csv",
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            encoding=encoding,
            delimiter=_display_delimiter(delimiter),
            warnings=[],
        )

    elif extension == ".tsv":
        dataframe, encoding, delimiter = _load_delimited_text(
            raw_bytes=raw_bytes,
            preferred_delimiter="\t",
            require_multiple_columns=False,
        )
        metadata = FileLoadMetadata(
            filename=cleaned_filename,
            extension=extension,
            format="tsv",
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            encoding=encoding,
            delimiter=_display_delimiter(delimiter),
            warnings=[],
        )

    elif extension == ".txt":
        dataframe, encoding, delimiter = _load_delimited_text(
            raw_bytes=raw_bytes,
            preferred_delimiter=None,
            require_multiple_columns=True,
        )
        metadata = FileLoadMetadata(
            filename=cleaned_filename,
            extension=extension,
            format="delimited_text",
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            encoding=encoding,
            delimiter=_display_delimiter(delimiter),
            warnings=[],
        )

    elif extension == ".json":
        dataframe, encoding, record_path, warnings = _load_json(raw_bytes)
        metadata = FileLoadMetadata(
            filename=cleaned_filename,
            extension=extension,
            format="json",
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            encoding=encoding,
            json_record_path=record_path,
            warnings=warnings,
        )

    elif extension == ".xlsx":
        dataframe, selected_sheet, sheets, warnings = _load_xlsx(
            raw_bytes=raw_bytes,
            sheet_name=sheet_name,
        )
        metadata = FileLoadMetadata(
            filename=cleaned_filename,
            extension=extension,
            format="xlsx",
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            selected_sheet=selected_sheet,
            available_sheets=sheets,
            warnings=warnings,
        )

    else:
        raise ValueError(f"Unsupported file extension: {extension}")

    dataframe = _clean_dataframe(dataframe)
    metadata.row_count = int(len(dataframe))
    metadata.column_count = int(len(dataframe.columns))

    return LoadedDataset(
        dataframe=dataframe,
        metadata=metadata,
    )


def list_excel_sheets(
    uploaded_file: BinaryIO,
    filename: str,
) -> list[str]:
    """Return worksheet names without selecting or loading one."""

    _, extension = _validate_filename(filename)

    if extension != ".xlsx":
        raise ValueError("Worksheet discovery is available only for XLSX files.")

    raw_bytes = _read_uploaded_bytes(uploaded_file)

    try:
        workbook = pd.ExcelFile(
            io.BytesIO(raw_bytes),
            engine="openpyxl",
        )
    except Exception as error:
        raise ValueError(
            "Unable to open the Excel workbook. "
            f"Technical details: {error}"
        ) from error

    return list(workbook.sheet_names)


def _validate_filename(filename: str) -> tuple[str, str]:
    """Validate a filename and return the clean name and extension."""

    if not isinstance(filename, str):
        raise TypeError("filename must be a string")

    cleaned_filename = filename.strip()

    if not cleaned_filename:
        raise ValueError("The uploaded filename is empty.")

    extension = Path(cleaned_filename).suffix.casefold()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{extension or 'unknown'}'. "
            f"Supported types: {supported}."
        )

    return cleaned_filename, extension


def _read_uploaded_bytes(uploaded_file: BinaryIO) -> bytes:
    """Read all bytes from a seekable uploaded file."""

    if uploaded_file is None:
        raise ValueError("No uploaded file was provided.")

    try:
        uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)
    except (AttributeError, OSError) as error:
        raise ValueError(
            "The uploaded file stream is not readable or seekable."
        ) from error

    if isinstance(raw_bytes, str):
        raw_bytes = raw_bytes.encode("utf-8")

    if not isinstance(raw_bytes, bytes):
        raise ValueError("The uploaded file did not return binary content.")

    if not raw_bytes:
        raise ValueError("The uploaded file is empty.")

    return raw_bytes


def _decode_text(raw_bytes: bytes) -> tuple[str, str]:
    """Decode uploaded text with common encoding fallbacks."""

    failures: list[str] = []

    for encoding in TEXT_ENCODINGS:
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError as error:
            failures.append(f"{encoding}: {error}")

    raise ValueError(
        "The text encoding could not be detected. "
        + " | ".join(failures)
    )


def _detect_delimiter(
    text: str,
    preferred_delimiter: str | None,
) -> str:
    """Detect a delimiter while respecting a format-specific preference."""

    sample = text[:16384]

    if preferred_delimiter and preferred_delimiter in sample:
        return preferred_delimiter

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters="".join(DELIMITER_CANDIDATES),
        )
        return dialect.delimiter
    except csv.Error:
        candidate_counts = {
            delimiter: sample.count(delimiter)
            for delimiter in DELIMITER_CANDIDATES
        }
        best = max(candidate_counts, key=candidate_counts.get)

        if candidate_counts[best] == 0:
            raise ValueError(
                "A tabular delimiter could not be detected. "
                "TXT files must contain comma, tab, semicolon, or pipe "
                "separated rows with a header."
            )

        return best


def _load_delimited_text(
    raw_bytes: bytes,
    preferred_delimiter: str | None,
    require_multiple_columns: bool,
) -> tuple[pd.DataFrame, str, str]:
    """Load CSV, TSV, or TXT-delimited data."""

    text, encoding = _decode_text(raw_bytes)

    if not text.strip():
        raise ValueError("The uploaded text file contains no readable text.")

    delimiter = _detect_delimiter(text, preferred_delimiter)

    try:
        dataframe = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
        )
    except pd.errors.EmptyDataError as error:
        raise ValueError("The file contains no readable tabular data.") from error
    except pd.errors.ParserError as error:
        raise ValueError(
            "The delimited text structure could not be parsed. "
            f"Detected delimiter: {_display_delimiter(delimiter)}. "
            f"Technical details: {error}"
        ) from error
    except Exception as error:
        raise ValueError(
            f"Unable to read the delimited text file: {error}"
        ) from error

    if require_multiple_columns and len(dataframe.columns) < 2:
        raise ValueError(
            "The TXT file was read as a single column. "
            "Provide comma, tab, semicolon, or pipe separated tabular data."
        )

    return dataframe, encoding, delimiter


def _load_json(
    raw_bytes: bytes,
) -> tuple[pd.DataFrame, str, str, list[str]]:
    """Load flat or nested JSON records into a DataFrame."""

    text, encoding = _decode_text(raw_bytes)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The JSON file is invalid. "
            f"Line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error

    records, record_path, warnings = _extract_json_records(payload)

    try:
        dataframe = pd.json_normalize(
            records,
            sep=".",
            max_level=None,
        )
    except Exception as error:
        raise ValueError(
            f"The JSON records could not be normalized: {error}"
        ) from error

    if dataframe.empty and len(dataframe.columns) == 0:
        raise ValueError("The JSON file contains no tabular records.")

    return dataframe, encoding, record_path, warnings


def _extract_json_records(
    payload: Any,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Find a list of records in common JSON structures."""

    if isinstance(payload, list):
        if not payload:
            raise ValueError("The JSON array is empty.")

        if all(isinstance(item, dict) for item in payload):
            return payload, "$", []

        return (
            [{"value": item} for item in payload],
            "$",
            [
                "The root JSON array contained scalar values; "
                "they were loaded into a 'value' column."
            ],
        )

    if not isinstance(payload, dict):
        return (
            [{"value": payload}],
            "$",
            [
                "The JSON root was a scalar value; it was loaded as one row."
            ],
        )

    for key in COMMON_RECORD_KEYS:
        value = payload.get(key)

        if isinstance(value, list) and value:
            if all(isinstance(item, dict) for item in value):
                return value, f"$.{key}", []

            return (
                [{"value": item} for item in value],
                f"$.{key}",
                [
                    f"The JSON list at '{key}' contained scalar values; "
                    "they were loaded into a 'value' column."
                ],
            )

    list_candidates = [
        (key, value)
        for key, value in payload.items()
        if isinstance(value, list) and value
    ]

    for key, value in list_candidates:
        if all(isinstance(item, dict) for item in value):
            return (
                value,
                f"$.{key}",
                [
                    f"The record list was inferred from the JSON key '{key}'."
                ],
            )

    return (
        [payload],
        "$",
        [
            "The JSON object contained no record array; "
            "the root object was loaded as one normalized row."
        ],
    )


def _load_xlsx(
    raw_bytes: bytes,
    sheet_name: str | int | None,
) -> tuple[pd.DataFrame, str, list[str], list[str]]:
    """Load an explicitly selected or first non-empty Excel worksheet."""

    try:
        workbook = pd.ExcelFile(
            io.BytesIO(raw_bytes),
            engine="openpyxl",
        )
    except Exception as error:
        raise ValueError(
            "Unable to open the Excel workbook. "
            f"Technical details: {error}"
        ) from error

    sheets = list(workbook.sheet_names)

    if not sheets:
        raise ValueError("The Excel workbook contains no worksheets.")

    warnings: list[str] = []

    if sheet_name is not None:
        selected_sheet = _resolve_sheet_name(sheet_name, sheets)
        dataframe = _read_excel_sheet(workbook, selected_sheet)

        if dataframe.empty and len(dataframe.columns) == 0:
            raise ValueError(
                f"Worksheet '{selected_sheet}' contains no tabular data."
            )

        return dataframe, selected_sheet, sheets, warnings

    empty_sheets: list[str] = []

    for candidate in sheets:
        dataframe = _read_excel_sheet(workbook, candidate)

        if not dataframe.empty or len(dataframe.columns) > 0:
            if candidate != sheets[0]:
                warnings.append(
                    f"The first non-empty worksheet '{candidate}' was selected."
                )
            elif len(sheets) > 1:
                warnings.append(
                    "The workbook contains multiple worksheets; "
                    f"'{candidate}' was selected automatically."
                )

            return dataframe, candidate, sheets, warnings

        empty_sheets.append(candidate)

    raise ValueError(
        "The Excel workbook contains no readable tabular data. "
        "Empty worksheets: "
        + ", ".join(empty_sheets)
    )


def _resolve_sheet_name(
    sheet_name: str | int,
    available_sheets: list[str],
) -> str:
    """Resolve a sheet name or zero-based index safely."""

    if isinstance(sheet_name, int):
        if sheet_name < 0 or sheet_name >= len(available_sheets):
            raise ValueError(
                f"Worksheet index {sheet_name} is outside the valid range."
            )
        return available_sheets[sheet_name]

    cleaned = str(sheet_name).strip()

    if cleaned not in available_sheets:
        raise ValueError(
            f"Worksheet '{cleaned}' was not found. "
            f"Available worksheets: {available_sheets}."
        )

    return cleaned


def _read_excel_sheet(
    workbook: pd.ExcelFile,
    sheet_name: str,
) -> pd.DataFrame:
    """Read and remove completely empty rows and columns from one sheet."""

    try:
        dataframe = pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            engine="openpyxl",
        )
    except Exception as error:
        raise ValueError(
            f"Unable to read worksheet '{sheet_name}': {error}"
        ) from error

    dataframe = dataframe.dropna(how="all")
    dataframe = dataframe.dropna(axis=1, how="all")
    return dataframe


def _clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply safe structural cleanup without changing analytical values."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("The loader did not produce a Pandas DataFrame.")

    dataframe = dataframe.copy()
    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    if len(dataframe.columns) == 0:
        raise ValueError("The uploaded file contains no columns.")

    if any(not column for column in dataframe.columns):
        dataframe.columns = [
            column or f"unnamed_column_{index + 1}"
            for index, column in enumerate(dataframe.columns)
        ]

    duplicate_columns = dataframe.columns[
        dataframe.columns.duplicated()
    ].tolist()

    if duplicate_columns:
        raise ValueError(
            "The uploaded file contains duplicate column names after "
            f"normalization: {duplicate_columns}."
        )

    return dataframe.reset_index(drop=True)


def _display_delimiter(delimiter: str) -> str:
    """Return a readable delimiter name for metadata and errors."""

    return {
        "\t": "tab",
        ",": "comma",
        ";": "semicolon",
        "|": "pipe",
    }.get(delimiter, repr(delimiter))
