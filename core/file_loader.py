"""Safe file-loading utilities for uploaded tabular datasets."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import BinaryIO

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}
CSV_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "latin-1",
)


def load_uploaded_file(
    uploaded_file: BinaryIO,
    filename: str,
) -> pd.DataFrame:
    """Load an uploaded CSV or XLSX file into a DataFrame.

    Args:
        uploaded_file: Streamlit UploadedFile or another seekable binary file.
        filename: Original filename, used to determine the file type.

    Returns:
        Loaded Pandas DataFrame.

    Raises:
        TypeError: If filename is not a string.
        ValueError: If the file is empty, unsupported, or cannot be parsed.
    """

    if not isinstance(filename, str):
        raise TypeError("filename must be a string")

    cleaned_filename = filename.strip()

    if not cleaned_filename:
        raise ValueError("The uploaded filename is empty.")

    extension = Path(cleaned_filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{extension or 'unknown'}'. "
            f"Supported types: {supported}."
        )

    _rewind(uploaded_file)
    raw_bytes = uploaded_file.read()

    if not raw_bytes:
        raise ValueError("The uploaded file is empty.")

    if extension == ".csv":
        dataframe = _load_csv(raw_bytes)
    else:
        dataframe = _load_xlsx(raw_bytes)

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    if len(dataframe.columns) == 0:
        raise ValueError("The uploaded file contains no columns.")

    return dataframe


def _rewind(uploaded_file: BinaryIO) -> None:
    """Move a seekable uploaded file back to its beginning."""

    try:
        uploaded_file.seek(0)
    except (AttributeError, OSError) as error:
        raise ValueError(
            "The uploaded file stream cannot be read from the beginning."
        ) from error


def _decode_csv(raw_bytes: bytes) -> tuple[str, str]:
    """Decode CSV bytes with common encoding fallbacks."""

    errors: list[str] = []

    for encoding in CSV_ENCODINGS:
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error}")

    raise ValueError(
        "The CSV encoding could not be detected. "
        + " | ".join(errors)
    )


def _detect_delimiter(text: str) -> str:
    """Detect a common CSV delimiter, falling back to a comma."""

    sample = text[:8192]

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=",;\t|",
        )
        return dialect.delimiter
    except csv.Error:
        return ","


def _load_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Read CSV bytes into a DataFrame."""

    decoded_text, encoding = _decode_csv(raw_bytes)
    delimiter = _detect_delimiter(decoded_text)

    try:
        return pd.read_csv(
            io.StringIO(decoded_text),
            sep=delimiter,
            encoding=encoding,
        )
    except pd.errors.EmptyDataError as error:
        raise ValueError("The CSV file contains no readable data.") from error
    except pd.errors.ParserError as error:
        raise ValueError(
            "The CSV structure could not be parsed. "
            f"Detected delimiter: {repr(delimiter)}. "
            f"Technical details: {error}"
        ) from error
    except Exception as error:
        raise ValueError(
            f"Unable to read the CSV file: {error}"
        ) from error


def _load_xlsx(raw_bytes: bytes) -> pd.DataFrame:
    """Read the first non-empty worksheet from an XLSX workbook."""

    try:
        workbook = pd.ExcelFile(
            io.BytesIO(raw_bytes),
            engine="openpyxl",
        )
    except Exception as error:
        raise ValueError(
            "Unable to open the Excel workbook. "
            "Confirm that the file is a valid XLSX file. "
            f"Technical details: {error}"
        ) from error

    if not workbook.sheet_names:
        raise ValueError("The Excel workbook contains no worksheets.")

    empty_sheet_names: list[str] = []

    for sheet_name in workbook.sheet_names:
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

        if not dataframe.empty or len(dataframe.columns) > 0:
            return dataframe

        empty_sheet_names.append(sheet_name)

    raise ValueError(
        "The Excel workbook contains no readable tabular data. "
        "Empty worksheets: "
        + ", ".join(empty_sheet_names)
    )


if __name__ == "__main__":
    sample = io.BytesIO(
        b"product,region,quantity,unit_price\n"
        b"Laptop,Yangon,2,850\n"
        b"Mouse,Mandalay,5,20\n"
    )

    frame = load_uploaded_file(sample, "sample.csv")
    print(frame)
