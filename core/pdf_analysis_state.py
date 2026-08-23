"""Current-DataFrame state and quality validation for PDF analytics."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.analytics_quality_gate import (
    AnalyticsQualityGateResult,
    validate_profit_readiness,
    validate_revenue_readiness,
)


@dataclass
class CurrentPdfAnalysisState:
    """Validated state for the exact PDF table used by analytics."""

    state_key: str
    filename: str
    ocr_language: str
    table_index: int
    row_count: int
    column_count: int
    columns: list[str]
    numeric_counts: dict[str, int]
    dataframe_fingerprint: str
    revenue_gate: dict[str, Any]
    profit_gate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_pdf_state_key(
    filename: str,
    pdf_bytes: bytes,
    ocr_language: str,
    table_index: int,
) -> str:
    """Create a key tied to bytes, OCR language, and table selection."""

    if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
        raise ValueError("pdf_bytes must contain a non-empty PDF upload")

    digest = hashlib.sha256(pdf_bytes).hexdigest()[:20]
    safe_name = Path(filename).stem.replace(" ", "_")[:40]
    language = str(ocr_language).strip() or "unknown"
    return (
        f"insightflow_pdf_{safe_name}_{digest}_"
        f"{language}_table_{int(table_index)}"
    )


def dataframe_fingerprint(dataframe: pd.DataFrame) -> str:
    """Fingerprint DataFrame values, columns, index, and row order."""

    _validate_dataframe(dataframe)
    digest = hashlib.sha256()
    digest.update(
        "|".join(str(column) for column in dataframe.columns).encode("utf-8")
    )
    digest.update(str(dataframe.shape).encode("ascii"))
    hashed_rows = pd.util.hash_pandas_object(
        dataframe,
        index=True,
        categorize=True,
    )
    digest.update(hashed_rows.to_numpy().tobytes())
    return digest.hexdigest()


def numeric_evidence_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    """Count usable values in financial input columns."""

    _validate_dataframe(dataframe)
    counts: dict[str, int] = {}

    for column in ("quantity", "unit_price", "unit_cost"):
        if column not in dataframe.columns:
            counts[column] = 0
        else:
            counts[column] = int(
                pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                ).notna().sum()
            )

    return counts


def validate_current_pdf_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
    pdf_bytes: bytes,
    ocr_language: str,
    table_index: int = 0,
) -> CurrentPdfAnalysisState:
    """Recompute both gates from the exact selected DataFrame."""

    _validate_dataframe(dataframe)
    revenue_gate = validate_revenue_readiness(dataframe)
    profit_gate = validate_profit_readiness(dataframe)

    return CurrentPdfAnalysisState(
        state_key=build_pdf_state_key(
            filename,
            pdf_bytes,
            ocr_language,
            table_index,
        ),
        filename=filename,
        ocr_language=ocr_language,
        table_index=int(table_index),
        row_count=int(len(dataframe)),
        column_count=int(len(dataframe.columns)),
        columns=[str(column) for column in dataframe.columns],
        numeric_counts=numeric_evidence_counts(dataframe),
        dataframe_fingerprint=dataframe_fingerprint(dataframe),
        revenue_gate=revenue_gate.to_dict(),
        profit_gate=profit_gate.to_dict(),
    )


def validate_before_metric(
    dataframe: pd.DataFrame,
    metric_name: str | None,
) -> AnalyticsQualityGateResult | None:
    """Gate the exact current DataFrame immediately before calculation."""

    _validate_dataframe(dataframe)
    metric = str(metric_name or "").strip().casefold()

    if metric == "revenue":
        result = validate_revenue_readiness(dataframe)
        result.raise_if_failed()
        return result

    if metric == "profit":
        result = validate_profit_readiness(dataframe)
        result.raise_if_failed()
        return result

    return None


def state_matches_dataframe(
    state: CurrentPdfAnalysisState | dict[str, Any],
    dataframe: pd.DataFrame,
) -> bool:
    """Return True only if cached state belongs to this DataFrame."""

    _validate_dataframe(dataframe)
    expected = (
        state.dataframe_fingerprint
        if isinstance(state, CurrentPdfAnalysisState)
        else state.get("dataframe_fingerprint")
    )
    return bool(expected and expected == dataframe_fingerprint(dataframe))


def _validate_dataframe(dataframe: pd.DataFrame) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a Pandas DataFrame")
    if dataframe.empty:
        raise ValueError("The selected analytical DataFrame contains no rows.")
