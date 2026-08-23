"""Regression tests for deterministic InsightFlow financial analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.agents.question_parser import parse_question
from app.services.filtered_executor import execute_filtered_analysis
from core.analytics_quality_gate import (
    require_profit_ready,
    require_revenue_ready,
    validate_profit_readiness,
    validate_revenue_readiness,
)
from core.pdf_table_extractor import extract_pdf_tables


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

CSV_PATH = SAMPLE_DATA_DIR / "sales_test.csv"
HIGH_RESOLUTION_PDF_PATH = (
    SAMPLE_DATA_DIR
    / "scanned_sales_report_highres_searchable.pdf"
)
LOW_RESOLUTION_PDF_PATH = (
    SAMPLE_DATA_DIR
    / "scanned_sales_report_searchable.pdf"
)

EXPECTED_COLUMNS = [
    "order_id",
    "order_date",
    "product",
    "category",
    "region",
    "quantity",
    "unit_price",
    "unit_cost",
]

EXPECTED_PRODUCT_REVENUE = [
    {"product": "Laptop", "revenue": 4250},
    {"product": "Monitor", "revenue": 1200},
    {"product": "Mouse", "revenue": 500},
    {"product": "Keyboard", "revenue": 405},
]


def require_fixture(path: Path) -> Path:
    """Return a required fixture path or skip with a clear message."""

    if not path.exists():
        pytest.skip(f"Required test fixture does not exist: {path}")

    return path


def load_csv_dataframe() -> pd.DataFrame:
    """Load the trusted source CSV fixture."""

    path = require_fixture(CSV_PATH)
    return pd.read_csv(path)


def load_pdf_dataframe(path: Path) -> pd.DataFrame:
    """Extract the highest-confidence table from a PDF fixture."""

    fixture_path = require_fixture(path)
    extraction = extract_pdf_tables(fixture_path)

    assert extraction.success, extraction.metadata()
    assert extraction.tables, extraction.metadata()

    table = extraction.best_table()
    dataframe = table.dataframe.copy()

    assert not dataframe.empty

    return dataframe


def analyze(dataframe: pd.DataFrame, question: str) -> Any:
    """Parse and execute one deterministic analytical question."""

    plan = parse_question(question)
    return execute_filtered_analysis(dataframe, plan)


def summary_value(result: Any) -> float:
    """Extract the numeric value from a one-row summary result."""

    records = result.analysis["result"]

    assert len(records) == 1
    assert "value" in records[0]

    return float(records[0]["value"])


def quality_gate_ready(gate: Any) -> bool:
    """Read readiness across compatible quality-gate result versions."""

    if hasattr(gate, "analytics_ready"):
        return bool(gate.analytics_ready)

    if hasattr(gate, "ready"):
        return bool(gate.ready)

    metadata = gate.to_dict()
    return bool(
        metadata.get(
            "analytics_ready",
            metadata.get("ready", metadata.get("success", False)),
        )
    )


def usable_ratio(gate: Any, column: str) -> float:
    """Read the usable numeric ratio for one required column."""

    if hasattr(gate, "usable_numeric_ratios"):
        ratios = gate.usable_numeric_ratios
    else:
        ratios = gate.to_dict().get("usable_numeric_ratios", {})

    return float(ratios[column])


def unusable_numeric_columns(gate: Any) -> list[str]:
    """Read unusable numeric columns across result versions."""

    if hasattr(gate, "unusable_numeric_columns"):
        return list(gate.unusable_numeric_columns)

    return list(
        gate.to_dict().get("unusable_numeric_columns", [])
    )


def gate_duplicate_rows(gate: Any) -> int:
    """Read duplicate-row count across result versions."""

    if hasattr(gate, "duplicate_rows"):
        return int(gate.duplicate_rows)

    return int(gate.to_dict().get("duplicate_rows", 0))


def gate_warnings(gate: Any) -> list[str]:
    """Read quality-gate warnings across result versions."""

    if hasattr(gate, "warnings"):
        return list(gate.warnings)

    return list(gate.to_dict().get("warnings", []))


def test_csv_fixture_shape_and_quality() -> None:
    dataframe = load_csv_dataframe()

    assert dataframe.shape == (12, 8)
    assert dataframe.columns.tolist() == EXPECTED_COLUMNS
    assert int(dataframe.duplicated().sum()) == 1
    assert int(dataframe.isna().sum().sum()) == 1


def test_csv_total_revenue() -> None:
    result = analyze(
        load_csv_dataframe(),
        "Show total revenue.",
    )

    assert result.success
    assert summary_value(result) == pytest.approx(6355.0)


def test_csv_total_profit() -> None:
    result = analyze(
        load_csv_dataframe(),
        "Show total profit.",
    )

    assert result.success
    assert summary_value(result) == pytest.approx(1335.0)


def test_high_resolution_pdf_shape_and_schema() -> None:
    dataframe = load_pdf_dataframe(HIGH_RESOLUTION_PDF_PATH)

    assert dataframe.shape == (12, 8)
    assert dataframe.columns.tolist() == EXPECTED_COLUMNS
    assert int(dataframe.duplicated().sum()) == 1


def test_high_resolution_pdf_quality_gates() -> None:
    dataframe = load_pdf_dataframe(HIGH_RESOLUTION_PDF_PATH)

    revenue_gate = validate_revenue_readiness(dataframe)
    profit_gate = validate_profit_readiness(dataframe)

    assert quality_gate_ready(revenue_gate)
    assert quality_gate_ready(profit_gate)

    assert usable_ratio(revenue_gate, "quantity") == pytest.approx(1.0)
    assert usable_ratio(revenue_gate, "unit_price") == pytest.approx(1.0)
    assert usable_ratio(profit_gate, "quantity") == pytest.approx(1.0)
    assert usable_ratio(profit_gate, "unit_price") == pytest.approx(1.0)
    assert usable_ratio(profit_gate, "unit_cost") == pytest.approx(1.0)

    require_revenue_ready(dataframe)
    require_profit_ready(dataframe)


def test_high_resolution_pdf_total_revenue() -> None:
    dataframe = load_pdf_dataframe(HIGH_RESOLUTION_PDF_PATH)
    require_revenue_ready(dataframe)

    result = analyze(dataframe, "Show total revenue.")

    assert result.success
    assert summary_value(result) == pytest.approx(6355.0)


def test_high_resolution_pdf_total_profit() -> None:
    dataframe = load_pdf_dataframe(HIGH_RESOLUTION_PDF_PATH)
    require_profit_ready(dataframe)

    result = analyze(dataframe, "Show total profit.")

    assert result.success
    assert summary_value(result) == pytest.approx(1335.0)


def test_product_revenue_ranking() -> None:
    dataframe = load_pdf_dataframe(HIGH_RESOLUTION_PDF_PATH)
    require_revenue_ready(dataframe)

    result = analyze(
        dataframe,
        "Show the top 5 products by revenue.",
    )

    assert result.success
    assert result.analysis["result"] == EXPECTED_PRODUCT_REVENUE


def test_myanmar_yangon_revenue() -> None:
    dataframe = load_pdf_dataframe(HIGH_RESOLUTION_PDF_PATH)
    require_revenue_ready(dataframe)

    result = analyze(
        dataframe,
        "ရန်ကုန်ဒေသ၏ ဝင်ငွေကို ပြပါ",
    )

    assert result.success
    assert result.filtered_rows == 5
    assert summary_value(result) == pytest.approx(4415.0)


def test_low_resolution_pdf_revenue_is_blocked() -> None:
    dataframe = load_pdf_dataframe(LOW_RESOLUTION_PDF_PATH)
    revenue_gate = validate_revenue_readiness(dataframe)

    assert not quality_gate_ready(revenue_gate)

    unusable_columns = unusable_numeric_columns(revenue_gate)
    assert "quantity" in unusable_columns

    with pytest.raises(ValueError):
        require_revenue_ready(dataframe)


def test_low_resolution_pdf_profit_is_blocked() -> None:
    dataframe = load_pdf_dataframe(LOW_RESOLUTION_PDF_PATH)
    profit_gate = validate_profit_readiness(dataframe)

    assert not quality_gate_ready(profit_gate)

    with pytest.raises(ValueError):
        require_profit_ready(dataframe)


def test_duplicate_row_warning() -> None:
    dataframe = load_pdf_dataframe(HIGH_RESOLUTION_PDF_PATH)
    gate = validate_revenue_readiness(dataframe)

    assert gate_duplicate_rows(gate) == 1
    assert any(
        "duplicate row" in warning.casefold()
        for warning in gate_warnings(gate)
    )
