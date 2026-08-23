"""End-to-end regression tests for InsightFlow AI PDF analytics.

Run from the project root:
    python -m scripts.test_pdf_analytics_regression

The tests verify:
- high-resolution searchable PDF table extraction;
- exact 12 x 8 analytical schema;
- current-DataFrame revenue and profit quality gates;
- total revenue, total profit, product ranking, and Yangon filtering;
- low-resolution PDF safety blocking;
- DataFrame fingerprint consistency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.question_parser import parse_question
from app.services.filtered_executor import execute_filtered_analysis
from core.analytics_quality_gate import (
    validate_profit_readiness,
    validate_revenue_readiness,
)
from core.pdf_analysis_state import (
    state_matches_dataframe,
    validate_before_metric,
    validate_current_pdf_dataframe,
)
from core.pdf_table_extractor import extract_pdf_tables


HIGH_RES_PDF = (
    PROJECT_ROOT
    / "sample_data"
    / "scanned_sales_report_highres_searchable.pdf"
)

LOW_RES_PDF = (
    PROJECT_ROOT
    / "sample_data"
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

EXPECTED_QUANTITIES = [2, 5, 3, 2, 1, 8, 3, 4, 2, 6, 6, 2]
EXPECTED_TOTAL_REVENUE = 6355.0
EXPECTED_TOTAL_PROFIT = 1335.0
EXPECTED_YANGON_REVENUE = 4415.0
EXPECTED_PRODUCT_REVENUE = {
    "Laptop": 4250.0,
    "Monitor": 1200.0,
    "Mouse": 500.0,
    "Keyboard": 405.0,
}


def section(title: str) -> None:
    """Print a consistent test section heading."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def check(condition: bool, message: str) -> None:
    """Raise a readable assertion failure."""

    if not condition:
        raise AssertionError(message)

    print(f"PASS: {message}")


def find_summary_value(result: Any) -> float:
    """Extract the numeric value from an executor summary result."""

    records = result.analysis.get("result", [])
    check(len(records) == 1, "Summary analysis returned exactly one record")

    value = records[0].get("value")
    check(isinstance(value, (int, float)), "Summary record contains a numeric value")
    return float(value)


def load_best_table(pdf_path: Path) -> tuple[Any, pd.DataFrame]:
    """Extract and return the highest-confidence table."""

    check(pdf_path.exists(), f"Input PDF exists: {pdf_path.name}")
    extraction = extract_pdf_tables(pdf_path)
    check(extraction.success, f"PDF extraction succeeded: {pdf_path.name}")
    check(bool(extraction.tables), "At least one PDF table was extracted")

    table = extraction.best_table()
    dataframe = table.dataframe.copy()

    print(
        json.dumps(
            table.metadata(),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    return table, dataframe


def test_high_resolution_extraction() -> pd.DataFrame:
    """Validate exact schema, row count, numeric values, and known warnings."""

    section("1. High-Resolution PDF Table Extraction")
    table, dataframe = load_best_table(HIGH_RES_PDF)

    check(len(dataframe) == 12, "High-resolution table contains 12 rows")
    check(len(dataframe.columns) == 8, "High-resolution table contains 8 columns")
    check(list(dataframe.columns) == EXPECTED_COLUMNS, "Extracted schema matches expected column order")
    check(table.confidence >= 0.95, "Extraction confidence is at least 95%")

    quantities = (
        pd.to_numeric(dataframe["quantity"], errors="coerce")
        .astype("Int64")
        .tolist()
    )
    check(quantities == EXPECTED_QUANTITIES, "All 12 quantity values match the source")

    numeric_counts = {
        column: int(pd.to_numeric(dataframe[column], errors="coerce").notna().sum())
        for column in ("quantity", "unit_price", "unit_cost")
    }
    check(
        numeric_counts == {"quantity": 12, "unit_price": 12, "unit_cost": 12},
        "Required financial columns contain 12 usable numeric values each",
    )
    check(int(dataframe.duplicated().sum()) == 1, "One duplicate row is detected")
    check(int(dataframe["region"].isna().sum()) == 1, "One missing region is detected")

    return dataframe


def test_current_dataframe_quality_gates(dataframe: pd.DataFrame) -> None:
    """Validate current-table state, fingerprints, and both financial gates."""

    section("2. Current DataFrame Quality Gates")

    revenue_gate = validate_revenue_readiness(dataframe)
    profit_gate = validate_profit_readiness(dataframe)

    check(revenue_gate.analytics_ready, "Revenue quality gate passed")
    check(profit_gate.analytics_ready, "Profit quality gate passed")

    state = validate_current_pdf_dataframe(
        dataframe=dataframe,
        filename=HIGH_RES_PDF.name,
        pdf_bytes=HIGH_RES_PDF.read_bytes(),
        ocr_language="eng+mya",
        table_index=0,
    )

    check(
        state.numeric_counts
        == {"quantity": 12, "unit_price": 12, "unit_cost": 12},
        "Current PDF state contains correct numeric evidence counts",
    )
    check(state_matches_dataframe(state, dataframe), "PDF state fingerprint matches the current DataFrame")

    revenue_precheck = validate_before_metric(dataframe, "revenue")
    profit_precheck = validate_before_metric(dataframe, "profit")
    check(bool(revenue_precheck and revenue_precheck.analytics_ready), "Pre-analysis revenue gate passed")
    check(bool(profit_precheck and profit_precheck.analytics_ready), "Pre-analysis profit gate passed")


def test_total_revenue(dataframe: pd.DataFrame) -> None:
    """Verify deterministic revenue summary."""

    section("3. Total Revenue")
    plan = parse_question("Show total revenue.")
    result = execute_filtered_analysis(dataframe, plan)
    value = find_summary_value(result)

    check(result.success, "Revenue analysis succeeded")
    check(value == EXPECTED_TOTAL_REVENUE, "Total revenue equals 6,355")
    check(result.source_rows == 12, "Revenue analysis used 12 source rows")
    check(result.filtered_rows == 12, "Revenue analysis retained 12 rows")


def test_total_profit(dataframe: pd.DataFrame) -> None:
    """Verify deterministic profit summary."""

    section("4. Total Profit")
    plan = parse_question("Show total profit.")
    result = execute_filtered_analysis(dataframe, plan)
    value = find_summary_value(result)

    check(result.success, "Profit analysis succeeded")
    check(value == EXPECTED_TOTAL_PROFIT, "Total profit equals 1,335")


def test_product_ranking(dataframe: pd.DataFrame) -> None:
    """Verify product ranking values and order."""

    section("5. Product Revenue Ranking")
    plan = parse_question("Show the top 5 products by revenue.")
    result = execute_filtered_analysis(dataframe, plan)
    records = result.analysis.get("result", [])

    check(result.success, "Product ranking analysis succeeded")
    check(len(records) == 4, "Product ranking returned four available products")

    actual = {
        str(record["product"]): float(record["revenue"])
        for record in records
    }
    check(actual == EXPECTED_PRODUCT_REVENUE, "Product revenue totals match expected values")
    check(
        [record["product"] for record in records]
        == ["Laptop", "Monitor", "Mouse", "Keyboard"],
        "Product ranking order is descending by revenue",
    )


def test_yangon_filter(dataframe: pd.DataFrame) -> None:
    """Verify Myanmar-language region filtering and revenue."""

    section("6. Myanmar Yangon Revenue Filter")
    plan = parse_question("ရန်ကုန်ဒေသ၏ ဝင်ငွေကို ပြပါ")
    result = execute_filtered_analysis(dataframe, plan)
    value = find_summary_value(result)

    check(result.success, "Myanmar filtered analysis succeeded")
    check(result.filtered_rows == 5, "Yangon filter matched five rows")
    check(value == EXPECTED_YANGON_REVENUE, "Yangon revenue equals 4,415")
    check(len(result.applied_filters) == 1, "Exactly one region filter was applied")


def test_low_resolution_safety_block() -> None:
    """Verify incomplete OCR data cannot produce financial calculations."""

    section("7. Low-Resolution PDF Safety Block")
    _, dataframe = load_best_table(LOW_RES_PDF)

    revenue_gate = validate_revenue_readiness(dataframe)
    profit_gate = validate_profit_readiness(dataframe)

    check(not revenue_gate.analytics_ready, "Low-resolution revenue gate is blocked")
    check(not profit_gate.analytics_ready, "Low-resolution profit gate is blocked")

    try:
        validate_before_metric(dataframe, "revenue")
    except ValueError as error:
        print(f"EXPECTED BLOCK: {error}")
    else:
        raise AssertionError("Low-resolution revenue calculation was not blocked")


def main() -> None:
    """Run the complete PDF analytics regression suite."""

    print("=" * 80)
    print("InsightFlow AI PDF Analytics Regression Suite")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")

    dataframe = test_high_resolution_extraction()
    test_current_dataframe_quality_gates(dataframe)
    test_total_revenue(dataframe)
    test_total_profit(dataframe)
    test_product_ranking(dataframe)
    test_yangon_filter(dataframe)
    test_low_resolution_safety_block()

    section("Final Result")
    print("ALL PDF ANALYTICS REGRESSION TESTS PASSED")


if __name__ == "__main__":
    main()
