"""Schema linking between an NLP analysis plan and dataset columns."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import pandas as pd


COLUMN_ALIASES = {
    "product": [
        "product",
        "product_name",
        "item",
        "item_name",
        "sku_name",
    ],
    "category": [
        "category",
        "product_category",
        "item_category",
    ],
    "region": [
        "region",
        "state",
        "state_name",
        "location",
        "area",
        "city",
    ],
    "order_date": [
        "order_date",
        "date",
        "sale_date",
        "transaction_date",
        "created_at",
    ],
    "quantity": [
        "quantity",
        "qty",
        "units",
        "units_sold",
        "sales_quantity",
    ],
    "unit_price": [
        "unit_price",
        "selling_price",
        "sale_price",
        "price",
    ],
    "unit_cost": [
        "unit_cost",
        "cost_price",
        "purchase_price",
        "cost",
    ],
    "revenue": [
        "revenue",
        "sales",
        "sales_amount",
        "total_sales",
        "income",
    ],
    "profit": [
        "profit",
        "net_profit",
        "gross_profit",
        "profit_amount",
    ],
}


DERIVED_METRICS = {
    "revenue": {
        "required_fields": ["quantity", "unit_price"],
        "formula": "quantity * unit_price",
    },
    "profit": {
        "required_fields": [
            "quantity",
            "unit_price",
            "unit_cost",
        ],
        "formula": "quantity * (unit_price - unit_cost)",
    },
}


@dataclass
class FieldLink:
    semantic_field: str
    source_column: str | None
    match_type: str
    confidence: float
    formula: str | None = None
    required_columns: list[str] | None = None
    warning: str | None = None

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)


def normalize_column_name(column_name: str) -> str:
    """Normalize a dataset column name for matching."""
    normalized = str(column_name).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def build_column_index(df: pd.DataFrame) -> dict[str, str]:
    """Map normalized column names to original column names."""
    return {
        normalize_column_name(column): str(column)
        for column in df.columns
    }


def find_direct_column(
    semantic_field: str,
    column_index: dict[str, str],
) -> FieldLink | None:
    """Find an exact semantic or alias match."""
    candidates = [semantic_field]
    candidates.extend(COLUMN_ALIASES.get(semantic_field, []))

    for candidate in candidates:
        normalized_candidate = normalize_column_name(candidate)

        if normalized_candidate in column_index:
            match_type = (
                "exact"
                if normalized_candidate == normalize_column_name(semantic_field)
                else "alias"
            )
            confidence = 1.0 if match_type == "exact" else 0.9

            return FieldLink(
                semantic_field=semantic_field,
                source_column=column_index[normalized_candidate],
                match_type=match_type,
                confidence=confidence,
            )

    return None


def link_derived_metric(
    metric: str,
    column_index: dict[str, str],
) -> FieldLink | None:
    """Link a metric that must be calculated from source columns."""
    definition = DERIVED_METRICS.get(metric)

    if definition is None:
        return None

    required_columns: list[str] = []
    missing_fields: list[str] = []

    for required_field in definition["required_fields"]:
        link = find_direct_column(required_field, column_index)

        if link is None or link.source_column is None:
            missing_fields.append(required_field)
        else:
            required_columns.append(link.source_column)

    if missing_fields:
        return FieldLink(
            semantic_field=metric,
            source_column=None,
            match_type="unresolved",
            confidence=0.0,
            formula=definition["formula"],
            required_columns=required_columns,
            warning=(
                "Cannot derive "
                f"{metric}. Missing fields: {', '.join(missing_fields)}."
            ),
        )

    source_formula = definition["formula"]

    for required_field, source_column in zip(
        definition["required_fields"],
        required_columns,
    ):
        source_formula = re.sub(
            rf"\b{re.escape(required_field)}\b",
            f"`{source_column}`",
            source_formula,
        )

    return FieldLink(
        semantic_field=metric,
        source_column=None,
        match_type="derived",
        confidence=0.95,
        formula=source_formula,
        required_columns=required_columns,
    )


def link_field(
    semantic_field: str,
    df: pd.DataFrame,
    allow_derived: bool = True,
) -> FieldLink:
    """Link one semantic field to the uploaded dataset."""
    column_index = build_column_index(df)

    direct_link = find_direct_column(
        semantic_field,
        column_index,
    )

    if direct_link is not None:
        return direct_link

    if allow_derived:
        derived_link = link_derived_metric(
            semantic_field,
            column_index,
        )

        if derived_link is not None:
            return derived_link

    return FieldLink(
        semantic_field=semantic_field,
        source_column=None,
        match_type="unresolved",
        confidence=0.0,
        warning=(
            f"No dataset column could be linked to '{semantic_field}'."
        ),
    )


def link_analysis_plan(
    metric: str | None,
    dimension: str | None,
    df: pd.DataFrame,
) -> dict:
    """Link the metric and dimension from an analysis plan."""
    metric_link = (
        link_field(metric, df, allow_derived=True)
        if metric
        else None
    )

    dimension_link = (
        link_field(dimension, df, allow_derived=False)
        if dimension
        else None
    )

    warnings: list[str] = []

    for link in [metric_link, dimension_link]:
        if link is not None and link.warning:
            warnings.append(link.warning)

    resolved = all(
        link is None or link.match_type != "unresolved"
        for link in [metric_link, dimension_link]
    )

    return {
        "resolved": resolved,
        "metric": metric_link.to_dict() if metric_link else None,
        "dimension": dimension_link.to_dict() if dimension_link else None,
        "warnings": warnings,
    }
