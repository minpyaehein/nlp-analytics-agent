"""
Pydantic models for structured analytical plans.
"""

from enum import Enum

from pydantic import BaseModel, Field


class AnalysisIntent(str, Enum):
    """
    Supported analytical question types.
    """

    SUMMARY = "summary"
    TREND = "trend"
    COMPARISON = "comparison"
    RANKING = "ranking"
    DISTRIBUTION = "distribution"
    DATA_QUALITY = "data_quality"
    CORRELATION = "correlation"
    UNKNOWN = "unknown"


class SortDirection(str, Enum):
    """
    Supported result sorting directions.
    """

    ASCENDING = "ascending"
    DESCENDING = "descending"


class VisualizationType(str, Enum):
    """
    Supported default visualization types.
    """

    KPI = "kpi"
    LINE = "line"
    BAR = "bar"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    TABLE = "table"


class FilterCondition(BaseModel):
    """
    A filter extracted from a user's analytical question.
    """

    column: str

    operator: str = "equals"

    value: str | int | float


class AnalysisPlan(BaseModel):
    """
    Structured representation of the user's analytical question.
    """

    original_question: str

    normalized_question: str

    language: str = Field(
        description=(
            "Detected question language: "
            "en, my, or mixed."
        )
    )

    intent: AnalysisIntent

    metric: str | None = None

    dimension: str | None = None

    aggregation: str = "sum"

    sort_direction: SortDirection | None = None

    limit: int | None = Field(
        default=None,
        ge=1,
        le=1000,
    )

    visualization: VisualizationType = (
        VisualizationType.TABLE
    )

    filters: list[FilterCondition] = Field(
        default_factory=list
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    warnings: list[str] = Field(
        default_factory=list
    )
