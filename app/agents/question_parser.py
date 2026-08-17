"""Rule-based bilingual parser for analytics questions."""

import re

from app.models.analysis_plan import (
    AnalysisIntent,
    AnalysisPlan,
    SortDirection,
    VisualizationType,
)
from app.services.text_normalizer import (
    detect_language,
    normalize_question,
)


METRIC_KEYWORDS = {
    "revenue": [
        "revenue",
        "sales revenue",
        "sales",
        "sale amount",
        "income",
    ],
    "profit": [
        "profit",
        "earnings",
        "net income",
    ],
    "quantity": [
        "quantity",
        "units sold",
        "units",
        "unit sold",
        "items sold",
    ],
    "unit_price": [
        "unit price",
        "selling price",
        "price",
    ],
    "unit_cost": [
        "unit cost",
        "cost price",
        "cost",
    ],
}


DIMENSION_KEYWORDS = {
    "product": [
        "product",
        "products",
        "item",
        "items",
    ],
    "category": [
        "category",
        "categories",
    ],
    "region": [
        "region",
        "regions",
        "location",
        "locations",
        "area",
        "areas",
    ],
    "month": [
        "month",
        "months",
        "monthly",
    ],
    "year": [
        "year",
        "years",
        "yearly",
        "annual",
    ],
    "order_date": [
        "order date",
        "date",
    ],
}


def find_keyword_mapping(
    text: str,
    keyword_map: dict[str, list[str]],
) -> str | None:
    """Return the canonical field for the first matching keyword."""

    mappings: list[tuple[str, str]] = []

    for standard_name, keywords in keyword_map.items():
        for keyword in keywords:
            mappings.append((standard_name, keyword))

    mappings.sort(
        key=lambda item: len(item[1]),
        reverse=True,
    )

    for standard_name, keyword in mappings:
        pattern = rf"\b{re.escape(keyword)}\b"

        if re.search(pattern, text, flags=re.IGNORECASE):
            return standard_name

    return None


def detect_intent(text: str) -> AnalysisIntent:
    """Detect the primary analytical intent from normalized text."""

    data_quality_terms = [
        "missing",
        "duplicate",
        "data quality",
        "null value",
        "null values",
        "empty value",
        "empty values",
    ]

    if any(term in text for term in data_quality_terms):
        return AnalysisIntent.DATA_QUALITY

    correlation_terms = [
        "correlation",
        "relationship between",
        "related to",
        "association between",
    ]

    if any(term in text for term in correlation_terms):
        return AnalysisIntent.CORRELATION

    ranking_terms = [
        "top",
        "bottom",
        "highest",
        "lowest",
        "best",
        "worst",
        "rank",
        "ranking",
    ]

    if any(term in text for term in ranking_terms):
        return AnalysisIntent.RANKING

    trend_terms = [
        "trend",
        "monthly",
        "yearly",
        "annual",
        "over time",
        "change over",
    ]

    if any(term in text for term in trend_terms):
        return AnalysisIntent.TREND

    comparison_terms = [
        "compare",
        "comparison",
        "versus",
        " vs ",
        "difference between",
    ]

    if any(term in text for term in comparison_terms):
        return AnalysisIntent.COMPARISON

    distribution_terms = [
        "distribution",
        "histogram",
        "frequency",
        "spread",
    ]

    if any(term in text for term in distribution_terms):
        return AnalysisIntent.DISTRIBUTION

    summary_terms = [
        "summary",
        "summarize",
        "overview",
        "describe",
        "total",
    ]

    if any(term in text for term in summary_terms):
        return AnalysisIntent.SUMMARY

    return AnalysisIntent.UNKNOWN


def extract_limit(text: str) -> int | None:
    """Extract a result limit such as top 5, bottom 10, or 5 products."""

    patterns = [
        r"\b(?:top|bottom|highest|lowest|best|worst)\s+(\d+)\b",
        r"\b(\d+)\s+(?:products|items|categories|regions|customers)\b",
        r"(?:အများဆုံး|အနည်းဆုံး|အကောင်းဆုံး|အဆိုးဆုံး)\s*(\d+)",
        r"(\d+)\s*(?:မျိုး|ခု|ယောက်)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return int(match.group(1))

    return None


def detect_sort_direction(
    text: str,
) -> SortDirection | None:
    """Determine the requested ranking direction."""

    ascending_terms = [
        "bottom",
        "lowest",
        "worst",
        "ascending",
    ]

    descending_terms = [
        "top",
        "highest",
        "best",
        "descending",
    ]

    if any(term in text for term in ascending_terms):
        return SortDirection.ASCENDING

    if any(term in text for term in descending_terms):
        return SortDirection.DESCENDING

    return None


def select_visualization(
    intent: AnalysisIntent,
) -> VisualizationType:
    """Select a default visualization for an analytical intent."""

    visualization_map = {
        AnalysisIntent.SUMMARY: VisualizationType.KPI,
        AnalysisIntent.TREND: VisualizationType.LINE,
        AnalysisIntent.COMPARISON: VisualizationType.BAR,
        AnalysisIntent.RANKING: VisualizationType.BAR,
        AnalysisIntent.DISTRIBUTION: VisualizationType.HISTOGRAM,
        AnalysisIntent.DATA_QUALITY: VisualizationType.TABLE,
        AnalysisIntent.CORRELATION: VisualizationType.SCATTER,
        AnalysisIntent.UNKNOWN: VisualizationType.TABLE,
    }

    return visualization_map[intent]


def calculate_confidence(
    intent: AnalysisIntent,
    metric: str | None,
    dimension: str | None,
) -> float:
    """Calculate a basic deterministic parsing-confidence score."""

    score = 0.0

    if intent != AnalysisIntent.UNKNOWN:
        score += 0.4

    if metric is not None:
        score += 0.3

    if dimension is not None:
        score += 0.3

    return round(score, 2)


def build_warnings(
    intent: AnalysisIntent,
    metric: str | None,
    dimension: str | None,
    limit: int | None,
) -> list[str]:
    """Generate warnings for missing or uncertain plan fields."""

    warnings: list[str] = []

    if intent == AnalysisIntent.UNKNOWN:
        warnings.append(
            "The analytical intent could not be identified."
        )

    if metric is None and intent != AnalysisIntent.DATA_QUALITY:
        warnings.append(
            "No analytical metric was detected."
        )

    intents_requiring_dimension = {
        AnalysisIntent.RANKING,
        AnalysisIntent.COMPARISON,
        AnalysisIntent.TREND,
        AnalysisIntent.DISTRIBUTION,
    }

    if intent in intents_requiring_dimension and dimension is None:
        warnings.append(
            "No analysis dimension was detected."
        )

    if intent == AnalysisIntent.RANKING and limit is None:
        warnings.append(
            "No ranking limit was provided. Defaulting to 5."
        )

    return warnings


def parse_question(question: str) -> AnalysisPlan:
    """Convert a natural-language question into an AnalysisPlan."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    language = detect_language(cleaned_question)
    normalized_question = normalize_question(cleaned_question)
    intent = detect_intent(normalized_question)

    metric = find_keyword_mapping(
        normalized_question,
        METRIC_KEYWORDS,
    )

    dimension = find_keyword_mapping(
        normalized_question,
        DIMENSION_KEYWORDS,
    )

    limit = extract_limit(normalized_question)
    sort_direction = detect_sort_direction(normalized_question)
    visualization = select_visualization(intent)

    confidence = calculate_confidence(
        intent=intent,
        metric=metric,
        dimension=dimension,
    )

    warnings = build_warnings(
        intent=intent,
        metric=metric,
        dimension=dimension,
        limit=limit,
    )

    if intent == AnalysisIntent.RANKING and limit is None:
        limit = 5

    return AnalysisPlan(
        original_question=cleaned_question,
        normalized_question=normalized_question,
        language=language,
        intent=intent,
        metric=metric,
        dimension=dimension,
        aggregation="sum",
        sort_direction=sort_direction,
        limit=limit,
        visualization=visualization,
        confidence=confidence,
        warnings=warnings,
    )
