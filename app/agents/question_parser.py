"""Rule-based bilingual analytics question parser."""

import re

from app.models.analysis_plan import (
    AnalysisIntent,
    AnalysisPlan,
    FilterCondition,
    SortDirection,
    VisualizationType,
)
from app.services.text_normalizer import detect_language, normalize_question


METRIC_KEYWORDS: dict[str, list[str]] = {
    "revenue": [
        "sales revenue",
        "total sales",
        "sales amount",
        "sale amount",
        "revenue",
        "sales",
        "income",
    ],
    "profit": [
        "net profit",
        "gross profit",
        "profit amount",
        "profit",
        "earnings",
        "net income",
    ],
    "quantity": [
        "units sold",
        "items sold",
        "sales quantity",
        "unit sold",
        "quantity",
        "units",
    ],
    "unit_price": [
        "unit price",
        "selling price",
        "sale price",
        "price",
    ],
    "unit_cost": [
        "unit cost",
        "cost price",
        "purchase price",
        "cost",
    ],
}


DIMENSION_KEYWORDS: dict[str, list[str]] = {
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
        "monthly",
        "month",
        "months",
    ],
    "year": [
        "yearly",
        "annual",
        "year",
        "years",
    ],
    "order_date": [
        "order date",
        "date",
    ],
}


LOCATION_KEYWORDS: dict[str, list[str]] = {
    "Yangon": ["yangon", "ရန်ကုန်"],
    "Mandalay": ["mandalay", "မန္တလေး"],
    "Naypyidaw": ["naypyidaw", "nay pyi taw", "နေပြည်တော်"],
    "Bago": ["bago", "ပဲခူး"],
    "Mon": ["mon state", "မွန်ပြည်နယ်"],
    "Kayin": ["kayin state", "ကရင်ပြည်နယ်"],
    "Shan": ["shan state", "ရှမ်းပြည်နယ်"],
    "Rakhine": ["rakhine state", "ရခိုင်ပြည်နယ်"],
    "Ayeyarwady": ["ayeyarwady", "irrawaddy", "ဧရာဝတီ"],
    "Sagaing": ["sagaing", "စစ်ကိုင်း"],
    "Magway": ["magway", "မကွေး"],
    "Tanintharyi": ["tanintharyi", "တနင်္သာရီ"],
    "Kachin": ["kachin", "ကချင်"],
    "Chin": ["chin state", "ချင်းပြည်နယ်"],
    "Kayah": ["kayah", "ကယား"],
}


def _contains(text: str, keyword: str) -> bool:
    """Return True when a complete keyword occurs in text."""

    normalized_text = text.casefold()
    normalized_keyword = keyword.casefold().strip()

    if not normalized_keyword:
        return False

    contains_myanmar = bool(
        re.search(
            r"[\u1000-\u109F\uAA60-\uAA7F]",
            normalized_keyword,
        )
    )

    if contains_myanmar:
        return normalized_keyword in normalized_text

    pattern = rf"(?<![A-Za-z0-9_]){re.escape(normalized_keyword)}(?![A-Za-z0-9_])"
    return re.search(pattern, normalized_text) is not None


def find_keyword_mapping(
    text: str,
    keyword_map: dict[str, list[str]],
) -> str | None:
    """Map a phrase in the question to a canonical semantic field."""

    mappings = [
        (canonical_name, keyword)
        for canonical_name, keywords in keyword_map.items()
        for keyword in keywords
    ]
    mappings.sort(key=lambda item: len(item[1]), reverse=True)

    for canonical_name, keyword in mappings:
        if _contains(text, keyword):
            return canonical_name

    return None


def detect_intent(text: str) -> AnalysisIntent:
    """Detect the primary analytical intent."""

    normalized = text.casefold().strip()

    data_quality_terms = [
        "missing",
        "duplicate",
        "data quality",
        "null value",
        "empty value",
    ]
    if any(term in normalized for term in data_quality_terms):
        return AnalysisIntent.DATA_QUALITY

    correlation_terms = [
        "correlation",
        "relationship between",
        "association between",
        "related to",
    ]
    if any(term in normalized for term in correlation_terms):
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
    if any(_contains(normalized, term) for term in ranking_terms):
        return AnalysisIntent.RANKING

    trend_terms = [
        "trend",
        "monthly",
        "yearly",
        "annual",
        "over time",
        "change over",
    ]
    if any(term in normalized for term in trend_terms):
        return AnalysisIntent.TREND

    comparison_terms = [
        "compare",
        "comparison",
        "versus",
        "difference between",
    ]
    if _contains(normalized, "vs") or any(
        term in normalized for term in comparison_terms
    ):
        return AnalysisIntent.COMPARISON

    distribution_terms = [
        "distribution",
        "histogram",
        "frequency",
        "spread",
    ]
    if any(term in normalized for term in distribution_terms):
        return AnalysisIntent.DISTRIBUTION

    summary_terms = [
        "summary",
        "summarize",
        "overview",
        "describe",
        "total",
        "overall",
        "sum",
        "analyze",
    ]
    if any(_contains(normalized, term) for term in summary_terms):
        return AnalysisIntent.SUMMARY

    return AnalysisIntent.UNKNOWN


def extract_limit(text: str) -> int | None:
    """Extract a ranking limit from English or normalized Myanmar text."""

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


def detect_sort_direction(text: str) -> SortDirection | None:
    """Determine ranking direction."""

    ascending_terms = ["bottom", "lowest", "worst", "ascending"]
    descending_terms = ["top", "highest", "best", "descending"]

    if any(_contains(text, term) for term in ascending_terms):
        return SortDirection.ASCENDING

    if any(_contains(text, term) for term in descending_terms):
        return SortDirection.DESCENDING

    return None


def select_visualization(intent: AnalysisIntent) -> VisualizationType:
    """Choose a default visualization for the detected intent."""

    return {
        AnalysisIntent.SUMMARY: VisualizationType.KPI,
        AnalysisIntent.TREND: VisualizationType.LINE,
        AnalysisIntent.COMPARISON: VisualizationType.BAR,
        AnalysisIntent.RANKING: VisualizationType.BAR,
        AnalysisIntent.DISTRIBUTION: VisualizationType.HISTOGRAM,
        AnalysisIntent.DATA_QUALITY: VisualizationType.TABLE,
        AnalysisIntent.CORRELATION: VisualizationType.SCATTER,
        AnalysisIntent.UNKNOWN: VisualizationType.TABLE,
    }[intent]


def extract_location_filters(original_text: str) -> list[FilterCondition]:
    """Extract known geographic filters from the original question."""

    filters: list[FilterCondition] = []

    for location, aliases in LOCATION_KEYWORDS.items():
        if any(_contains(original_text, alias) for alias in aliases):
            filters.append(
                FilterCondition(
                    column="region",
                    operator="equals",
                    value=location,
                )
            )

    return filters


def calculate_confidence(
    intent: AnalysisIntent,
    metric: str | None,
    dimension: str | None,
) -> float:
    """Calculate deterministic parser confidence."""

    score = 0.0

    if intent != AnalysisIntent.UNKNOWN:
        score += 0.4

    if metric is not None or intent in {
        AnalysisIntent.DATA_QUALITY,
        AnalysisIntent.CORRELATION,
    }:
        score += 0.3

    if dimension is not None or intent in {
        AnalysisIntent.SUMMARY,
        AnalysisIntent.DATA_QUALITY,
        AnalysisIntent.CORRELATION,
    }:
        score += 0.3

    return round(score, 2)


def build_warnings(
    intent: AnalysisIntent,
    metric: str | None,
    dimension: str | None,
    limit: int | None,
) -> list[str]:
    """Build warnings for missing parser fields."""

    warnings: list[str] = []

    if intent == AnalysisIntent.UNKNOWN:
        warnings.append(
            "The analytical intent could not be identified."
        )

    if metric is None and intent not in {
        AnalysisIntent.DATA_QUALITY,
        AnalysisIntent.CORRELATION,
    }:
        warnings.append("No analytical metric was detected.")

    if intent in {
        AnalysisIntent.RANKING,
        AnalysisIntent.COMPARISON,
        AnalysisIntent.TREND,
        AnalysisIntent.DISTRIBUTION,
    } and dimension is None:
        warnings.append("No analysis dimension was detected.")

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
    filters = extract_location_filters(cleaned_question)

    warnings = build_warnings(
        intent,
        metric,
        dimension,
        limit,
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
        visualization=select_visualization(intent),
        filters=filters,
        confidence=calculate_confidence(
            intent,
            metric,
            dimension,
        ),
        warnings=warnings,
    )


if __name__ == "__main__":
    test_questions = [
        "Show the top 5 products by revenue.",
        "Analyze missing values and duplicate rows.",
        "Show total revenue.",
        "အမြတ်အများဆုံး ကုန်ပစ္စည်း ၅ ခုကို ပြပါ",
        "ရန်ကုန်ဒေသ၏ ဝင်ငွေကို ခွဲခြမ်းစိတ်ဖြာပါ",
    ]

    for test_question in test_questions:
        print("=" * 80)
        print(
            parse_question(test_question).model_dump_json(
                indent=2

            )
        )
