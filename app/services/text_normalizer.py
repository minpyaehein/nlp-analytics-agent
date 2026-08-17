"""Text normalization for English, Myanmar, and mixed analytics questions."""

import re
import unicodedata


MYANMAR_DIGIT_MAP = str.maketrans(
    {
        "၀": "0",
        "၁": "1",
        "၂": "2",
        "၃": "3",
        "၄": "4",
        "၅": "5",
        "၆": "6",
        "၇": "7",
        "၈": "8",
        "၉": "9",
    }
)


BURMESE_TERM_MAP = {
    "ရောင်းရငွေ": "revenue",
    "ရောင်းအား": "revenue",
    "အရောင်းပမာဏ": "revenue",
    "ဝင်ငွေ": "revenue",
    "အသားတင်အမြတ်": "profit",
    "အမြတ်": "profit",
    "ကုန်ကျစရိတ်": "cost",
    "အသုံးစရိတ်": "cost",
    "ကုန်ပစ္စည်းများ": "products",
    "ကုန်ပစ္စည်း": "product",
    "ပစ္စည်းများ": "products",
    "ပစ္စည်း": "product",
    "ကုန်ပစ္စည်းအမျိုးအစား": "category",
    "အမျိုးအစား": "category",
    "ဒေသများ": "regions",
    "ဒေသ": "region",
    "နယ်မြေ": "region",
    "တည်နေရာ": "location",
    "ဖောက်သည်များ": "customers",
    "ဖောက်သည်": "customer",
    "ဝယ်ယူသူများ": "customers",
    "ဝယ်ယူသူ": "customer",
    "အရေအတွက်": "quantity",
    "ရောင်းချရသည့်အရေအတွက်": "quantity",
    "စျေးနှုန်း": "price",
    "ဈေးနှုန်း": "price",
    "လစဉ်": "monthly",
    "လအလိုက်": "monthly",
    "တစ်လချင်း": "monthly",
    "နှစ်စဉ်": "yearly",
    "နှစ်အလိုက်": "yearly",
    "တစ်နှစ်ချင်း": "yearly",
    "အများဆုံး": "top",
    "အနည်းဆုံး": "bottom",
    "အကောင်းဆုံး": "best",
    "အဆိုးဆုံး": "worst",
    "အမြင့်ဆုံး": "highest",
    "အနိမ့်ဆုံး": "lowest",
    "နှိုင်းယှဉ်": "compare",
    "ဆက်စပ်မှု": "correlation",
    "ဖြန့်ဝေမှု": "distribution",
    "ပျောက်ဆုံးတန်ဖိုး": "missing value",
    "ထပ်နေသောအတန်း": "duplicate row",
    "အကျဉ်းချုပ်": "summary",
    "ခွဲခြမ်းစိတ်ဖြာ": "analyze",
    "ပြပေးပါ": "show",
    "ပြပါ": "show",
}


def contains_burmese(text: str) -> bool:
    """Return True when text contains Myanmar Unicode characters."""

    return bool(
        re.search(
            r"[\u1000-\u109F\uAA60-\uAA7F]",
            text,
        )
    )


def contains_english(text: str) -> bool:
    """Return True when text contains English alphabetic characters."""

    return bool(re.search(r"[A-Za-z]", text))


def detect_language(text: str) -> str:
    """Detect English, Myanmar, mixed-language, or unknown text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    has_burmese = contains_burmese(text)
    has_english = contains_english(text)

    if has_burmese and has_english:
        return "mixed"

    if has_burmese:
        return "my"

    if has_english:
        return "en"

    return "unknown"


def normalize_myanmar_digits(text: str) -> str:
    """Convert Myanmar digits to ASCII digits."""

    return text.translate(MYANMAR_DIGIT_MAP)


def normalize_question(text: str) -> str:
    """Normalize an English, Myanmar, or mixed analytics question."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalize_myanmar_digits(normalized)
    normalized = normalized.replace("\u00A0", " ")
    normalized = normalized.replace("\u200B", "")
    normalized = normalized.lower().strip()

    sorted_terms = sorted(
        BURMESE_TERM_MAP.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for burmese_term, english_term in sorted_terms:
        normalized = normalized.replace(
            burmese_term,
            f" {english_term} ",
        )

    normalized = re.sub(
        r"[၊။,.!?;:]+",
        " ",
        normalized,
    )
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()
