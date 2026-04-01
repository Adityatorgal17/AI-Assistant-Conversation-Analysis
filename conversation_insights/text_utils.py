from __future__ import annotations

import re
from collections import Counter
from urllib.parse import urlparse


MARKDOWN_LINK_RE = re.compile(r"\[\*\*(.+?)\*\*\]\((https?://[^\s)]+)\)")
URL_RE = re.compile(r"https?://[^\s)\]]+")
SLUG_RE = re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+){2,}\b")
EMBEDDED_JSON_RE = re.compile(r"\s*(?:End of stream\s*)?(\{(?:\"type\"|\"data\"|\"products\"))", re.IGNORECASE)

HINGLISH_TOKENS = {
    "mujhe",
    "krna",
    "nhi",
    "kya",
    "aap",
    "hai",
    "kaam",
    "mujh",
    "diya",
    "aya",
    "abhi",
    "samajhiye",
    "namaste",
}

ORDER_KEYWORDS = (
    "order",
    "refund",
    "cancel",
    "track",
    "edit",
    "delivery",
    "shipment",
    "shipped",
    "delivered",
)

PRODUCT_DISCOVERY_KEYWORDS = (
    "suggest",
    "recommend",
    "best for",
    "which is best",
    "weight loss",
    "weight lose",
    "oily skin",
    "acne",
    "dark circles",
    "pigmentation",
    "live in conditioner",
    "how to use",
)

PRODUCT_PAGE_PREFIXES = (
    "what are the key ingredients?",
    "how long does it take to see results?",
    "can this be used in a daily routine?",
    "how do i use it daily?",
    "who should use this product?",
    "what health benefits does it offer?",
    "is it safe for long-term use?",
)

LOGIN_HINTS = (
    "account/login",
    "sign in to your account",
    "log in",
    "login",
    "sign in",
    "account access",
)


def clean_agent_text(raw_text: str) -> str:
    marker_match = EMBEDDED_JSON_RE.search(raw_text)
    if marker_match:
        cleaned = raw_text[: marker_match.start()]
    else:
        cleaned = raw_text
    return normalize_whitespace(cleaned)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return normalize_whitespace(text)


def detect_language_style(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return "unknown"
    tokens = set(normalized.split())
    hinglish_hits = len(tokens & HINGLISH_TOKENS)
    if hinglish_hits >= 2:
        return "hinglish"
    if hinglish_hits == 1:
        return "mixed"
    return "english"


def extract_product_names(text: str) -> list[str]:
    names = [
        normalize_whitespace(name)
        for name, link in MARKDOWN_LINK_RE.findall(text)
        if is_product_link(link)
    ]
    if text.startswith("Viewed product:"):
        names.append(normalize_whitespace(text.replace("Viewed product:", "", 1)))
    if text.startswith("Requested similar products to"):
        names.append(normalize_whitespace(text.replace("Requested similar products to", "", 1)))
    deduped = []
    seen = set()
    for name in names:
        if name and name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def extract_links(text: str) -> list[str]:
    links = URL_RE.findall(text)
    for _, link in MARKDOWN_LINK_RE.findall(text):
        if link not in links:
            links.append(link)
    return links


def is_product_link(value: str) -> bool:
    lowered = value.lower()
    parsed = urlparse(value)
    path = parsed.path.lower()
    return any(
        token in lowered or token in path
        for token in ("/products/", "/product/", "/item/", "viewed product:", "requested similar products to")
    )


def is_login_link(value: str) -> bool:
    lowered = value.lower()
    parsed = urlparse(value)
    path = parsed.path.lower()
    return any(token in lowered or token in path for token in LOGIN_HINTS)


def infer_brand_names_by_widget(grouped_links: dict[str, list[str]]) -> dict[str, str | None]:
    ignored_hosts = {
        "cdn.shopify.com",
        "api.whatsapp.com",
        "wa.me",
        "www.facebook.com",
        "www.instagram.com",
        "tracker.shadowfax.in",
        "www.delhivery.com",
        "www.bluedart.com",
        "shiprocket.co",
        "pikndel.com",
    }
    inferred: dict[str, str | None] = {}
    for widget_id, links in grouped_links.items():
        host_counts: Counter[str] = Counter()
        for link in links:
            host = urlparse(link).netloc.lower()
            if not host or host in ignored_hosts:
                continue
            host_counts[host] += 1
        if not host_counts:
            inferred[widget_id] = None
            continue
        best_host = host_counts.most_common(1)[0][0]
        parts = best_host.split(".")
        if len(parts) >= 3 and parts[-2] in {"co", "com", "org", "net"}:
            brand_token = parts[-3]
        else:
            brand_token = parts[-2] if len(parts) >= 2 else best_host
        inferred[widget_id] = brand_token.replace("-", " ").title()
    return inferred


def has_slug_like_token(text: str) -> bool:
    return bool(SLUG_RE.search(text.lower()))
