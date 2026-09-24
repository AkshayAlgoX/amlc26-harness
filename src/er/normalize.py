"""Normalization module for business_name, business_address, and country.

Designed for noisy entity resolution with abbreviations, typos, and variations.
Keeps country open-set (never filters or assumes fixed country categories).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Set

import pandas as pd

# Legal suffixes and abbreviations mapping to canonical tokens
LEGAL_SUFFIX_MAP: Dict[str, str] = {
    "corp": "corporation",
    "corporation": "corporation",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "ltd": "limited",
    "limited": "limited",
    "pvt": "private",
    "private": "private",
    "llc": "llc",
    "llp": "llp",
    "co": "company",
    "company": "company",
    "gmbh": "gmbh",
    "sa": "sa",
    "sarl": "sarl",
    "sas": "sas",
    "plc": "plc",
    "holdings": "holdings",
    "group": "group",
    "enterprises": "enterprises",
    "services": "services",
    "solutions": "solutions",
    "technologies": "technologies",
}

# Common street/address abbreviations to canonical forms
ADDRESS_ABBR_MAP: Dict[str, str] = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "bldg": "building",
    "fl": "floor",
    "apt": "apartment",
    "ste": "suite",
    "hwy": "highway",
    "pkwy": "parkway",
    "po": "po",
    "box": "box",
    "opp": "opposite",
    "nr": "near",
}

# Country canonicalization aliases (open-set: preserves unknown countries unchanged)
COUNTRY_MAP: Dict[str, str] = {
    "us": "US",
    "usa": "US",
    "united states": "US",
    "united states of america": "US",
    "in": "India",
    "ind": "India",
    "india": "India",
    "fr": "France",
    "fra": "France",
    "france": "France",
}

RE_NON_ALPHANUM = re.compile(r"[^\w\s]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"\b\d+\b")


def clean_text(text: object) -> str:
    """Normalize unicode, convert & to 'and', lowercase, and trim."""
    if text is None or pd.isna(text):
        return ""
    s = str(text)
    # Normalize unicode accents (NFKD)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    # Replace ampersands
    s = s.replace("&", " and ")
    s = s.lower().strip()
    return s


def tokenize(text: str) -> List[str]:
    """Tokenize cleaned text on whitespace after removing punctuation."""
    text_clean = RE_NON_ALPHANUM.sub(" ", text)
    return [t for t in RE_WHITESPACE.split(text_clean) if t]


def normalize_name(name: object) -> str:
    """Normalize business name: handle &, punctuation, suffixes, and whitespace."""
    s = clean_text(name)
    if not s:
        return ""

    tokens = tokenize(s)
    norm_tokens = [LEGAL_SUFFIX_MAP.get(t, t) for t in tokens]
    return " ".join(norm_tokens)


def extract_name_core(name: object) -> str:
    """Extract core business name by removing legal entity suffixes and company keywords."""
    s = clean_text(name)
    if not s:
        return ""

    tokens = tokenize(s)
    # Filter out common legal suffixes
    core = [t for t in tokens if t not in LEGAL_SUFFIX_MAP]
    if not core:
        core = tokens  # fallback if all tokens were legal words
    return " ".join(core)


def normalize_address(address: object) -> str:
    """Normalize business address: expand abbreviations, clean punctuation."""
    s = clean_text(address)
    if not s:
        return ""

    tokens = tokenize(s)
    norm_tokens = [ADDRESS_ABBR_MAP.get(t, t) for t in tokens]
    return " ".join(norm_tokens)


def extract_numbers(text: object) -> List[str]:
    """Extract list of numeric strings (postal codes, plot numbers, street numbers)."""
    if text is None or pd.isna(text):
        return []
    return RE_NUMBERS.findall(str(text))


def normalize_country(country: object) -> str:
    """Normalize country string.

    Open-set: standardizes known aliases but leaves any unseen country as clean string.
    """
    if country is None or pd.isna(country):
        return ""
    c = str(country).strip()
    lower = c.lower()
    return COUNTRY_MAP.get(lower, c)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add standardized normalized columns to an entity DataFrame."""
    out = df.copy()

    out["norm_business_name"] = out["business_name"].apply(normalize_name)
    out["core_business_name"] = out["business_name"].apply(extract_name_core)
    out["norm_business_address"] = out["business_address"].apply(normalize_address)
    out["norm_country"] = out["country"].apply(normalize_country)

    out["name_tokens"] = out["norm_business_name"].apply(tokenize)
    out["core_tokens"] = out["core_business_name"].apply(tokenize)
    out["address_tokens"] = out["norm_business_address"].apply(tokenize)
    out["address_numbers"] = out["business_address"].apply(extract_numbers)

    return out
