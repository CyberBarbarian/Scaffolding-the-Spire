"""Shared checks for model-visible localization integrity."""

from __future__ import annotations

import re
from collections.abc import Iterable


MISSING_LOCALIZATION_MARKER = "Localization dir not found:"
_RAW_LOCALIZATION_TOKEN = re.compile(
    r"(?<![\w.])(?:[A-Za-z_][A-Za-z0-9_]*\.)+(?:title|description|name)(?![\w.])"
)


def unresolved_localization_tokens(texts: Iterable[str]) -> list[str]:
    """Return unresolved localization keys exposed to the model or user."""

    return sorted(
        {
            match.group(0)
            for text in texts
            if isinstance(text, str)
            for match in _RAW_LOCALIZATION_TOKEN.finditer(text)
        }
    )


def localization_integrity(texts: Iterable[str], *, stderr: str = "") -> dict[str, object]:
    """Summarize whether a run exposed complete localized text."""

    tokens = unresolved_localization_tokens(texts)
    missing_directory_warning = MISSING_LOCALIZATION_MARKER in stderr
    return {
        "valid": not tokens and not missing_directory_warning,
        "missing_directory_warning": missing_directory_warning,
        "unresolved_tokens": tokens,
    }
