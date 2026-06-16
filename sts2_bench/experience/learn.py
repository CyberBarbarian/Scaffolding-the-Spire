"""CLI entry point for reflecting traces and promoting new experience memories."""

from __future__ import annotations

from .reflection import learn_main


if __name__ == "__main__":
    raise SystemExit(learn_main())
