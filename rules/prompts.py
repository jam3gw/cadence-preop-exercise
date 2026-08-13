"""Jinja2 prompt templates.

Prompts live in `rules/templates/*.j2` rather than inline string literals so
that prompt copy is versioned as its own artifact -- readable in diffs, and
editable without touching call-site code.

`StrictUndefined` is intentional: a typo'd or missing template variable raises
instead of silently rendering an empty string into a prompt.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@lru_cache(maxsize=1)
def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
        autoescape=False,
    )


def render(template_name: str, /, **context: object) -> str:
    """Render `rules/templates/<template_name>` and strip surrounding blanks."""

    return _environment().get_template(template_name).render(**context).strip()
