"""Jinja2 environment for evaluation prompts.

Mirrors `rules.prompts` but rooted in `evals/templates`, so judge prompts stay
out of the production prompt directory -- nothing here runs in the triage path.
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
    return _environment().get_template(template_name).render(**context).strip()
