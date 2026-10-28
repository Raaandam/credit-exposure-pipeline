"""Write measured results into README placeholder sections.

The README never contains hand-typed metrics: sections between
``<!-- RESULTS:<name>:start -->`` and ``<!-- RESULTS:<name>:end -->`` are overwritten by the
evaluation / timing scripts with whatever they actually measured, including sample sizes.
"""
from __future__ import annotations

import re
from pathlib import Path

from exposure_pipeline.config import PROJECT_ROOT

README = PROJECT_ROOT / "README.md"


def update_readme_section(name: str, markdown: str, readme: Path = README) -> bool:
    start, end = f"<!-- RESULTS:{name}:start -->", f"<!-- RESULTS:{name}:end -->"
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        return False
    new = pattern.sub(lambda _: f"{start}\n{markdown.strip()}\n{end}", text)
    readme.write_text(new, encoding="utf-8")
    return True
