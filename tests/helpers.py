"""Test helpers shared by the test modules."""

from rag_engine.models import Hit


def section(heading: str, text: str, distance: float) -> Hit:
    """A retrieved section as the knowledge layer returns it: a label line, then the text."""
    return Hit("doc.md", heading, f"Doc > {heading}\n\n{text}", distance)
