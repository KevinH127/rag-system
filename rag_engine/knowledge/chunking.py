import re
from dataclasses import dataclass

MAX_CHARS = 1500


@dataclass(frozen=True)
class Chunk:
    ord: int
    heading: str
    content: str


def chunk_markdown(text: str, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Split markdown into one chunk per `##` section.

    Each chunk is prefixed with the document title and section heading so it
    stays meaningful on its own. Oversized sections are split on blank lines.
    """
    title = ""
    sections: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        elif line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
        elif line.strip() and not title:
            continue
        elif line.strip():
            sections.append((title, [line]))

    chunks: list[Chunk] = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        for part in _split(body, max_chars):
            label = f"{title} > {heading}" if heading != title else title
            chunks.append(Chunk(len(chunks), heading, f"{label}\n\n{part}"))
    return chunks


def _split(body: str, max_chars: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    parts: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", body):
        if current and len(current) + len(para) + 2 > max_chars:
            parts.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        parts.append(current)
    return parts
