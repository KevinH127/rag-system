from rag_engine.knowledge.chunking import chunk_markdown

DOC = """# Fees

Intro text.

## How fees work
Pay after delivery.

## Refunds
No refunds.
"""


def test_one_chunk_per_section_with_title_prefix():
    chunks = chunk_markdown(DOC)
    assert [c.heading for c in chunks] == ["Fees", "How fees work", "Refunds"]
    assert chunks[1].content.startswith("Fees > How fees work\n\n")
    assert "Pay after delivery." in chunks[1].content
    assert [c.ord for c in chunks] == [0, 1, 2]


def test_oversized_section_is_split_on_paragraphs():
    body = "\n\n".join(f"paragraph {i} " + "x" * 80 for i in range(10))
    chunks = chunk_markdown(f"# T\n\n## S\n{body}", max_chars=300)
    assert len(chunks) > 1
    assert all(len(c.content) < 400 for c in chunks)


def test_empty_sections_are_skipped():
    assert chunk_markdown("# T\n\n## Empty\n\n## Full\nhello")[0].heading == "Full"
