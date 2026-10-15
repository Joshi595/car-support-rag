"""Unit tests for the pure logic (no Ollama needed): section splitting, RRF fusion, citation parsing."""

from langchain_core.documents import Document

from rag.graph import cited_sources
from rag.ingest import load_chunks, split_doc
from rag.retrieval import rrf

SAMPLE = """Title: Warranty Coverage\r
Category: Warranty\r
\r
We provide coverage for all new vehicles:\r
\r
Standard Warranty:\r
- Duration: 2 years\r
\r
Paintwork Warranty:\r
- Duration: 3 years\r
\r
All claims go through authorized partners.\r
"""


def test_split_by_section():
    chunks = split_doc(SAMPLE.replace("\r\n", "\n"), "doc_x")
    assert [c.metadata["section"] for c in chunks] == ["Standard Warranty", "Paintwork Warranty", "General"]
    # lead-in line is glued to the next section, title header is prefixed, fact stays with its section
    assert chunks[0].page_content.startswith("Warranty Coverage › Standard Warranty\nWe provide coverage")
    assert "3 years" in chunks[1].page_content and "3 years" not in chunks[0].page_content
    assert [c.id for c in chunks] == ["doc_x#0", "doc_x#1", "doc_x#2"]


def test_real_knowledge_base_parses():
    chunks = load_chunks()
    assert len({c.metadata["source"] for c in chunks}) == 6
    assert len({c.id for c in chunks}) == len(chunks)
    assert any(c.metadata["section"] == "Electric Vehicle Battery Warranty" for c in chunks)


def _doc(i):
    return Document(page_content=f"chunk {i}", id=str(i))


def test_rrf_rewards_agreement():
    a, b, c = _doc("a"), _doc("b"), _doc("c")
    fused = rrf([[a, b, c], [b, c]])  # b ranks high in both lists → beats a, which only one list found
    assert [d.id for d in fused] == ["b", "c", "a"]


def test_cited_sources():
    docs = [_doc(i) for i in range(3)]
    found = cited_sources("Two years [1]. Paint is three [1, 3]; bogus [9].", docs)
    assert [n for n, _ in found] == [1, 3]
    assert cited_sources("no citations here", docs) == []
