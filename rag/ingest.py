"""Ingestion pipeline: load the knowledge base → split by section → embed → persist in Chroma.

Run directly to (re)build the index:  python -m rag.ingest
"""

import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from rag import config


def split_doc(text: str, source: str) -> list[Document]:
    """Split one knowledge-base file into one chunk per section.

    The docs are structured: a `Title:`/`Category:` header, then blank-line separated
    blocks that start with a heading ending in ":" (e.g. "Paintwork Warranty:").
    Splitting on those boundaries keeps each fact next to the heading that gives it
    meaning ("3 years" belongs to *paintwork*, not to the standard warranty), which
    fixed-size character chunking cannot guarantee. Each chunk is prefixed with
    "Title › Section" so both BM25 and the embedding see that context.
    """
    blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", text.strip())]
    header = dict(re.findall(r"^(Title|Category):\s*(.+)$", blocks[0], re.M))
    title, category = header["Title"].strip(), header["Category"].strip()

    chunks, lead_in = [], ""
    for block in blocks[1:]:
        first, _, body = block.partition("\n")
        if first.endswith(":") and not body:
            # A lone lead-in line ("We offer multiple support channels:") → glue onto the next block.
            lead_in += first + "\n"
            continue
        if first.endswith(":"):
            section = first.rstrip(":").strip()
        else:
            section, body = "General", block
        chunks.append(Document(
            page_content=f"{title} › {section}\n{lead_in}{body.strip()}",
            metadata={"source": source, "title": title, "category": category, "section": section},
            id=f"{source}#{len(chunks)}",
        ))
        lead_in = ""
    # ponytail: no size-based sub-splitting; every section here is < 500 chars.
    # Add RecursiveCharacterTextSplitter per section if long documents arrive.
    return chunks


def load_chunks() -> list[Document]:
    """Parse every .txt file in the knowledge base into section chunks."""
    return [c for path in sorted(config.KB_DIR.glob("*.txt"))
            for c in split_doc(path.read_text(encoding="utf-8"), path.stem)]


def embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)


def open_store(chunks: list[Document]) -> Chroma:
    """Open the persisted Chroma index, rebuilding it if missing or out of date with `chunks`."""
    store = Chroma(collection_name=_collection_name(), embedding_function=embeddings(),
                   persist_directory=str(config.CHROMA_DIR))
    # ponytail: staleness check compares chunk ids only; edits inside a section need `python -m rag.ingest`.
    if set(store.get()["ids"]) != {c.id for c in chunks}:
        store.reset_collection()
        store.add_documents(chunks, ids=[c.id for c in chunks])
    return store


def _collection_name() -> str:
    # One collection per embedding model, so switching models never mixes vector spaces.
    return "kb_" + re.sub(r"[^A-Za-z0-9]+", "_", config.EMBED_MODEL).strip("_")


if __name__ == "__main__":
    shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
    chunks = load_chunks()
    open_store(chunks)
    print(f"Indexed {len(chunks)} chunks from {len({c.metadata['source'] for c in chunks})} documents "
          f"with {config.EMBED_MODEL} → {config.CHROMA_DIR}")
    for c in chunks:
        print(f"  {c.id:34} {c.metadata['title']} › {c.metadata['section']}")
