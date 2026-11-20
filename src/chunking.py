from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "Header 1"), ("##", "Header 2")]


@dataclass
class Chunk:
    text: str
    source_file: str
    section: str
    chunk_id: str = ""
    metadata: dict = field(default_factory=dict)


def load_corpus(corpus_dir: str | Path) -> list[tuple[str, str]]:
    docs = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def _finish(chunks: list[Chunk]) -> list[Chunk]:
    for i, chunk in enumerate(chunks):
        chunk.chunk_id = f"{chunk.source_file}:{chunk.section}:{i}"
    return chunks


def header_aware_chunks(docs: list[tuple[str, str]], chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    # split on structure first so a chunk never mixes two sections, then cap size
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks: list[Chunk] = []
    for name, text in docs:
        for doc in md_splitter.split_text(text):
            section = doc.metadata.get("Header 2") or doc.metadata.get("Header 1") or "intro"
            for piece in size_splitter.split_text(doc.page_content):
                if piece.strip():
                    chunks.append(Chunk(text=piece, source_file=name, section=section))
    return _finish(chunks)


def fixed_chunks(docs: list[tuple[str, str]], chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks: list[Chunk] = []
    for name, text in docs:
        for piece in splitter.split_text(text):
            if piece.strip():
                chunks.append(Chunk(text=piece, source_file=name, section="fixed"))
    return _finish(chunks)
