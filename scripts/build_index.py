import argparse
from pathlib import Path
from typing import List

from chromadb import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader


def load_documents(source_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif suffix in [".txt", ".md"]:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
        elif suffix == ".csv":
            docs.extend(CSVLoader(str(path)).load())
    return docs


def enrich_metadata(docs: List[Document]) -> List[Document]:
    for idx, doc in enumerate(docs):
        metadata = doc.metadata or {}
        if "source" not in metadata:
            metadata["source"] = metadata.get("file_path", f"document_{idx}")
        metadata["chunk_id"] = idx
        doc.metadata = metadata
    return docs


def build_index(source_dir: Path, persist_dir: Path, embedding_model: str, chunk_size: int, chunk_overlap: int):
    docs = load_documents(source_dir)
    if not docs:
        raise ValueError(f"No documents found in {source_dir}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)
    split_docs = enrich_metadata(split_docs)

    embeddings = OpenAIEmbeddings(model=embedding_model)
    Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        client_settings=Settings(anonymized_telemetry=False),
    )
    print(f"Indexed {len(split_docs)} chunks into {persist_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build Chroma index for course materials.")
    parser.add_argument("--source", required=True, help="Directory with source docs (pdf/txt/md/csv).")
    parser.add_argument("--persist-dir", required=True, help="Output Chroma directory.")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=180)
    args = parser.parse_args()

    build_index(
        source_dir=Path(args.source),
        persist_dir=Path(args.persist_dir),
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
