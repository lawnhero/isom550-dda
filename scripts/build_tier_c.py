#!/usr/bin/env python3
"""
Build the Tier C index: long-form course documents.

Tier C holds the prose that does not fit anywhere else -- what was said in each
class, and what each assignment actually asks for. Both come from Canvas via
course_data/schedule.json, so nothing here is hand-maintained and nothing goes
stale on its own.

    python scripts/sync_canvas.py --course-id 162137     # refresh the source
    python scripts/build_tier_c.py                       # rebuild the index

Why this is a separate index from Tier B (data/contents):
  Tier B page_content is a QUESTION and the answer lives in metadata; Tier C
  page_content is a PASSAGE. Similarity scores across those two shapes are not
  comparable, so blending them in one collection makes ranking meaningless.

Chunking rule, which is the opposite of Tier B's:
  Tier B  -> one row = one chunk, never split.
  Tier C  -> prose, split only when long, and every chunk carries the document
             title as a header. A chunk that says "Slide 3: comment on the
             correlation (2 points)" with no title attached is unusable; that
             orphaning is exactly what a naive splitter produces.
"""

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_SCHEDULE = Path("course_data/schedule.json")
DEFAULT_PERSIST = Path("data/tier_c")

# Documents shorter than this stay whole. Sized so a typical class recap or
# assignment brief is never split -- splitting a 900-character assignment into
# "objective" and "tasks" halves is strictly worse than one slightly long chunk.
MAX_WHOLE = 1800
# When a document does exceed that, aim for chunks around this size.
TARGET_CHUNK = 1200


def _ymd(iso):
    """ISO timestamp -> sortable int like 20260724, or 0 when undated.

    Chroma metadata filters work cleanly on numbers, and "what did we cover in
    the last two weeks" is a range filter, not a similarity query.
    """
    if not iso:
        return 0
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", iso)
    return int(m.group(1) + m.group(2) + m.group(3)) if m else 0


def _split(text):
    """Split on blank lines, packing paragraphs up to TARGET_CHUNK."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) + 2 > TARGET_CHUNK:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks or [text]


def _emit(doc_id, title, body, meta, out):
    """Add one document, splitting only if long. Every chunk keeps the title."""
    body = (body or "").strip()
    if not body:
        return
    pieces = [body] if len(body) <= MAX_WHOLE else _split(body)
    total = len(pieces)
    for i, piece in enumerate(pieces):
        header = title if total == 1 else f"{title} (part {i + 1} of {total})"
        out.append(
            {
                "id": f"{doc_id}#{i}",
                "text": f"{header}\n\n{piece}",
                "metadata": {**meta, "chunk": i, "n_chunks": total},
            }
        )


def build_documents(schedule):
    """schedule.json -> list of {id, text, metadata}. Pure, so it is testable."""
    out = []

    for i, a in enumerate(schedule.get("announcements") or []):
        _emit(
            doc_id=f"ann-{i}",
            title=a.get("title", "").strip(),
            body=a.get("body", ""),
            meta={
                "doc_type": "announcement",
                "title": a.get("title", "").strip(),
                "url": a.get("url", ""),
                "ymd": _ymd(a.get("posted_utc")),
            },
            out=out,
        )

    for i, a in enumerate(schedule.get("assignments") or []):
        _emit(
            doc_id=f"asg-{i}",
            title=f"Assignment: {a.get('name', '').strip()}",
            body=a.get("instructions", ""),
            meta={
                "doc_type": "assignment",
                "title": a.get("name", "").strip(),
                "url": a.get("url", ""),
                "ymd": _ymd(a.get("due_utc")),
            },
            out=out,
        )

    return out


def main():
    ap = argparse.ArgumentParser(description="Build the Tier C document index.")
    ap.add_argument("--schedule", default=str(DEFAULT_SCHEDULE))
    ap.add_argument("--persist-dir", default=str(DEFAULT_PERSIST))
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be indexed, embed nothing")
    args = ap.parse_args()

    schedule = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
    docs = build_documents(schedule)
    if not docs:
        sys.exit(
            f"No indexable documents in {args.schedule}.\n"
            "Announcement bodies and assignment instructions both come from the "
            "Canvas sync -- run scripts/sync_canvas.py first."
        )

    by_type = {}
    for d in docs:
        by_type[d["metadata"]["doc_type"]] = by_type.get(d["metadata"]["doc_type"], 0) + 1
    chars = sum(len(d["text"]) for d in docs)
    print(f"{len(docs)} chunks  ({', '.join(f'{v} {k}' for k, v in sorted(by_type.items()))})")
    print(f"{chars:,} characters total, longest chunk {max(len(d['text']) for d in docs):,}")

    if args.dry_run:
        for d in docs[:5]:
            m = d["metadata"]
            print(f"\n  [{m['doc_type']}] {m['title']}  ymd={m['ymd']}  "
                  f"chunk {m['chunk'] + 1}/{m['n_chunks']}  {len(d['text'])}ch")
            print("   " + d["text"][:160].replace("\n", " ") + "...")
        return

    # Imported here so --dry-run works without the embedding stack installed.
    from chromadb import Settings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings

    persist = Path(args.persist_dir)
    if persist.exists():
        # Rebuild from scratch: ids are positional, so a removed announcement
        # would otherwise leave an orphaned chunk behind forever.
        import shutil

        shutil.rmtree(persist)

    Chroma.from_documents(
        documents=[
            Document(page_content=d["text"], metadata=d["metadata"], id=d["id"])
            for d in docs
        ],
        embedding=OpenAIEmbeddings(model=args.embedding_model),
        persist_directory=str(persist),
        client_settings=Settings(anonymized_telemetry=False),
    )
    print(f"\nwrote {persist}")


if __name__ == "__main__":
    main()
