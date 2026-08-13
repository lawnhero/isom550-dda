#!/usr/bin/env python3
"""
Build the Tier B concept index from course_data/concepts.csv.

    python scripts/build_concepts.py --dry-run   # report + lint, embed nothing
    python scripts/build_concepts.py             # write the index
    python scripts/build_concepts.py --strict    # exit non-zero on any lint error

ONE VECTOR PER CONCEPT, embedded from `title` + `body`.

    `topic`, `managerial_phrasing` and `common_mistake` are NOT embedded. They
    ride along as metadata and come back with the concept, so the model gets
    the memo language and the misconception correction whenever the concept is
    retrieved -- they just do not compete for the match.

WHAT FILTERS, WHAT DOES NOT

    module   the only search filter. The router names one; retrieval narrows to
             it. Eight controlled values, defined by utils.concept_taxonomy.
    status   `planned` rows are declared but never indexed. They exist so the
             coverage report can say a module is empty instead of that fact
             being invisible.

    Everything else is carried, not filtered. `topic` in particular is free
    text with no controlled vocabulary -- useful to read, unusable to select on.

COVERAGE IS DECLARED, NOT INFERRED

    Retrieval bands a query by its distance to the nearest chunk, which measures
    topical similarity and not whether anything was written. A question about
    logistic regression sits close to every regression concept in the index and
    scores like a good match; the old CSV rated it 1.371 -- STRONG -- against
    descriptive-statistics content. Only an inventory can know the difference,
    which is what the `planned` rows and the coverage report are for.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.concept_taxonomy import list_modules, load_rows  # noqa: E402

DEFAULT_CSV = Path("course_data/concepts.csv")
DEFAULT_PERSIST = Path("data/concepts")

# Metadata that rides on the vector and comes back with it.
CARRIED = ("topic", "managerial_phrasing", "common_mistake", "updated")


def _clean(value: str) -> str:
    return (value or "").strip()


def build_documents(rows):
    """CSV rows -> [{id, text, metadata}]. Pure, so it is testable.

    Skips `planned` rows and anything with no body: an empty vector would be
    indistinguishable from a real concept at query time, which is the exact
    confusion this index exists to remove.
    """
    out = []
    for row in rows:
        cid, body = _clean(row.get("id")), _clean(row.get("body"))
        title, module = _clean(row.get("title")), _clean(row.get("module"))
        if _clean(row.get("status")) == "planned" or not body:
            continue
        out.append({
            "id": cid,
            # Title first: it is the most compressed statement of what the
            # concept is, and putting it in front of the body gives the vector
            # a clear subject rather than opening mid-explanation.
            "text": f"{title}\n\n{body}",
            "metadata": {
                "concept_id": cid,
                "title": title,
                "module": module,
                "status": _clean(row.get("status")) or "core",
                "body": body,
                **{k: _clean(row.get(k)) for k in CARRIED},
            },
        })
    return out


def lint(rows):
    """Problems that break retrieval silently. Returns a list of messages."""
    problems = []
    seen = {}
    known = {m["id"] for m in list_modules()}

    # Row number, not file line: bodies span lines, so the two diverge fast.
    for i, row in enumerate(rows, start=1):
        cid = _clean(row.get("id"))
        where = f"row {i} ({cid or 'no id'})"
        if not cid:
            problems.append(f"{where}: missing id")
            continue
        if cid in seen:
            problems.append(f"{where}: duplicate id, first seen at line {seen[cid]}")
        seen[cid] = i

        module = _clean(row.get("module"))
        if not module:
            problems.append(f"{where}: missing module")
        elif module not in known:
            problems.append(f"{where}: module {module!r} is not a known module id")
        elif module != row.get("module", ""):
            problems.append(f"{where}: module has surrounding whitespace")
        elif " " in module:
            # The bug that started this check: one row said "simple regression"
            # while every other said "simple-regression", so its filter matched
            # nothing and the concept was unreachable.
            problems.append(f"{where}: module {module!r} contains a space; ids use hyphens")

        status = _clean(row.get("status"))
        if status not in {"core", "extension", "planned"}:
            problems.append(f"{where}: status {status!r} is not core/extension/planned")
        if status != "planned" and not _clean(row.get("body")):
            problems.append(f"{where}: status is {status!r} but body is empty")
        if status == "planned" and _clean(row.get("body")):
            problems.append(f"{where}: status is 'planned' but a body is written")
        if not _clean(row.get("title")):
            problems.append(f"{where}: missing title")
        if _clean(row.get("body")) and not _clean(row.get("updated")):
            problems.append(f"{where}: written concept has no `updated` date")
        # Spaces and tabs only -- \s would match the blank line between
        # paragraphs, which is deliberate formatting in most bodies.
        if re.search(r"[ \t]{2,}", _clean(row.get("body")) or ""):
            problems.append(f"{where}: body has a double space")
    return problems


def report(rows, docs):
    print(f"{len(rows)} rows -> {len(docs)} vectors (one per written concept)\n")
    print("COVERAGE BY MODULE")
    for m in list_modules():
        mark = "  " if m["n_written"] else "!!"
        planned = f"  (+{m['n_planned']} planned)" if m["n_planned"] else ""
        print(f"  {mark} {m['id']:26s} {m['n_written']:2d} written{planned}")

    empty = [m for m in list_modules() if not m["n_written"]]
    if empty:
        print(f"\n!! {len(empty)} modules have nothing written:")
        for m in empty:
            print(f"     {m['id']} — {m['label']}")
        print("   The router is told not to use these. Retrieval alone cannot\n"
              "   detect them: their questions score like good matches.")

    filled = sum(1 for d in docs if d["metadata"]["managerial_phrasing"])
    mistakes = sum(1 for d in docs if d["metadata"]["common_mistake"])
    print(f"\ncarried on retrieval: {filled} managerial_phrasing, {mistakes} common_mistake")


def main():
    ap = argparse.ArgumentParser(description="Build the Tier B concept index.")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--persist-dir", default=str(DEFAULT_PERSIST))
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on lint errors")
    args = ap.parse_args()

    rows = load_rows(Path(args.csv))
    if not rows:
        sys.exit(f"No rows in {args.csv}")
    docs = build_documents(rows)
    if not docs:
        sys.exit(f"No written concepts in {args.csv} -- every row is planned or empty.")

    report(rows, docs)

    problems = lint(rows)
    if problems:
        print(f"\nLINT ({len(problems)})")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nLINT clean")

    if args.dry_run:
        print("\nSAMPLE")
        for d in docs[:3]:
            m = d["metadata"]
            print(f"  {m['concept_id']:32s} [{m['module']}] {d['text'][:58]!r}...")
        return
    if args.strict and problems:
        sys.exit("\nstrict: refusing to build with lint errors")

    from chromadb import Settings
    from dotenv import load_dotenv
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings

    load_dotenv()
    persist = Path(args.persist_dir)
    if persist.exists():
        # Rebuild from scratch: a concept deleted from the CSV would otherwise
        # linger in the index forever.
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
    print("Re-derive the abstention thresholds before trusting them:\n"
          f"    python scripts/calibrate_retrieval.py --probe --db {persist}")


if __name__ == "__main__":
    main()
