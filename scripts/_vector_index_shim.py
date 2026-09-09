"""Shared plumbing for the two build scripts.

Both are break-glass wrappers over vat-research/vector_index. That checkout is
private and instructor-only, which is the same access boundary as the OpenAI
key a rebuild already requires -- a cloner of this public repo could not run
one regardless.
"""

import os
import sys
from pathlib import Path

DEFAULT_SYNC_HOME = Path.home() / "AI_projects" / "vat-research"


def _load():
    home = Path(os.getenv("CANVAS_SYNC_HOME") or DEFAULT_SYNC_HOME).expanduser()
    if not (home / "vector_index").is_dir():
        sys.exit(
            f"vector_index not found at {home}.\n"
            "Set CANVAS_SYNC_HOME to your vat-research checkout, or rebuild through "
            "the vector-index MCP, which does not need this script."
        )
    sys.path.insert(0, str(home))
    try:
        from dotenv import load_dotenv

        load_dotenv()  # this repo's .env carries OPENAI_API_KEY
    except ImportError:
        pass
    import course_registry as registry
    from vector_index import adapters, engine

    return registry, adapters, engine


def run_build(course_name, kind, dry_run=False, strict=False):
    registry, adapters, engine = _load()
    course = registry.resolve(course_name)
    index = course.index(kind)

    result = engine.build(index, adapters.get(kind), dry_run=dry_run)

    print(f"{result['n_chunks']} chunks from {result['source']}")
    print(f"  {result['total_characters']:,} characters, "
          f"longest chunk {result['longest_chunk']:,}")
    if result.get("by_doc_type"):
        print("  " + ", ".join(f"{v} {k}" for k, v in sorted(result["by_doc_type"].items())))
    for row in result.get("coverage") or []:
        mark = "!!" if not row["written"] else "  "
        extra = f"  (+{row['planned']} planned)" if row["planned"] else ""
        print(f"  {mark} {row['module']:26s} {row['written']:2d} written{extra}")
    if result.get("outline"):
        print("\n  Pills the app will offer (module -> topics):")
        for module in result["outline"]:
            topics = ", ".join(
                f"{t['label']} ({t['n_concepts']})" for t in module["topics"]
            )
            print(f"    {module['module']}: {topics}")
    if result.get("empty_modules"):
        print(f"\n!! {len(result['empty_modules'])} modules have nothing written: "
              + ", ".join(result["empty_modules"]))
        print("   The router is told not to use these. Retrieval alone cannot")
        print("   detect them: their questions score like good matches.")

    for w in result.get("warnings") or []:
        print(f"\n!! {w}")

    problems = result.get("problems") or []
    if problems:
        print(f"\nLINT ({len(problems)})")
        for p in problems:
            print(f"  - {p}")
        if strict:
            sys.exit("\nstrict: refusing to build with lint errors")

    if dry_run:
        print("\ndry run: nothing embedded")
        return
    print(f"\nwrote {result['persist']}  (built_at {result['built_at']})")
