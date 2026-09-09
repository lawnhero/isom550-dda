#!/usr/bin/env python3
"""
Sync course facts from Canvas into course_data/schedule.json.

INSTRUCTOR-SIDE ONLY. This script needs a Canvas API token; the Streamlit app
does not, and must never be given one. A Canvas personal access token inherits
every permission of the user who created it -- rosters, grades, submissions,
quiz answer keys, and every other course you teach. Keeping it here, offline,
and out of the deployed app is the entire point of this design.

WHERE THE LOGIC LIVES NOW

    The fetch/strip/validate implementation moved to vat-research/canvas_sync/
    when a second course repo (isom352) needed the same pipeline. Keeping one
    copy is the point: two copies of a due-date formatter drift, and the whole
    reason dates are computed at sync time is that they have to be exact.

    The normal way to run a sync is the `canvas-course-sync` skill, which calls
    the Canvas MCP's sync_course_snapshot tool and writes this file directly.
    This script is the break-glass path for when the MCP is not running.

    It therefore needs the vat-research checkout on disk. That repo is private
    and instructor-only, which is the same access boundary as the Canvas token
    this script already requires -- a cloner of this public repo could not run
    a sync regardless.

Usage:
    export CANVAS_API_TOKEN=...          # or put it in vat-research/.env
    python scripts/sync_canvas.py --course-id 165666
    git diff course_data/schedule.json   # REVIEW before committing
    git add course_data/schedule.json && git commit

    # if vat-research lives somewhere else:
    export CANVAS_SYNC_HOME=/path/to/vat-research

The design rules this pipeline enforces (endpoint allowlist, field allowlist,
unpublished items dropped, `verifier=` token scrubbing, local-time formatting,
deterministic ordering, no LLM in the path) are documented and tested in
canvas_sync/snapshot.py and tests/test_canvas_snapshot.py over there.
"""

import argparse
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

DEFAULT_TZ = "America/New_York"
DEFAULT_OUT = Path("course_data/schedule.json")
DEFAULT_SYNC_HOME = Path.home() / "AI_projects" / "vat-research"


def _load_snapshot_module():
    home = Path(os.getenv("CANVAS_SYNC_HOME") or DEFAULT_SYNC_HOME).expanduser()
    if not (home / "canvas_sync" / "snapshot.py").exists():
        sys.exit(
            f"canvas_sync not found at {home}.\n"
            "Set CANVAS_SYNC_HOME to your vat-research checkout, or run the sync "
            "through the Canvas MCP (sync_course_snapshot), which does not need this script."
        )
    sys.path.insert(0, str(home))
    try:
        from dotenv import load_dotenv

        load_dotenv(home / ".env")
    except ImportError:
        pass
    from canvas_sync import snapshot

    return snapshot


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--course-id", required=True)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--timezone", default=DEFAULT_TZ)
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args()

    snapshot = _load_snapshot_module()

    snap = snapshot.build_snapshot(
        snapshot.make_session(), args.course_id, ZoneInfo(args.timezone)
    )

    # This repo is PUBLIC. The validator's leak and verifier-token checks are
    # sized for that; see the `public` argument in canvas_sync/snapshot.py.
    errs = snapshot.validate(snap, public=True)
    if errs:
        for e in errs:
            print(f"VALIDATION: {e}", file=sys.stderr)
        sys.exit("refusing to write an invalid snapshot")

    text = snapshot.render(snap)
    if args.dry_run:
        print(text)
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    c = snap["course"]
    print(
        f"wrote {out}  ({c['term']})\n"
        f"  {len(snap['assignments'])} assignments, {len(snap['modules'])} modules, "
        f"{len(snap['pages'])} pages, {len(snap['announcements'])} announcements\n"
        f"  next: git diff {out}   <-- review before committing"
    )


if __name__ == "__main__":
    main()
