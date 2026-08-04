#!/usr/bin/env python3
"""
Sync course facts from Canvas into course_data/schedule.json.

INSTRUCTOR-SIDE ONLY. This script needs a Canvas API token; the Streamlit app
does not, and must never be given one. A Canvas personal access token inherits
every permission of the user who created it -- rosters, grades, submissions,
quiz answer keys, and every other course you teach. Keeping it here, offline,
and out of the deployed app is the entire point of this design.

Usage:
    export CANVAS_API_TOKEN=...          # or put it in .env
    python scripts/sync_canvas.py --course-id 162137
    git diff course_data/schedule.json   # REVIEW before committing
    git add course_data/schedule.json && git commit

Design rules enforced below:
  1. Strict field allowlist. Output dicts are built key by key; Canvas
     responses are never dumped wholesale, so new upstream fields cannot
     leak in silently.
  2. Endpoint allowlist. _get() refuses any path not in ALLOWED_ENDPOINTS.
  3. Unpublished items are dropped at sync time and no `published` field is
     written, so a draft cannot reach students and presence == published.
  4. Due dates are stored pre-formatted in course-local time. Six of twelve
     deadlines in this course fall on a DIFFERENT CALENDAR DAY in UTC than
     students see; date math is done here, once, not by an LLM at inference.
  5. Deterministic ordering + indent=2, so `git diff` is reviewable.
  6. No LLM anywhere in this path. A due date has to be exact.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv()

CANVAS_BASE = os.getenv("CANVAS_BASE_URL", "https://canvas.emory.edu").rstrip("/")
DEFAULT_TZ = "America/New_York"
DEFAULT_OUT = Path("course_data/schedule.json")

# --------------------------------------------------------------------------
# SAFETY: the only endpoints this script may call.
#
# Deliberately absent, and never to be added:
#   /enrollments  /users  /students        -> rosters, names, SIS ids
#   /submissions  /gradebook  /grades      -> student work and scores
#   /quizzes/:id/questions                 -> answer keys
#   /discussion_topics (without the announcement filter) -> student posts
# --------------------------------------------------------------------------
ALLOWED_ENDPOINTS = (
    r"^/api/v1/courses/\d+$",
    r"^/api/v1/courses/\d+/assignments$",
    r"^/api/v1/courses/\d+/assignment_groups$",
    r"^/api/v1/courses/\d+/modules$",
    r"^/api/v1/courses/\d+/pages$",
    r"^/api/v1/courses/\d+/discussion_topics$",  # announcements only, see _get
)


_BLOCK_TAGS = ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "ul", "ol")


class _Stripper(HTMLParser):
    """Minimal HTML -> text. Canvas announcement and assignment bodies are HTML.

    Two behaviours that matter:
      - <script>/<style> content is dropped. Canvas injects a global CSS and JS
        tag into every assignment description; without this, raw CSS rules end
        up in the indexed text.
      - <a href> targets are kept, but only the path. Canvas file links carry a
        `verifier=` query token that grants UNAUTHENTICATED access to the file.
        This repo is public, so publishing one would publish the course file
        itself. The bare path still requires a Canvas login, which is correct.
    """

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = 0
        self._href = ""

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
            return
        if tag == "a":
            href = dict(attrs).get("href") or ""
            self._href = href.split("?", 1)[0]
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag == "a" and self._href:
            self.parts.append(f" ({self._href})")
            self._href = ""

    def handle_data(self, d):
        if self._skip:
            return
        self.parts.append(d)

    def text(self):
        out = "".join(self.parts)
        out = out.replace("\xa0", " ")
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r"\n\s*\n\s*\n+", "\n\n", out)
        return out.strip()


def html_to_text(html):
    if not html:
        return ""
    p = _Stripper()
    p.feed(html)
    return p.text()


def _get(session, path, params=None):
    """Paginated Canvas GET, restricted to the endpoint allowlist."""
    if not any(re.match(p, path) for p in ALLOWED_ENDPOINTS):
        raise RuntimeError(f"Refusing to call non-allowlisted endpoint: {path}")
    if path.endswith("/discussion_topics") and not (params or {}).get("only_announcements"):
        raise RuntimeError("discussion_topics may only be fetched with only_announcements=true")

    url = CANVAS_BASE + path
    params = dict(params or {})
    params.setdefault("per_page", 100)
    out = []
    while url:
        r = session.get(url, params=params, timeout=30)
        r.raise_for_status()
        body = r.json()
        out.extend(body if isinstance(body, list) else [body])
        url = (r.links.get("next") or {}).get("url")
        params = None  # the next link already carries them
    return out


def to_local(iso, tz):
    """Canvas ISO-8601 UTC -> (utc_iso, 'Mon Aug 3, 2026, 11:59 PM EDT')."""
    if not iso:
        return None, None
    utc = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    loc = utc.astimezone(tz)
    hour = loc.hour % 12 or 12
    pretty = (
        f"{loc.strftime('%a %b')} {loc.day}, {loc.year}, "
        f"{hour}:{loc.minute:02d} {loc.strftime('%p')} {loc.strftime('%Z')}"
    )
    return utc.isoformat().replace("+00:00", "Z"), pretty


def build_snapshot(session, course_id, tz):
    # include[]=term is required: without it Canvas returns only
    # enrollment_term_id, and course.term stays empty -- which validate()
    # rejects, because facts.toml's staleness check compares against it.
    course = _get(session, f"/api/v1/courses/{course_id}", {"include[]": "term"})[0]

    groups_raw = _get(session, f"/api/v1/courses/{course_id}/assignment_groups")
    weighted = bool(course.get("apply_assignment_group_weights"))
    groups = {
        str(g["id"]): {"name": g.get("name", ""), "weight": g.get("group_weight")}
        for g in groups_raw
    }

    assignments = []
    for a in _get(session, f"/api/v1/courses/{course_id}/assignments"):
        if not a.get("published"):
            continue  # rule 3
        due_utc, due_local = to_local(a.get("due_at"), tz)
        assignments.append(
            {
                "name": (a.get("name") or "").strip(),
                "group": groups.get(str(a.get("assignment_group_id")), {}).get("name", ""),
                "kind": "quiz" if a.get("is_quiz_assignment") else "assignment",
                "due_utc": due_utc,
                "due_local": due_local,
                "points": a.get("points_possible"),
                "url": a.get("html_url", ""),
                # Full written instructions. Not used in the Tier A prompt block
                # (far too long); this is the source Tier C indexes.
                "instructions": html_to_text(a.get("description")),
            }
        )
    # undated last, then chronological, then by name -- stable diffs
    assignments.sort(key=lambda x: (x["due_utc"] is None, x["due_utc"] or "", x["name"]))

    modules = [
        {"position": m.get("position"), "name": (m.get("name") or "").strip()}
        for m in _get(session, f"/api/v1/courses/{course_id}/modules")
        if m.get("published")
    ]
    modules.sort(key=lambda m: (m["position"] is None, m["position"]))

    pages = [
        {"title": (p.get("title") or "").strip(),
         "url": f"{CANVAS_BASE}/courses/{course_id}/pages/{p.get('url')}"}
        for p in _get(session, f"/api/v1/courses/{course_id}/pages")
        if p.get("published")
    ]
    pages.sort(key=lambda p: p["title"].lower())

    announcements = []
    for d in _get(
        session,
        f"/api/v1/courses/{course_id}/discussion_topics",
        {"only_announcements": "true"},
    ):
        if not d.get("published", True):
            continue
        # Canvas supports scheduled announcements. One that has not gone live
        # yet must not reach students early via this file -- same rule as
        # unpublished assignments. Fall back to created_at only when there is
        # no delayed post scheduled.
        delayed = d.get("delayed_post_at")
        if delayed:
            when = datetime.fromisoformat(delayed.replace("Z", "+00:00"))
            if when > datetime.now(timezone.utc):
                continue
            stamp = delayed
        else:
            stamp = d.get("posted_at") or d.get("created_at")
        posted_utc, posted_local = to_local(stamp, tz)
        announcements.append(
            {
                "title": (d.get("title") or "").strip(),
                "posted_utc": posted_utc,
                "posted_local": posted_local,
                "url": d.get("html_url", ""),
                # Body is captured here for Tier C ingestion. The prompt
                # renderer uses title + date only; see render_schedule().
                "body": html_to_text(d.get("message")),
            }
        )
    announcements.sort(key=lambda a: a["posted_utc"] or "", reverse=True)

    snap = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source": "canvas",
        "course": {
            "canvas_id": str(course_id),
            "code": course.get("course_code", ""),
            "name": course.get("name", ""),
            "term": (course.get("term") or {}).get("name", ""),
            "timezone": str(tz),
            "url": f"{CANVAS_BASE}/courses/{course_id}",
        },
        "grading_weights": (
            [{"group": g["name"], "weight": g["weight"]} for g in groups.values()]
            if weighted
            else []
        ),
        "modules": modules,
        "assignments": assignments,
        "pages": pages,
        "announcements": announcements,
    }
    return snap


LEAK_KEYS = {
    "email", "sis_user_id", "login_id", "sortable_name", "user_id",
    "score", "grade", "submission", "enrollment", "answers", "correct",
}


def validate(snap):
    """Fail loudly rather than write a bad or leaky artifact."""
    errs = []
    for key in ("generated_at", "course", "modules", "assignments", "pages", "announcements"):
        if key not in snap:
            errs.append(f"missing top-level key: {key}")
    if not snap.get("course", {}).get("term"):
        errs.append("course.term is empty -- the staleness check in facts.toml needs it")

    for a in snap.get("assignments", []):
        if not a.get("name"):
            errs.append("assignment with empty name")
        if a.get("due_utc"):
            try:
                datetime.fromisoformat(a["due_utc"].replace("Z", "+00:00"))
            except ValueError:
                errs.append(f"unparseable due_utc on {a['name']!r}")
        if a.get("due_utc") and not a.get("due_local"):
            errs.append(f"{a['name']!r} has UTC but no local time")

    blob = json.dumps(snap).lower()
    for k in LEAK_KEYS:
        if f'"{k}"' in blob:
            errs.append(f"possible student-data key in output: {k!r}")

    # Canvas file links carry a `verifier=` token granting unauthenticated access
    # to the file. This repo is public; one of these in a commit publishes the
    # course file. _Stripper drops query strings, so this should never fire --
    # it is here to catch a regression in that stripping.
    if "verifier=" in blob:
        errs.append("a Canvas file verifier token reached the output -- would leak file access")

    if not snap.get("assignments"):
        print("WARNING: no published assignments found", file=sys.stderr)
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--course-id", required=True)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--timezone", default=DEFAULT_TZ)
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args()

    token = os.getenv("CANVAS_API_TOKEN")
    if not token:
        sys.exit("CANVAS_API_TOKEN is not set (env or .env).")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    snap = build_snapshot(session, args.course_id, ZoneInfo(args.timezone))

    errs = validate(snap)
    if errs:
        for e in errs:
            print(f"VALIDATION: {e}", file=sys.stderr)
        sys.exit("refusing to write an invalid snapshot")

    text = json.dumps(snap, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
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
