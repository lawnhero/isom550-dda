"""
Tier A course context: the facts the tutor should always know.

Loads course_data/schedule.json (machine-written by scripts/sync_canvas.py) and
course_data/facts.toml (hand-written), and renders one compact block for the
prompt. Nothing here is embedded or retrieved -- these are facts, and facts get
looked up, not matched by cosine similarity.

Two rules this module exists to enforce:

1. ALL DATE REASONING HAPPENS HERE, IN PYTHON. Canvas returns UTC, and six of
   this course's twelve deadlines fall on a different calendar DAY in UTC than
   students see (a 11:59 PM EDT deadline is 03:59 UTC the next day). The
   snapshot stores a pre-formatted local string; this module only sorts and
   selects. An LLM is never asked to compare or convert a timestamp.

2. STALE DATA MUST ANNOUNCE ITSELF. If the snapshot is old, or if facts.toml
   and schedule.json disagree about the term, the rendered block leads with an
   instruction telling the model not to state specific dates. A wrong deadline
   is the worst failure this app has; "check Canvas" is a much cheaper one.

The rendering functions are pure and take `now` explicitly so they can be
tested at any point in the term. Only get_course_context() touches Streamlit.
"""

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

SCHEDULE_PATH = Path("course_data/schedule.json")
FACTS_PATH = Path("course_data/facts.toml")

# The class-document index is BUILT from schedule.json, not read from it. Tier A is read
# live and is current the moment a sync writes; the document index does not move
# until it is rebuilt. Syncing without rebuilding therefore leaves the tutor stating this
# week's due dates while quoting last week's announcements, with nothing to say
# so. The build writes this sidecar; comparing it to the snapshot is how that
# silence gets broken.
DOCUMENTS_PROVENANCE_PATH = Path("data/documents/provenance.json")

# Fallback only. The real value comes from `stale_after_days` in facts.toml,
# because the right number depends on the term: at 2 classes/week the normal
# sync gap is 3-4 days in any term, but fall/spring breaks stretch it to 9-11.
STALE_AFTER_DAYS = 7

# Announcements whose title matches this are class recaps ("Class 5 (7/13) ...").
# Everything else is an admin notice ("new due date", "final is open").
CLASS_RECAP_RE = re.compile(r"^\s*Class\s+\d+", re.I)

# Canvas pages that are software walkthroughs. Surfaced only to the software
# route, which does not retrieve -- it just needs the link menu.
SOFTWARE_PAGE_RE = re.compile(r"jmp|excel|treeplan|mysql|install|software", re.I)


@dataclass
class Advisory:
    """A reason to distrust the data, plus what the tutor should do about it."""

    level: str  # "conflict" | "ended" | "drift" | "index"
    message: str
    instruction: str


def _parse(iso):
    if not iso:
        return None
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _fmt_day(dt):
    """'Mon Aug 3, 2026' -- platform-independent (no %-d)."""
    return f"{dt.strftime('%a %b')} {dt.day}, {dt.year}"


def _with_url(line: str, item: dict) -> str:
    """Append a Canvas URL when the snapshot carried one."""
    url = str((item or {}).get("url") or "").strip()
    return f"{line} — {url}" if url else line


def load(schedule_path=SCHEDULE_PATH, facts_path=FACTS_PATH):
    """Return (schedule, facts). Either may be None if absent or unparseable.

    Missing files are not fatal: the app ran without them before, and a broken
    course_data/ should degrade the tutor, not take it down.
    """
    schedule = facts = None
    try:
        schedule = json.loads(Path(schedule_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    try:
        with open(facts_path, "rb") as fh:
            facts = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        pass
    return schedule, facts


def _stale_limit(facts, override=None):
    """Days-until-stale: explicit override > facts.toml > module fallback."""
    if override is not None:
        return override
    try:
        value = int((facts or {}).get("stale_after_days"))
        return value if value > 0 else STALE_AFTER_DAYS
    except (TypeError, ValueError):
        return STALE_AFTER_DAYS


def _course_ended(schedule, now):
    """Last deadline, if every dated assignment is already past. Else None.

    A far better end-of-term signal than days-since-sync: exact, and derived
    from the data itself. Returns None when nothing is dated yet, so a course
    in week one is never mistaken for a finished one.
    """
    dated = [
        _parse(a["due_utc"])
        for a in (schedule or {}).get("assignments", [])
        if a.get("due_utc")
    ]
    if not dated:
        return None
    last = max(dated)
    return last if last < now else None


def advisories(schedule, facts, now=None, stale_after_days=None):
    """Reasons to distrust the loaded data, most severe first.

    Three distinct situations, deliberately not collapsed into one flag,
    because the right tutor behavior differs for each:

      conflict -> the two files disagree; no date can be trusted
      ended    -> the term is over; dates are correct but historical
      drift    -> mid-term with an old snapshot; dates are probably still fine
      index    -> Tier A is fresh but the document index was not rebuilt
    """
    now = now or datetime.now(timezone.utc)
    stale_after_days = _stale_limit(facts, stale_after_days)
    out = []

    # 1. CONFLICT. The exact bug that motivated splitting these files: the FAQ
    #    claimed "Spring 2026" for a Summer 2026 course, for a whole term.
    if schedule and facts:
        sched_term = (schedule.get("course") or {}).get("term", "")
        facts_term = facts.get("term", "")
        if sched_term and facts_term and sched_term != facts_term:
            out.append(
                Advisory(
                    "conflict",
                    f"Course facts are labeled {facts_term!r} but the schedule is "
                    f"{sched_term!r}. One of them is stale.",
                    "Do NOT state specific due dates, times, or office hours. "
                    "Tell the student to confirm on Canvas.",
                )
            )

    ended = _course_ended(schedule, now) if schedule else None

    # 2. ENDED. Matters most between terms: the app stays deployed for months
    #    after the last class, and without this it goes on describing a
    #    finished course as though it were live.
    if ended:
        course = schedule.get("course") or {}
        tz = ZoneInfo(course.get("timezone", "America/New_York"))
        out.append(
            Advisory(
                "ended",
                f"This schedule is for {course.get('term', 'a previous term')}, which "
                f"ended {_fmt_day(ended.astimezone(tz))}. Every deadline below is past.",
                "Refer to deadlines in the past tense. If the student seems to be in a "
                "current offering, say this schedule is from a previous term and point "
                "them to Canvas.",
            )
        )

    # 3. DRIFT. Suppressed once the course has ended, where age grows without
    #    bound and stops meaning anything.
    if schedule and not ended:
        generated = _parse(schedule.get("generated_at"))
        if generated is None:
            out.append(
                Advisory(
                    "drift",
                    "The schedule has no sync timestamp.",
                    "Tell the student to confirm dates on Canvas.",
                )
            )
        elif (now - generated).days > stale_after_days:
            out.append(
                Advisory(
                    "drift",
                    f"The schedule was last synced {(now - generated).days} days ago.",
                    "Dates below are probably still correct -- state them, but tell "
                    "the student to confirm on Canvas if the exact date matters.",
                )
            )

    # 4. INDEX. Distinct from `drift`: the snapshot can be current while the
    #    index built from it is not. Tier A answers stay correct, so nothing
    #    else here fires, but answer_course_documents is quietly serving the
    #    previous sync's announcements.
    out.extend(_index_advisories(schedule))

    return out


def _documents_provenance(path=DOCUMENTS_PROVENANCE_PATH):
    """The class-document index build stamp, or None when absent/unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _index_advisories(schedule, schedule_path=SCHEDULE_PATH,
                      provenance_path=DOCUMENTS_PROVENANCE_PATH):
    """Warn when the document index is older than the snapshot it came from.

    Compared on content hash, not timestamps. mtime moves on any checkout or
    rebase without the content changing, and `generated_at` moves on every sync
    even when nothing about the course did -- either would cry wolf until the
    warning stopped being read. The hash changes exactly when the indexed
    content would differ.

    Cost is one sha256 of a ~50 KB file, and the only caller is cached on file
    mtime, so this runs when the file changes rather than on every turn.
    """
    if not schedule:
        return []
    prov = _documents_provenance(provenance_path)
    if not prov:
        # No stamp at all. Either the index predates stamping or it was never
        # built. Both mean the document route cannot be vouched for, but
        # neither is worth silencing the tutor over -- Tier A is unaffected.
        return [
            Advisory(
                "index",
                "The class-document index carries no build stamp, so it cannot be "
                "checked against the current schedule.",
                "Answers drawn from class recaps and assignment briefs may be out "
                "of date; say so if one looks inconsistent with the dates above.",
            )
        ]

    try:
        current = hashlib.sha256(Path(schedule_path).read_bytes()).hexdigest()
    except OSError:
        return []

    built_from = prov.get("sha256")
    if built_from and built_from != current:
        return [
            Advisory(
                "index",
                f"The class-document index was built from an older schedule "
                f"(index built {prov.get('built_at', 'at an unknown time')}; the "
                f"snapshot has changed since).",
                "Course facts and due dates above are current. Answers drawn from "
                "class recaps and assignment briefs are NOT -- prefer the facts "
                "above when the two disagree, and point the student to Canvas.",
            )
        ]
    return []


def _render_facts(facts, lines, verbose=False):
    if not facts:
        return
    inst = facts.get("instructor") or {}
    if inst.get("name"):
        bits = [inst["name"]]
        if inst.get("email"):
            bits.append(inst["email"])
        if inst.get("office"):
            bits.append(f"office {inst['office']}")
        lines.append("Instructor: " + " | ".join(bits))
    if inst.get("office_hours"):
        lines.append(f"Office hours: {inst['office_hours']}")

    # A placeholder TA entry ("TBD", no email) renders as "TAs: TBD ()", which
    # the model then repeats to students as though it were an answer.
    tas = [
        t for t in (facts.get("tas") or [])
        if (t.get("name") or "").strip() and (t.get("name") or "").strip().lower() != "tbd"
    ]
    if tas:
        lines.append(
            "TAs: " + "; ".join(
                f"{t['name'].strip()} ({t['email'].strip()})" if (t.get("email") or "").strip()
                else t["name"].strip()
                for t in tas
            )
        )

    weights = (facts.get("grading") or {}).get("weights", "").strip()
    if weights:
        lines.append("")
        lines.append("GRADING")
        lines.append(weights)

    # Objectives are ~19% of this block's tokens and drew 1 query out of 1,322
    # in the logged history, so they are opt-in. Materials stay: only 6 queries,
    # but the block carries the software requirements, which matter more often.
    sections = [("materials", "MATERIALS")]
    if verbose:
        sections.append(("objectives", "COURSE OBJECTIVES"))
    for key, header in sections:
        body = (facts.get(key) or {}).get("body", "").strip()
        if body:
            lines.append("")
            lines.append(header)
            lines.append(body)

    policies = {k: v for k, v in (facts.get("policies") or {}).items() if v.strip()}
    extra = {k: v for k, v in (facts.get("grading") or {}).items()
             if k != "weights" and isinstance(v, str) and v.strip()}
    policies.update(extra)
    if policies:
        lines.append("")
        lines.append("POLICIES")
        for k, v in policies.items():
            lines.append(f"- {k.replace('_',' ')}: {v}")


def render(schedule, facts, now=None, stale_after_days=None, verbose=False):
    """The always-in-context Tier A block. Returns '' if there is nothing to say.

    `verbose=True` adds the course objectives section (see _render_facts).
    """
    if not schedule and not facts:
        return ""
    now = now or datetime.now(timezone.utc)

    tz = ZoneInfo((schedule.get("course") or {}).get("timezone", "America/New_York")) \
        if schedule else ZoneInfo("America/New_York")
    today_local = now.astimezone(tz)

    lines = []
    for adv in advisories(schedule, facts, now, stale_after_days):
        lines.append(f"!! SCHEDULE RELIABILITY [{adv.level}]")
        lines.append(f"   {adv.message}")
        lines.append(f"   -> {adv.instruction}")
        lines.append("")

    if schedule:
        c = schedule.get("course") or {}
        lines.append(f"COURSE: {c.get('name','')} ({c.get('term','')})")
        lines.append(f"Canvas: {c.get('url','')}")
    elif facts:
        c = facts.get("course") or {}
        lines.append(f"COURSE: {c.get('code','')} {c.get('title','')} ({facts.get('term','')})")

    lines.append(f"Today is {_fmt_day(today_local)}.")
    if schedule and _parse(schedule.get("generated_at")):
        lines.append(
            f"Schedule last synced {_fmt_day(_parse(schedule['generated_at']).astimezone(tz))}."
        )
    if facts and (facts.get("course") or {}).get("meeting"):
        lines.append(f"Meets: {facts['course']['meeting']}")

    _render_facts(facts, lines, verbose=verbose)

    if schedule:
        modules = schedule.get("modules") or []
        if modules:
            lines.append("")
            lines.append("MODULES RELEASED SO FAR")
            lines.append("  " + " | ".join(m.get("name", "") for m in modules))

        dated = [a for a in schedule.get("assignments", []) if a.get("due_utc")]
        undated = [a for a in schedule.get("assignments", []) if not a.get("due_utc")]
        upcoming = [a for a in dated if _parse(a["due_utc"]) >= now]
        past = [a for a in dated if _parse(a["due_utc"]) < now]

        lines.append("")
        lines.append("UPCOMING DEADLINES")
        if upcoming:
            for a in upcoming:
                lines.append(_with_url(
                    f"  - {a['name']} — due {a['due_local']} — "
                    f"{a.get('points') or 0:g} pts ({a.get('kind','assignment')})",
                    a,
                ))
        else:
            lines.append("  (none remaining)")

        if past:
            lines.append("")
            lines.append("PAST DEADLINES")
            for a in past:
                lines.append(_with_url(
                    f"  - {a['name']} — was due {a['due_local']} — "
                    f"{a.get('points') or 0:g} pts",
                    a,
                ))
        if undated:
            lines.append("")
            lines.append("NO DUE DATE SET")
            for a in undated:
                lines.append(_with_url(
                    f"  - {a['name']} — {a.get('points') or 0:g} pts",
                    a,
                ))

        # Only announcements that have actually gone out by `now`. The sync
        # already drops scheduled-but-unposted ones, but render(now=X) must
        # describe the course as of X -- otherwise a class recap can appear
        # before that class has happened.
        anns = [
            a for a in (schedule.get("announcements") or [])
            if not a.get("posted_utc") or _parse(a["posted_utc"]) <= now
        ]
        recaps = [a for a in anns if CLASS_RECAP_RE.match(a.get("title", ""))]
        notices = [a for a in anns if not CLASS_RECAP_RE.match(a.get("title", ""))]

        if recaps:
            lines.append("")
            lines.append("CLASS SESSIONS SO FAR (most recent last)")
            # Titles already carry the class number and date, and the title's
            # date is more accurate than posted_at (recaps are often posted the
            # next morning). Show the title alone rather than two dates.
            for a in sorted(recaps, key=lambda x: x.get("posted_utc") or ""):
                lines.append(_with_url(f"  - {a['title'].strip()}", a))
        if notices:
            lines.append("")
            lines.append("RECENT COURSE NOTICES")
            for a in notices[:5]:
                day = _parse(a.get("posted_utc"))
                stamp = _fmt_day(day.astimezone(tz)) if day else ""
                lines.append(_with_url(f"  - {stamp}: {a['title'].strip()}", a))

    return "\n".join(lines).strip()


def course_date_span(schedule):
    """(min_ymd, max_ymd) covered by the course, or (0, 0).

    Used to resolve a bare "july 30" to a year. A student never types the year,
    and the router has no calendar awareness by design, so the span is what
    makes the resolution deterministic.
    """
    ymds = []
    for a in (schedule or {}).get("announcements", []):
        ymds.append(_ymd_int(a.get("posted_utc")))
    for a in (schedule or {}).get("assignments", []):
        ymds.append(_ymd_int(a.get("due_utc")))
    ymds = [y for y in ymds if y]
    return (min(ymds), max(ymds)) if ymds else (0, 0)


def _ymd_int(iso):
    dt = _parse(iso)
    return int(dt.strftime("%Y%m%d")) if dt else 0


def render_software_context(schedule, facts, limit=None):
    """Grounding for the software route: versions, conventions, walkthrough links.

    The software route does NOT retrieve -- the model already knows JMP and
    Excel far better than any 21-row index could teach it. What it cannot know
    is which version this course assumes, where this course diverges from the
    tool defaults, and that the instructor already wrote a walkthrough for the
    exact task being asked about. That is all this block supplies.
    """
    lines = []

    software = (facts or {}).get("software") or {}
    if software:
        lines.append("SOFTWARE THIS COURSE USES")
        for key, label in (("excel", "Excel"), ("jmp", "JMP"), ("addins", "Add-ins")):
            if software.get(key):
                lines.append(f"  {label}: {software[key]}")

    conventions = (software.get("conventions") or "").strip()
    lines.append("")
    if conventions:
        lines.append("COURSE CONVENTIONS (these override tool defaults)")
        lines.append(conventions)
    else:
        # Said explicitly. With the block simply absent, the software route
        # invented one ("the course expects you to report RSquare, RMSE and the
        # parameter estimates") and presented it as policy.
        lines.append(
            "COURSE CONVENTIONS: none recorded for this course. Do not claim the "
            "course expects a particular option, report, or output."
        )

    pages = [
        p for p in (schedule or {}).get("pages", [])
        if SOFTWARE_PAGE_RE.search(p.get("title", ""))
    ]
    if limit:
        pages = pages[:limit]
    if pages:
        lines.append("")
        lines.append("COURSE WALKTHROUGHS (link to one when it matches the task)")
        for p in pages:
            lines.append(f"  - {p['title']}: {p['url']}")

    return "\n".join(lines).strip()


# --------------------------------------------------------------------------
# Streamlit-facing wrapper. Cached on file mtime so edits to either file show
# up immediately in local dev without a stale-cache mystery.
# --------------------------------------------------------------------------
def _mtimes():
    out = []
    for p in (SCHEDULE_PATH, FACTS_PATH):
        try:
            out.append(Path(p).stat().st_mtime)
        except OSError:
            out.append(0.0)
    return tuple(out)


@st.cache_data(show_spinner=False)
def _cached_context(mtimes, day_key, verbose):
    """Cache key is (file mtimes, UTC date, verbose) -- see callers below."""
    schedule, facts = load()
    return render(schedule, facts, verbose=verbose)


@st.cache_data(show_spinner=False)
def _cached_date_span(mtimes):
    schedule, _ = load()
    return course_date_span(schedule)


@st.cache_data(show_spinner=False)
def _cached_software_context(mtimes):
    schedule, facts = load()
    return render_software_context(schedule, facts)


@st.cache_data(show_spinner=False)
def _cached_links(mtimes):
    """Canvas URL and instructor email, for UI escape hatches.

    The prompt blocks already carry these, but the UI needs them as data: when
    the tutor abstains, the student should get a real link and a real mailto,
    not a sentence telling them to go find one.
    """
    schedule, facts = load()
    return {
        "canvas_url": ((schedule or {}).get("course") or {}).get("url", ""),
        "instructor_email": ((facts or {}).get("instructor") or {}).get("email", ""),
    }


def get_course_context(verbose=False):
    """Rendered Tier A block for the current moment. Safe to call every rerun."""
    # day_key busts the cache at midnight so "Today is ..." never goes stale,
    # and mtimes bust it the moment either source file is edited.
    return _cached_context(
        _mtimes(), datetime.now(timezone.utc).date().isoformat(), verbose
    )


def get_course_date_span():
    """(min_ymd, max_ymd) of the indexed course. Safe to call every rerun."""
    return _cached_date_span(_mtimes())


def get_software_context():
    """Grounding block for the software route. Safe to call every rerun."""
    return _cached_software_context(_mtimes())


def get_course_links():
    """{'canvas_url', 'instructor_email'} for student-facing fallbacks."""
    return _cached_links(_mtimes())


@st.cache_data(show_spinner=False)
def _cached_banner(mtimes):
    """What the sidebar footer states about the course: code, term, when the
    schedule was last synced, and who teaches it. Read from the same two files
    as the prompt block, so the footer can never disagree with the tutor."""
    schedule, facts = load()
    course = (schedule or {}).get("course") or {}
    inst = (facts or {}).get("instructor") or {}
    synced = _parse((schedule or {}).get("generated_at"))
    tz = ZoneInfo(course.get("timezone", "America/New_York"))
    return {
        "code": ((facts or {}).get("course") or {}).get("code") or course.get("code", ""),
        "term": course.get("term") or (facts or {}).get("term", ""),
        "synced_on": _fmt_day(synced.astimezone(tz)) if synced else "",
        "canvas_url": course.get("url", ""),
        "instructor_name": inst.get("name", ""),
        "instructor_email": inst.get("email", ""),
    }


def get_course_banner():
    """Sidebar footer facts. Safe to call every rerun."""
    return _cached_banner(_mtimes())
