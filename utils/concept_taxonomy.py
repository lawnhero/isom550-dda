"""Tier B taxonomy, derived entirely from course_data/concepts.csv.

Three things read this taxonomy and all three used to have their own copy:

  the router      is told which modules exist and passes one back as
                  `module`; retrieval turns that into a Chroma filter.
  the pills       "Explain a concept" / "Practice question" offer a module,
                  then a topic inside it.
  topic inference the keyword guess behind `learning_objective` and the
                  practice-topic fallback.

Both ends have to agree with the CSV, and the way they agree is that NOTHING
here is written by hand -- not the list, not the order, not the labels, not
the keywords.

That is not hypothetical tidiness. An earlier version kept its own taxonomy
table, and when concepts were re-filed in the CSV the two drifted: the router
was still offered `data-basics`, `model-building`, `decision-analysis` and
`value-of-information`, none of which any row claimed any more. A module id
matching no row produces an empty filter, zero results, and an abstention that
looks exactly like "we don't teach that". The pills drifted the same way one
layer up: they offered "Probability -> Bayes theorem" and "Hypothesis testing
-> ANOVA" while no such concept existed, so the very first click abstained.

COLUMNS THIS MODULE DEPENDS ON

  module   the id the router and the filter use. Hyphenated, lowercase. Its
           student-facing label is derived ("simple-regression" -> "Simple
           regression"); rename the id to rename the pill.
  topic    the STUDENT-FACING subtopic label, and the grouping key under a
           module. Rows sharing a module and a topic become one pill. So the
           value must read as a label a student would click ("Central
           tendency", "p-values", "Two-variable data table"), written
           identically on every row of the group. The build lint in
           vat-research/vector_index/adapters/concepts.py enforces that.
  title    the concept's own name. Feeds inference keywords; shown as the
           practice topic after a strong concept answer.
  status   `planned` rows are counted but never offered.

ORDER is first appearance in the file, which is already teaching order
because that is how the CSV is sorted. Nothing needs to restate the calendar.

Reads are cached on the file's mtime: the pills render on every Streamlit
rerun, and editing the CSV takes effect on the next rerun with no rebuild.
"""

import csv
import hashlib
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

DEFAULT_PATH = Path("course_data/concepts.csv")


def _mtime(path) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


@lru_cache(maxsize=8)
def _rows_at(path_str: str, mtime: float):
    """Cached by (path, mtime); `mtime` is only there to key the cache."""
    try:
        with open(path_str, newline="", encoding="utf-8-sig") as fh:
            return tuple(
                {k: (v or "") for k, v in r.items()}
                for r in csv.DictReader(fh)
                if (r.get("id") or "").strip()
            )
    except OSError:
        return ()


def load_rows(path=DEFAULT_PATH):
    """Concept rows from the CSV. utf-8-sig because the source file has carried
    a BOM before, and a BOM in the header makes every column name wrong."""
    path = Path(path)
    return [dict(r) for r in _rows_at(str(path), _mtime(path))]


def _clean(value: str) -> str:
    return (value or "").strip()


def _written(row) -> bool:
    return _clean(row.get("status")) != "planned" and bool(_clean(row.get("body")))


def module_label(module_id: str) -> str:
    """'simple-regression' -> 'Simple regression'. Derived, never stored."""
    text = (module_id or "").replace("-", " ").replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else ""


def list_modules(path=DEFAULT_PATH):
    """[{id, label, n_written, n_planned}, ...] in first-appearance order.

    `n_written` counts rows that have a body, which is the distinction the
    router needs: a module of nothing but `planned` rows can be named but
    cannot be answered from.
    """
    order, counts = [], {}
    for row in load_rows(path):
        mid = _clean(row.get("module"))
        if not mid:
            continue
        if mid not in counts:
            counts[mid] = {"written": 0, "planned": 0}
            order.append(mid)
        counts[mid]["written" if _written(row) else "planned"] += 1

    return [
        {
            "id": mid,
            "label": module_label(mid),
            "n_written": counts[mid]["written"],
            "n_planned": counts[mid]["planned"],
        }
        for mid in order
    ]


def valid_module_ids(path=DEFAULT_PATH):
    return {m["id"] for m in list_modules(path)}


def normalize_module(module: str, path=DEFAULT_PATH) -> str:
    """Canonical module id, or '' when unknown.

    Tolerant of the ways a model (or a spreadsheet) mangles an id: case, outer
    whitespace, and spaces where the id uses hyphens. That last one is not
    theoretical -- one CSV row carried `simple regression` against every other
    row's `simple-regression`, and matched nothing without complaint. The
    derived label ("Simple regression") resolves too, so a pill can be mapped
    back to its module.
    """
    module = _clean(module)
    if not module:
        return ""
    if module in valid_module_ids(path):
        return module
    candidate = module.lower().replace(" ", "-").replace("_", "-")
    for item in list_modules(path):
        if candidate == item["id"].lower() or module.lower() == item["label"].lower():
            return item["id"]
    return ""


def build_concept_filter(module: str, path=DEFAULT_PATH):
    """Chroma `where` for one module, or None when the id is unknown/empty.

    None means "search everything", which is the right failure mode: a module
    the router invented should widen the search, never silently empty it.
    """
    module = normalize_module(module, path)
    return {"module": {"$eq": module}} if module else None


def format_modules_for_prompt(path=DEFAULT_PATH) -> str:
    """The module list injected into the agent system prompt."""
    lines = []
    for item in list_modules(path):
        gap = "" if item["n_written"] else " — NOT WRITTEN YET, do not use"
        lines.append(f"- {item['id']}{gap}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Pills
# --------------------------------------------------------------------------
def outline(path=DEFAULT_PATH):
    """Modules with their written topics, in teaching order.

    [{id, label, topics: [{label, n_concepts}, ...]}, ...]

    This is the pill tree. Only written rows count: offering a topic the
    index cannot answer from is the drift this module exists to prevent.
    """
    topics = defaultdict(list)
    seen = set()
    for row in load_rows(path):
        mid, topic = _clean(row.get("module")), _clean(row.get("topic"))
        if not (mid and topic and _written(row)):
            continue
        if (mid, topic) not in seen:
            seen.add((mid, topic))
            topics[mid].append({"label": topic, "n_concepts": 0})
        next(t for t in topics[mid] if t["label"] == topic)["n_concepts"] += 1
    return [
        {"id": m["id"], "label": m["label"], "topics": topics.get(m["id"], [])}
        for m in list_modules(path)
        if m["n_written"]
    ]


def curriculum_topics(path=DEFAULT_PATH):
    """Top-level pill labels: the written modules, in teaching order."""
    return [m["label"] for m in outline(path)]


def subtopics(module_label_or_id: str, path=DEFAULT_PATH):
    """Subtopic pill labels under one module, or [] when unknown."""
    mid = normalize_module(module_label_or_id, path)
    for m in outline(path):
        if m["id"] == mid:
            return [t["label"] for t in m["topics"]]
    return []


def split_focus(focus: str, path=DEFAULT_PATH):
    """'Simple regression: Slope' -> ('simple-regression', 'Slope').

    The pills compose a topic as "Module label: Topic label". A tool that
    receives one can recover the module and use it as a retrieval FILTER --
    the one thing the module is actually good for. Anything that does not
    start with a known module label comes back as ('', focus) unchanged.
    """
    focus = _clean(focus)
    head, sep, tail = focus.partition(":")
    if sep:
        mid = normalize_module(head, path)
        if mid:
            return mid, _clean(tail) or module_label(mid)
    return normalize_module(focus, path), focus


# --------------------------------------------------------------------------
# Keyword inference, derived from the same rows
# --------------------------------------------------------------------------
_STOP = frozenset("""
a an the is are was were be been of in on at to for from with and or but if
this that these those it its i my me we our you your can could should would
will do does did what when where why who which how explain tell show help
about mean means use using get interpreting interpret describing describe
understanding understand what one two per
""".split())

# No "es": it trims "z-scores" to "z-scor" while the query's "z-score" stays
# whole; the plain "s" rule handles both.
_SUFFIXES = ("ance", "ant", "ing", "ed", "s")


def _stem(word: str) -> str:
    """Just enough to let 'significant' meet 'significance' and 'z-scores'
    meet 'z-score'. Not a stemmer; a suffix trim with a length floor."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _terms(text: str):
    return [
        _stem(w)
        for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", (text or "").lower())
        if w not in _STOP and len(w) > 1
    ]


@lru_cache(maxsize=8)
def _keyword_table(path_str: str, mtime: float):
    """{module_id: {phrase: weight}} from module ids, topic labels and titles.

    Unigrams and bigrams of the content words. Weight is the phrase length
    divided by how many modules use it, so "standard deviation" is decisive
    and "regression" (three modules) only breaks ties -- in favour of the
    first module in teaching order, which is the right default for a bare
    "how do I run a regression".
    """
    phrases = defaultdict(set)
    order = []
    for row in _rows_at(path_str, mtime):
        mid = _clean(row.get("module"))
        if not mid:
            continue
        if mid not in order:
            order.append(mid)
        for text in (mid.replace("-", " "), row.get("topic"), row.get("title")):
            words = _terms(text)
            phrases[mid].update(words)
            phrases[mid].update(f"{a} {b}" for a, b in zip(words, words[1:]))
    df = defaultdict(int)
    for mid in order:
        for p in phrases[mid]:
            df[p] += 1
    return [(mid, {p: len(p) / df[p] for p in phrases[mid]}) for mid in order]


def infer_module(text: str, path=DEFAULT_PATH) -> str:
    """Best-guess module id for free text, or '' when nothing matches.

    Replaces a hand-written keyword table that had its own taxonomy and its
    own drift. Scored by summed phrase weight; ties go to teaching order.
    """
    terms = " ".join(_terms(text))
    if not terms:
        return ""
    padded = f" {terms} "
    best, best_score = "", 0.0
    for mid, weights in _keyword_table(str(Path(path)), _mtime(path)):
        score = sum(w for p, w in weights.items() if f" {p} " in padded)
        if score > best_score:
            best, best_score = mid, score
    return best


def infer_module_label(text: str, path=DEFAULT_PATH) -> str:
    return module_label(infer_module(text, path))


# --------------------------------------------------------------------------
# Index freshness
# --------------------------------------------------------------------------
CONCEPTS_PROVENANCE_PATH = Path("data/concepts/provenance.json")


def index_drift(path=DEFAULT_PATH, provenance_path=CONCEPTS_PROVENANCE_PATH) -> str:
    """Why the concept index cannot be trusted to match the CSV, or "".

    Everything in this module reads the CSV live; the index's `module`
    metadata is a copy frozen at build time. Rename a module in the CSV and
    the router is offered the new id while the filter looks for it in an
    index that only knows the old one -- observed as an abstention on a
    concept the index held. Compared on content hash, like the documents
    index, so a checkout or a touch does not cry wolf.
    """
    try:
        prov = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "The concept index carries no build stamp; rebuild it to be sure it matches concepts.csv."
    try:
        current = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""
    if prov.get("sha256") and prov["sha256"] != current:
        return (
            f"concepts.csv has changed since the concept index was built "
            f"({prov.get('built_at', 'unknown time')}). Module filters may miss "
            "renamed modules until it is rebuilt (build_concept_index)."
        )
    return ""
