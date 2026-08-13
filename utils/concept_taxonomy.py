"""Tier B module taxonomy, derived entirely from course_data/concepts.csv.

The router is told which modules exist and passes one back as `module`;
retrieval turns that into a Chroma filter. Both ends have to agree with the
CSV, and the way they agree is that NOTHING here is written by hand -- not the
list, not the order, not the labels.

That is not hypothetical tidiness. An earlier version kept its own taxonomy
table, and when concepts were re-filed in the CSV the two drifted: the router
was still offered `data-basics`, `model-building`, `decision-analysis` and
`value-of-information`, none of which any row claimed any more. A module id
matching no row produces an empty filter, zero results, and an abstention that
looks exactly like "we don't teach that".

Module ORDER is first appearance in the file, which is already teaching order
because that is how the CSV is sorted. Nothing needs to restate the calendar:
class numbers and dates change every term, they were a second thing to keep in
sync, and the router does not need them to pick a module.
"""

import csv
from pathlib import Path

DEFAULT_PATH = Path("course_data/concepts.csv")


def load_rows(path=DEFAULT_PATH):
    """Concept rows from the CSV. utf-8-sig because the source file has carried
    a BOM before, and a BOM in the header makes every column name wrong."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            return [r for r in csv.DictReader(fh) if (r.get("id") or "").strip()]
    except OSError:
        return []


def _clean(value: str) -> str:
    return (value or "").strip()


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
        written = _clean(row.get("status")) != "planned" and bool(_clean(row.get("body")))
        counts[mid]["written" if written else "planned"] += 1

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
    row's `simple-regression`, and matched nothing without complaint.
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


def module_is_unwritten(module: str, path=DEFAULT_PATH) -> bool:
    """True when the module exists but has no concept with a body yet.

    This is the coverage signal distance cannot provide. A question about
    logistic regression is topically close to every regression concept in the
    index, so it scores like a good match; only the inventory knows there is
    nothing there to have matched.
    """
    module = normalize_module(module, path)
    if not module:
        return False
    return any(m["id"] == module and m["n_written"] == 0 for m in list_modules(path))


def format_modules_for_prompt(path=DEFAULT_PATH) -> str:
    """The module list injected into the agent system prompt."""
    lines = []
    for item in list_modules(path):
        gap = "" if item["n_written"] else " — NOT WRITTEN YET, do not use"
        lines.append(f"- {item['id']}{gap}")
    return "\n".join(lines)
