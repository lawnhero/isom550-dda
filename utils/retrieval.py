import math
import re
import traceback
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


# --------------------------------------------------------------------------
# Retrieval quality
#
# The old abstention test was `total characters < 80`. With 21 chunks in Tier B,
# 32 in Tier C, and chunk sizes of 200-2700 characters, k=4 could never produce
# fewer than 80 characters -- the tutor had never abstained on a concept
# question in its life. It answered every out-of-domain question with whatever
# four chunks came back, in the confident house style.
#
# What follows thresholds the ACTUAL relevance signal instead. Three rules earn
# their complexity, and the first two were set from measurement
# (scripts/calibrate_retrieval.py):
#
# 1. Thresholds are on the RAW Chroma distance, not the blended score below.
#    `_normalize_scores` is min-max over the candidate set, so the top hit
#    always scores ~1.0 no matter how bad it is. The blend ranks; it cannot
#    judge. Raw distance is absolute and separates cleanly: across in-domain
#    probes the worst top-hit was 1.47, while the best out-of-domain top-hit
#    was 1.63.
#
# 2. Abstention needs a query with enough substance to judge. Mid-conversation
#    fragments ("Binary!", "$1,800 is the difference", "it is simple") are 4%
#    of the real query log; their meaning lives in the conversation, not the
#    text, so their retrieval scores are noise. Abstaining on those would be a
#    regression -- the tutoring prompts already refuse to answer from thin
#    context, but nothing recovers from a wrong "not covered".
#
# 3. A DATE filter has already decided relevance, so distance must not overrule
#    it. fetch_by_filter says this for the no-topic case ("the FILTER
#    established relevance") but that reasoning does not stop being true when a
#    topic is also present: "what did we cover on 7/30 about sensitivity"
#    narrows to the one recap from that day and then scores it -- and if the
#    student's wording is far enough from the instructor's, abstains on the
#    document the filter had already identified as the answer. Naming a day is
#    a lookup. See `date_filtered` below.
#
#    Deliberately DATE only, not any filter. doc_type='assignment' narrows a
#    category without offering any evidence the documents match the question,
#    so flooring on it would answer "how do I make sourdough" with assignment
#    briefs. A date is something the student actually asserted.
# --------------------------------------------------------------------------

# Raw squared-L2 distance from Chroma, over text-embedding-3-small vectors
# (unit length, so distance = 2 - 2*cos and the usable range is 0..2).
# Re-derive with scripts/calibrate_retrieval.py after changing the embedding
# model, the chunker, or the collection's hnsw:space -- these numbers are
# meaningless under a different distance metric.
STRONG_MAX_DISTANCE = 1.40
ABSTAIN_MIN_DISTANCE = 1.58

# A distinctive-token match can carry a query the embedding misses: Tier C is
# full of literals ("Pilgrim Bank", "BTG", "Store24") that vector search ranks
# poorly. Requires MIN_CONTENT_TOKENS as well -- overlap on a one-token query
# saturates at 1.0 and rescued nonsense in testing.
RESCUE_MIN_OVERLAP = 0.60
MIN_CONTENT_TOKENS = 3

# Below this the chunks are degenerate regardless of score.
MIN_CONTEXT_CHARS = 80

# Stripped before measuring overlap. Without this, "What is the capital of
# Mongolia?" scored 0.67 against the stats index -- higher than several real
# course questions -- purely on "what/is/the/of".
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "what", "when", "where", "why", "who", "which", "how",
    "of", "in", "on", "at", "to", "for", "from", "with", "and", "or", "but",
    "if", "this", "that", "these", "those", "it", "its", "i", "my", "me", "we",
    "our", "you", "your", "can", "could", "should", "would", "will",
    "explain", "tell", "show", "help", "about", "mean", "means", "use",
    "using", "get",
})


def content_tokens(text: str) -> set:
    """Tokens that carry topic meaning: no stopwords, nothing under 3 chars."""
    return {t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 2}


def content_overlap(query: str, text: str) -> float:
    """Share of the query's content tokens present in text. 0.0 when none."""
    tokens = content_tokens(query)
    if not tokens:
        return 0.0
    return len(tokens & set(_tokenize(text))) / len(tokens)


@dataclass
class RetrievalResult:
    """Ranked documents plus an absolute judgement of whether they are any good.

    `quality` is what callers should branch on:
      "strong" -- ground the answer and present it normally
      "weak"   -- answer, but tell the student the match was loose
      "none"   -- abstain
    """

    docs: List = field(default_factory=list)
    best_distance: float = math.inf
    best_overlap: float = 0.0
    query_tokens: int = 0
    quality: str = "none"
    # "hybrid" -- ranked by similarity, judged by distance
    # "filter" -- selected by an exact metadata filter, distance not consulted
    mode: str = "hybrid"
    # True when the distance said "none" and a date filter overruled it. Worth
    # surfacing in diagnostics: it is the one place a result is kept on
    # something other than its own score.
    floored_by_date: bool = False
    # The module filter that was dropped because it matched nothing, or "".
    # Set by search_concepts; see the note there.
    widened_from_module: str = ""

    def __bool__(self) -> bool:
        return bool(self.docs)

    def as_trace(self) -> Dict[str, Any]:
        """Compact record for the instructor diagnostics panel."""
        trace = {
            "quality": self.quality,
            "mode": self.mode,
            "best_distance": (
                round(self.best_distance, 3)
                if self.best_distance != math.inf
                else None
            ),
            "best_overlap": round(self.best_overlap, 2),
            "query_tokens": self.query_tokens,
        }
        if self.floored_by_date:
            trace["floored_by_date"] = True
        if self.widened_from_module:
            trace["widened_from_module"] = self.widened_from_module
        return trace


def assess(
    docs: List,
    distances: List[float],
    query: str,
    *,
    in_conversation: bool = False,
    date_filtered: bool = False,
) -> RetrievalResult:
    """Band a retrieval into strong / weak / none.

    `in_conversation` gates the short-query exemption. A two-word turn only
    deserves the benefit of the doubt when there is a conversation carrying its
    meaning; as an opening question it is just a short question, and "What is a
    Kubernetes pod?" should abstain rather than be excused for brevity.

    `date_filtered` says these documents were selected by a date the student
    named or a recency window they asked for. That is a lookup whose answer set
    is already settled, so distance may downgrade the result to "weak" but not
    abstain on it -- see rule 3 above.
    """
    if not docs:
        return RetrievalResult(query_tokens=len(content_tokens(query)))

    best_distance = min(distances) if distances else math.inf
    best_overlap = max((content_overlap(query, getattr(d, "page_content", "")) for d in docs),
                       default=0.0)
    n_tokens = len(content_tokens(query))
    total_chars = sum(len(getattr(d, "page_content", "") or "") for d in docs)

    result = RetrievalResult(
        docs=docs,
        best_distance=best_distance,
        best_overlap=best_overlap,
        query_tokens=n_tokens,
    )

    if total_chars < MIN_CONTEXT_CHARS:
        result.quality = "none"
    elif best_distance <= STRONG_MAX_DISTANCE:
        result.quality = "strong"
    elif best_distance < ABSTAIN_MIN_DISTANCE:
        result.quality = "weak"
    elif n_tokens >= MIN_CONTENT_TOKENS and best_overlap >= RESCUE_MIN_OVERLAP:
        # Distinctive words matched even though the embedding did not.
        result.quality = "weak"
    elif in_conversation and n_tokens < MIN_CONTENT_TOKENS:
        # Too little to judge -- mid-conversation this is almost always a
        # follow-up ("Binary!", "$1,800 is the difference") whose meaning lives
        # in the history, not the query text, so its scores are noise. Answer
        # with a caveat rather than refusing.
        result.quality = "weak"
    else:
        result.quality = "none"

    # Last: the date filter overrules an abstention, but never the degenerate
    # check above -- a filter that matched 40 characters of nothing is still
    # nothing. "weak" rather than "strong" because the topic half of the
    # question genuinely did not match; doc_chain is instructed to say so when
    # the documents do not cover what was asked.
    if result.quality == "none" and date_filtered and total_chars >= MIN_CONTEXT_CHARS:
        result.quality = "weak"
        result.floored_by_date = True
    return result


def _keyword_score(query: str, text: str) -> float:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(_tokenize(text))
    overlap = len(query_tokens.intersection(text_tokens))
    return overlap / max(len(query_tokens), 1)


def _normalize_scores(values: List[float]) -> List[float]:
    if not values:
        return values
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return [0.5 for _ in values]
    return [(v - minimum) / (maximum - minimum) for v in values]


# --------------------------------------------------------------------------
# One document must not own the whole context window.
#
# Tier C splits long documents into parts, and the parts of one document score
# alike -- "Group Pilgrim bank - Predictive" is 3 chunks, and five class recaps
# are 2 each. An unconstrained top_k=4 can therefore spend three slots on one
# assignment brief and return a single other document, which is exactly wrong
# for "what did we cover the week of July 27".
#
# The top-ranked document is exempt and keeps every chunk it has. hybrid_retrieve
# always runs with a topic (the no-topic path goes to fetch_by_filter instead),
# so the best match is usually the document the student named, and returning two
# thirds of an assignment brief is a worse failure than the crowding this rule
# prevents -- doc_chain is asked to "list every task".
# --------------------------------------------------------------------------
MAX_CHUNKS_PER_DOCUMENT = 2


def _document_key(doc, fallback):
    """Identify the source DOCUMENT a chunk came from.

    Tier C chunks carry `title`, which is precisely this. Tier B has no title
    and stores one CSV row per document, so `source` alone would collapse the
    whole index into a single "document" and cap all of Tier B at
    MAX_CHUNKS_PER_DOCUMENT -- the row is what separates them. Anything with
    neither falls back to a unique key, so an index shape this function has not
    seen is never grouped by accident.
    """
    metadata = getattr(doc, "metadata", {}) or {}
    title = metadata.get("title")
    if title:
        return ("title", str(title))
    for key in ("row", "chunk_id"):
        if metadata.get(key) is not None:
            source = metadata.get("source") or metadata.get("file_path") or ""
            return ("row", str(source), str(metadata[key]))
    return ("unique", fallback)


def _as_int(value) -> int:
    """Metadata coerced to a sort key. Chroma round-trips ints, but a hand-built
    or older index can hold a string, and sorted() raises on mixed types."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _chunk_index(doc) -> int:
    return _as_int((getattr(doc, "metadata", {}) or {}).get("chunk"))


def _select_diverse(ranked: List, top_k: int) -> List:
    """Take top_k of `ranked`, capping how much of it one document may supply.

    `ranked` is a list of (blended_score, doc, distance), best first.

    Documents appear in the order their best-scoring chunk ranked, but the
    chunks WITHIN a document come back in part order, not score order:
    `_format_docs` concatenates page_content verbatim, so "(part 2 of 3)"
    sitting above "(part 1 of 3)" reads to the model as a non sequitur, and
    nothing downstream reorders it.

    A capped document contributes its best-SCORING parts, which need not be
    contiguous -- parts 1 and 3 of 3 is a normal result. That is preferred over
    forcing contiguity: the parts that matched are the parts worth sending, and
    build_documents writes the part number into every chunk header, so the gap is
    visible to the model rather than papered over.
    """
    groups: Dict[Any, List] = {}
    order = []
    for i, item in enumerate(ranked):
        key = _document_key(item[1], i)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)

    selected = []
    for position, key in enumerate(order):
        members = groups[key]
        limit = len(members) if position == 0 else MAX_CHUNKS_PER_DOCUMENT
        selected.extend(sorted(members[:limit], key=lambda x: _chunk_index(x[1])))
        if len(selected) >= top_k:
            break
    return selected[:top_k]


def concept_payload(metadata: Dict[str, Any]) -> str:
    """What the model receives for one retrieved concept.

    The Tier B index embeds `title` + `body` only, but a concept row also
    carries two things that answer questions the body does not:

      managerial_phrasing  the memo language -- "what words do I use for this?"
                           is one of the most frequent asks in the query log,
                           and the body states the idea without naming the
                           phrase to write.
      common_mistake       the misconception and its correction, which is what
                           a student needs when they arrive already believing
                           the wrong thing.

    They are excluded from the embedding on purpose -- they should not compete
    for the match -- but there is no reason to withhold them once the concept
    has been chosen. Labelled rather than concatenated, so the model can tell
    the instructor's explanation from the phrasing note and the correction.
    """
    title = str(metadata.get("title") or "").strip()
    parts = [f"{title}\n\n{str(metadata.get('body') or '').strip()}" if title
             else str(metadata.get("body") or "").strip()]
    phrasing = str(metadata.get("managerial_phrasing") or "").strip()
    mistake = str(metadata.get("common_mistake") or "").strip()
    if phrasing:
        parts.append(f"How to phrase it: {phrasing}")
    if mistake:
        parts.append(f"Common student mistake: {mistake}")
    return "\n\n".join(p for p in parts if p)


def _collapse_concepts(ranked: List) -> List:
    """Collapse concept-index hits to one entry each, best rank first.

    One vector per concept today, so collapsing is usually a no-op -- but it is
    what makes the vector layout an implementation detail. Splitting a concept
    into several vectors (a title vector, keyword vectors) would otherwise let
    one concept occupy every slot in top_k, and this keeps top_k counting
    concepts either way. It also swaps the embedded text for the full payload,
    which is what carries managerial_phrasing and common_mistake through.

    `ranked` is (blended, doc, distance), best first; the distance kept is the
    best-ranked vector's, the one that earned the match. Candidates from any
    other index pass through untouched.
    """
    out, seen = [], set()
    for score, doc, distance in ranked:
        metadata = getattr(doc, "metadata", {}) or {}
        concept_id = metadata.get("concept_id")
        if not (concept_id and metadata.get("body")):
            out.append((score, doc, distance))
            continue
        if concept_id in seen:
            continue
        seen.add(concept_id)
        out.append((score, _Doc(concept_payload(metadata), metadata), distance))
    return out


def _Doc(page_content: str, metadata: Dict[str, Any]):
    """A Document, imported lazily so this module stays importable standalone."""
    from langchain_core.documents import Document

    return Document(page_content=page_content, metadata=metadata)


def _extract_source_label(doc, fallback_index: int) -> str:
    """Human-readable name for the document a chunk came from.

    `title` is checked FIRST because Tier C carries the real document name
    ("Class 10 (7/30) Sensitivity, Value of Information, and Exam Review") and
    carries no `source` at all -- so this fell straight through to the
    positional fallback and every Tier C citation reached the student as
    "Source 1". A numbered placeholder for a document that has a name and a
    Canvas link is worse than no sources expander at all: it tells the student
    the answer is grounded and then refuses to say in what.
    """
    metadata = getattr(doc, "metadata", {}) or {}
    title = str(metadata.get("title") or "").strip()
    if title:
        return title
    source = metadata.get("source") or metadata.get("file_path") or metadata.get("filename")
    page = metadata.get("page")
    if source and page is not None:
        return f"{source} (page {page})"
    if source:
        return str(source)
    return f"Source {fallback_index + 1}"


def _extract_source_url(doc) -> str:
    """The document's own link, when its index carries one. Tier C only today.

    Scheme-checked because this string is used as a markdown link target in the
    sources expander and is placed in front of the model in the prompt. The
    values come from our own Canvas sync, so this guards against a malformed
    snapshot rather than a hostile one.
    """
    url = str((getattr(doc, "metadata", {}) or {}).get("url") or "").strip()
    return url if url.startswith(("http://", "https://")) else ""


def format_source_line(doc) -> str:
    """One provenance line to sit above a chunk in the PROMPT, or "".

    doc_chain is instructed to "name the document you are drawing on and
    include its link when one is provided". No link was ever provided:
    `_format_docs` concatenates page_content and drops metadata on the floor,
    so the Canvas URL that Tier C stores for every chunk never reached the
    model, and rule 3 was unfulfillable by construction.

    Requires a URL. Both indexes already open their text with the document
    title -- Tier C bakes it into every chunk header, Tier B's concept payload
    leads with it -- so a title-only line is pure duplication, and the one
    thing the model genuinely cannot see without help is the link.
    """
    url = _extract_source_url(doc)
    if not url:
        return ""
    title = str((getattr(doc, "metadata", {}) or {}).get("title") or "").strip()
    return f"Source: {title} — {url}" if title else f"Source: {url}"


def hybrid_retrieve(
    vector_db,
    query: str,
    top_k: int = 4,
    candidate_k: int = 12,
    vector_weight: float = 0.65,
    where=None,
    in_conversation: bool = False,
    date_filtered: bool = False,
) -> RetrievalResult:
    """
    Run a lightweight hybrid retrieval:
    - semantic similarity from vector DB
    - keyword overlap score from query

    Returns a RetrievalResult: the blended score decides the ORDER, the raw
    distances decide whether the results are worth using at all. Both are
    needed -- see the notes above the threshold constants for why the blend
    cannot do the second job.

    `date_filtered` reports that `where` constrains a date, which changes how
    the distances are judged (rule 3 above). The caller passes it rather than
    this function inspecting `where`, because only the caller knows whether the
    date came from the student or from a default.
    """
    kwargs = {"k": candidate_k}
    if where:
        kwargs["filter"] = where
    docs_with_scores = vector_db.similarity_search_with_score(query, **kwargs)
    if not docs_with_scores:
        return assess([], [], query, in_conversation=in_conversation)

    semantic_scores = []
    lexical_scores = []
    for doc, distance in docs_with_scores:
        # Chroma returns lower distance for better match.
        semantic_scores.append(1.0 / (1.0 + float(distance)))
        lexical_scores.append(_keyword_score(query, getattr(doc, "page_content", "")))

    semantic_norm = _normalize_scores(semantic_scores)
    lexical_norm = _normalize_scores(lexical_scores)

    ranked = []
    for i, (doc, distance) in enumerate(docs_with_scores):
        blended = vector_weight * semantic_norm[i] + (1.0 - vector_weight) * lexical_norm[i]
        ranked.append((blended, doc, float(distance)))

    ranked.sort(key=lambda x: x[0], reverse=True)
    # Concept vectors collapse to whole concepts first, so top_k counts
    # concepts rather than alias phrasings. No-op for every other index.
    top = _select_diverse(_collapse_concepts(ranked), top_k)
    return assess(
        [doc for _, doc, _ in top],
        [d for _, _, d in top],
        query,
        in_conversation=in_conversation,
        date_filtered=date_filtered,
    )


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_course_date(text: str, span=None) -> int:
    """Free-text date -> sortable ymd int (20260730), or 0 if unparseable.

    Accepts "2026-07-30", "7/30", "july 30", "30 July", "Jul 30 2026".

    The year is usually absent, because a student types "july 30", not the year.
    `span` is the (min_ymd, max_ymd) of the indexed course; when the year is
    missing we pick the candidate year that lands inside that span. A single
    term has exactly one, and a term crossing New Year still resolves because
    only one of the two candidates falls in range.

    All of this is deliberately here rather than in a prompt: the router only
    extracts the words "july 30" from the question, which needs no calendar
    awareness, and Python does the resolution.
    """
    text = (text or "").strip().lower()
    if not text:
        return 0

    y = m = d = 0
    iso = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if iso:
        y, m, d = (int(g) for g in iso.groups())
    else:
        named = re.search(r"([a-z]{3,9})\.?\s+(\d{1,2})|(\d{1,2})\s+([a-z]{3,9})", text)
        numeric = re.search(r"\b(\d{1,2})[-/](\d{1,2})\b", text)
        if named:
            word = (named.group(1) or named.group(4) or "")[:3]
            day = named.group(2) or named.group(3)
            m, d = _MONTHS.get(word, 0), int(day)
        elif numeric:
            m, d = int(numeric.group(1)), int(numeric.group(2))
        year = re.search(r"\b(20\d{2})\b", text)
        if year:
            y = int(year.group(1))

    if not (1 <= m <= 12 and 1 <= d <= 31):
        return 0
    if y:
        return y * 10000 + m * 100 + d

    if span and span[0] and span[1]:
        lo, hi = int(span[0]), int(span[1])
        for candidate in range(lo // 10000, hi // 10000 + 1):
            ymd = candidate * 10000 + m * 100 + d
            if lo <= ymd <= hi:
                return ymd
        # Outside the course window -- still resolve against its first year so
        # the caller gets a filter that correctly matches nothing.
        return (lo // 10000) * 10000 + m * 100 + d
    return 0


def resolve_date_range(text: str, span=None, granularity: str = "day"):
    """A named date plus a granularity word -> inclusive (from_ymd, to_ymd).

    "Week of July 30" needs a range anchored on a NAMED date. `on_date` alone
    gives one day, and `days_back` anchors on today -- neither can express it,
    and days_back returns nothing at all once the term is over.

    Both inputs come straight out of the student's sentence ("july 30", "week"),
    so the router still copies words through and Python does the calendar work.

    A week is the ISO week CONTAINING the date (Mon-Sun), not the seven days
    starting from it: "week of July 30" (a Thursday) then covers Mon Jul 27
    through Sun Aug 2, which is how the phrase is normally meant.
    """
    ymd = parse_course_date(text, span)
    if not ymd:
        return 0, 0

    g = (granularity or "day").strip().lower()
    if g not in {"day", "week", "month"}:
        g = "day"
    if g == "day":
        return ymd, ymd

    try:
        anchor = date(ymd // 10000, (ymd // 100) % 100, ymd % 100)
    except ValueError:
        return ymd, ymd

    if g == "week":
        start = anchor - timedelta(days=anchor.weekday())
        end = start + timedelta(days=6)
    else:
        start = anchor.replace(day=1)
        end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)

    as_ymd = lambda d: int(d.strftime("%Y%m%d"))
    return as_ymd(start), as_ymd(end)


def build_document_filter(
    doc_type: str = "", since_ymd: int = 0, from_ymd: int = 0, to_ymd: int = 0
):
    """Chroma `where` filter for the Tier C index, or None for no filter.

    "What did we cover in the last two weeks" is a RANGE query, not a similarity
    query -- no embedding reliably surfaces recency. Tier C stores a sortable
    `ymd` integer so the date part is an exact filter and only the topic part
    goes through the vector search.
    """
    clauses = []
    if doc_type:
        clauses.append({"doc_type": {"$eq": doc_type}})
    if from_ymd and to_ymd:
        # A named date or date range is a lookup, not a ranking problem, and it
        # overrides any recency window -- naming a day means that day.
        if from_ymd == to_ymd:
            clauses.append({"ymd": {"$eq": int(from_ymd)}})
        else:
            clauses.append({"ymd": {"$gte": int(from_ymd)}})
            clauses.append({"ymd": {"$lte": int(to_ymd)}})
    elif since_ymd:
        clauses.append({"ymd": {"$gte": int(since_ymd)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def fetch_by_filter(vector_db, where, top_k: int = 8) -> RetrievalResult:
    """Fetch documents by metadata filter alone, newest first.

    "What did we cover last week" has no topic term -- the answer is defined
    entirely by a date range. Ranking those by embedding similarity to an empty
    string is noise, and judging them by distance would abstain on a lookup
    that is exactly correct: the FILTER established relevance, so quality is
    settled before any vector is involved.

    Sorted by `ymd` descending because a recency question wants the most recent
    session first, which similarity search has no way to express. Ties break on
    title and then part number, so a document split across chunks arrives whole
    and in order rather than interleaved with whatever else shares its date.
    """
    if not where:
        return RetrievalResult(mode="filter")
    try:
        from langchain_core.documents import Document

        # A date window can legitimately hold several sessions, each split into
        # chunks, so ask for more than the ranked path would.
        got = vector_db.get(where=where, limit=max(top_k, 8) * 3)
    except Exception:
        print("[fetch_by_filter] query failed:\n" + traceback.format_exc())
        return RetrievalResult(mode="filter")

    texts = got.get("documents") or []
    metas = got.get("metadatas") or [{} for _ in texts]
    rows = sorted(
        zip(texts, metas),
        key=lambda pair: (
            -_as_int((pair[1] or {}).get("ymd")),
            str((pair[1] or {}).get("title") or ""),
            _as_int((pair[1] or {}).get("chunk")),
        ),
    )[:max(top_k, 8)]

    docs = [Document(page_content=text or "", metadata=meta or {}) for text, meta in rows]
    if not docs:
        return RetrievalResult(mode="filter")
    return RetrievalResult(docs=docs, quality="strong", mode="filter")


def search_concepts(
    vector_db,
    query: str,
    module: str = "",
    top_k: int = 4,
    in_conversation: bool = False,
) -> RetrievalResult:
    """Hybrid search over Tier B, optionally narrowed to one module first.

    `module` is a module id from course_data/concepts.csv (e.g.
    `simple-regression`), chosen by the router. When set, Chroma filters on
    metadata `module` before the embedding search runs, so unrelated modules
    cannot crowd the top_k.

    An unrecognised module yields no filter rather than an empty one -- see
    concept_taxonomy.build_concept_filter. A module the router invented should
    widen the search, never silently return nothing and read as "not covered".

    The same rule applies when a RECOGNISED module matches nothing. Observed:
    "What does a p-value of 0.03 mean?" routed to `hypothesis-testing`, a
    module that exists in the CSV -- but the index had been built before that
    module was renamed from `inference`, so the filter returned zero rows and
    the tutor abstained on a concept it holds. A wrong-but-plausible module
    from the router (the p-value concept is filed under simple-regression)
    fails the same way with a fresh index. Either way the answer is in the
    collection; the filter just hid it. So an empty filtered result is retried
    unfiltered, and the trace records that the module was dropped.
    """
    from utils.concept_taxonomy import build_concept_filter, normalize_module

    resolved = normalize_module(module)
    where = build_concept_filter(resolved) if resolved else None
    print(
        "[search_concepts]",
        {
            "query": (query or "").strip() or None,
            "module": resolved or None,
            "top_k": top_k,
            "where": where,
            "in_conversation": in_conversation,
        },
    )

    def _search(filt):
        return hybrid_retrieve(
            vector_db, query, top_k=top_k, where=filt, in_conversation=in_conversation,
        )

    try:
        found = _search(where)
        if where and not found.docs:
            print(f"[search_concepts] module {resolved!r} matched nothing; widening")
            found = _search(None)
            found.widened_from_module = resolved
        return found
    except Exception:
        print("[search_concepts] giving up:\n" + traceback.format_exc())
        return assess([], [], query, in_conversation=in_conversation)


def search_documents(
    vector_db,
    query: str = "",
    doc_type: str = "",
    since_ymd: int = 0,
    from_ymd: int = 0,
    to_ymd: int = 0,
    top_k: int = 4,
    in_conversation: bool = False,
) -> RetrievalResult:
    """Hybrid search over Tier C, narrowed by document type and date.

    Uses hybrid_retrieve rather than plain similarity_search: Tier C queries are
    full of literal tokens -- "7/30", "Pilgrim Bank", "BTG" -- and the keyword
    half of the blend is what makes those land. Pure vector search was a
    regression against Tier B, which has had the blend all along.

    With no topic term but a filter, drops to fetch_by_filter -- see there for
    why ranking an empty query by similarity is the wrong question.
    """
    where = build_document_filter(doc_type, since_ymd, from_ymd, to_ymd)
    # A date the student named or a window they asked for. doc_type alone does
    # not count: it narrows a category without asserting anything about the
    # question, so it must not soften the abstention (rule 3 at the top).
    date_filtered = bool(since_ymd or (from_ymd and to_ymd))
    search_mode = "filter" if not (query or "").strip() else "hybrid"
    print(
        "[search_documents]",
        {
            "mode": search_mode,
            "query": (query or "").strip() or None,
            "top_k": top_k,
            "where": where,
            "date_filtered": date_filtered,
            "in_conversation": in_conversation,
        },
    )
    try:
        if not (query or "").strip():
            return fetch_by_filter(vector_db, where, top_k=top_k)
        return hybrid_retrieve(
            vector_db, query, top_k=top_k, where=where,
            in_conversation=in_conversation,
            date_filtered=date_filtered,
        )
    except Exception:
        # Still degrade to "no sources" so a missing index cannot take the turn
        # down -- but the traceback above is what tells us *why*.
        print("[search_documents] giving up:\n" + traceback.format_exc())
        return assess([], [], query, in_conversation=in_conversation)


def build_source_block(docs: List) -> str:
    if not docs:
        return "_No sources available._"
    lines = []
    for i, doc in enumerate(docs):
        label = _extract_source_label(doc, i)
        preview = (getattr(doc, "page_content", "") or "").strip().replace("\n", " ")
        preview = preview[:180] + ("..." if len(preview) > 180 else "")
        lines.append(f"- **[{i + 1}] {label}**: {preview}")
    return "\n".join(lines)


def retrieval_debug_rows(docs: List) -> List[Dict[str, str]]:
    """The rows the sources expander renders, one per retrieved chunk.

    Carries `url` so the expander can link the document rather than name it and
    leave the student to go find it on Canvas. Empty string when the index has
    no link for the chunk, which the renderer treats as plain text.
    """
    rows = []
    for i, doc in enumerate(docs):
        rows.append(
            {
                "rank": str(i + 1),
                "source": _extract_source_label(doc, i),
                "url": _extract_source_url(doc),
                "preview": (getattr(doc, "page_content", "") or "").strip()[:120],
            }
        )
    return rows
