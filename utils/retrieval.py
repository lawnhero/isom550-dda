import math
import re
from typing import Dict, List


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


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


def _extract_source_label(doc, fallback_index: int) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    source = metadata.get("source") or metadata.get("file_path") or metadata.get("filename")
    page = metadata.get("page")
    if source and page is not None:
        return f"{source} (page {page})"
    if source:
        return str(source)
    return f"Source {fallback_index + 1}"


def hybrid_retrieve(
    vector_db,
    query: str,
    top_k: int = 4,
    candidate_k: int = 12,
    vector_weight: float = 0.65,
) -> List:
    """
    Run a lightweight hybrid retrieval:
    - semantic similarity from vector DB
    - keyword overlap score from query
    """
    docs_with_scores = vector_db.similarity_search_with_score(query, k=candidate_k)
    if not docs_with_scores:
        return []

    semantic_scores = []
    lexical_scores = []
    for doc, distance in docs_with_scores:
        # Chroma returns lower distance for better match.
        semantic_scores.append(1.0 / (1.0 + float(distance)))
        lexical_scores.append(_keyword_score(query, getattr(doc, "page_content", "")))

    semantic_norm = _normalize_scores(semantic_scores)
    lexical_norm = _normalize_scores(lexical_scores)

    ranked = []
    for i, (doc, _distance) in enumerate(docs_with_scores):
        blended = vector_weight * semantic_norm[i] + (1.0 - vector_weight) * lexical_norm[i]
        ranked.append((blended, doc))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def build_document_filter(doc_type: str = "", since_ymd: int = 0):
    """Chroma `where` filter for the Tier C index, or None for no filter.

    "What did we cover in the last two weeks" is a RANGE query, not a similarity
    query -- no embedding reliably surfaces recency. Tier C stores a sortable
    `ymd` integer so the date part is an exact filter and only the topic part
    goes through the vector search.
    """
    clauses = []
    if doc_type:
        clauses.append({"doc_type": {"$eq": doc_type}})
    if since_ymd:
        clauses.append({"ymd": {"$gte": int(since_ymd)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def search_documents(
    vector_db,
    query: str,
    doc_type: str = "",
    since_ymd: int = 0,
    top_k: int = 4,
) -> List:
    """Similarity search over Tier C, narrowed by document type and date."""
    where = build_document_filter(doc_type, since_ymd)
    try:
        if where:
            return vector_db.similarity_search(query, k=top_k, filter=where)
        return vector_db.similarity_search(query, k=top_k)
    except Exception:
        # An empty or missing collection should degrade to "no sources", not
        # take the turn down. The index does not exist until build_tier_c runs.
        return []


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
    rows = []
    for i, doc in enumerate(docs):
        rows.append(
            {
                "rank": str(i + 1),
                "source": _extract_source_label(doc, i),
                "preview": (getattr(doc, "page_content", "") or "").strip()[:120],
            }
        )
    return rows
