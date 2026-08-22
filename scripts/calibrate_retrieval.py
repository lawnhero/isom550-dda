"""
Re-derive the retrieval abstention thresholds in utils/retrieval.py.

Those thresholds are raw Chroma distances. They are only meaningful for the
embedding model and distance metric the indexes were built with, so any change
to the embedding model, the chunker, or the collection's `hnsw:space` makes
them wrong -- silently, because a wrong threshold does not raise, it just
abstains on good questions or answers out-of-domain ones with confidence.

Two modes:

  --probe   Score three labelled question sets (in-domain / adjacent /
            out-of-domain) and print the separation. This is how
            STRONG_MAX_DISTANCE and ABSTAIN_MIN_DISTANCE were chosen: put the
            abstain line above the worst in-domain hit and below the best
            out-of-domain hit.

  --replay  Run a sample of the real query log through the current thresholds
            and report the band distribution, with examples. Use this to catch
            a threshold that looks fine on invented questions but abstains on
            a fifth of real traffic.

Usage:
    python scripts/calibrate_retrieval.py --probe
    python scripts/calibrate_retrieval.py --replay --sample 250
    python scripts/calibrate_retrieval.py --probe --db data/documents
"""

import argparse
import collections
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chromadb import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from utils.retrieval import (
    ABSTAIN_MIN_DISTANCE,
    MIN_CONTENT_TOKENS,
    RESCUE_MIN_OVERLAP,
    STRONG_MAX_DISTANCE,
    content_overlap,
    content_tokens,
)

load_dotenv()

# Labelled probe sets. IN should all land well below the abstain line and OUT
# well above it; the gap between the two is the margin the thresholds sit in.
# ADJACENT is deliberately ambiguous -- plausible business-analytics-shaped
# questions this course does not actually teach.
PROBES = {
    "IN": [
        "What does R-squared mean?",
        "How do I interpret a regression coefficient?",
        "Explain hypothesis testing and p-values",
        "What is a decision tree and expected value?",
        "How do I read the output of a t-test?",
        "What is multicollinearity?",
        "Explain sensitivity analysis",
        "descriptive statistics mean median mode",
    ],
    "LITERAL": [
        "What did we do in the Pilgrim Bank case?",
        "What is BTG about?",
        "what did we cover on 7/30",
        "assignment 3 requirements",
    ],
    # TAUGHT IN THIS COURSE, ABSENT FROM THIS INDEX. The category that was
    # missing, and the one that matters most.
    #
    # IN / ADJACENT / OUT only ever tested whether the index could reject
    # material from OTHER domains, which it does well. They could not catch the
    # actual failure: Tier B held nothing for Classes 6-10, and rated
    # "what is logistic regression used for" a STRONG match (d=1.371) against
    # descriptive-statistics chunks, answering it in the confident house style.
    #
    # Do not expect to tune a threshold until these abstain. Measured on
    # held-out phrasings, GAP and IN OVERLAP in every index shape tried --
    # worst IN 1.368 vs best GAP 1.325 on the CSV index, and the multi-vector
    # concept index widens the overlap rather than closing it. Distance to the
    # nearest chunk measures topical similarity, not coverage, and a question
    # about a topic this course teaches is topically similar by definition.
    # What this set is for is watching that overlap, and proving that the
    # coverage inventory in course_data/concepts.toml -- not a threshold -- is
    # what has to carry the decision.
    "GAP": [
        "What is logistic regression used for?",
        "How do I read a residual plot?",
        "What is the value of information?",
        "How does sensitivity analysis change my decision?",
        "How do I build a decision tree?",
    ],
    "ADJACENT": [
        "How do I train a convolutional neural network?",
        "Explain the Black-Scholes option pricing model",
        "What is a Kubernetes pod?",
        "How does double-entry bookkeeping work?",
    ],
    "OUT": [
        "How do I make sourdough bread?",
        "What is the offside rule in soccer?",
        "Write me a Python web scraper for Instagram",
        "What were the causes of the French Revolution?",
        "How do I change the oil in a Honda Civic?",
        "What is the capital of Mongolia?",
    ],
}


def open_db(path: str, model: str):
    return Chroma(
        persist_directory=path,
        embedding_function=OpenAIEmbeddings(model=model),
        client_settings=Settings(anonymized_telemetry=False),
    )


def score(db, query: str, k: int = 12):
    """(best_distance, best_content_overlap, n_content_tokens) for one query."""
    hits = db.similarity_search_with_score(query, k=k)
    if not hits:
        return None, 0.0, len(content_tokens(query))
    best_d = min(float(d) for _, d in hits)
    best_lex = max(content_overlap(query, doc.page_content) for doc, _ in hits)
    return best_d, best_lex, len(content_tokens(query))


def band(best_d, best_lex, n_tokens, in_conversation=False):
    """Mirror of retrieval.assess, but naming which rule fired.

    Probe questions are scored as openers (in_conversation=False); the replay
    treats them as mid-conversation, which is where most real traffic sits.
    """
    if best_d is None:
        return "none"
    if best_d <= STRONG_MAX_DISTANCE:
        return "strong"
    if best_d < ABSTAIN_MIN_DISTANCE:
        return "weak"
    if n_tokens >= MIN_CONTENT_TOKENS and best_lex >= RESCUE_MIN_OVERLAP:
        return "weak (lexical rescue)"
    if in_conversation and n_tokens < MIN_CONTENT_TOKENS:
        return "weak (too short to judge)"
    return "none"


def run_probe(db):
    print(f"thresholds: strong <= {STRONG_MAX_DISTANCE} | abstain > {ABSTAIN_MIN_DISTANCE}"
          f" | rescue >= {RESCUE_MIN_OVERLAP} with >= {MIN_CONTENT_TOKENS} content tokens\n")
    worst_in, best_out = 0.0, 99.0
    for label, queries in PROBES.items():
        print(f"--- {label} ---")
        for q in queries:
            d, lex, n = score(db, q)
            print(f"  {q[:46]:48s} d={d:.3f}  overlap={lex:.2f}  tokens={n:2d}  -> {band(d, lex, n)}")
            if label == "IN" and d is not None:
                worst_in = max(worst_in, d)
            if label == "OUT" and d is not None:
                best_out = min(best_out, d)
        print()
    print(f"worst in-domain distance : {worst_in:.3f}")
    print(f"best out-of-domain       : {best_out:.3f}")
    print(f"usable margin            : {best_out - worst_in:.3f}"
          f"   (abstain line at {ABSTAIN_MIN_DISTANCE})")
    if not worst_in < ABSTAIN_MIN_DISTANCE < best_out:
        print("\n!! ABSTAIN_MIN_DISTANCE is outside the separating gap -- retune it.")


def run_replay(db, csv_path: str, sample_size: int, seed: int):
    rows = list(csv.DictReader(open(csv_path)))
    seen, uniq = set(), []
    for row in rows:
        q = (row.get("query") or "").strip()
        if 3 < len(q) < 400 and q.lower() not in seen:
            seen.add(q.lower())
            uniq.append(q)
    random.seed(seed)
    sample = random.sample(uniq, min(sample_size, len(uniq)))
    print(f"{len(rows)} logged rows, {len(uniq)} unique queries, replaying {len(sample)}\n")

    bands = collections.Counter()
    examples = collections.defaultdict(list)
    for q in sample:
        d, lex, n = score(db, q)
        b = band(d, lex, n, in_conversation=True)
        bands[b] += 1
        if len(examples[b]) < 15:
            examples[b].append((round(d, 2) if d is not None else None, round(lex, 2), q))

    total = len(sample)
    for name, count in bands.most_common():
        print(f"  {name:26s} {count:4d}  {count / total:5.1%}")
    for name in bands:
        print(f"\n--- {name} ---")
        for d, lex, q in examples[name]:
            print(f"  d={d} overlap={lex}  {q[:95]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/contents",
                    help="Chroma directory to calibrate against (default: data/contents)")
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--probe", action="store_true", help="Score the labelled question sets.")
    ap.add_argument("--replay", action="store_true", help="Replay the real query log.")
    ap.add_argument("--queries", default="analytics/isom550_queries.csv")
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not args.probe and not args.replay:
        args.probe = True

    db = open_db(args.db, args.embedding_model)
    try:
        print(f"index: {args.db}  ({db._collection.count()} chunks)\n")
    except Exception:
        pass

    if args.probe:
        run_probe(db)
    if args.replay:
        print()
        run_replay(db, args.queries, args.sample, args.seed)


if __name__ == "__main__":
    main()
