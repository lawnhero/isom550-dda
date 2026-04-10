import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import certifi
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


def connect(uri: str, database: str, collection: str):
    client = MongoClient(uri, server_api=ServerApi("1"), tlsCAFile=certifi.where())
    return client[database][collection]


def infer_length_bucket(query: str) -> str:
    words = len((query or "").split())
    if words < 8:
        return "short"
    if words < 20:
        return "medium"
    return "long"


def generate_baseline(collection):
    docs = list(collection.find({}))
    queries = [d.get("query", "") for d in docs if d.get("query")]
    ts_values = [d.get("timestamp") for d in docs if d.get("timestamp")]

    buckets = Counter(infer_length_bucket(q) for q in queries)
    baseline = {
        "generated_at": datetime.now().isoformat(),
        "total_documents": len(docs),
        "query_count": len(queries),
        "avg_query_words": round(sum(len(q.split()) for q in queries) / max(len(queries), 1), 2),
        "query_length_distribution": dict(buckets),
        "time_range": {
            "first_event": min(ts_values).isoformat() if ts_values else None,
            "last_event": max(ts_values).isoformat() if ts_values else None,
        },
    }
    return baseline


def main():
    parser = argparse.ArgumentParser(description="Generate baseline metrics from existing query logs.")
    parser.add_argument("--mongo-uri", required=True)
    parser.add_argument("--database", default="user_queries_db")
    parser.add_argument("--collection", default="BUS 350")
    parser.add_argument("--output", default="analytics/baseline_metrics.json")
    args = parser.parse_args()

    collection = connect(args.mongo_uri, args.database, args.collection)
    baseline = generate_baseline(collection)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"Wrote baseline metrics to {output_path}")


if __name__ == "__main__":
    main()
