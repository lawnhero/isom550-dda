import argparse
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import certifi
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


def connect(uri: str, database: str, collection: str):
    client = MongoClient(uri, server_api=ServerApi("1"), tlsCAFile=certifi.where())
    return client[database][collection]


def normalize_topic(event):
    objective = event.get("learning_objective")
    if objective:
        return objective
    route = event.get("route_label")
    return route or "unknown"


def run_report(collection, output_path: Path, days: int):
    cutoff = datetime.now() - timedelta(days=days)
    events = list(collection.find({"timestamp": {"$gte": cutoff}}))

    query_events = [e for e in events if e.get("event_type") == "query"]
    feedback_events = [e for e in events if e.get("event_type") == "feedback"]

    topic_counter = Counter(normalize_topic(e) for e in query_events)
    unresolved_counter = Counter(
        normalize_topic(e) for e in query_events if e.get("resolved") is False
    )
    helpful_counter = Counter(
        e.get("metadata", {}).get("helpful", "Not submitted") for e in feedback_events
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "window_days": days,
        "total_events": len(events),
        "query_events": len(query_events),
        "feedback_events": len(feedback_events),
        "top_topics": topic_counter.most_common(10),
        "high_friction_topics": unresolved_counter.most_common(10),
        "feedback_breakdown": dict(helpful_counter),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote weekly report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate weekly learning analytics summary.")
    parser.add_argument("--mongo-uri", required=True)
    parser.add_argument("--database", default="user_queries_db")
    parser.add_argument("--collection", default="BUS 350")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--output", default="analytics/weekly_report.json")
    args = parser.parse_args()

    collection = connect(args.mongo_uri, args.database, args.collection)
    run_report(collection, Path(args.output), args.days)


if __name__ == "__main__":
    main()
