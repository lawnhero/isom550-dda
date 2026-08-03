import argparse
import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import certifi
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()

CSV_COLUMNS = [
    "_id",
    "record_type",
    "timestamp",
    "event_id",
    "event_type",
    "session_id",
    "mode",
    "response_mode",
    "query",
    "route_label",
    "learning_objective",
    "learner_level",
    "resolved",
    "tools_used",
    "tool_calls",
    "source_count",
    "attachment_count",
    "attachment_names",
    "interaction_id",
    "helpful",
    "feedback_note",
    "metadata_json",
]


def connect(uri: str, database: str, collection: str):
    client = MongoClient(uri, server_api=ServerApi("1"), tlsCAFile=certifi.where())
    return client[database][collection]


def _resolve_mongo_uri(explicit_uri: Optional[str]) -> str:
    uri = (explicit_uri or os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        raise ValueError(
            "MongoDB URI not configured. Pass --mongo-uri or set MONGODB_URI in .env."
        )
    return uri


def _json_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _format_timestamp(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _infer_record_type(doc: Dict[str, Any]) -> str:
    event_type = doc.get("event_type")
    if event_type == "query":
        return "query_event"
    if event_type == "feedback":
        return "feedback"
    if doc.get("query"):
        return "legacy_query"
    return "other"


def _flatten_document(doc: Dict[str, Any]) -> Dict[str, str]:
    metadata = doc.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    row = {
        "_id": str(doc.get("_id", "")),
        "record_type": _infer_record_type(doc),
        "timestamp": _format_timestamp(doc.get("timestamp")),
        "event_id": _json_value(doc.get("event_id")),
        "event_type": _json_value(doc.get("event_type")),
        "session_id": _json_value(doc.get("session_id")),
        "mode": _json_value(doc.get("mode")),
        "response_mode": _json_value(doc.get("response_mode")),
        "query": _json_value(doc.get("query")),
        "route_label": _json_value(doc.get("route_label")),
        "learning_objective": _json_value(doc.get("learning_objective")),
        "learner_level": _json_value(doc.get("learner_level")),
        "resolved": _json_value(doc.get("resolved")),
        "tools_used": _json_value(metadata.get("tools_used")),
        "tool_calls": _json_value(metadata.get("tool_calls")),
        "source_count": _json_value(metadata.get("source_count")),
        "attachment_count": _json_value(metadata.get("attachment_count")),
        "attachment_names": _json_value(metadata.get("attachment_names")),
        "interaction_id": _json_value(metadata.get("interaction_id")),
        "helpful": _json_value(metadata.get("helpful")),
        "feedback_note": _json_value(metadata.get("note")),
        "metadata_json": _json_value(metadata),
    }
    return row


def _build_query_filter(
    *,
    include_feedback: bool,
    days: Optional[int],
) -> Dict[str, Any]:
    filters: List[Dict[str, Any]] = []

    if include_feedback:
        filters.append({"query": {"$exists": True, "$ne": ""}})
        filters.append({"event_type": "feedback"})
    else:
        filters.append({"query": {"$exists": True, "$ne": ""}})

    mongo_filter: Dict[str, Any]
    if len(filters) == 1:
        mongo_filter = filters[0]
    else:
        mongo_filter = {"$or": filters}

    if days is not None:
        cutoff = datetime.now() - timedelta(days=days)
        mongo_filter = {"$and": [mongo_filter, {"timestamp": {"$gte": cutoff}}]}

    return mongo_filter


def export_queries(
    collection,
    output_path: Path,
    *,
    include_feedback: bool = False,
    days: Optional[int] = None,
) -> int:
    mongo_filter = _build_query_filter(
        include_feedback=include_feedback,
        days=days,
    )
    docs = list(collection.find(mongo_filter).sort("timestamp", 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for doc in docs:
            writer.writerow(_flatten_document(doc))

    return len(docs)


def main():
    parser = argparse.ArgumentParser(
        description="Export query records from a MongoDB collection to CSV."
    )
    parser.add_argument(
        "--mongo-uri",
        default="",
        help="MongoDB URI. Defaults to MONGODB_URI from .env.",
    )
    parser.add_argument("--database", default="user_queries_db")
    parser.add_argument("--collection", default="ISOM 550")
    parser.add_argument(
        "--output",
        default="analytics/isom550_queries.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Optional rolling window in days. Export all records when omitted.",
    )
    parser.add_argument(
        "--include-feedback",
        action="store_true",
        help="Also export feedback events (event_type=feedback).",
    )
    args = parser.parse_args()

    mongo_uri = _resolve_mongo_uri(args.mongo_uri or None)
    collection = connect(mongo_uri, args.database, args.collection)
    output_path = Path(args.output)
    count = export_queries(
        collection,
        output_path,
        include_feedback=args.include_feedback,
        days=args.days,
    )
    print(f"Exported {count} records to {output_path}")


if __name__ == "__main__":
    main()
