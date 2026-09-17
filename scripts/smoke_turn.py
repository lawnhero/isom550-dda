#!/usr/bin/env python3
"""Run one real turn through app.py headlessly and print what the student saw.

Uses Streamlit's AppTest, so it exercises the whole turn -- router, tools,
section streaming, footers -- without a browser, and surfaces the
exception the in-app fallback path would otherwise swallow. Needs the API
keys and MongoDB URI from .env, and costs a few model calls per run.

    python scripts/smoke_turn.py
    python scripts/smoke_turn.py "what does a p-value of 0.03 mean"
    python scripts/smoke_turn.py --mode "Direct answer" "when is the next quiz due"

Not a pytest test on purpose: it is a live check to run after touching
app.py's turn loop or a chain template, not something CI should pay for.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest  # noqa: E402

DEFAULT_QUERY = "how to do regression in JMP and how to interpret R2"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    ap.add_argument("--mode", default=None, help="response style, e.g. 'Direct answer'")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    at = AppTest.from_file("app.py", default_timeout=args.timeout)
    at.run()
    if args.mode:
        at.session_state["response_mode"] = args.mode
    started = time.perf_counter()
    at.chat_input[0].set_value(args.query).run()
    elapsed = time.perf_counter() - started

    print(f"\n=== {args.query!r} -- {elapsed:.1f}s including rerun ===")
    if at.exception:
        print("EXCEPTION:", *[e.value for e in at.exception], sep="\n")
    seen = set()
    for block in at.markdown:
        text = block.value.strip()
        if text and text not in seen and not text.startswith(args.query):
            seen.add(text)
            print(text)
            print("-" * 60)
    for status in at.status:
        print("STATUS:", status.label)
        break
    return 1 if at.exception else 0


if __name__ == "__main__":
    sys.exit(main())
