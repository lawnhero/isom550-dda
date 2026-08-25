# ISOM 550 Virtual TA

Student-centered RAG chatbot for MBA Data and Decision Analytics.

## What is new in this revamp
- One tool-calling router and seven LCEL tutoring tools (`answer_course_facts`, `answer_course_documents`, `answer_software`, `answer_concept`, `generate_practice`, `coach_practice`, `check_attempt`). A turn is: route → run the chosen tools in order → stream one answer section per tool → log
- The practice question on screen is held in session state (`utils/practice.py`), so hints, "a harder one", and attempt checks all refer to the same question
- One guidance switch ("Show me how to work through it", on by default) instead of three response modes; concept and assignment answers end with an ordered plan when it is on, and with one check when it is off. Facts, software steps and practice are unaffected either way
- Bounded recent-chat window (8 messages) in every prompt
- Hybrid retrieval with injected context and explicit source blocks for logistics answers
- Event-based learning analytics with weekly report scripts
- No student identity storage (anonymous `session_id` only)

## Chat UI (Streamlit 1.57+)
- Every answer carries a provenance badge naming its source, so students can
  tell grounded course facts from general software knowledge
- Three follow-up chips after every answer: explain a concept, practice this, check my work
- Sources expander on any retrieval-backed answer
- Abstained answers get a recovery panel with deep links into Canvas (Announcements, Modules) and a mailto
- `st.status` line naming what the tutor is looking up; `st.feedback` thumbs per answer
- Pinned composer with `st.bottom` (chips + chat input stay visible)
- Clarifying turns for topic/attempt, with a "Never mind" exit and automatic
  escape when the student types a new question instead of an answer
- Attachments: `.txt`, `.csv`, `.pdf` are read and handed to the checker; screenshots go to vision builds of the check / software / concept chains, which transcribe what they read before interpreting it
- Course-themed light/dark settings in `.streamlit/config.toml`

See `docs/ux_audit.md` for the full route map, the student-experience trace,
and the structural changes proposed but not yet implemented.

## App runtime
- Main app (orchestration + session state): `app.py`
- Presentation layer (badges, chips, sources, diagnostics): `utils/ui.py`
- Router (one tool-calling model call, then the tools in order): `utils/router.py`
- Agent tools (prepare a chain payload per call): `utils/ta_tools.py`
- LCEL chains (one prompt per route): `utils/chains_lcel.py`
- Tier A course context: `utils/course_context.py`
- Tier B module taxonomy, derived from `concepts.csv`: `utils/concept_taxonomy.py`
- Retrieval, quality bands, date resolution: `utils/retrieval.py`
- Held practice question: `utils/practice.py`
- Attachment text / image extraction: `utils/attachments.py`
- Model wiring and fallback wrapper: `utils/llm_models.py`
- Logging and DB helpers: `utils/utils.py`

## Tests

Pure-Python coverage of the tool payloads, chain templates, retrieval date
logic, Tier A rendering, the practice session, and the model fallback wrapper.
No API keys, index, or Streamlit runtime needed:

```bash
python -m pytest tests -q
```

`tests/test_chain_contracts.py` streams every tool's payload through the chain
it names on a fake model, so a renamed payload key or a new `{placeholder}` in
a template fails here instead of in front of a student.

## Running locally

```bash
streamlit run app.py
```

Add `?debug=1` to the URL to unlock instructor diagnostics (router decision,
per-tool trace, retrieved chunks). In a deployed app, set a `diagnostics_token`
secret and pass it as the `debug` value instead.

## Models
Wired in `app.py`; the instances live in `utils/llm_models.py`.
- Main tutoring (`doc_chain`, `concept_chain`, `practice_chain`, `check_chain`, `coach_chain`): `deepseek-v4-pro` (fallback: `gpt-5.6-luna`)
- Light routes (`facts_chain`, `software_chain`): `deepseek-v4-flash` (fallback: `gpt-5.6-luna`)
- Routing (tool dispatch): `gpt-5.6-luna`, no fallback -- a routing failure drops the turn to the ungrounded `class_chain`
- Screenshot turns: `gpt-5.6-luna` with the full token budget, regardless of which model is tutoring

`ModelWithFallback` falls back on both `invoke` and `stream`; the stream case
is caught on the first token, since a generator cannot fail at call time.

## Secrets and environment
Local dev: create `.env` with `MONGODB_URI`, `OPENAI_API_KEY`, and `DEEPSEEK_API_KEY`. Two providers, two keys.

Deployed (Streamlit Cloud): set `mongodb_uri` in app secrets (see `.streamlit/secrets.toml.example`). API keys can live in `.env` locally or in Streamlit secrets on deploy.

## Build/rebuild the knowledge indexes
Two vector indexes are live. Tier A is not one of them: course facts are read
from `course_data/` at render time, never embedded.

Tier B — the curated concept index (`data/concepts`), read by
`answer_concept`. Source is `course_data/concepts.csv`, hand-maintained.
The same file drives the "Explain a concept" / "Practice question" pills:
`module` ids become the top row (label derived: `simple-regression` →
"Simple regression"), distinct `topic` values per module become the second
row, so `topic` must be a label a student can click, spelled identically on
every row of its group. Pills need no rebuild — the app reads the CSV live —
but the dry run below lints the labels and prints the pill tree:

```bash
python scripts/build_concepts.py --dry-run   # report + lint, embed nothing
python scripts/build_concepts.py
```

Tier C — class recaps and assignment briefs (`data/documents`), read by
`answer_course_documents`. Built from the Canvas snapshot, so sync first:

```bash
python scripts/sync_canvas.py --course-id 165666
python scripts/build_documents.py
```

Both scripts are thin wrappers over `vat-research/vector_index/`, where the
chunking, embedding and provenance stamping live so both course repos share one
implementation. The usual way to run a refresh after a class session is the
`canvas-course-sync` skill: it resolves the course, writes the snapshot via the
Canvas MCP, rebuilds via the vector-index MCP's `build_document_index`, and stops at the diff for review.
The scripts are the break-glass path; see their docstrings.

Each build writes `data/<index>/provenance.json` recording the source hash,
model and chunk count. `utils/course_context.py` compares it to the live
snapshot and adds an advisory when the index is older, so a sync without a
rebuild announces itself instead of quietly serving last week's announcements.

After changing the embedding model or either chunker, re-derive the abstention
thresholds — they are raw distances and go wrong silently:

```bash
python scripts/calibrate_retrieval.py --probe --db data/concepts
```

## Analytics scripts
- Weekly learning report:

```bash
python scripts/generate_weekly_report.py --mongo-uri "<YOUR_MONGO_URI>"
```

- Export the query log to CSV:

```bash
python scripts/export_queries.py --mongo-uri "<YOUR_MONGO_URI>"
```

Outputs are written to `analytics/`.

## Live check

```bash
python scripts/smoke_turn.py "when are office hours?"
```

Runs one real turn through `app.py` headlessly (router, tools, streamed
sections, footers) and prints what the student would have seen, plus any
exception the in-app fallback would otherwise swallow.
