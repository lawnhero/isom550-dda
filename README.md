# ISOM 550 Virtual TA

Student-centered RAG chatbot for MBA Data and Decision Analytics.

## What is new in this revamp
- Hybrid LangGraph agent with LCEL tutoring tools (`answer_course_facts`, `answer_course_documents`, `answer_software`, `answer_concept`, `generate_practice`, `check_attempt`)
- Session-scoped continue-practice loop after practice/check turns
- Adaptive tutoring controls: `Direct answer`, `Hint-first`, `Teach me step-by-step`
- Memory continuity with rolling summary (instead of hard short truncation)
- Hybrid retrieval with injected context and explicit source blocks for logistics answers
- Objective-aware tutoring prompts with attempt-check feedback behavior
- Event-based learning analytics with weekly report scripts
- No student identity storage (anonymous `session_id` only)

## Chat UI (Streamlit 1.57+)
- Every answer carries a provenance badge naming its source, so students can
  tell grounded course facts from general software knowledge
- Follow-up chips are derived from the route that answered the last turn
- Sources expander on any retrieval-backed answer
- Abstained answers get a recovery panel with the real Canvas link and a mailto
- `st.status` progress line during routing; `st.feedback` thumbs per answer
- Pinned composer with `st.bottom` (chips + chat input stay visible)
- Clarifying turns for topic/attempt, with a "Never mind" exit and automatic
  escape when the student types a new question instead of an answer
- Attachments: `.txt`, `.csv`, `.pdf` are read; images are not, and the app says so
- Course-themed light/dark settings in `.streamlit/config.toml`

See `docs/ux_audit.md` for the full route map, the student-experience trace,
and the structural changes proposed but not yet implemented.

## App runtime
- Main app (orchestration + session state): `app.py`
- Presentation layer (badges, chips, sources, diagnostics): `utils/ui.py`
- LangGraph agent: `utils/agent_graph.py`
- Agent tools: `utils/ta_tools.py`
- LCEL chains: `utils/chains_lcel.py`
- Tier A course context: `utils/course_context.py`
- Retrieval helpers: `utils/retrieval.py`
- Attachment text extraction: `utils/attachments.py`
- Logging and DB helpers: `utils/utils.py`

## Running locally

```bash
streamlit run app.py
```

Add `?debug=1` to the URL to unlock instructor diagnostics (router decision,
per-tool trace, retrieved chunks). In a deployed app, set a `diagnostics_token`
secret and pass it as the `debug` value instead.

## Models
- Main tutoring / step guidance: `grok-4.5` (fallback: `claude-sonnet-5`)
- Course RAG + memory/recap: `claude-haiku-4-5` (fallback: `gpt-5.6-luna`)
- Agent tool dispatch: `gpt-5.6-luna`

## Secrets and environment
Local dev: create `.env` with `MONGODB_URI`, `XAI_API_KEY`, `ANTHROPIC_API_KEY`, and `OPENAI_API_KEY`.

Deployed (Streamlit Cloud): set `mongodb_uri` in app secrets (see `.streamlit/secrets.toml.example`). API keys can live in `.env` locally or in Streamlit secrets on deploy.

## Build/rebuild the knowledge indexes
Two vector indexes are live. Tier A is not one of them: course facts are read
from `course_data/` at render time, never embedded.

Tier B — class content Q&A (`data/contents`), read by `answer_concept`:

```bash
python scripts/build_index.py \
  --source data/raw/course_materials \
  --persist-dir data/contents
```

Tier C — class recaps and assignment briefs (`data/tier_c`), read by
`answer_course_documents`. Built from the Canvas snapshot, so sync first:

```bash
python scripts/sync_canvas.py --course-id 162137
python scripts/build_tier_c.py
```

After changing the embedding model or either chunker, re-derive the abstention
thresholds — they are raw distances and go wrong silently:

```bash
python scripts/calibrate_retrieval.py --probe --db data/contents
```

## Analytics scripts
- Baseline from existing logs:

```bash
python scripts/baseline_metrics.py --mongo-uri "<YOUR_MONGO_URI>"
```

- Weekly learning report:

```bash
python scripts/generate_weekly_report.py --mongo-uri "<YOUR_MONGO_URI>"
```

- Prompt style behavior check:

```bash
python scripts/eval_prompt_styles.py
```

Outputs are written to `analytics/`.
