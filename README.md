# BUS 350 Virtual TA

Student-centered RAG chatbot for MBA Data and Decision Analytics.

## What is new in this revamp
- Hybrid LangGraph agent with LCEL tutoring tools (`answer_logistics`, `answer_concept`, `generate_practice`, `check_attempt`)
- Session-scoped continue-practice loop after practice/check turns
- Adaptive tutoring controls: `Direct answer`, `Hint-first`, `Teach me step-by-step`
- Memory continuity with rolling summary (instead of hard short truncation)
- Hybrid retrieval with injected context and explicit source blocks for logistics answers
- Objective-aware tutoring prompts with attempt-check feedback behavior
- Event-based learning analytics with weekly report scripts
- No student identity storage (anonymous `session_id` only)

## Chat UI (Streamlit 1.57+)
- Pinned composer with `st.bottom` (quick actions + chat input stay visible)
- Continue-practice buttons after practice/check turns
- Segmented response-style control in the sidebar
- Loading skeleton / shimmer while the agent runs tools
- Course-themed light/dark settings in `.streamlit/config.toml`
- Chat input file paste/attach plus `submit_mode="stop"` for long answers
- Quick actions use clarifying turns: ask for topic/attempt when context is missing

## App runtime
- Main app: `app.py`
- LangGraph agent: `utils/agent_graph.py`
- Agent tools: `utils/ta_tools.py`
- LCEL chains: `utils/chains_lcel.py`
- Retrieval helpers: `utils/retrieval.py`
- Logging and DB helpers: `utils/utils.py`

## Models
- Main tutoring / step guidance: `grok-4.5` (fallback: `claude-sonnet-5`)
- Course RAG + memory/recap: `claude-haiku-4-5` (fallback: `gpt-4o-mini`)
- Agent tool dispatch: `gpt-4o-mini`
- Requires env vars: `XAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`

## Build/rebuild the knowledge index
Use this script with your raw course material folder:

```bash
python scripts/build_index.py \
  --source data/raw/course_materials \
  --persist-dir data/course
```

You can build a second index for assignment/content materials by changing `--source` and `--persist-dir`.

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
