# BUS 350 Virtual TA

Student-centered RAG chatbot for MBA Data and Decision Analytics.

## What is new in this revamp
- Adaptive tutoring controls: `Direct answer`, `Hint-first`, `Teach me step-by-step`
- Memory continuity with rolling summary (instead of hard short truncation)
- Hybrid retrieval diagnostics and explicit source blocks for course-mode answers
- Objective-aware tutoring prompts with attempt-check feedback behavior
- Event-based learning analytics with weekly report scripts

## App runtime
- Main app: `app.py`
- LCEL chains: `utils/chains_lcel.py`
- Retrieval helpers: `utils/retrieval.py`
- Logging and DB helpers: `utils/utils.py`

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
