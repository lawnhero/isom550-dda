# Architecture audit — ISOM 550 Virtual TA

Date: 2026-08-23
Scope: agent structure, chain/prompt/payload configuration, and the three
course-information tiers. Companion to `ux_audit.md` (student experience).

---

## 1. How the agent is structured

One turn is two LLM phases with a Python boundary between them:

```
student text (+ attachment text, + image marker)
   │
   ▼
LangGraph: agent ──► tools ──► (agent again ONLY if a tool call failed) ──► END
   gpt-5.6-luna        7 StructuredTools, run concurrently by ToolNode
   picks 0..n tools    each one PREPARES a payload and returns a receipt
   │
   ▼
app.py streams each prepared step through its LCEL chain, in the order the
router listed the calls, with one provenance badge + sources expander per step
```

Things that are right about it and worth preserving:

- **The router is a dispatcher, not a writer.** It sees a receipt
  (`tool_name`, `stream_ready`, `step_id`), never the retrieved text, so there
  is no second router pass that rewrites or previews the answer. The hand-built
  graph (`_after_tools`) exists precisely to skip the unconditional
  `tools -> agent` edge of the prebuilt ReAct agent.
- **Per-call `ToolStep`s keyed by `step_id`** fixed the multi-tool race
  documented in `ux_audit.md` §7; order comes from LangGraph's emission order,
  payload from the side channel.
- **All calendar arithmetic is in Python.** The router copies "july 30"
  through; `retrieval.resolve_date_range` resolves it against the course span.
  Tier A timestamps are pre-formatted local strings and the facts prompt
  forbids date conversion.
- **Images and attachments travel by closure, not by tool argument.** The
  router sees a one-line marker and routes; the chain sees the pixels.

Weak points, ordered by impact:

| # | Finding | Status |
|---|---|---|
| A1 | `ModelWithFallback.stream` never fell back. `BaseChatModel.stream` is a generator function, so the `try` around the *call* could not catch anything; failures surface on the first `next()`. Every tutoring chain streams, so the main tutor's fallback (`deepseek-v4-pro -> grok-4.5`) was dead in practice. | **Fixed** — first chunk is pulled inside the `try`; tests in `tests/test_llm_fallback.py`. |
| A2 | Router capped at 300 output tokens while instructed to copy attached files (up to 8,000 chars) into `check_attempt(attempt_text=...)`. A long attachment truncated the tool-call JSON. Observed live even after relaxing the instruction: the router copied 882 of 1,925 chars and stopped. | **Fixed** — attachment text reaches `check_attempt` through the tool closure (same path as images); `_merge_attachment` treats the closure copy as authoritative and discards partial router copies. Router budget raised to 900 for typed attempts. |
| A3 | Light model (`facts_chain`, `software_chain`, `recap_chain`) was `deepseek_v4_flash` with no fallback; a `deepseek_flash_with_fallback` wrapper existed unused. A DeepSeek outage meant no deadline answers. | **Fixed** — `app.py` uses the wrapper. |
| A4 | Agent LLM (`gpt-5.6-luna`) has no fallback. A dispatch failure drops the whole turn to the ungrounded `class_chain`. | Open. A `ModelWithFallback` does not forward `bind_tools`, so wrapping needs a small extension. See §5. |
| A5 | `_stream_answer` emitted a progress event — and re-rendered `st.status` — on every streamed token. | **Fixed** — one event on the first real token. |
| A6 | The "no tool prepared an answer → class_chain" branch in `app.py` is unreachable: `_final_answer` always returns non-empty text. Harmless, but misleading to read. | Open (cosmetic). |
| A7 | Leftover `print('-----turn_result')` / `print('-----chain')` debugging in the turn loop. | **Fixed** — removed. |
| A8 | A module filter that matches nothing abstained on a concept the index holds. Observed: "What does a p-value of 0.03 mean?" → router passes `hypothesis-testing` (from the CSV), the index still says `inference` (built before the rename), filter returns zero rows, tool abstains. A plausible-but-wrong module from the router fails the same way on a fresh index. | **Fixed** — `search_concepts` retries unfiltered when the filtered search is empty and records `widened_from_module` in the trace; `concept_taxonomy.index_drift()` warns (diagnostics view) when the CSV hash no longer matches the concept index's provenance stamp. |
| A9 | Three response modes (Direct / Hint-first / Step-by-step) for a setting only two of nine chains read. Logs: 459 / 48 / **1** uses across 508 turns. | **Fixed** — one switch, "Show me how to work through it" (on by default), mapped onto the two `response_mode` values the chains and analytics already use. Hint-first branches removed from the three prompts that had them; hints live in `coach_practice`, where withholding teaches something. |

---

## 2. Chains — purpose, prompt, payload

Payload keys are built in `ta_tools.py` and consumed by `RunnableParallel`
projections in `chains_lcel.py`; they agree by name only.
`tests/test_chain_contracts.py` now streams every tool's real payload through
its chain on a fake model, so a renamed key or a new `{placeholder}` fails in
CI rather than in front of a student.

| Chain | Tool | Model | Grounding | Honours response mode? | Verdict |
|---|---|---|---|---|---|
| `facts_chain` | `answer_course_facts` | light | Tier A block, whole | No (by design) | Correct. Forced **Answer / Check yourself** shape, Canvas link rule, reliability-advisory override rule 3. |
| `doc_chain` | `answer_course_documents` | main | Tier C chunks with `Source:` provenance lines | Only for the explanation after the facts | Correct. Filter-only questions get a synthesised query. |
| `software_chain` (+vision) | `answer_software` | light | model knowledge + versions/conventions/walkthrough links | No (by design) | Correct. "Never invent a menu path" rule is the right bar. |
| `concept_chain` (+vision) | `answer_concept` | main | Tier B concept payload (body + phrasing + mistake) | After the answer only | Correct. Prompt knows the three-part context shape. |
| `practice_chain` | `generate_practice` | main | the concept the topic names (Tier B, when within distance 1.0) + `previous_question` from session | n/a | **Fixed**: was ungrounded, and the "Practice this" chip asked for a keyword-inferred label ("Regression") rather than the concept just explained ("Interpreting R-squared"). `_concept_focus` now names the retrieved concept as the practice topic, and the tool grounds the question in that concept's body and common mistake. Short topic strings score closer to everything, so grounding uses its own bar (`PRACTICE_GROUND_MAX_DISTANCE = 1.0`, vs 1.40 for answering). |
| `coach_chain` | `coach_practice` | main | held question from session | n/a | Correct. Hint / clarify / worked_step with escalation after 3 hints. |
| `check_chain` (+vision) | `check_attempt` | main | held question **or** placeholder | n/a | **Fixed**: rule 0 told the model to announce it was grading blind whenever no practice question was open — which is the normal homework-check case. Now an explicit "no practice question is open; take the task from the attempt" placeholder (`NO_HELD_QUESTION`). |
| `class_chain` | fallback only | main | none | Yes | Still carries the literal **Step 1** heading `concept_chain`'s docstring complains about. Only reachable on exceptions; low priority. |
| `recap_chain` | every 8 turns | light | chat history | n/a | Fine. |

Cross-cutting prompt observations:

- `SHARED_POLICY` and `DAYTON_PERSONA` are injected into every chain; good.
  `VISION_POLICY` is a separate system turn so vision/non-vision builds cannot
  drift; good.
- `infer_learner_level` is a five-keyword heuristic ("sql join" is not in
  this course). It labels most MBA students "novice" in the prompt. Harmless
  but not informative; see §5 for what to do instead.
- `infer_curriculum_topic` returned **Descriptive statistics** for "What does
  an R-squared of 0.62 mean?" because the keyword `mean` was checked before
  `r-squared`. That set `last_practice_topic`, so the "Practice this" chip
  under an R-squared explanation drilled the wrong topic. **Fixed** —
  keyword matches are scored by total matched length with word boundaries.

---

## 3. The three course-information tiers

### Tier A — schedule and facts (`course_data/schedule.json` + `facts.toml`)

Rendered by `utils/course_context.render()` into a ~4 KB block and passed
whole to `facts_chain`. Not embedded. Canvas-owned facts (deadlines, modules,
announcements, pages) come from the sync; human-owned facts (office hours,
grading weights, policies, software versions) from the TOML. Reliability
advisories (`conflict` / `ended` / `drift` / `index`) lead the block and the
prompt is told to obey them over the dates below.

This is the strongest part of the system. Findings:

| # | Finding | Status |
|---|---|---|
| T1 | `TAs: TBD ()` was rendered from the placeholder entry, and would be repeated to students. | **Fixed** — placeholder/blank TA entries are skipped. |
| T2 | **No midterm date exists anywhere** (not a Canvas assignment, not in `facts.toml`), yet the first starter prompt was "When is the midterm exam?". The very first suggested click abstained. | **Fixed** the prompt (now office hours). **Open** on the data side: add the exam dates to `facts.toml` (see §5). |
| T3 | Canvas `grading_weights` (Engagement 10 / Individual 20 / Group 20 / Course Assessment 50) are synced but never rendered; `facts.toml` restates them with the 50% split into Midterm 20 / Final 30. The two agree today, and `facts.toml`'s own header says to delete the block once Canvas owns it. | Open — instructor decision. |
| T4 | `late_policy`, `regrade`, `academic_integrity` are empty and `ai_use` says "See syllabus". The `facts.toml` comments already flag these as the most-asked logistics questions. | Open — data entry. |
| T5 | `?debug=1` unlocks diagnostics for anyone when no `diagnostics_token` secret is set. README says to set it on deploy; worth confirming it is set. | Open — ops. |

### Tier B — curated concepts (`course_data/concepts.csv` → `data/concepts`)

43 rows, 8 modules, one vector per concept embedded from title + body;
`managerial_phrasing` and `common_mistake` ride as metadata and are
re-attached after retrieval. Module taxonomy is derived from the CSV
(`concept_taxonomy.py`) so the router's module list cannot drift from the
index. Quality bands are on raw Chroma distance, calibrated by script.

| # | Finding | Status |
|---|---|---|
| T6 | **Fixed (P1).** The student-facing topic pills were a different taxonomy from the Tier B modules. `CURRICULUM_TOPICS` / `CURRICULUM_SUBTOPICS` in `chains_lcel.py` are hand-written and offer *Probability → Bayes theorem / Random variables*, *Hypothesis testing → ANOVA / t-tests*, *Regression → Assumptions and diagnostics* — none of which exist in `concepts.csv`. "Explain a concept → Probability → Bayes theorem" routes to `answer_concept` and abstains. This was exactly the drift `concept_taxonomy.py` was written to prevent, one layer up. | **Fixed** — see §5 P1. |
| T7 | `module_is_unwritten` exists but is never called; today every module has written rows so nothing is lost, and an unwritten module would still abstain via an empty filter. | Open (cosmetic). |
| T8 | Tier B sources carry no URL, so the sources expander names the concept but cannot link it. Fine unless concepts get a Canvas page. | No action. |

### Tier C — class recaps and assignment briefs (`schedule.json` → `data/documents`)

Announcement bodies and assignment instructions, split only past 1,800 chars
with the title re-attached to every part, `ymd` stored as an int for exact
date filters, Canvas URL per chunk. Hybrid retrieval with a
chunks-per-document cap; date-filtered lookups cannot be abstained on by
distance. Provenance stamp compared by content hash so a sync without a
rebuild announces itself.

| # | Finding | Status |
|---|---|---|
| T9 | Canvas **pages** (25 of them — the JMP/Excel walkthroughs) are synced as title + URL only. Their bodies are not indexed, so `answer_software` can link a walkthrough but not quote it, and `answer_course_documents` cannot find "how did the instructor say to build a data table". | Open — see §5. |
| T10 | Assignments are dated by **due date**. "What did we cover last week" with `doc_type=""` also returns briefs that were *due* last week. Mostly harmless; the router usually sets `doc_type='announcement'` for that phrasing. | No action. |
| T11 | `days_back` is computed from UTC midnight; a student asking at 11 PM EDT on Sunday gets a window that starts a day early. | No action (one-day fuzz on a recency window). |

---

## 4. What was changed in this pass

Code:
- `utils/llm_models.py` — streaming fallback actually falls back; router budget 300 → 900 (`ROUTER_MAX_TOKENS`).
- `utils/attachments.py` — `extract_attachments` returns bare file blocks; `attachment_query_block()` adds the router-facing header.
- `utils/ta_tools.py` — `attachment_text` closure parameter; `_merge_attachment`; `NO_HELD_QUESTION` placeholder for `check_chain`; `_concept_focus` and concept-grounded `generate_practice`.
- `utils/agent_graph.py` — forwards `attachment_text`; router prompt no longer asks for attachments to be copied.
- `utils/chains_lcel.py` — `infer_curriculum_topic` scoring; `check_chain` rule 0; `practice_chain` CLASS MATERIAL block; doc_chain typo.
- `utils/course_context.py` — skip placeholder TA entries.
- `utils/sidebar.py` — help dialog no longer claims screenshots are unreadable; practice badges listed.
- `utils/ui.py` — first starter prompt is answerable from current data.
- `app.py` — light model gets its fallback wrapper; one progress event per stream; debug prints removed; attachment wiring.
- `README.md` — models, tool list, tests section; removed the false "rolling summary" claim.

Tests (`tests/`, 82 cases, no keys or index needed): chain/payload contracts,
tool behaviour with stub index, fallback wrapper, topic inference, Tier A
rendering and advisories, practice session, attachments, date resolution.

Verified live: app boots; "When are office hours, and where?" → `facts_chain`
via the new light-model wrapper with the blue provenance badge; a scripted
router turn with a 2,350-char attachment → `check_attempt` with
`attempt_text=""` from the router and the full attachment in the chain payload.

---

## 5. Proposed, not implemented (need a decision)

**P1. Derive the topic pills from `concepts.csv` (T6). — DONE 2026-08-23.**
`utils/concept_taxonomy.py` now yields the pill tree (`outline()`), the
top-level labels (`curriculum_topics()`), the subtopics per module, a
"Module: Topic" focus parser (`split_focus()`), and CSV-derived keyword
inference (`infer_module()`) — unigrams and bigrams from module ids, topic
labels and titles, weighted by how many modules share them. The hand-written
`CURRICULUM_TOPICS`, `CURRICULUM_SUBTOPICS` and `_TOPIC_KEYWORDS` tables are
gone from `chains_lcel.py`. The CSV's `topic` column was relabelled into
student-facing labels (25 distinct values, e.g. `one-variable-data-table` →
"One-variable data table", `evaluate extreme` → "Extreme values and
z-scores"); it is now the pill label and the grouping key. Reads are cached
on the file's mtime, so editing the CSV changes the pills on the next rerun
with no rebuild. On the vat-research side the concepts adapter's build
report carries `outline` (the same tree) and the lint flags id-looking
labels, two spellings of one label in a module, and written rows with no
topic, so `build_concept_index(dry_run=True)` previews the pills. Practice
grounding uses the module from a pill focus as a Chroma *filter* (bar 1.2
in-module vs 1.0 unfiltered); measured over all 25 pills every one grounds on
a correct concept. The `learning_objective` analytics vocabulary changed from
the six old labels to the eight module labels plus "JMP / Excel workflows".

**P2. Add exam facts to `facts.toml` (T2).** An `[exams]` table (midterm
date/format/coverage, final date/format) renders into Tier A under a new
EXAMS header. Ten lines of TOML plus ten of Python; the only reason it is not
done is that the dates are not in the repo.

**P3. Index Canvas page bodies into Tier C as `doc_type='page'` (T9).** The
sync would fetch page bodies (the Canvas MCP exposes `get_page`), the
documents adapter would emit them, and `answer_course_documents` would accept
`doc_type='page'`. Walkthrough pages are the instructor's own words on exactly
the JMP/Excel tasks students ask about most. Needs a change in
`vat-research/vector_index` and the sync, so it crosses repos.

**P4. Fallback for the router (A4).** Extend `ModelWithFallback` with
`bind_tools` that binds both models and returns a wrapper, then route
`gpt-5.6-luna → deepseek-v4-flash`. Small, but tool-call fidelity differs
between providers, so it wants a few routed test questions before shipping.

**P5. Replace `infer_learner_level`.** Either drop the field from prompts or
replace the keyword heuristic with something the app already knows: number of
attempts checked, hints used, abstentions. Today it mostly says "novice".

**P6. Keep scaffolding out of model history** (`ux_audit.md` P2). Clarify
turns ("Practice question" / "Which part of Regression?") are fed to every
chain as real conversation. Tagging them with `additional_kwargs={"scaffold":
True}` and filtering in `format_chat_history` / `_history_to_messages` is a
small diff.

**P7. Grading weights: pick one owner (T3).** Either delete the table from
`facts.toml` and render Canvas `grading_weights`, or keep the TOML (more
detailed) and add a one-line advisory when the Canvas totals disagree with it.

---

## 6. Compound turns (implemented 2026-08-23)

Measured before, on "how to do regression in JMP and how to interpret R2":
router 2.2 s, then `software_chain` 6.2 s / 357 words, then `concept_chain`
3.8 s / 155 words -- ~12 s wall clock, the JMP section explaining R-squared
before the R-squared section did, both parts signing off with their own
question, and no cap on the software section's length. In the query log, 28
of 238 turns (12%) called two tools.

Two changes, independent and additive:

**Compound-aware sections.** Every tool records `ToolStep.covers` (what its
section answers, usually the router's own query argument).
`ta_tools.annotate_compound_turn`, called from `run_ta_turn`, writes a
`turn_context` block into each streamed section's payload when there are two
or more: part N of M, what the other parts cover, a bold heading, a
150-word cap, no greeting, and a follow-up question on the last part only.
Every chain template carries a `{turn_context}` slot that renders "" on a
single-tool turn. `software_chain` also gained a 200-word cap and an explicit
rule for when no course conventions are recorded (it had invented one).

**Concurrent streaming.** `app._stream_sections` starts one worker thread
per section, all feeding a single queue; the script thread drains it into
placeholders created up front in router order, then draws each section's
footer. A section whose chain fails shows a one-line apology and the turn
continues; only if every section fails does the turn fall to `class_chain`.

After: **5.5-5.8 s** wall clock, JMP part ~110 words ending "see the
R-squared part below", R-squared part ~120 words with the only follow-up
question, both badges and the sources popover intact. Conventions from the
updated `facts.toml` (Minimum Report, Indicator Parameterization) appear in
the JMP steps.

`scripts/smoke_turn.py` runs a turn through `app.py` headlessly with
Streamlit's AppTest and prints every section and any swallowed exception --
this is how the signature drift in `ui.render_sources` was caught.

---

## 7. Prototype trim (2026-08-23)

The app is a teaching prototype, not a product, and was trimmed to the code
that tutors before being forked for ISOM 352. Each removal below had earned
its place by fixing an observed problem; what it cost in reading and
maintenance outweighed that for a prototype. The decision list that preceded
this pass is the "ISOM 550 Trim List" note; this records what was done.

**Removed.**
- The LangGraph state machine (`utils/agent_graph.py`), the progress reporter
  (`utils/progress.py`), and the threaded concurrent section streaming in
  `app.py`. These were one decision: the graph only ever ran
  agent → tools → END, its `ToolNode` thread pool was the reason progress
  events needed thread-aware buffering, and the streaming threads were the
  other thread-heavy piece. `utils/router.py` is the same control flow as a
  plain loop -- one `bind_tools().invoke()`, then each tool call in order on
  the main thread -- and `app._stream_sections` is a `for` loop. Cost: the
  12% of turns that call two tools now wait for the sections in sequence
  (§6 measured 5.5-5.8 s concurrent; sequential is ~9-10 s), and a tool call
  the router mis-formats is dropped rather than retried. The `st.status`
  line is updated at three points (routed, writing, done) instead of per event.
- The learner profile: `infer_learner_level`, `detect_attempt_check`,
  `build_learning_profile`, and the `learner_level` / `learning_objective` /
  `attempt_check` keys in five prompt templates (P5). `learning_objective`
  survives only as the analytics column the weekly report groups by, computed
  once in `app.py`.
- The route-keyed follow-up chip table (14 chips, `ui.follow_ups_for`).
  Three fixed chips after every answer -- explain a concept, practice this,
  check my work -- reusing the quick-action intents. The clarify flow (topic
  → subtopic pills) was kept.
- The Chroma reopen/retry wrappers in `retrieval.py`; `search_concepts` and
  `search_documents` call the store directly and still degrade to "no
  sources" on an exception.
- Dead code (`render_recap`, `get_sidebar_settings`, `update_session_stats`,
  `process_and_store_query`, `format_source_block_from_debug`,
  `module_is_unwritten`) and stale scripts (`build_index.py`,
  `eval_prompt_styles.py`, `baseline_metrics.py`).

**Simplified.** Two providers instead of four: DeepSeek V4 Pro/Flash write
the answers, GPT Luna routes, reads screenshots, and is the fallback for both
DeepSeek wrappers. `langgraph` and `langchain_anthropic` left
`requirements.txt`; `XAI_API_KEY` and `ANTHROPIC_API_KEY` are no longer read.

**Unchanged on purpose.** The three course-information tiers and their
reliability advisories, the retrieval quality bands and their refinements,
the held practice question, attachments and vision, provenance badges and the
abstain panel, compound-turn annotation, instructor diagnostics, the
CSV-derived taxonomy, MongoDB event logging, the shared index builder in
vat-research, and the tests.

Verified after the pass: 102 tests pass; `smoke_turn.py` on a facts question
(2.4 s, blue badge), a two-tool question (9.8 s, two sections with the
compound annotation -- part 1 ends "see the next part", only part 2 asks the
follow-up), a no-tool greeting (router text under the grey badge), and a
practice question (green badge, grounded on one concept).
