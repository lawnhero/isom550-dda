# Developer reference — ISOM 550 Virtual TA

A lookup document, not a narrative: every functional unit in the turn
pipeline, its exact inputs and outputs, and where it lives. Cross-reference
with `docs/architecture_audit.md` (why things are built this way, and known
issues) and `docs/ux_audit.md` (student-facing route map).

Regenerate the parts of this file that quote code (payload key lists, route
tables) whenever a chain's `RunnableParallel` setup or a tool's signature
changes — they are transcribed from source, not aspirational.

---

## 1. Turn pipeline, top to bottom

```
app.main()
  -> extract_attachments(uploaded_files)              [utils/attachments.py]
  -> build_ta_agent(...)                                [utils/router.py]
       -> build_ta_tools(...)                           [utils/ta_tools.py]
       -> agent_llm.bind_tools(tools)
  -> run_ta_turn(agent, query, chat_history, artifacts, on_route)
       -> ONE model call: the router returns 0..N tool calls (and no student text
          unless it returns zero)
       -> on_route(tool names)  -> app updates the st.status label
       -> each tool call runs IN ORDER on the main thread; each PREPARES a
          StreamSpec on its ToolStep and returns a JSON receipt
       -> annotate_compound_turn(steps)                 [if >=2 sections]
  -> app._stream_sections(answerable_steps, status)
       -> for each section, in router order: chains_lcel.<chain>.stream(payload),
          then ui.render_answer_footer(...) (badge, sources, thumbs)
  -> practice session state written (utils/practice.py)
  -> event logged to MongoDB (utils/utils.py)
  -> st.session_state.message_meta[index] written, st.rerun()
```

Two LLM phases per turn: **router** (picks tools, writes no student-facing
text) then **N chain calls** (write the actual answer, streamed). The router
never sees retrieved text; each tool returns only a receipt
(`{answer, abstained, tool_name, stream_ready, step_id}`) — see
`ta_tools._serialize_tool_result`.

---

## 2. Router

**File:** `utils/router.py`
**Model:** `agent_llm` = `llms.openai_gpt56_luna` (no fallback — a routing failure drops the turn to `class_chain`)
**Control flow:** one `model.invoke(messages)`; then `for call in reply.tool_calls: tools[name].invoke(args)`. No loop, no retry: a tool that raises is logged and its section is dropped; the other sections still stream.

| | |
|---|---|
| **Input** | `SystemMessage(AGENT_SYSTEM_PROMPT + Tier B module list)` + last `memory_window` messages of `chat_history` + `HumanMessage(query_for_model)` |
| **`query_for_model` composition** (built in `app.py`) | `user_query + attachment_note + describe_images(images) + attachment_query_block(attachment_text)` |
| **Output** | Zero or more tool calls (name + args), OR plain text (shown to student only when no tool was called) |
| **Tools bound** | The 7 `StructuredTool`s from `build_ta_tools` (§3) |
| **`on_route`** | Optional callback, called with the tool names after the router answers and before any tool runs. `app.py` uses it to set the status label (`ui.working_label`) |
| **`run_ta_turn` return dict** | `steps, answerable_steps, answer, tools_used, tool_calls, route_label, sources, retrieval_debug, practice_topic, abstained, retrieval_quality, router_ms, router_text, trace` |

**System prompt structure** (`AGENT_SYSTEM_PROMPT`, `utils/router.py`): tool-selection guide, then 9 numbered rules — the deadline gate (rule 1), "what did we cover" overview-vs-detail split (rule 2), assignment-vs-concept (rule 3), software-vs-concept + compound-question instruction (rule 4), practice/coach/check-attempt boundaries (rule 5), attachment handling (rule 6), dispatcher-not-writer (rule 7), no-tool-call behavior (rule 8), tone (rule 9). Appended at call time: the Tier B module list from `concept_taxonomy.format_modules_for_prompt()`.

---

## 3. Tools (`utils/ta_tools.build_ta_tools`)

All 7 are closures built fresh per turn, sharing `chat_history`, `response_mode`, `course_context`, `software_context`, `course_span`, `practice_session`, `attachment_text`, `images`, `artifacts` via the closure. Each tool: (1) claims a `ToolStep` via `artifacts.new_step(name)`, (2) either sets `step.static_answer` (abstain / ask-for-input) or `step.stream_spec = StreamSpec(chain_key, payload)`, (3) returns a JSON receipt via `_serialize_tool_result`.

### 3.1 `answer_course_facts(query: str)`
| | |
|---|---|
| Purpose | Dates, deadlines, office hours, people, grading weights, materials, "what's covered so far" |
| Retrieval | None — Tier A context block passed whole |
| Abstains when | `course_context` is empty (no snapshot loaded) |
| Chain | `facts_chain` |
| Payload keys | `course_context, chat_history, query, turn_context` |
| `step.covers` | the query verbatim |

### 3.2 `answer_course_documents(query="", doc_type="", days_back=0, on_date="", date_span="day")`
| | |
|---|---|
| Purpose | Assignment briefs and class-session recaps (Tier C) |
| Date resolution | Done in Python — `days_back` → `since_ymd`; `on_date` + `date_span` → `(from_ymd, to_ymd)` via `retrieval.resolve_date_range(on_date, course_span, date_span)` |
| `top_k` | 8 if a date range spans >1 day, else 4 |
| Abstains when | no query and no filter (asks for one); or retrieval quality is `"none"` |
| No-query path | `query=""` + a filter present → `retrieval.fetch_by_filter` (sorted by `ymd` desc, `quality="strong"` unconditionally — the filter established relevance) |
| Chain | `doc_chain` |
| Payload keys | `context, query, chat_history, response_mode, turn_context` |
| `step.covers` | the query, or `_filter_only_question(doc_type, days_back, on_date, date_span)` when query was empty |

### 3.3 `answer_software(query: str)`
| | |
|---|---|
| Purpose | JMP/Excel how-to |
| Retrieval | None, deliberately (model's own tool knowledge, grounded by `software_context`) |
| Chain | `software_chain` (or `software_chain_vision` if `images` present) |
| Payload keys | `software_context, chat_history, query, turn_context` (+ `images` if vision) |

### 3.4 `answer_concept(query: str, module: str = "")`
| | |
|---|---|
| Purpose | What a statistic/concept MEANS |
| Retrieval | `_prepare_rag_tool` → `retrieval.search_concepts(contents_db, query, module, top_k=4)`. `module` narrows via a Chroma `where` filter (`concept_taxonomy.build_concept_filter`) |
| Abstains when | `RetrievalResult.quality == "none"` |
| `practice_topic` | `_concept_focus(found)` — the retrieved concept's own title (strong match) or its module label (weak match), NOT a keyword-inferred label |
| Chain | `concept_chain` (or `_vision`) |
| Payload keys | `context, query, chat_history, response_mode, turn_context` (+ `images` if vision) |
| `step.covers` | the query verbatim |

### 3.5 `generate_practice(topic: str, difficulty: str = "same")`
| | |
|---|---|
| Purpose | Write ONE new practice question |
| Topic default | `chains.infer_topic_from_history(chat_history)` if `topic` empty |
| Difficulty | must be in `practice.DIFFICULTIES` (`easier, same, harder`), else `"same"` |
| Grounding | `search_concepts(contents_db, topic, module=module_id, top_k=1)`; module parsed from a `"Module: Topic"` focus string via `concept_taxonomy.split_focus`. Grounds only if `best_distance <= bar`, where `bar = PRACTICE_GROUND_MAX_DISTANCE_IN_MODULE (1.2)` with a module filter or `PRACTICE_GROUND_MAX_DISTANCE (1.0)` without — short topic strings score closer to everything, so this needs a tighter bar than answering |
| `probable_misroute` (trace only, not blocking) | true when a practice session is active AND `hints_given > 0` — student likely said "I'm stuck", not "give me another" |
| Chain | `practice_chain` |
| Payload keys | `topic, difficulty, chat_history, previous_question, concept_context, turn_context` |
| `previous_question` | `practice.question_of(practice_session)` — so "harder" escalates instead of repeating |
| `step.covers` | `f"a new practice question on {topic}"` |

### 3.6 `coach_practice(query: str, request: str = "hint")`
| | |
|---|---|
| Purpose | Help on the practice question already on screen — never generates a new one |
| Abstains when | `not practice.is_active(practice_session)` → static "no practice question open" |
| `request` resolution | `practice.effective_request(session, request)` — escalates a bare `"hint"` to `"worked_step"` once `hints_given >= practice.MAX_HINTS (3)` |
| Chain | `coach_chain` |
| Payload keys | `topic, request, question_block, chat_history, query, turn_context` |
| `question_block` | `practice.prompt_block(session)` — the held question + hint/attempt counts |
| `step.covers` | `f"help ({resolved}) with the practice question on screen"` |

### 3.7 `check_attempt(attempt_text: str, topic: str = "")`
| | |
|---|---|
| Purpose | Rubric-style feedback on a student attempt |
| Attachment merge | `_merge_attachment(attempt_text, attachment_text)` — the closure's attachment text is authoritative; router's copy is kept, replaced, or appended depending on containment (see docstring — this exists because the router used to truncate long attachments mid-copy) |
| Empty-attempt fallback | if no text but `images` present → `"(The student's work is in the attached screenshot...)"`; if still nothing → static "please paste your attempt" |
| Topic default | `chains.infer_topic_from_history(chat_history)` if empty |
| `question` payload | `practice.question_of(session) or NO_HELD_QUESTION` — an explicit "no practice question open, grade the attempt on its own terms" placeholder, not silence |
| Chain | `check_chain` (or `_vision`) |
| Payload keys | `topic, attempt_text, question, chat_history, turn_context` (+ `images` if vision) |
| `step.covers` | `f"feedback on the student's attempt ({topic})"` |

### 3.8 Tool receipt (what the router sees)
```json
{"answer": "...", "abstained": bool, "tool_name": "...", "stream_ready": bool, "step_id": "..."}
```
Deliberately NOT the retrieved text or the chain payload — those stay on `TurnArtifacts.steps[step_id]`, read directly by `app.py` after the graph finishes. This is what keeps the router from writing a redundant, discarded answer.

---

## 4. Chains (`utils/chains_lcel.py`)

Every chain is `RunnableParallel(setup) | ChatPromptTemplate | llm | StrOutputParser()`; vision variants wrap with `_with_vision` (prepends `VISION_POLICY` as a system turn, converts the human turn to `[text, image_url...]` parts). `_turn_context(payload)` is mixed into every `setup` dict and reads `payload.get("turn_context") or ""` — renders empty on ordinary single-tool turns.

| Chain | Called by | Model (see §5) | Grounding | Honours `response_mode`? |
|---|---|---|---|---|
| `class_chain` | no-tool / fallback paths only | main | none | Yes |
| `facts_chain` | `answer_course_facts` | light | Tier A block (whole) | No — always the same shape |
| `software_chain`(`_vision`) | `answer_software` | light | model knowledge + `software_context` | No |
| `doc_chain` | `answer_course_documents` | main | Tier C chunks | Only the explanation added after the facts |
| `concept_chain`(`_vision`) | `answer_concept` | main | Tier B concept payload | Only what's added after the answer |
| `practice_chain` | `generate_practice` | main | optional Tier B concept (`concept_context`) | n/a |
| `coach_chain` | `coach_practice` | main | held practice question | n/a |
| `check_chain`(`_vision`) | `check_attempt` | main | held question or `NO_HELD_QUESTION` | n/a |

### 4.1 `class_chain(llm)`
- **Input keys:** `query, chat_history, turn_context, response_mode`
- **Output shape:** varies by `response_mode` — `**Answer**/**Check yourself**` (Direct), `**Step 1**/**Checkpoint**` (Teach step-by-step). ≤180 words.
- **Built from:** `build_chain_payload(query, chat_history, response_mode, context="", memory_window)` in `chains_lcel.py`.

### 4.2 `facts_chain(llm)`
- **Input keys:** `course_context, chat_history, turn_context, query`
- **Output shape:** plain prose fact, then `"You can verify at <where, with Canvas link if present>"`. No response-mode variation. Under 120 words.
- **Special rule:** if `course_context` opens with a `!! SCHEDULE RELIABILITY` advisory block, the model must obey its `->` instruction (e.g. "do not state specific dates") before answering — see `course_context.advisories`.

### 4.3 `software_chain(llm, vision=False)`
- **Input keys:** `software_context, chat_history, turn_context, query` (+ `images` if vision)
- **Output shape:** numbered concrete steps, version assumption stated, "never invent a menu path" (says so if unsure), course-convention override rule, walkthrough link if one matches. Capped at 200 words. Explicit rule: if no COURSE CONVENTIONS are listed, do not claim the course expects anything.

### 4.4 `doc_chain(llm)`
- **Input keys:** `context, query, chat_history, turn_context, response_mode`
- **Output shape:** reports document content plainly first (names the document + Canvas link), THEN adapts only the follow-on explanation to `response_mode`. Under 200 words unless asked to list/summarise multiple documents.

### 4.5 `concept_chain(llm, vision=False)`
- **Input keys:** `context, query, chat_history, turn_context, response_mode` (+ `images` if vision)
- **Context shape it expects:** up to 3 labelled parts per concept — instructor explanation, `"How to phrase it: ..."`, `"Common student mistake: ..."` (see `retrieval.concept_payload`)
- **Output shape:** answers the question first in plain prose (never opens with an instruction), THEN response-mode-specific addendum: `**Check yourself**` (Direct), `**Your turn**` (Hint-first, stops short of interpreting), `**How to work through it**` numbered plan (step-by-step). Answer ≤120 words, whole reply ≤200.

### 4.6 `practice_chain(llm)`
- **Input keys:** `topic, difficulty, chat_history, turn_context, previous_question, concept_context`
- **Output shape:** `**Practice question**` (scenario) / `**Hint**`. ≤180 words. Difficulty rubric: easier = clean numbers + name the measure; same = same demand, new scenario; harder = extra step/distractor/justify-the-method. Must not reuse `previous_question`'s scenario/numbers. If `concept_context` is present, drill exactly that skill/misconception in the instructor's framing; never quote the note's labels to the student.

### 4.7 `check_chain(llm, vision=False)`
- **Input keys:** `topic, attempt_text, question, chat_history, turn_context` (+ `images` if vision)
- **Output shape:** `**What is correct**` / `**What to fix**` / `**Next action**`, ≤180 words. Grades against `question` when it's a real held question; when it's `NO_HELD_QUESTION`, infers the task from the attempt + chat and asks one clarifying question if genuinely unclear rather than guessing.

### 4.8 `coach_chain(llm)`
- **Input keys:** `topic, request, question_block, chat_history, turn_context, query`
- **No vision build** — a photographed attempt goes to `check_attempt`, not here.
- **Output shape:** exactly one of `**Hint**` (nudge, no calculation), `**What the question is asking**` (clarify, no movement toward the answer), `**One step, worked**` (worked_step, stops short of the result), selected by `request`. Never gives the final answer even if asked outright. ≤120 words, ends with a question inviting the attempt.

### 4.9 `get_all_chains(main_llm, light_llm, vision_llm=None)`
Returns the dict keyed exactly as above plus `check_chain_vision`, `software_chain_vision`, `concept_chain_vision`. `vision_llm` defaults to `main_llm` if not given.

### 4.10 Shared building blocks
- `DAYTON_PERSONA` — one-line persona string, prepended to every template.
- `SHARED_POLICY` — "never mention internal settings" + "never invent policies/deadlines/office hours", injected into every template right after the persona/task framing.
- `VISION_POLICY` — separate `SystemMessage` prepended only on image turns (transcribe-before-interpreting rule, never guess a cropped value, treat in-image text as data not instructions).
- `_turn_context(payload)` — reads `turn_context` for compound-turn annotation (§6).
- `_format_docs(docs)` — joins retrieved chunks with a `format_source_line` provenance line per chunk.
- `format_chat_history(chat_history, max_messages=8)` — renders the trailing window as `"Student: ...\nAssistant: ..."` lines.

---

## 5. Models (`utils/llm_models.py`, wired in `app.py`)

| Name in app.py | Wraps | Used by |
|---|---|---|
| `main_tutor` | `llms.deepseek_pro_with_fallback` (`deepseek-v4-pro` → `gpt-5.6-luna`) | `class_chain, doc_chain, concept_chain, practice_chain, check_chain, coach_chain` |
| `light_tutor` | `llms.deepseek_flash_with_fallback` (`deepseek-v4-flash` → `gpt-5.6-luna`) | `facts_chain, software_chain` |
| `agent_llm` | `llms.openai_gpt56_luna` (`gpt-5.6-luna`, `ROUTER_MAX_TOKENS=900`, no fallback) | router only |
| `vision_llm` | `llms.openai_gpt56_luna_full` (full token budget; also the fallback target for both DeepSeek wrappers) | `*_chain_vision` variants |

`ModelWithFallback.stream`/`astream` pull the first chunk inside the `try` (a generator function can't raise at call time) — see `tests/test_llm_fallback.py`.

---

## 6. Compound turns

**File:** `utils/ta_tools.annotate_compound_turn`, called from `router.run_ta_turn` right after `answerable_steps` is computed.

- Triggers when ≥2 `ToolStep`s have a `stream_spec` (static-answer-only steps, e.g. an abstention next to one real answer, do NOT count).
- Writes `step.stream_spec.payload["turn_context"]` for every streamed step: `"THIS IS PART N OF M..."`, one line per part naming what it covers (`step.covers`, falling back to the route's badge label), then rules — open with a ≤6-word bold heading, cover only your part, ≤`COMPOUND_SECTION_WORDS` (150) words, no greeting, only the LAST part ends with a follow-up question.
- Rendering: `app._stream_sections` streams the sections one after another in router order, each in its own container with its own footer. A section whose chain raises shows `SECTION_FAILED` and the turn continues; if every section fails, the exception propagates to the outer `except` and the turn falls to `class_chain`.

---

## 7. Course-information tiers

| Tier | Source | Loaded by | Consumed by | Retrieval? |
|---|---|---|---|---|
| **A — facts** | `course_data/schedule.json` (synced) + `course_data/facts.toml` (hand-written) | `course_context.get_course_context()` → `render()`, cached on file mtime + UTC date | `answer_course_facts` → `facts_chain` | None — whole block passed |
| **B — concepts** | `course_data/concepts.csv` → `data/concepts` (Chroma) | `load_db('data/concepts')` in `app.py` | `answer_concept`, `generate_practice` (grounding) | Hybrid vector+keyword, `search_concepts` |
| **C — documents** | `course_data/schedule.json` → `data/documents` (Chroma) | `load_db('data/documents')` in `app.py` | `answer_course_documents` | Hybrid vector+keyword or exact filter, `search_documents` / `fetch_by_filter` |

`course_context.render()` also emits `!! SCHEDULE RELIABILITY` advisories (`conflict`/`ended`/`drift`/`index` — see `course_context.advisories`) that `facts_chain` must obey before stating any date.

`concept_taxonomy.py` derives the module list, curriculum pill labels/subtopics, and keyword-based topic inference (`infer_module`) entirely from `concepts.csv` — nothing is hand-maintained in `chains_lcel.py` any more (`curriculum_topics`, `get_subtopics`, `infer_curriculum_topic` are thin wrappers over it).

---

## 8. Session state written by a turn (`app.py`)

| Key | Written by | Read by |
|---|---|---|
| `chat_history` | `main()`, appended every turn | everything (rendering, chain payloads) |
| `message_meta[index]` | `_set_meta` at end of turn | `_render_ai_message` on rerun (badges, sources, diagnostics) |
| `practice_session` | `practice.start/record_hint/record_attempt`, called from `main()` based on which route answered | `coach_practice`, `check_attempt`, `generate_practice` (all read via the `practice_session` closure param) |
| `last_practice_topic` | `main()`, from `turn_result["practice_topic"]` or history inference | `"Practice this"` follow-up chip, quick-action topic default |
| `pending_intent` | `_append_clarifying_turn` | clarify-flow pill rendering, `_resolve_pending_intent` |

---

## 9. Where to look for more

- `docs/architecture_audit.md` — findings, fixes, and the compound-turn design rationale with before/after timing.
- `docs/ux_audit.md` — student-facing entry points, route map, provenance badges, follow-up chips.
- `tests/test_chain_contracts.py` — streams every tool's real payload through its named chain on a fake model; the fastest way to see a payload/template mismatch.
- `tests/test_ta_tools.py` — tool-level behavior (abstention, attachment merge, compound annotation) without any LLM.
- `scripts/smoke_turn.py` — runs one real turn through `app.py` headlessly (Streamlit `AppTest`) and prints every section plus any swallowed exception.
