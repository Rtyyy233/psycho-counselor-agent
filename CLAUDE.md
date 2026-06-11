# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run CLI mode
cd src && python user_interface.py

# Run Web server
cd src && python -m web.main

# Run all tests
cd src && python -m pytest ../test/ -v

# Run a single test file
cd src && python -m pytest ../test/test_supervisor.py -v

# Run a single test
cd src && python -m pytest ../test/test_supervisor.py::test_xxx -v

# Type check (no config provided, use pyright or mypy as available)
# Lint
ruff check src/

# Format
ruff format src/
```

## Architecture

A Chinese-language psychological counselor AI agent using a **2-agent observer pattern** over an asynchronous event-driven core. The analyst and supervisor roles have been merged into a single unified Supervisor agent.

### Core Loop (src/user_interface.py)

1. User sends a message → `SharedContext.add_message()` appends it and fires `supervisor_trigger` event
2. Background task `call_supervisor` starts an asyncio task — waits on the trigger, examines recent messages, and may set `PromptInjection` on the context
3. `chatter.ainvoke()` runs synchronously with the user message — it receives injections (appended to the prompt) and produces the final reply
4. The reply is added to context and the loop repeats

### Two Agents

- **chatter.py** — Frontline counselor. Uses `ToolStrategy(ChatterOutput)` with structured output (`reply`, `should_retrieve`, `retrieve_query`). Has `read_file_tool`, `retrieve_user_profile_tool`, `lookup_skill`, `get_available_skills`. System prompt: integrative therapy expert grounded in person-centered + existential base, with 15 progressive-disclosure therapy skills. Never reveals multi-agent architecture to user.

- **supervisor.py** — Unified analyst + supervisor (replaces both old `analysist.py` and old `supervisor.py`). Uses `ToolStrategy(SupervisionOutput)` with structured output (`should_inject`, `injection_content`, `should_retrieve`, `retrieve_query`). Has all 3 retrieval tools, read_file, store tools, `lookup_skill`, `get_available_skills`. 3-tier monitoring framework: L1 base monitoring (alliance/process-quality), L2 on-demand (countertransference/pattern-recognition), L3 crisis (crisis-detection). Locked to prevent concurrency.

### analysist.py (deprecated)

Now a backward-compatibility shim: re-exports `supervisor.SupervisionOutput` as `analysis`, `supervisor.supervisor` as `analysist`, and `call_supervisor` as `call_analysist`. All analyst functionality has been merged into `supervisor.py`.

### supervisoner.py (separate module)

A different supervisor implementation using `with_structured_output(SupervisorResult)` for LLM-based judgment with heuristic fallback. Used by `web/main.py` rather than supervisor.py.

### Skill System (src/skill_loader.py)

Progressive disclosure for therapy and supervision skills using `skillkit`:

- **L1** (metadata, ~100 tokens) — always in system prompt via skill catalog
- **L2** (full instructions) — loaded on-demand via `lookup_skill(name)` tool
- **L3** (resources) — referenced within full instructions

Two tool functions exposed to agents:
- `lookup_skill(name)` — load a skill's full guide (max 3 calls per round)
- `get_available_skills()` — list all skills with trigger conditions (max 1 call per round)

Chatter skills (15) in `src/skills/chatter/`: person-centered, existential, psychodynamic, adlerian, gestalt, cbt, behavioral-third-wave, choice-reality, sfbt, narrative, feminist, family-systems, alliance-repair, clinical-interviewing, crisis-intervention.

Supervisor skills (5) in `src/skills/supervisor/`: alliance-monitoring, process-quality, countertransference, pattern-recognition, crisis-detection.

### Merge Router (src/merge_router.py)

When both chatter and supervisor independently decide to retrieve, `route_queries()` uses an LLM to decide whether to merge semantically similar queries into one retrieval, avoiding duplicate Chroma calls.

### User Profile System (src/user_profile/)

Systematic psychological profiling with 14 domains across 4 categories:

| Category | Domains |
|----------|---------|
| baseline (stable traits) | 2-Growth/Development, 3-Predisposing, 7-Relational Patterns, 10-Cultural/Contextual, 11-Personality Impression |
| dynamic (changes with therapy) | 1-Presenting Complaint, 4-Precipitating, 5-Perpetuating, 6-Protective, 8-Intervention Response, 9-Risk Assessment, 14-Help-Seeking & Change |
| emotional_world | 12-Emotional World |
| communication | 13-Communication Style |

Key modules:
- `profile_models.py` — Pydantic models: Fact (6 types), ProfileDomain, Profile (with version history), 14-domain registry
- `profile_collector.py` — `CollectedData` dataclass with 4 collection strategies (PAIP scan, diary scan, material semantic search, conversation semantic search). Both `collect_all_data()` and `collect_targeted_data()` public APIs.
- `profile_generator.py` — 3-round LLM generation: R1 (domains 1-5 from PAIP+diary), R2 (6-10 from R1+materials+PAIP), R3 (11-14 from R1-10+conversation chunks). Plus `generate_domain()` and `screen_domains_for_update()` for incremental updates.
- `profiler.py` — Orchestrator: `init_profile()` (full rebuild, version=1), `update_profile()` (incremental via screening → targeted collect → single-domain update), `get_or_init_profile()`.
- `mem_retrieve_user_profile.py` — Retrieval: file-first (JSON from `database/profiles/<user_id>/profile.json`), Chroma-fallback. Returns `ProfileRetrievalResult` with domain-level access. Evidence resolution via `lookup_evidence_content()`.

### Memory System

Three data types, each with a store module and a retrieval module:

| Type | Store | Retrieval | Chroma Collection(s) |
|------|-------|-----------|---------------------|
| Diary | `mem_store_diary.py` — date splitting, semantic chunking, LLM extracts emotion/cognition/behavior/scene tags | `mem_retrieve_diary.py` — LangGraph state machine (planner → route → search nodes) | `original_diary`, `diary_annotation` |
| Materials | `mem_store_material.py` — type inference, parent-child semantic chunking | `mem_retrieve_material.py` — LangGraph state machine (semantic → parent_lookup small-to-big) | `child_chunks`, `parent_chunks` |
| Conversation Outline | `mem_store_conv_outline.py` — PAIP (Problem/Assessment/Intervention/Plan) extraction + parent-child chunking | `mem_retrieve_conv_outline.py` — LangGraph state machine (semantic → paip_outline_lookup) | `conv_outline` |

All retrieval modules use the same pattern: an LLM planner generates a multi-step plan, a dispatcher routes each step to the correct node, and an `after_execution` function advances `current_step_idx`.

### SharedContext (src/SharedContext.py)

Thread-safe async context container. Key features:
- `asyncio.Lock`-protected message list with timestamp-based expiry
- `PromptInjection` dataclass carrying supervisor injection content with 5-min TTL
- Token tracking with optional DeepSeek V3 tokenizer (falls back to char estimation)
- `cleanup_context()` — removes oldest messages against a target usage %, storing a summary via callback

### Web Layer (src/web/)

FastAPI server at port 8000. WebSocket chat (`/ws/chat`), file upload (`/api/upload`), session CRUD. Static files in `src/web/static/`. Session persistence as JSON in `web/sessions/`. Uses `supervisoner.py` (not supervisor.py) for the supervisor role.

### Configuration

- `src/config.py` — centralized: file limits, timeouts, paths, keyword patterns for file type detection
- `src/mem_integration.py` — module that wires together Chroma collections (created with OllamaEmbeddings), tool definitions, and the memory_manager agent (deprecated)
- `.env` — `DEEPSEEK_API_KEY`, `DATA_DIR=database`, LangSmith settings

### Dependencies

- **LLM**: DeepSeek via `langchain-deepseek` (model `deepseek-v4-flash`)
- **Embeddings**: Ollama `qwen3-embedding:4b` (must run `ollama serve` locally)
- **Vector DB**: ChromaDB (SQLite-backed)
- **Framework**: LangChain agents + LangGraph state machines
- **Web**: FastAPI + WebSocket
- **Skill system**: `skillkit` for progressive disclosure

### Testing

Tests are in `test/` using pytest. Since the system is heavily async, many tests use `pytest-asyncio`. Test filenames mirror source modules.

Some tests may require a running Ollama instance and the embedding model pulled. Unit tests that mock external services exist in several test files.

### Project Structure

```
src/
  user_interface.py       — main CLI entry point
  config.py               — centralized config constants
  SharedContext.py        — thread-safe async context
  session_manager.py      — JSON session persistence
  conversation_manager.py — context with automatic summarization
  chatter.py              — frontend conversation agent (integrative, 15 skills)
  supervisor.py           — unified analyst+supervisor agent (5 skills, 3-tier monitoring)
  analysist.py            — DEPRECATED backward-compat shim → re-exports from supervisor
  supervisoner.py         — supervision with structured output (used by web)
  skill_loader.py         — progressive disclosure via skillkit (L1→L2→L3)
  merge_router.py         — LLM-based query dedup for chatter+supervisor retrievals
  mem_integration.py      — Chroma init, tool definitions (central hub)
  mem_store_diary.py      — diary storage (date/event splitting + LLM annotation)
  mem_store_material.py   — material storage (parent-child semantic chunks)
  mem_store_conv_outline.py — conversation outline storage (PAIP)
  mem_retrieve_diary.py   — diary retrieval (LangGraph state machine)
  mem_retrieve_material.py — material retrieval (LangGraph)
  mem_retrieve_conv_outline.py — conv outline retrieval (LangGraph)
  mem_retrieve_user_profile.py — user profile retrieval (file-first, Chroma-fallback)
  read_file.py            — multi-format file reader (txt/pdf/md/csv/docx)
  user_profile/
    __init__.py
    schema.md             — 14-domain schema documentation
    profile_models.py     — Fact, ProfileDomain, Profile, 14-domain registry
    profile_collector.py  — CollectedData, 4 collection strategies
    profile_generator.py  — 3-round LLM generation + incremental update
    profiler.py           — orchestrator (init/update/get_or_init)
  skills/
    chatter/              — 15 therapy skills (cbt, gestalt, sfbt, etc.)
      SKILL.md            — therapy skills index
    supervisor/           — 5 supervision skills
      SKILL.md            — supervision skills index
  web/
    main.py               — FastAPI app with WebSocket
    session_manager.py    — web session CRUD
    static/               — HTML/CSS/JS frontend
  _book_extract/          — extracted textbook content for RAG
    interviewing/         — Sommers-Flanagan chapters
    supervision/          — Falender & Shafranske chapters
web/
  sessions/               — persisted session files (gitignored)
database/                 — Chroma SQLite files + profiles (gitignored)
test/                     — pytest test files
```

### Common Patterns

- Retrieval functions wrap synchronous Chroma calls via `asyncio.get_running_loop().run_in_executor(None, _sync_fn)`
- Chroma collections are global singletons created at module import in `mem_integration.py`
- All retrieval modules use a singleton graph pattern (build once, cache in module-level `_graph`/`_conv_graph`/`_material_graph`)
- File type detection uses keyword matching against known lists in `config.py`
- SkillManager instances (`_chatter_manager`, `_supervisor_manager`) are created at module import, discovered once, and reused
- `reset_skill_lookup_counts()` must be called before each agent invocation to reset lookup limits
- Profile persistence is file-first (JSON under `database/profiles/<user_id>/`) with version snapshots in `versions/` subdirectory
