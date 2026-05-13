# Agent guidance for this repository

This repo is a collection of small GenAI experiments and coursework-style demos. There is no single top-level package or unified dependency manifest; treat each area as its own mini-project.

## Layout

| Path | What it is |
|------|------------|
| `project/` | **HomeMatch** — real estate listing personalization (offline TF‑IDF path and online OpenAI + Chroma path). Start here for the most complete README and runnable pipeline. |
| `src/` | Standalone scripts and prototypes (vector search, sentiment, chains, multimodal search, wellness agent, etc.). Dependencies and entry points vary by file. |

Authoritative setup and run instructions for HomeMatch live in `project/README.md`. The restaurant sentiment demo documents itself in `src/restaurants/README.md`.

## Environment and dependencies

- Use **Python 3.10+** for new work unless a subdirectory explicitly targets an older version.
- Prefer a **virtual environment** per area you are changing (for example, one venv while working in `project/`).
- `project/requirements.txt` applies to the online HomeMatch flow; offline mode there only needs `scikit-learn` (see `project/README.md`).
- Scripts under `src/` often assume `pip install` of a few packages (for example `openai`); read the file header or nearby README before assuming imports.

## Secrets and generated files

- Do **not** commit API keys or `.env` files. `.gitignore` already ignores `*.env` and common generated paths under `project/` (for example `listings.json`, `chroma_db*`).
- Use `project/env.example` as the template for local configuration when working on HomeMatch.

## Conventions for changes

- Match the **style and patterns** of the file or folder you edit (imports, error handling, logging).
- Avoid drive-by refactors across unrelated `src/` demos when fixing or extending one script.
- If you add a new runnable script, document prerequisites (Python version, env vars, pip packages) in a short README next to the script or in an existing README in that folder.

## Verification

- After edits, run the smallest relevant command (for example the script the user cares about, or `python homematch_offline.py` from `project/` when touching HomeMatch offline code).
- There is no repo-wide test suite; `src/test/function-test.py` is ad hoc and not a guaranteed gate for all modules.
