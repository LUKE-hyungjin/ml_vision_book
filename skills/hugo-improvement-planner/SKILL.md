---
name: hugo-improvement-planner
description: Plan iterative improvements for this Hugo-based ml_vision_book project by reading STRUCTURE.md, CLAUDE.md, and content trees to choose the next high-impact docs/content-quality tasks without breaking i18n or link rules. Use when selecting what to improve next in scheduled automation.
---

# Hugo Improvement Planner

Use this skill to pick the **next small, safe, high-impact content task**.

## Read order
1. `STRUCTURE.md`
2. `CLAUDE.md`
3. target path in `content/ko/docs` and matching `content/en/docs`

## Task selection rules
- Prefer existing files marked `[ ]` over `*(planned)*`.
- Prefer tasks with clear prerequisites already completed.
- Keep scope to 1~2 docs per cycle.
- Do not choose tasks that require broad refactors in one run.

## Safety rules
- Preserve ko/en path symmetry.
- Preserve Hugo Book link style (`/ko/...`, `/en/...`).
- Keep front matter valid (`title`, `weight`, `math` when needed).
- Avoid menu structure changes unless explicitly requested.

## Output format (for Telegram planning message)
Return only:
1. `이번 사이클 제안:` one-line summary
2. `수정 대상:` bullet list (1~2 files)
3. `이유:` one short sentence
4. `완료 조건:` checklist (2~4 items)

Do not edit files when running in planning-only mode.