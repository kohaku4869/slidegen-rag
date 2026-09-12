# Baseline snapshot — legacy CRAG pipeline (frozen for A/B)

Frozen at git sha in `GIT_SHA.txt`, before the AgenticRAG retrofit began.
Pipeline: 5-node LangGraph state machine
(`retrieve -> tool -> grade_documents -> context_evaluator -> {generate | retrieve}`),
`MAX_RETRIEVE = 3`, index backend = `tree/` (heading-based), 30-question Vietnamese
legal Q&A set.

## Measured numbers (30 questions, `trajectory.txt`)

| Metric | Value |
|---|---|
| Runs ending `HẾT LƯỢT RETRIEVE` (best-effort, context_sufficient=False, 3/3) | 18/30 (60%) |
| `context_sufficient` False : True (all evaluator calls) | 74 : 12 |
| Avg total tokens / query | 44,802 |
| Avg wall time / query | 38.6s |
| Avg LLM calls / query | 65.8 (Gemini ~4.9, local grader ~60.9 — 93%) |
| Chunks graded (30 questions) | 1,827 total, 480 kept (74% discarded) |
| Grading rounds keeping 0 chunks | 20/68 (29%) |
| LLM judge (README, self-reported) | Accuracy 9.2, Completeness 8.9, Relevance 9.4, Faithfulness 9.3 /10; Keyword Coverage 88.5% |

These numbers are the baseline every later ablation config (`legacy_crag`,
`legacy_crag_no_grader`, `agentic_full`, ...) must be compared against. Do not
regenerate `answer.txt`/`chunks.txt`/`trajectory.txt` in the parent directory without
first diffing against this snapshot — they are also live eval-cache files reused by
`evaluate/evaluate.py`.
