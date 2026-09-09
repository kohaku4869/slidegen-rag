# benchmarks/

Ad-hoc scripts that measure throughput, latency, and fallback behavior against a
**live** LLM/embedding endpoint (`LLM_BASE_URL`, `EMBEDDING_BASE_URL`, ...). They
are infra load tests, not correctness tests — most are run directly
(`python benchmarks/<file>.py`), a couple use `unittest` internally but still
self-run via `if __name__ == "__main__"`.

**Never run by pytest / CI.** Hermetic unit tests live in `tests/` instead. See
the restructuring plan for the split rationale.

The one exception: `test_batch_embedding.py` mixes pure batching-logic
assertions (worth promoting to `tests/unit/`) with a live-endpoint smoke test;
it hasn't been split yet.
