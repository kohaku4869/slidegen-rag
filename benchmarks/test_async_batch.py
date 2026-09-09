"""
test_async_batch.py — Async batch load test
8 workers in parallel, each worker sequentially sends 4 async requests (total 32 requests)
using LLMModel.ainvoke().
"""
import asyncio
import io
import sys
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage
from slidegen_rag.agent.models.llm import LLMModel

import argparse

parser = argparse.ArgumentParser(description="Async Batch Load Test")
parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")
parser.add_argument("--reqs", type=int, default=10, help="Requests per worker")
args = parser.parse_args()

TOTAL_WORKERS = args.workers
REQS_PER_WORKER = args.reqs
TOTAL_REQUESTS = TOTAL_WORKERS * REQS_PER_WORKER

PROMPT = "Viết bài thơ 4 câu về Hà Nội."

async def worker_task(worker_id: int, llm: LLMModel):
    latencies = []
    print(f"  ▶ Worker #{worker_id} started (4 requests)...")
    for i in range(REQS_PER_WORKER):
        t0 = time.perf_counter()
        try:
            res = await llm.ainvoke([HumanMessage(content=PROMPT)])
            dt = (time.perf_counter() - t0) * 1000
            latencies.append(dt)
            print(f"    [Worker #{worker_id}] Req {i+1}/4 done in {dt:.0f} ms")
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000
            print(f"    [Worker #{worker_id}] Req {i+1}/4 ERROR in {dt:.0f} ms: {e}")
    return latencies

async def main():
    print(f"============================================================")
    print(f"  Async Load Test: {TOTAL_WORKERS} workers x {REQS_PER_WORKER} reqs = {TOTAL_REQUESTS} total requests")
    print(f"  Prompt: '{PROMPT}'")
    print(f"============================================================")

    llm = LLMModel()
    
    t_start = time.perf_counter()
    
    tasks = [worker_task(w_id + 1, llm) for w_id in range(TOTAL_WORKERS)]
    results = await asyncio.gather(*tasks)
    
    t_total = time.perf_counter() - t_start

    all_latencies = [lat for worker_lats in results for lat in worker_lats]
    successful = len(all_latencies)

    print("\n============================================================")
    print("  TEST RESULTS")
    print("============================================================")
    print(f"  Total requests completed : {successful}/{TOTAL_REQUESTS}")
    print(f"  Total duration           : {t_total:.2f} seconds")
    print(f"  Throughput               : {successful / t_total:.2f} req/s")
    if all_latencies:
        print(f"  Avg request latency      : {sum(all_latencies)/len(all_latencies):.0f} ms")
        print(f"  Min request latency      : {min(all_latencies):.0f} ms")
        print(f"  Max request latency      : {max(all_latencies):.0f} ms")
    print("============================================================")

if __name__ == "__main__":
    asyncio.run(main())
