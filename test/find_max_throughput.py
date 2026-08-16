"""
find_max_throughput.py — Server Maximum Throughput Benchmark

Stepped concurrency load test to find the server's maximum throughput (peak req/s)
and saturation point.
"""
import asyncio
import io
import sys
import time
from typing import List, Tuple
import urllib.request
import json

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

URL = "http://171.226.10.154:8080/v1/chat/completions"
TOKEN = "458d41f783134eb3715cc8f371af8351"
PROMPT = "Viết bài thơ 4 câu về Hà Nội."

CONCURRENCY_LEVELS = [5, 10, 20, 40, 60, 80, 100]
REQUESTS_PER_LEVEL = 40  # send 40 requests per test level

async def send_single_request(client_session) -> Tuple[bool, float]:
    """Sends a single request using asyncio + urllib in executor to avoid extra dependencies."""
    payload = json.dumps({
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "thinking": False,
    }).encode("utf-8")
    
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }

    loop = asyncio.get_running_loop()
    
    def _do_request():
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
                latency = (time.perf_counter() - t0) * 1000
                return True, latency
        except Exception:
            latency = (time.perf_counter() - t0) * 1000
            return False, latency

    return await loop.run_in_executor(None, _do_request)


async def run_benchmark_level(concurrency: int, total_requests: int) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    
    async def worker():
        async with semaphore:
            return await send_single_request(None)

    print(f"  ▶ Testing concurrency = {concurrency:>3d} ({total_requests} requests)...", end="", flush=True)
    
    t_start = time.perf_counter()
    tasks = [worker() for _ in range(total_requests)]
    results = await asyncio.gather(*tasks)
    t_total = time.perf_counter() - t_start

    successes = [lat for ok, lat in results if ok]
    failures = len(results) - len(successes)
    
    throughput = len(successes) / t_total if t_total > 0 else 0
    avg_latency = sum(successes) / len(successes) if successes else 0
    
    print(f" Done in {t_total:.2f}s | Throughput: {throughput:.2f} req/s | Avg Latency: {avg_latency:.0f} ms | Failed: {failures}")

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful": len(successes),
        "failed": failures,
        "duration_s": t_total,
        "throughput_rps": throughput,
        "avg_latency_ms": avg_latency,
    }


async def main():
    print("==========================================================================")
    print("       MAXIMUM THROUGHPUT BENCHMARK — STEPPED CONCURRENCY TEST            ")
    print("==========================================================================")
    print(f"  Target Endpoint : {URL}")
    print(f"  Test Levels     : {CONCURRENCY_LEVELS} workers")
    print(f"  Reqs per Level  : {REQUESTS_PER_LEVEL}")
    print("==========================================================================\n")

    summary_results = []
    
    for level in CONCURRENCY_LEVELS:
        res = await run_benchmark_level(level, REQUESTS_PER_LEVEL)
        summary_results.append(res)
        # Brief pause between levels to let server rest
        await asyncio.sleep(1.0)

    # ── Report Summary Table ──
    print("\n==========================================================================")
    print("                         BENCHMARK SUMMARY REPORT                         ")
    print("==========================================================================")
    print(f" {'Workers':<10} | {'Throughput (req/s)':<20} | {'Avg Latency (ms)':<18} | {'Success Rate':<12}")
    print("--------------------------------------------------------------------------")
    
    max_rps = 0.0
    best_concurrency = 0

    for r in summary_results:
        success_pct = (r['successful'] / r['total_requests']) * 100
        print(f" {r['concurrency']:<10d} | {r['throughput_rps']:<20.2f} | {r['avg_latency_ms']:<18.0f} | {success_pct:.1f}%")
        if r['throughput_rps'] > max_rps:
            max_rps = r['throughput_rps']
            best_concurrency = r['concurrency']

    print("==========================================================================")
    print(f"  🔥 THÔNG LƯỢNG TỐI ĐA ĐẠT ĐƯỢC: {max_rps:.2f} req/s (tại {best_concurrency} workers song song)")
    print("==========================================================================\n")


if __name__ == "__main__":
    asyncio.run(main())
