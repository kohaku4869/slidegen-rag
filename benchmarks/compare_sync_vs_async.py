"""
compare_sync_vs_async.py — So sánh tốc độ giữa batch_summarize (Sync ThreadPool) và async_batch_summarize (Asyncio)
"""

import asyncio
import io
import sys
import time
from pathlib import Path

# Fix Windows cp1252 stdout encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slidegen_rag.tree.models.SummarizationModels import ChatSummarizationModel

SAMPLE_TEXTS = [
    "Nhà nước pháp quyền xã hội chủ nghĩa Việt Nam là nhà nước của nhân dân, do nhân dân, vì nhân dân.",
    "Hệ thống pháp luật Việt Nam bao gồm các quy phạm pháp luật, chế định pháp luật và các ngành luật.",
    "Hiến pháp là đạo luật cơ bản của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam, có hiệu lực pháp lý cao nhất.",
    "Quyền con người, quyền công dân về chính trị, dân sự, kinh tế, văn hóa, xã hội được công nhận và bảo đảm.",
    "Quốc hội là cơ quan đại biểu cao nhất của Nhân dân, cơ quan quyền lực nhà nước cao nhất của nước CHXHCN Việt Nam.",
    "Chính phủ là cơ quan hành chính nhà nước cao nhất, thực hiện quyền hành pháp, là cơ quan chấp hành của Quốc hội.",
    "Tòa án nhân dân là cơ quan xét xử của nước Cộng hòa xã hội chủ nghĩa Việt Nam, thực hiện quyền tư pháp.",
    "Viện kiểm sát nhân dân thực hành quyền công tố, kiểm sát hoạt động tư pháp theo quy định của Hiến pháp và pháp luật.",
    "Hội đồng nhân dân là cơ quan quyền lực nhà nước ở địa phương, đại diện cho ý chí, nguyện vọng của Nhân dân địa phương.",
    "Ủy ban nhân dân ở cấp chính quyền địa phương do Hội đồng nhân dân cùng cấp bầu là cơ quan chấp hành của HĐND.",
] * 2  # Total 20 contexts

async def run_benchmark():
    model = ChatSummarizationModel(max_workers=20, timeout=60)

    print("=" * 70)
    print(f"  BENCHMARK: Sync ThreadPool vs Asyncio ({len(SAMPLE_TEXTS)} prompts, concurrency=20)")
    print("=" * 70)

    # 1. Test Sync ThreadPool
    print("\n--- 1. Testing Sync ThreadPool (batch_summarize) ---")
    t0 = time.perf_counter()
    sync_results = model.batch_summarize(SAMPLE_TEXTS, max_tokens=150)
    t_sync = time.perf_counter() - t0
    sync_success = sum(1 for r in sync_results if r.strip())
    print(f"-> Sync Time: {t_sync:.2f}s | Success: {sync_success}/{len(SAMPLE_TEXTS)} | Throughput: {sync_success/t_sync:.2f} req/s")

    # Cooldown
    await asyncio.sleep(2)

    # 2. Test Asyncio
    print("\n--- 2. Testing Asyncio (async_batch_summarize) ---")
    t0 = time.perf_counter()
    async_results = await model.async_batch_summarize(SAMPLE_TEXTS, max_tokens=150, num_workers=20)
    t_async = time.perf_counter() - t0
    async_success = sum(1 for r in async_results if r.strip())
    print(f"-> Async Time: {t_async:.2f}s | Success: {async_success}/{len(SAMPLE_TEXTS)} | Throughput: {async_success/t_async:.2f} req/s")

    # Comparison summary
    print("\n" + "=" * 70)
    print("                         SUMMARY COMPARISON")
    print("=" * 70)
    print(f"  * Sync ThreadPool (batch_summarize)       : {t_sync:.2f}s ({sync_success/t_sync:.2f} req/s)")
    print(f"  * Asyncio (async_batch_summarize)         : {t_async:.2f}s ({async_success/t_async:.2f} req/s)")
    diff_pct = ((t_sync - t_async) / t_sync) * 100 if t_sync > 0 else 0
    if t_async < t_sync:
        print(f"  🔥 Asyncio nhanh hơn Sync ThreadPool: {t_sync - t_async:.2f}s ({diff_pct:.1f}%)")
    else:
        print(f"  ⚠️ Sync nhanh hơn hoặc tương đương Asyncio: {t_async - t_sync:.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
