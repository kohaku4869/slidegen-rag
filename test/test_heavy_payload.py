"""
test_heavy_payload.py — Benchmark heavy payloads (~2000 input tokens, ~1000 output tokens)
"""
import asyncio
import io
import sys
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage
from agent.models.llm import LLMModel

# Generate ~2000 input tokens (~1500 words of Vietnamese text)
LONG_PROMPT = """Hãy phân tích chi tiết và viết nội dung slide thuyết trình về chủ đề: "Ứng dụng Trí tuệ Nhân tạo trong Y tế hiện đại".
""" + ("Hà Nội là thủ đô của Việt Nam với lịch sử ngàn năm văn hiến, phát triển mạnh mẽ về kinh tế và công nghệ y tế. " * 75) + """
Yêu cầu chi tiết bài viết:
1. Viết bài phân tích chuyên sâu ít nhất 800-1000 từ.
2. Bao gồm các mục: Giới thiệu, Phân tích dữ liệu y tế, Chẩn đoán hình ảnh qua AI, Thách thức bảo mật dữ liệu, và Tương lai phát triển.
3. Trình bày chi tiết từng mục.
"""

async def test_concurrency(concurrency: int, total: int):
    llm = LLMModel(max_tokens=1000)
    sem = asyncio.Semaphore(concurrency)

    async def _worker(i):
        async with sem:
            t0 = time.perf_counter()
            try:
                res = await llm.ainvoke([HumanMessage(content=LONG_PROMPT)])
                dt = (time.perf_counter() - t0) * 1000
                tokens = getattr(res, "usage_metadata", {})
                return True, dt, tokens
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000
                return False, dt, str(e)

    print(f"\n--- Benchmark Heavy Request (~2000 in, ~1000 out) | Concurrency = {concurrency} ---")
    t_start = time.perf_counter()
    results = await asyncio.gather(*[_worker(i) for i in range(total)])
    t_total = time.perf_counter() - t_start

    successes = [r for r in results if r[0]]
    lats = [r[1] for r in successes]
    print(f"Total time: {t_total:.2f}s | Success: {len(successes)}/{total}")
    if lats:
        print(f"Avg latency per request: {sum(lats)/len(lats):.0f} ms ({sum(lats)/len(lats)/1000:.2f}s)")
        print(f"Throughput: {len(successes)/t_total:.2f} req/s")
        if successes[0][2]:
            print(f"Token sample usage: {successes[0][2]}")

async def main():
    await test_concurrency(1, 1)   # baseline 1 request
    await test_concurrency(5, 10)  # 5 workers

if __name__ == "__main__":
    asyncio.run(main())
