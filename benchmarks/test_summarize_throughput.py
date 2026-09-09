"""
test/test_summarize_throughput.py
===================================
Test throughput (req/s) of ChatSummarizationModel.async_batch_summarize()
"""
import asyncio
import io
import sys
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from tree.models.SummarizationModels import ChatSummarizationModel

# Sample contexts for summarization
SAMPLE_CONTEXT = """
Hà Nội là thủ đô, thành phố trực thuộc trung ương và là đô thị loại đặc biệt của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam.
Thành phố nằm ở trung tâm đồng bằng sông Hồng, là trung tâm chính trị, văn hóa và là trung tâm lớn về kinh tế, giao lưu quốc tế của Việt Nam.
Hà Nội là thành phố có diện tích lớn nhất Việt Nam với 3.359,82 km2 và là thành phố đông dân thứ hai với hơn 8,4 triệu người (năm 2022).
Vùng đất Hà Nội đã có lịch sử cư trú lâu đời từ thời tiền sử, tiêu biểu là văn hóa Phùng Nguyên, Đồng Đậu, Gò Mun.
Năm 1010, Vua Lý Thái Tổ quyết định dời đô từ Hoa Lư về Đại La và đổi tên thành Thăng Long, mở ra thời kỳ phát triển rực rỡ của kinh đô.
"""

NUM_CONTEXTS = 100  # Batch size
NUM_WORKERS = 100   # Concurrency

async def main():
    print("==========================================================================")
    print("      BENCHMARK THROUGHPUT: ChatSummarizationModel.async_batch_summarize   ")
    print("==========================================================================")
    print(f"  Total contexts (batch size) : {NUM_CONTEXTS}")
    print(f"  Num workers (concurrency)   : {NUM_WORKERS}")
    print("==========================================================================\n")

    model = ChatSummarizationModel(max_workers=NUM_WORKERS)
    contexts = [f"Context #{i+1}:\n" + SAMPLE_CONTEXT for i in range(NUM_CONTEXTS)]

    t0 = time.perf_counter()
    results = await model.async_batch_summarize(
        contexts=contexts,
        max_tokens=300,
        num_workers=NUM_WORKERS
    )
    t_total = time.perf_counter() - t0

    successful = [r for r in results if r.strip()]
    failed_count = len(contexts) - len(successful)
    throughput = len(successful) / t_total if t_total > 0 else 0
    avg_latency = (t_total / len(successful)) * 1000 if successful else 0

    print("\n==========================================================================")
    print("                         BENCHMARK RESULTS                                ")
    print("==========================================================================")
    print(f"  Total requests completed : {len(successful)}/{NUM_CONTEXTS}")
    print(f"  Failed requests          : {failed_count}")
    print(f"  Total duration           : {t_total:.2f} seconds")
    print(f"  🔥 Throughput            : {throughput:.2f} req/s")
    print(f"  Avg latency per request  : {avg_latency:.0f} ms")
    print("==========================================================================")
    if successful:
        print(f"\nSample summary output (Context #1):\n{successful[0][:200]}...\n")

if __name__ == "__main__":
    asyncio.run(main())
