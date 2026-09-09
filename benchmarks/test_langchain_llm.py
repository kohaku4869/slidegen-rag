import asyncio
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage
from agent.models.llm import LLMModel

def test_sync_invoke():
    print("\n--- 1. Testing Sync Invoke ---")
    llm = LLMModel()
    msg = HumanMessage(content="Chào bạn, hãy giới thiệu ngắn gọn 1 câu về Hà Nội.")
    response = llm.invoke([msg])
    print(f"Response: {response.content}")

def test_sync_stream():
    print("\n--- 2. Testing Sync Stream ---")
    llm = LLMModel()
    msg = HumanMessage(content="Viết bài thơ 4 câu về mùa thu.")
    print("Streaming: ", end="", flush=True)
    for chunk in llm.stream([msg]):
        print(chunk.content, end="", flush=True)
    print()

async def test_async_invoke():
    print("\n--- 3. Testing Async Invoke ---")
    llm = LLMModel()
    msg = HumanMessage(content="Thành phố nào là thủ đô của Việt Nam?")
    response = await llm.ainvoke([msg])
    print(f"Async Response: {response.content}")

def main():
    test_sync_invoke()
    test_sync_stream()
    asyncio.run(test_async_invoke())
    print("\n✅ Tất cả bài test LangChain LLMModel đều thành công!")

if __name__ == "__main__":
    main()
