import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
base_url = os.getenv("LLM_BASE_URL")
api_key = os.getenv("LLM_API_KEY")

print(f"Testing LLM endpoint: {base_url}")
client = OpenAI(base_url=base_url, api_key=api_key, timeout=180.0)

# sample sentence is 93 chars, ~13 tokens in Qwen tokenizer (approx 1.38 tokens per repeat of 18 chars)
sample_text = "Machine learning models require sufficient context window to process large documents accurately. "
# 1 repeat of sample_text gives ~13 tokens

def test_prompt_tokens(target_tokens: int):
    # Calculate repeats: 1 repeat ~ 13 tokens
    repeat_count = max(1, int(target_tokens / 13))
    prompt = sample_text * repeat_count
    
    t0 = time.time()
    try:
        completion = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Reply with one word: 'OK'."},
                {"role": "user", "content": f"Context:\n{prompt}\n\nTask: Say OK."}
            ],
            max_tokens=10,
            temperature=0.0
        )
        elapsed = time.time() - t0
        usage = completion.usage
        actual_tokens = usage.prompt_tokens if usage else "unknown"
        response_text = completion.choices[0].message.content.strip()
        print(f"[SUCCESS] Target: {target_tokens:7d} tokens | Actual: {actual_tokens:7d} prompt tokens | Time: {elapsed:.2f}s | Response: {response_text!r}", flush=True)
        return True, actual_tokens, None
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[FAILED ] Target: {target_tokens:7d} tokens | Time: {elapsed:.2f}s | Error: {e}", flush=True)
        return False, None, str(e)

print("\n--- Testing High Context Lengths (100k -> 300k) ---", flush=True)
test_targets = [
    100_000,
    128_000,
    131_072,
    140_000,
    160_000,
    200_000,
    262_144,
]

for target in test_targets:
    ok, actual, err = test_prompt_tokens(target)
    if not ok:
        print(f"\nReached context limit / error at target {target}: {err}", flush=True)
        break
