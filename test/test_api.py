"""
test_api.py - API Load & Performance Test Script
=================================================
Tests the chat completion API endpoint with:
  - Single request latency measurement
  - Throughput (requests/sec) under concurrency
  - Tree-based test organisation (TestNode / TestTree)
  - Detailed per-request & aggregate metrics
  - Rich terminal report

Usage:
    python test_api.py [--url URL] [--token TOKEN] [--concurrency N] [--total N] [--skip-flood]
"""

from __future__ import annotations

import sys
import io
# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import urllib.request
import urllib.error

# ──────────────────────────────────────────────
# Colour helpers (ANSI, works on modern Windows)
# ──────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DEFAULT_URL   = "http://171.226.10.154:8080/v1/chat/completions"
DEFAULT_TOKEN = "458d41f783134eb3715cc8f371af8351"

PROMPTS: List[str] = [
    "Viet bai tho ve Ha Noi.",
    "Mo ta ve dep cua mua thu Viet Nam.",
    "Ke mot cau chuyen ngan ve tinh ban.",
    "Giai thich khai niem hoc may bang ngon ngu don gian.",
    "Viet doan hoi thoai giua hai nguoi ban gap nhau sau 10 nam.",
]

# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────
@dataclass
class RequestResult:
    """Result of a single API request."""
    prompt:          str
    status_code:     int
    latency_ms:      float          # end-to-end ms
    ttfb_ms:         float          # time-to-first-byte ms
    tokens_total:    int = 0
    tokens_prompt:   int = 0
    tokens_response: int = 0
    response_text:   str = ""
    error:           Optional[str] = None
    success:         bool = True


@dataclass
class TestMetrics:
    """Aggregate metrics for a test suite."""
    total_requests:   int = 0
    successful:       int = 0
    failed:           int = 0
    latencies_ms:     List[float] = field(default_factory=list)
    start_epoch:      float = 0.0
    end_epoch:        float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_epoch - self.start_epoch

    @property
    def throughput_rps(self) -> float:
        return self.successful / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        idx = max(0, int(len(s) * 0.95) - 1)
        return s[idx]

    @property
    def p99_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        idx = max(0, int(len(s) * 0.99) - 1)
        return s[idx]

    @property
    def min_ms(self) -> float:
        return min(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0


# ──────────────────────────────────────────────
# Tree-based test organisation
# (mirrors tree/structure/tree_structure.py)
# ──────────────────────────────────────────────
class TestNode:
    """
    A node in the test tree.
    Each node represents one test case (an API call) and
    can have child test nodes for hierarchical organisation.
    """

    def __init__(
        self,
        name: str,
        prompt: str,
        url: str,
        token: str,
        index: int,
        children: Optional[Set[int]] = None,
        stream: bool = False,
        thinking: bool = False,
    ) -> None:
        self.name     = name
        self.prompt   = prompt
        self.url      = url
        self.token    = token
        self.index    = index
        self.children: Set[int] = children or set()
        self.stream   = stream
        self.thinking = thinking
        self.result:  Optional[RequestResult] = None

    def run(self) -> RequestResult:
        """Execute the API call and store the result."""
        payload = json.dumps({
            "messages": [{"role": "user", "content": self.prompt}],
            "stream":   self.stream,
            "thinking": self.thinking,
        }).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type":  "application/json",
        }

        req = urllib.request.Request(
            self.url, data=payload, headers=headers, method="POST"
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                ttfb = (time.perf_counter() - t0) * 1000
                body = resp.read().decode("utf-8")
                latency = (time.perf_counter() - t0) * 1000

                data = json.loads(body)
                usage = data.get("usage", {})
                content = ""
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")

                self.result = RequestResult(
                    prompt          = self.prompt,
                    status_code     = resp.status,
                    latency_ms      = latency,
                    ttfb_ms         = ttfb,
                    tokens_total    = usage.get("total_tokens", 0),
                    tokens_prompt   = usage.get("prompt_tokens", 0),
                    tokens_response = usage.get("completion_tokens", 0),
                    response_text   = content[:200],
                    success         = True,
                )

        except urllib.error.HTTPError as e:
            latency = (time.perf_counter() - t0) * 1000
            self.result = RequestResult(
                prompt      = self.prompt,
                status_code = e.code,
                latency_ms  = latency,
                ttfb_ms     = latency,
                error       = f"HTTP {e.code}: {e.reason}",
                success     = False,
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            self.result = RequestResult(
                prompt      = self.prompt,
                status_code = 0,
                latency_ms  = latency,
                ttfb_ms     = latency,
                error       = str(exc),
                success     = False,
            )

        return self.result


class TestTree:
    """
    Hierarchical collection of TestNodes.
    Mirrors the tree.structure.tree_structure.Tree design:
      all_nodes / root_nodes / leaf_nodes / num_layers / layer_to_nodes
    """

    def __init__(self) -> None:
        self.all_nodes:       Dict[int, TestNode]    = {}
        self.root_nodes:      Set[int]               = set()
        self.leaf_nodes:      Set[int]               = set()
        self.layer_to_nodes:  Dict[int, List[int]]   = {}

    def add_node(self, node: TestNode, layer: int = 0) -> None:
        self.all_nodes[node.index] = node
        self.layer_to_nodes.setdefault(layer, []).append(node.index)

    @property
    def num_layers(self) -> int:
        return len(self.layer_to_nodes)

    def build_from_prompts(
        self, prompts: List[str], url: str, token: str
    ) -> None:
        """
        Build a 2-layer test tree:
          Layer 0  ->  leaf nodes (individual prompt test cases)
          Layer 1  ->  one root node representing the full test suite
        """
        leaf_indices: Set[int] = set()

        for i, prompt in enumerate(prompts):
            label = (prompt[:40] + "...") if len(prompt) > 40 else prompt
            node = TestNode(
                name     = f"Test-{i + 1}: {label}",
                prompt   = prompt,
                url      = url,
                token    = token,
                index    = i,
                children = set(),
            )
            self.add_node(node, layer=0)
            leaf_indices.add(i)

        self.leaf_nodes = leaf_indices

        root_idx = len(prompts)
        root_node = TestNode(
            name     = "ROOT — Full Test Suite",
            prompt   = "",
            url      = url,
            token    = token,
            index    = root_idx,
            children = leaf_indices,
        )
        self.add_node(root_node, layer=1)
        self.root_nodes = {root_idx}

    # ── runners ──────────────────────────────────────────────────
    def run_sequential(self) -> TestMetrics:
        """Run all leaf nodes one by one and collect metrics."""
        metrics = TestMetrics(total_requests=len(self.leaf_nodes))
        metrics.start_epoch = time.time()

        for idx in sorted(self.leaf_nodes):
            node = self.all_nodes[idx]
            print(f"  {_c(CYAN, '>>')} {node.name}")
            result = node.run()
            self._record(result, metrics)
            self._print_result(result)
            print()

        metrics.end_epoch = time.time()
        return metrics

    def run_concurrent(
        self, concurrency: int, total_requests: int, prompt: str
    ) -> TestMetrics:
        """
        Flood test: send `total_requests` identical requests with
        `concurrency` parallel workers.
        """
        metrics = TestMetrics(total_requests=total_requests)
        # Borrow url/token from first leaf node
        ref = self.all_nodes[next(iter(sorted(self.leaf_nodes)))]

        nodes = [
            TestNode(
                name    = f"Concurrent-{i}",
                prompt  = prompt,
                url     = ref.url,
                token   = ref.token,
                index   = 10000 + i,
            )
            for i in range(total_requests)
        ]

        done = 0
        metrics.start_epoch = time.time()

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(n.run): n for n in nodes}
            for fut in as_completed(futures):
                result = fut.result()
                self._record(result, metrics)
                done += 1
                icon = _c(GREEN, "[OK]") if result.success else _c(RED, "[FAIL]")
                err  = f"  {_c(RED, result.error)}" if result.error else ""
                print(
                    f"  {icon} [{done:>3}/{total_requests}]"
                    f"  {result.latency_ms:>8.1f} ms"
                    f"  HTTP {result.status_code}"
                    f"{err}",
                    flush=True,
                )

        metrics.end_epoch = time.time()
        return metrics

    # ── helpers ──────────────────────────────────────────────────
    @staticmethod
    def _record(result: RequestResult, metrics: TestMetrics) -> None:
        if result.success:
            metrics.successful += 1
            metrics.latencies_ms.append(result.latency_ms)
        else:
            metrics.failed += 1

    @staticmethod
    def _print_result(result: RequestResult) -> None:
        if result.success:
            tok = (
                f"  tokens(prompt={result.tokens_prompt}"
                f" + completion={result.tokens_response}"
                f" = {result.tokens_total})"
                if result.tokens_total > 0 else ""
            )
            print(
                f"    {_c(GREEN, '[OK]')}  HTTP {result.status_code}"
                f"  latency={result.latency_ms:.0f} ms{tok}"
            )
            if result.response_text:
                preview = result.response_text.replace("\n", " ")[:120]
                print(f"    {_c(YELLOW, '[preview]')}  \"{preview}...\"")
        else:
            print(f"    {_c(RED, '[FAIL]')}  {result.error}")


# ──────────────────────────────────────────────
# Report printer
# ──────────────────────────────────────────────
def print_metrics(title: str, m: TestMetrics) -> None:
    bar = "=" * 62
    print(f"\n{_c(BOLD, bar)}")
    print(f"{_c(BOLD, f'  {title}')}")
    print(_c(BOLD, bar))
    print(f"  {'Total requests':<30} {m.total_requests}")
    print(f"  {'Successful':<30} {_c(GREEN, str(m.successful))}")
    fail_str = _c(RED, str(m.failed)) if m.failed else _c(GREEN, "0")
    print(f"  {'Failed':<30} {fail_str}")
    print(f"  {'Test duration (s)':<30} {m.duration_s:.3f}")
    print(f"  {'Throughput (req/s)':<30} {_c(CYAN, f'{m.throughput_rps:.3f}')}")
    print(f"  {'-' * 58}")
    print(f"  {'Latency avg (ms)':<30} {m.avg_latency_ms:.1f}")
    print(f"  {'Latency min (ms)':<30} {m.min_ms:.1f}")
    print(f"  {'Latency p50 median (ms)':<30} {m.p50_ms:.1f}")
    print(f"  {'Latency p95 (ms)':<30} {m.p95_ms:.1f}")
    print(f"  {'Latency p99 (ms)':<30} {m.p99_ms:.1f}")
    print(f"  {'Latency max (ms)':<30} {m.max_ms:.1f}")
    print(_c(BOLD, bar))


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chat Completion API — Load & Performance Test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url",         default=DEFAULT_URL,   help="API endpoint URL")
    p.add_argument("--token",       default=DEFAULT_TOKEN, help="Bearer token")
    p.add_argument("--concurrency", type=int, default=5,   help="Concurrent workers for flood test")
    p.add_argument("--total",       type=int, default=20,  help="Total requests for flood test")
    p.add_argument(
        "--skip-flood", action="store_true",
        help="Skip the concurrent flood test (Phase 2)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(_c(BOLD, "\n+============================================================+"))
    print(_c(BOLD,   "|       Chat Completion API  --  Performance Test Suite      |"))
    print(_c(BOLD,   "+============================================================+"))
    print(f"  Endpoint    : {_c(CYAN, args.url)}")
    print(f"  Token       : {args.token[:8]}...")
    print(f"  Concurrency : {args.concurrency}  |  Flood total : {args.total}")

    # ── Build test tree ──────────────────────────────────────────
    tree = TestTree()
    tree.build_from_prompts(PROMPTS, url=args.url, token=args.token)

    print(f"\n  [Tree Structure]")
    print(f"    Layers     : {tree.num_layers}")
    print(f"    Leaf nodes : {len(tree.leaf_nodes)}  (individual test cases)")
    print(f"    Root nodes : {len(tree.root_nodes)}")
    for layer, indices in sorted(tree.layer_to_nodes.items()):
        label = "leaf" if layer == 0 else "root"
        names = [tree.all_nodes[i].name for i in indices]
        print(f"    Layer {layer} ({label}): {names}")

    # ── Phase 1: Sequential varied prompts ─────────────────────
    print(_c(BOLD, "\n\n---- PHASE 1: Sequential Latency Test (varied prompts) ----\n"))
    seq_metrics = tree.run_sequential()
    print_metrics("PHASE 1 -- Sequential Results", seq_metrics)

    # ── Phase 2: Concurrent flood ───────────────────────────────
    if not args.skip_flood:
        flood_prompt = PROMPTS[0]
        print(_c(BOLD, "\n\n---- PHASE 2: Concurrent Flood Test ----"))
        print(f"  Workers     : {_c(CYAN, str(args.concurrency))}")
        print(f"  Total reqs  : {_c(CYAN, str(args.total))}")
        print(f"  Prompt      : \"{flood_prompt[:70]}...\"\n")

        flood_metrics = tree.run_concurrent(
            concurrency    = args.concurrency,
            total_requests = args.total,
            prompt         = flood_prompt,
        )
        print_metrics("PHASE 2 -- Concurrent Flood Results", flood_metrics)

    print(f"\n  {_c(GREEN, 'All tests completed.')}\n")


if __name__ == "__main__":
    main()
