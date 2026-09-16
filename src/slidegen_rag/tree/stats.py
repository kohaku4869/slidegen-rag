import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class LayerStat:
    """Statistics for a single layer in the RAPTOR tree hierarchy."""
    layer: int
    name: str
    num_nodes: int
    num_llm_calls: int
    summarize_time: float
    embedding_time: float
    total_time: float = 0.0

    def __post_init__(self):
        if not self.total_time:
            self.total_time = self.summarize_time + self.embedding_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "name": self.name,
            "num_nodes": self.num_nodes,
            "num_llm_calls": self.num_llm_calls,
            "summarize_time": round(self.summarize_time, 3),
            "embedding_time": round(self.embedding_time, 3),
            "total_time": round(self.total_time, 3),
        }


@dataclass
class BuildStats:
    """Comprehensive performance statistics for building a hierarchical tree."""
    filename: str = ""
    src_path: str = ""
    out_path: str = ""
    total_time: float = 0.0
    parse_time: float = 0.0
    total_summarize_time: float = 0.0
    total_embedding_time: float = 0.0
    save_time: float = 0.0
    file_size_bytes: int = 0
    total_nodes: int = 0
    total_llm_calls: int = 0
    num_layers: int = 0
    layers: List[LayerStat] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "src_path": self.src_path,
            "out_path": self.out_path,
            "timestamp": self.timestamp,
            "total_time": round(self.total_time, 3),
            "parse_time": round(self.parse_time, 3),
            "total_summarize_time": round(self.total_summarize_time, 3),
            "total_embedding_time": round(self.total_embedding_time, 3),
            "save_time": round(self.save_time, 3),
            "file_size_bytes": self.file_size_bytes,
            "total_nodes": self.total_nodes,
            "total_llm_calls": self.total_llm_calls,
            "num_layers": self.num_layers,
            "layers": [l.to_dict() for l in self.layers],
        }


def build_stats_from_dict(data: Dict[str, Any]) -> "BuildStats":
    """Nghịch đảo của `BuildStats.to_dict()` — đọc lại `document_indexes.build_stats`
    (jsonb) thành `BuildStats` để tái dùng `format_ingest_report()` nguyên
    vẹn ở tab Tổng quan của UI (xem plan6.md B3)."""
    layers = [
        LayerStat(
            layer=l["layer"],
            name=l["name"],
            num_nodes=l["num_nodes"],
            num_llm_calls=l["num_llm_calls"],
            summarize_time=l["summarize_time"],
            embedding_time=l["embedding_time"],
            total_time=l.get("total_time", 0.0),
        )
        for l in data.get("layers", [])
    ]
    return BuildStats(
        filename=data.get("filename", ""),
        src_path=data.get("src_path", ""),
        out_path=data.get("out_path", ""),
        total_time=data.get("total_time", 0.0),
        parse_time=data.get("parse_time", 0.0),
        total_summarize_time=data.get("total_summarize_time", 0.0),
        total_embedding_time=data.get("total_embedding_time", 0.0),
        save_time=data.get("save_time", 0.0),
        file_size_bytes=data.get("file_size_bytes", 0),
        total_nodes=data.get("total_nodes", 0),
        total_llm_calls=data.get("total_llm_calls", 0),
        num_layers=data.get("num_layers", 0),
        layers=layers,
        timestamp=data.get("timestamp", ""),
    )


def _format_size(size_bytes: int) -> str:
    """Format bytes into a human-readable string (KB, MB)."""
    if size_bytes <= 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def format_ingest_report(stats: BuildStats) -> str:
    """
    Renders a clean, highly informative ASCII dashboard report of the ingestion metrics.
    """
    total_time = max(stats.total_time, 0.0001)

    lines = []
    width = 96
    border = "=" * width
    sub_border = "-" * width

    lines.append(border)
    lines.append("INGESTION PERFORMANCE & METRICS REPORT".center(width))
    lines.append(border)

    # File and overview info
    src_display = stats.src_path or stats.filename or "N/A"
    out_display = stats.out_path or "N/A"
    lines.append(f" File   : {src_display}")
    lines.append(f" Output : {out_display}")
    lines.append(
        f" Date   : {stats.timestamp} | "
        f"Total Time: {stats.total_time:.2f}s | "
        f"Nodes: {stats.total_nodes} | "
        f"LLM Calls: {stats.total_llm_calls} | "
        f"Layers: {stats.num_layers}"
    )
    lines.append(sub_border)

    lines.append(" LAYER BREAKDOWN:")
    lines.append(" +-------+---------------------+---------+-----------+----------------+----------------+----------------+")
    lines.append(" | Layer | Role / Description  |   Nodes | LLM Calls | Summarize (s)  | Embedding (s)  | Total Time (s) |")
    lines.append(" +-------+---------------------+---------+-----------+----------------+----------------+----------------+")

    for l in stats.layers:
        sum_pct = (l.summarize_time / total_time) * 100
        emb_pct = (l.embedding_time / total_time) * 100
        layer_pct = (l.total_time / total_time) * 100

        sum_str = f"{l.summarize_time:6.2f}s ({sum_pct:4.1f}%)" if l.summarize_time > 0 else "        -       "
        emb_str = f"{l.embedding_time:6.2f}s ({emb_pct:4.1f}%)" if l.embedding_time > 0 else "        -       "
        tot_str = f"{l.total_time:6.2f}s ({layer_pct:4.1f}%)"

        layer_name = l.name[:19]
        lines.append(
            f" | {l.layer:5d} | {layer_name:<19} | {l.num_nodes:7d} | {l.num_llm_calls:9d} | {sum_str} | {emb_str} | {tot_str} |"
        )

    lines.append(" +-------+---------------------+---------+-----------+----------------+----------------+----------------+")

    total_sum_pct = (stats.total_summarize_time / total_time) * 100
    total_emb_pct = (stats.total_embedding_time / total_time) * 100
    all_layers_time = sum(l.total_time for l in stats.layers)
    all_layers_pct = (all_layers_time / total_time) * 100

    sum_total_str = f"{stats.total_summarize_time:6.2f}s ({total_sum_pct:4.1f}%)"
    emb_total_str = f"{stats.total_embedding_time:6.2f}s ({total_emb_pct:4.1f}%)"
    tot_total_str = f"{all_layers_time:6.2f}s ({all_layers_pct:4.1f}%)"

    lines.append(
        f" | TOTAL | All Layers          | {stats.total_nodes:7d} | {stats.total_llm_calls:9d} | {sum_total_str} | {emb_total_str} | {tot_total_str} |"
    )
    lines.append(" +-------+---------------------+---------+-----------+----------------+----------------+----------------+")
    lines.append("")

    # Phase time breakdown
    parse_pct = (stats.parse_time / total_time) * 100
    sum_pct = (stats.total_summarize_time / total_time) * 100
    emb_pct = (stats.total_embedding_time / total_time) * 100
    save_pct = (stats.save_time / total_time) * 100
    other_time = max(0.0, stats.total_time - (stats.parse_time + stats.total_summarize_time + stats.total_embedding_time + stats.save_time))
    other_pct = (other_time / total_time) * 100

    avg_llm_call = (
        f" [{stats.total_llm_calls} calls, avg {stats.total_summarize_time / stats.total_llm_calls:.2f}s/call]"
        if stats.total_llm_calls > 0
        else ""
    )
    size_str = f" [Size: {_format_size(stats.file_size_bytes)}]" if stats.file_size_bytes > 0 else ""

    lines.append(" PHASE TIME BREAKDOWN:")
    lines.append(f"   * Markdown Hierarchy Parsing : {stats.parse_time:6.2f}s ({parse_pct:5.1f}%)")
    lines.append(f"   * LLM Summarization Phase    : {stats.total_summarize_time:6.2f}s ({sum_pct:5.1f}%){avg_llm_call}")
    lines.append(f"   * Vector Embedding Phase     : {stats.total_embedding_time:6.2f}s ({emb_pct:5.1f}%) [{stats.total_nodes} nodes encoded]")
    lines.append(f"   * Disk Serialization & Save  : {stats.save_time:6.2f}s ({save_pct:5.1f}%){size_str}")
    if other_time >= 0.01:
        lines.append(f"   * Internal Overhead / Misc   : {other_time:6.2f}s ({other_pct:5.1f}%)")
    lines.append(f"   " + "-" * (width - 6))
    lines.append(f"   >> TOTAL INGESTION TIME      : {stats.total_time:6.2f}s (100.0%)")
    lines.append("")

    lines.append(" BOTTLENECK ANALYSIS & OPTIMIZATION INSIGHTS:")
    if stats.total_summarize_time > stats.total_embedding_time and stats.total_summarize_time > stats.parse_time:
        max_layer = max(stats.layers, key=lambda l: l.summarize_time) if stats.layers else None
        lines.append(f"   - Dominant Phase: LLM Summarization accounts for {sum_pct:.1f}% of total ingestion time.")
        if max_layer and max_layer.summarize_time > 0:
            layer_sum_pct = (max_layer.summarize_time / total_time) * 100
            lines.append(
                f"   - Major Bottleneck: {max_layer.name} took {max_layer.summarize_time:.2f}s ({layer_sum_pct:.1f}%) with {max_layer.num_llm_calls} LLM calls."
            )
        lines.append(
            "   - Recommendation  : Increase concurrency (async_workers / max_workers) or use a faster LLM endpoint."
        )
    elif stats.total_embedding_time > stats.total_summarize_time:
        lines.append(f"   - Dominant Phase: Vector Embedding accounts for {emb_pct:.1f}% of total ingestion time.")
        lines.append("   - Recommendation  : Increase embedding batch_size or use GPU/CUDA acceleration for SentenceTransformer.")
    else:
        lines.append("   - Runtime is evenly distributed across parsing, embedding, and serialization.")


    lines.append(border)
    return "\n".join(lines)


def format_multi_file_summary(stats_list: List[BuildStats]) -> str:
    """
    Renders an aggregated summary table when multiple files are indexed in a single run.
    """
    if not stats_list:
        return ""

    width = 96
    border = "=" * width
    sub_border = "-" * width
    lines = []

    lines.append("")
    lines.append(border)
    lines.append("BATCH INGESTION MULTI-FILE SUMMARY".center(width))
    lines.append(border)
    lines.append(" +---------------------------+---------+-----------+---------------+---------------+------------+")
    lines.append(" | File Name                 |   Nodes | LLM Calls | Summarize (s) | Embedding (s) | Total Time |")
    lines.append(" +---------------------------+---------+-----------+---------------+---------------+------------+")

    total_nodes = 0
    total_calls = 0
    total_sum_time = 0.0
    total_emb_time = 0.0
    total_time = 0.0

    for s in stats_list:
        name = (s.filename or "unknown")[:25]
        total_nodes += s.total_nodes
        total_calls += s.total_llm_calls
        total_sum_time += s.total_summarize_time
        total_emb_time += s.total_embedding_time
        total_time += s.total_time

        lines.append(
            f" | {name:<25} | {s.total_nodes:7d} | {s.total_llm_calls:9d} | {s.total_summarize_time:11.2f}s | {s.total_embedding_time:11.2f}s | {s.total_time:8.2f}s |"
        )

    lines.append(" +---------------------------+---------+-----------+---------------+---------------+------------+")
    lines.append(
        f" | TOTAL ({len(stats_list):2d} files)           | {total_nodes:7d} | {total_calls:9d} | {total_sum_time:11.2f}s | {total_emb_time:11.2f}s | {total_time:8.2f}s |"
    )
    lines.append(" +---------------------------+---------+-----------+---------------+---------------+------------+")
    lines.append(border)

    return "\n".join(lines)
