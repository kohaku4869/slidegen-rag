"""`BuildStats.to_dict()` <-> `build_stats_from_dict()` phải round-trip —
`document_indexes.build_stats` (jsonb) lưu bản `to_dict()`, tab Tổng quan
UI đọc lại bằng `build_stats_from_dict()` rồi tái dùng `format_ingest_report`
nguyên vẹn (xem plan6.md B3)."""

from slidegen_rag.tree.stats import (
    BuildStats,
    LayerStat,
    build_stats_from_dict,
    format_ingest_report,
)


def test_build_stats_roundtrip_via_dict():
    original = BuildStats(
        filename="pldc.md",
        src_path="data/src/pldc.md",
        out_path="data/index/pldc.pkl",
        total_time=12.345,
        parse_time=0.1,
        total_summarize_time=8.0,
        total_embedding_time=2.0,
        save_time=0.05,
        file_size_bytes=123456,
        total_nodes=320,
        total_llm_calls=42,
        num_layers=5,
        layers=[
            LayerStat(
                layer=0, name="Leaf Nodes", num_nodes=200, num_llm_calls=0,
                summarize_time=0.0, embedding_time=1.5,
            ),
            LayerStat(
                layer=1, name="Layer 1 Summaries", num_nodes=80, num_llm_calls=20,
                summarize_time=4.0, embedding_time=0.5,
            ),
        ],
    )

    restored = build_stats_from_dict(original.to_dict())

    assert restored.to_dict() == original.to_dict()
    # format_ingest_report phải chạy được trên bản restored y hệt bản gốc.
    assert format_ingest_report(restored) == format_ingest_report(original)
