"""
`run_config.json` — khóa để nhận biết "hai run cùng cấu hình" (xem plan5.md
C1/C2). Ghi 1 lần lúc kết thúc mỗi run, đọc lại để quyết định cache hit
(`config_fingerprint` khớp) hay chọn baseline so sánh (`compare.py`).
"""

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from slidegen_rag.evaluate.judge import JUDGE_FALLBACK_MODEL
from slidegen_rag.evaluate.runs import RunPaths

logger = logging.getLogger(__name__)

RUN_CONFIG_SCHEMA_VERSION = 1

# Các trường xác định "cùng cấu hình" — xem plan5.md C1. Không đưa
# started_at/finished_at/git_sha vào đây: khác thời điểm chạy không có nghĩa
# là khác cấu hình.
_FINGERPRINT_FIELDS = (
    "mode",
    "generate_provider",
    "gemini_model",
    "local_base_url",
    "index_source_sha1",
    "web_search_enabled",
    "max_turns",
    "questions_sha1",
)
# Alias công khai — evaluate/compare.py dùng để tính config_diff() trên ĐÚNG
# subset các trường ảnh hưởng tới config_fingerprint (không so sánh
# started_at/git_sha/... — khác thời điểm chạy không có nghĩa là khác cấu hình).
FINGERPRINT_FIELDS = _FINGERPRINT_FIELDS


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _sha1_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_index_fingerprint(ra, index_path: Path) -> dict:
    """Ưu tiên `tree.source_sha1` (đã có sẵn từ pickle v2, miễn phí — xem
    tree_builder.py), chỉ sha1 cả file `.pkl` khi thiếu (index cũ chưa
    migrate qua v2)."""
    source_sha1 = getattr(getattr(ra, "tree", None), "source_sha1", "") or None
    pkl_sha1 = None if source_sha1 else _sha1_file(index_path)
    return {
        "path": str(index_path),
        "source_sha1": source_sha1,
        "pkl_sha1": pkl_sha1,
    }


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def git_sha_and_dirty() -> tuple[Optional[str], Optional[bool]]:
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return None, None
    status = _git("status", "--porcelain")
    dirty = bool(status) if status is not None else None
    return sha, dirty


def config_fingerprint(cfg: dict) -> str:
    """sha1 của subset `_FINGERPRINT_FIELDS`, giá trị thiếu coi là None (ổn
    định giữa các lần chạy). Trả về None nếu BẤT KỲ trường bắt buộc nào
    thiếu (vd run đã migrate không rõ cấu hình — xem plan5.md C2: run migrate
    có config_fingerprint=null, không bao giờ dùng làm cache)."""
    subset = {k: cfg.get(k) for k in _FINGERPRINT_FIELDS}
    payload = json.dumps(subset, sort_keys=True, ensure_ascii=False)
    return _sha1_text(payload)


def build_run_config(
    *,
    run_paths: RunPaths,
    topic: str,
    started_at: str,
    finished_at: str,
    mode: str,
    generate_provider: str,
    lowcost: bool,
    max_turns: Optional[int],
    web_search_enabled: bool,
    max_retrieve: int,
    gemini_model: str,
    local_base_url: Optional[str],
    index_path: Path,
    index_source_sha1: Optional[str],
    index_pkl_sha1: Optional[str],
    questions_sha1: str,
    references_sha1: str,
    total_questions: int,
    source_run_id: Optional[str] = None,
    migrated: bool = False,
) -> dict:
    git_sha, git_dirty = git_sha_and_dirty()

    flat_for_fingerprint = {
        "mode": mode,
        "generate_provider": generate_provider,
        "gemini_model": gemini_model,
        "local_base_url": local_base_url,
        "index_source_sha1": index_source_sha1,
        "web_search_enabled": web_search_enabled,
        "max_turns": max_turns,
        "questions_sha1": questions_sha1,
    }

    cfg = {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "run_id": run_paths.run_id,
        "topic": topic,
        "started_at": started_at,
        "finished_at": finished_at,
        "mode": mode,
        "generate_provider": generate_provider,
        "lowcost": lowcost,
        "max_turns": max_turns,
        "web_search_enabled": web_search_enabled,
        "max_retrieve": max_retrieve,
        "models": {
            "gemini_model": gemini_model,
            "local_base_url": local_base_url,
            "judge_primary": os.getenv("LLM_BASE_URL", "").strip() or None,
            "judge_fallback": JUDGE_FALLBACK_MODEL,
        },
        "index": {
            "path": str(index_path),
            "source_sha1": index_source_sha1,
            "pkl_sha1": index_pkl_sha1,
        },
        "dataset": {
            "questions_sha1": questions_sha1,
            "references_sha1": references_sha1,
            "total_questions": total_questions,
        },
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "source_run_id": source_run_id,
        "migrated": migrated,
    }
    cfg["config_fingerprint"] = (
        None if migrated else config_fingerprint(flat_for_fingerprint)
    )
    return cfg


def save_run_config(run_paths: RunPaths, cfg: dict) -> None:
    run_paths.root.mkdir(parents=True, exist_ok=True)
    with open(run_paths.run_config, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_run_config(run_paths: RunPaths) -> Optional[dict]:
    if not run_paths.run_config.exists():
        return None
    try:
        with open(run_paths.run_config, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Không đọc được {run_paths.run_config}: {e}")
        return None


def sha1_of_file_text(path: Path) -> str:
    """sha1 nội dung text (question.txt/reference.txt) — dùng cho
    dataset.questions_sha1/references_sha1."""
    if not path.exists():
        return ""
    return _sha1_text(path.read_text(encoding="utf-8"))
