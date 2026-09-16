"""
Đường dẫn cho MỘT LẦN CHẠY (run) trong data/test/<topic>/runs/<run_id>/ — xem
plan5.md Phần C. Thay thế layout phẳng cũ (answer.txt/chunks.txt/
trajectory.txt nằm thẳng trong data/test/<topic>/ và bị ghi đè mỗi lần chạy)
bằng một thư mục tự chứa cho mỗi lần chạy, để scores.csv luôn nằm cạnh đúng
answer.txt/chunks.txt/trajectory.txt đã sinh ra nó.

`question.txt`/`reference.txt` KHÔNG nằm trong `runs/` — chúng là INPUT viết
tay, dùng chung cho mọi run của topic đó.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from slidegen_rag.config import LangSmithConfig
from slidegen_rag.evaluate.settings import (
    ANSWER_FILENAME,
    CHUNKS_FILENAME,
    CONVERSATION_FILENAME,
    DATA_TEST_DIR,
    GENERATE_STATE_FILENAME,
    LATEST_POINTER_FILENAME,
    QUESTION_FILENAME,
    REFERENCE_FILENAME,
    RUN_CONFIG_FILENAME,
    RUN_ID_FORMAT,
    RUNS_DIRNAME,
    SCORES_FILENAME,
    SUMMARY_FILENAME,
    TURN_METRICS_FILENAME,
)


def topic_dir(topic: str) -> Path:
    return DATA_TEST_DIR / topic


def question_file(topic: str) -> Path:
    return topic_dir(topic) / QUESTION_FILENAME


def reference_file(topic: str) -> Path:
    return topic_dir(topic) / REFERENCE_FILENAME


def runs_dir(topic: str) -> Path:
    return topic_dir(topic) / RUNS_DIRNAME


def list_run_ids(topic: str) -> list[str]:
    """Danh sách run_id đã có, MỚI -> CŨ. Chỉ tính thư mục có `summary.json`
    (run dở dang hoặc thư mục rác không được coi là run hợp lệ). Sort theo
    tên vì RUN_ID_FORMAT (%Y%m%d_%H%M%S) sort lexicographically đúng thứ tự
    thời gian."""
    d = runs_dir(topic)
    if not d.exists():
        return []
    ids = [
        p.name
        for p in d.iterdir()
        if p.is_dir() and (p / SUMMARY_FILENAME).exists()
    ]
    return sorted(ids, reverse=True)


@dataclass(frozen=True)
class RunPaths:
    """Toàn bộ đường dẫn file của MỘT run. `root` đã được tạo sẵn (mkdir) khi
    dựng qua `for_new_run()`; `for_run()` chỉ trỏ tới thư mục đã có, không
    kiểm tra tồn tại (dùng để đọc run cũ)."""

    topic: str
    run_id: str
    root: Path

    @property
    def answer(self) -> Path:
        return self.root / ANSWER_FILENAME

    @property
    def chunks(self) -> Path:
        return self.root / CHUNKS_FILENAME

    @property
    def generate_state(self) -> Path:
        return self.root / GENERATE_STATE_FILENAME

    @property
    def trajectory(self) -> Path:
        return self.root / LangSmithConfig.TRAJECTORY_FILENAME

    @property
    def summary(self) -> Path:
        return self.root / SUMMARY_FILENAME

    @property
    def scores(self) -> Path:
        return self.root / SCORES_FILENAME

    @property
    def run_config(self) -> Path:
        return self.root / RUN_CONFIG_FILENAME

    @property
    def conversation(self) -> Path:
        return self.root / CONVERSATION_FILENAME

    @property
    def turn_metrics(self) -> Path:
        return self.root / TURN_METRICS_FILENAME

    @classmethod
    def for_new_run(cls, topic: str, *, now: Optional[datetime] = None) -> "RunPaths":
        """Sinh run_id NGAY BÂY GIỜ (lúc bắt đầu run, không phải lúc export —
        xem plan5.md C1: run_id sinh lúc bắt đầu để artifact và điểm số cùng
        một id) và tạo sẵn thư mục root. Trùng giây thì thêm hậu tố _2, _3...
        thay vì ghi đè lên nhau."""
        base = (now or datetime.now()).strftime(RUN_ID_FORMAT)
        d = runs_dir(topic)
        d.mkdir(parents=True, exist_ok=True)
        run_id = base
        suffix = 1
        while (d / run_id).exists():
            suffix += 1
            run_id = f"{base}_{suffix}"
        root = d / run_id
        root.mkdir(parents=True, exist_ok=False)
        return cls(topic=topic, run_id=run_id, root=root)

    @classmethod
    def for_run(cls, topic: str, run_id: str) -> "RunPaths":
        return cls(topic=topic, run_id=run_id, root=runs_dir(topic) / run_id)


def resolve_latest(topic: str) -> Optional[str]:
    """Đọc LATEST.txt (đúng 1 dòng = run_id). Mất con trỏ hoặc trỏ tới run đã
    biến mất -> fallback `max(list_run_ids())`, đúng vì RUN_ID_FORMAT sort
    lexicographically theo thời gian."""
    pointer = runs_dir(topic) / LATEST_POINTER_FILENAME
    if pointer.exists():
        run_id = pointer.read_text(encoding="utf-8").strip()
        if run_id and (runs_dir(topic) / run_id / SUMMARY_FILENAME).exists():
            return run_id
    ids = list_run_ids(topic)
    return ids[0] if ids else None


def write_latest_pointer(topic: str, run_id: str) -> None:
    """Ghi ATOMICALLY (.tmp rồi os.replace) — không dùng symlink vì cần
    quyền admin trên Windows."""
    d = runs_dir(topic)
    d.mkdir(parents=True, exist_ok=True)
    pointer = d / LATEST_POINTER_FILENAME
    tmp = pointer.with_name(LATEST_POINTER_FILENAME + ".tmp")
    tmp.write_text(run_id + "\n", encoding="utf-8")
    os.replace(tmp, pointer)


def resolve_source_run(
    topic: str,
    *,
    explicit_run_id: Optional[str] = None,
    predicate: Optional[Callable[["RunPaths"], bool]] = None,
) -> tuple[Optional[str], str]:
    """Chọn run NGUỒN để đọc cache/replay — xem plan5.md C2.

    Thứ tự ưu tiên: run chọn tường minh (vd người dùng chọn trên UI) ->
    LATEST.txt -> run mới nhất (mới -> cũ) thỏa `predicate`. `predicate`
    nhận một `RunPaths` (đã biết tồn tại `summary.json`) và tự đọc thêm file
    nào cần để quyết định.

    Returns:
        (run_id | None, reason) — `reason` luôn có, kể cả khi không tìm được
        run nào, để log rõ vì sao (thà không có số còn hơn số gây hiểu nhầm).
    """
    if explicit_run_id:
        rp = RunPaths.for_run(topic, explicit_run_id)
        if not rp.summary.exists():
            return (
                None,
                f"run chỉ định '{explicit_run_id}' không tồn tại hoặc thiếu summary.json",
            )
        if predicate is not None and not predicate(rp):
            return None, f"run chỉ định '{explicit_run_id}' không thỏa điều kiện lọc"
        return explicit_run_id, f"dùng run được chỉ định tường minh '{explicit_run_id}'"

    latest = resolve_latest(topic)
    tried: list[str] = []
    if latest is not None:
        rp = RunPaths.for_run(topic, latest)
        if predicate is None or predicate(rp):
            return latest, f"dùng LATEST.txt -> '{latest}'"
        tried.append(latest)

    for run_id in list_run_ids(topic):
        if run_id in tried:
            continue
        rp = RunPaths.for_run(topic, run_id)
        if predicate is None or predicate(rp):
            reason = (
                f"LATEST.txt ('{latest}') không thỏa điều kiện, dùng run mới nhất "
                f"thỏa điều kiện: '{run_id}'"
                if latest is not None
                else f"không có LATEST.txt, dùng run mới nhất thỏa điều kiện: '{run_id}'"
            )
            return run_id, reason
        tried.append(run_id)

    all_ids = list_run_ids(topic)
    if not all_ids:
        return None, f"chủ đề '{topic}' chưa có run nào trong runs/"
    return (
        None,
        f"đã quét {len(all_ids)} run ({', '.join(all_ids)}) nhưng không run nào thỏa điều kiện lọc",
    )
