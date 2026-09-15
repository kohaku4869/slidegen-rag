"""
Đọc/ghi dữ liệu benchmark của một chủ đề trong data/test/<topic>/.

Mỗi chủ đề gồm: question.txt, reference.txt (đầu vào, viết tay) và answer.txt,
chunks.txt, generate_state.json, trajectory.txt (đầu ra do agent sinh, dùng
làm cache cho lần chấm sau).
"""

import json
import logging
import re
from pathlib import Path

from slidegen_rag.evaluate.settings import DATA_TEST_DIR

logger = logging.getLogger(__name__)


def get_available_topics() -> list[str]:
    """Tìm tất cả thư mục chủ đề trong data/test/."""
    if not DATA_TEST_DIR.exists():
        return []
    topics = []
    for d in sorted(DATA_TEST_DIR.iterdir()):
        if (
            d.is_dir()
            and (d / "question.txt").exists()
            and (d / "reference.txt").exists()
        ):
            topics.append(d.name)
    return topics


def load_questions(file_path: Path) -> list[dict]:
    """Đọc danh sách câu hỏi từ question.txt."""
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_blocks = re.split(r"\n(?=\d+\.\s+)", content.strip())
    questions = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        match = re.match(r"^(\d+)\.\s*(.+)$", block, re.DOTALL)
        if match:
            q_num = int(match.group(1))
            questions.append(
                {
                    "num": q_num,
                    "text": match.group(2).strip(),
                }
            )
    return questions


def load_references(file_path: Path) -> dict[int, str]:
    """Đọc đáp án tham chiếu từ reference.txt (-N. format)."""
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"(?:^|\n)-(\d+)\.[ \t]*", content.strip())
    references = {}
    i = 1
    while i < len(blocks) - 1:
        num = int(blocks[i])
        text = blocks[i + 1].strip()
        references[num] = text
        i += 2
    return references


def load_answers(file_path: Path) -> dict[int, str]:
    """Đọc câu trả lời đã lưu từ answer.txt."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"(?:^|\n)-(\d+)\.[ \t]*", content.strip())
    answers = {}
    if len(blocks) > 1:
        i = 1
        while i < len(blocks) - 1:
            num = int(blocks[i])
            text = blocks[i + 1].strip()
            answers[num] = text
            i += 2
    else:
        blocks = re.split(r"(?:^|\n)(\d+)\.[ \t]*", content.strip())
        i = 1
        while i < len(blocks) - 1:
            num = int(blocks[i])
            text = blocks[i + 1].strip()
            answers[num] = text
            i += 2
    return answers


def save_answers(file_path: Path, answers: dict[int, str]):
    """Ghi toàn bộ câu trả lời ra answer.txt theo format -N."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for num in sorted(answers.keys()):
            text = answers[num].strip()
            f.write(f"-{num}. {text}\n\n")


def load_chunks(file_path: Path) -> dict[int, list[str]]:
    """Đọc danh sách các context chunks đã sử dụng từ chunks.txt."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    q_blocks = re.split(r"(?:^|\n)===\s*(?:CÂU HỎI|QUESTION)\s*(\d+)\s*===", content)
    chunks_dict = {}
    i = 1
    while i < len(q_blocks) - 1:
        q_num = int(q_blocks[i])
        raw_text = q_blocks[i + 1].strip()
        chunk_items = re.split(r"\[CHUNK\s*\d+\]\n?", raw_text)
        cleaned_chunks = [c.strip() for c in chunk_items if c.strip()]
        chunks_dict[q_num] = cleaned_chunks
        i += 2
    return chunks_dict


def save_chunks(file_path: Path, chunks: dict[int, list[str]]):
    """Ghi các context chunks đã sử dụng ra chunks.txt."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for num in sorted(chunks.keys()):
            f.write(f"=== CÂU HỎI {num} ===\n\n")
            doc_list = chunks[num]
            if not doc_list:
                f.write("[CHUNK 1]\n(Không có chunk nào được sử dụng)\n\n")
            else:
                for idx, doc in enumerate(doc_list, 1):
                    f.write(f"[CHUNK {idx}]\n{doc.strip()}\n\n")


def load_generate_state(file_path: Path) -> dict[int, dict]:
    """
    Đọc generate_state.json — phần state cấp cha mà generate_node đọc NGOÀI
    `documents` (đã nằm trong chunks.txt): `facet_sufficiency` và
    `document_priorities`.

    Tồn tại để chế độ "chỉ chạy lại generate" (regenerate_answers_for_topic)
    dựng lại ĐÚNG input mà generate_node từng nhận, thay vì đoán. Thiếu 2
    field này thì generate chạy lại sẽ lệch so với lần chạy đầy đủ:
    facet_sufficiency rỗng khiến _format_sufficiency_note báo "ngữ cảnh ĐẦY
    ĐỦ, không cần rào đón" — che mất đúng hành vi rào đón/từ chối đang cần đo.
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Không đọc được {file_path.name}: {e} — bỏ qua cache này")
        return {}
    return {int(num): payload for num, payload in raw.items()}


def save_generate_state(file_path: Path, states: dict[int, dict]):
    """Ghi generate_state.json (xem load_generate_state)."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            {str(num): states[num] for num in sorted(states)},
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_trajectory_texts(file_path: Path) -> dict[int, str]:
    """
    Đọc lại nội dung trajectory.txt (text thô, đã render sẵn) theo từng câu
    hỏi — dùng để hiển thị lại trên dashboard khi chạy từ cache (không cần
    parse thành Trajectory object, chỉ cần hiển thị).
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    q_blocks = re.split(
        r"(?:^|\n)===\s*(?:CÂU HỎI|QUESTION)\s*(\d+)\s*===[ \t]*", content
    )
    texts: dict[int, str] = {}
    i = 1
    while i < len(q_blocks) - 1:
        q_num = int(q_blocks[i])
        texts[q_num] = q_blocks[i + 1].strip()
        i += 2
    return texts
