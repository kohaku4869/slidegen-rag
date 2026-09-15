import tempfile
from pathlib import Path

from slidegen_rag.evaluate.io import load_answers, load_references


def test_load_answers_normal():
    content = "-1. Câu trả lời 1\n\n-2. Câu trả lời 2\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, suffix=".txt"
    ) as f:
        f.write(content)
        path = Path(f.name)
    try:
        ans = load_answers(path)
        assert ans == {1: "Câu trả lời 1", 2: "Câu trả lời 2"}
    finally:
        path.unlink(missing_ok=True)


def test_load_answers_empty_answer_does_not_swallow_next_answer():
    content = "-9. Câu trả lời 9\n\n-10. \n\n-11. Câu trả lời 11\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, suffix=".txt"
    ) as f:
        f.write(content)
        path = Path(f.name)
    try:
        ans = load_answers(path)
        assert 10 in ans
        assert 11 in ans
        assert ans[10] == ""
        assert ans[11] == "Câu trả lời 11"
    finally:
        path.unlink(missing_ok=True)


def test_load_answers_fallback_format():
    content = "1. Câu trả lời 1\n\n2. \n\n3. Câu trả lời 3\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, suffix=".txt"
    ) as f:
        f.write(content)
        path = Path(f.name)
    try:
        ans = load_answers(path)
        assert ans[1] == "Câu trả lời 1"
        assert ans[2] == ""
        assert ans[3] == "Câu trả lời 3"
    finally:
        path.unlink(missing_ok=True)


def test_load_references_empty():
    content = "-1. Ref 1\n\n-2. \n\n-3. Ref 3\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, suffix=".txt"
    ) as f:
        f.write(content)
        path = Path(f.name)
    try:
        refs = load_references(path)
        assert refs[1] == "Ref 1"
        assert refs[2] == ""
        assert refs[3] == "Ref 3"
    finally:
        path.unlink(missing_ok=True)
