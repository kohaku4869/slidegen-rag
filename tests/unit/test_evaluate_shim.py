"""`evaluate.py` giờ chỉ là shim re-export sau khi package được tách.

Test này canh cho shim khỏi mục ruỗng: đổi tên/di chuyển một hàm ở module con
mà quên cập nhật shim thì lệnh chạy cũ `python -m slidegen_rag.evaluate.evaluate`
và mọi import cũ sẽ vỡ — nhưng vỡ lúc chạy dashboard chứ không phải lúc test,
nên không có test này thì không ai biết.
"""

from slidegen_rag.evaluate import evaluate as shim
from slidegen_rag.evaluate import io, judge, metrics, report, runner, settings


def test_shim_exports_every_name_in_all():
    missing = [name for name in shim.__all__ if not hasattr(shim, name)]
    assert missing == []


def test_shim_points_at_the_real_objects():
    """Không chỉ 'có tên' — phải đúng object của module con, không phải bản sao."""
    assert shim.load_answers is io.load_answers
    assert shim.calculate_answer_coverage is metrics.calculate_answer_coverage
    assert (
        shim.evaluate_single_answer_with_judge
        is judge.evaluate_single_answer_with_judge
    )
    assert shim.regenerate_answers_for_topic is runner.regenerate_answers_for_topic
    assert shim.export_evaluation_results is report.export_evaluation_results
    assert shim.DATA_TEST_DIR is settings.DATA_TEST_DIR
