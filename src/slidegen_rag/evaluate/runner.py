"""
Điều phối một lượt đánh giá chủ đề: lấy câu trả lời (chạy agent, đọc cache,
hoặc chỉ chạy lại generate) rồi chấm điểm và gom thành bảng chi tiết + summary.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Optional

import pandas as pd

from slidegen_rag.evaluate.io import (
    load_answers,
    load_chunks,
    load_generate_state,
    load_questions,
    load_references,
    load_trajectory_texts,
    save_answers,
    save_chunks,
    save_generate_state,
)
from slidegen_rag.evaluate.judge import (
    create_evaluator_llm,
    evaluate_single_answer_with_judge,
)
from slidegen_rag.evaluate.metrics import (
    calculate_answer_coverage,
    calculate_keyword_coverage,
    calculate_keyword_coverage_strict,
)
from slidegen_rag.evaluate.settings import (
    DATA_INDEX_DIR,
    DATA_TEST_DIR,
    GENERATE_STATE_FILENAME,
)

logger = logging.getLogger(__name__)


def get_answers_and_chunks_for_topic(
    topic: str,
    force_rerun: bool = False,
    progress_callback=None,
    lowcost: bool = False,
) -> tuple[list[dict], bool]:
    """
    Lấy danh sách câu trả lời và chunks cho chủ đề.
    Nếu file answer.txt và chunks.txt đã tồn tại đầy đủ và force_rerun=False,
    đọc trực tiếp từ file (cache).

    Args:
        lowcost: True → generate_node dùng model tự host (LLM_BASE_URL) thay
            Gemini (xem generate_node.set_generate_provider). Các node khác
            (retrieve/decompose/context_evaluator) không đổi — vẫn Gemini.

    Returns:
        (results, was_cached)
    """
    from slidegen_rag.main import load_ra
    from slidegen_rag.agent.graph.builder import build_graph
    from slidegen_rag.agent.nodes.generate_node import set_generate_provider
    from slidegen_rag.agent.tools.tools import (
        set_retrieval_augmentation,
        set_web_search_enabled,
    )
    from slidegen_rag.agent.observability.trajectory import (
        TrajectoryRecorder,
        Trajectory,
        make_initial_state,
        save_trajectories,
    )
    from slidegen_rag.config import LangSmithConfig

    question_file = DATA_TEST_DIR / topic / "question.txt"
    reference_file = DATA_TEST_DIR / topic / "reference.txt"
    answer_file = DATA_TEST_DIR / topic / "answer.txt"
    chunks_file = DATA_TEST_DIR / topic / "chunks.txt"
    trajectory_file = DATA_TEST_DIR / topic / LangSmithConfig.TRAJECTORY_FILENAME
    generate_state_file = DATA_TEST_DIR / topic / GENERATE_STATE_FILENAME
    v2_path = DATA_INDEX_DIR / f"{topic}_v2.pkl"
    index_path = v2_path if v2_path.exists() else DATA_INDEX_DIR / f"{topic}.pkl"

    questions = load_questions(question_file)
    references = load_references(reference_file)

    if not questions:
        raise ValueError(f"Không tìm thấy câu hỏi nào trong {question_file}")

    cached_answers = load_answers(answer_file)
    cached_chunks = load_chunks(chunks_file)

    all_cached = (
        not force_rerun
        and len(cached_answers) >= len(questions)
        and all(q["num"] in cached_answers for q in questions)
    )

    if all_cached:
        if progress_callback:
            progress_callback(
                f"[{topic}] Đã nạp {len(questions)} câu từ cache (answer.txt & chunks.txt)"
            )
            # answer.txt/chunks.txt chỉ cache theo topic, KHÔNG theo provider —
            # nếu bật lowcost mà dùng cache cũ, câu trả lời vẫn là của lần
            # chạy trước (có thể là Gemini), không phải của model tự host.
            if lowcost:
                progress_callback(
                    f"[{topic}] ⚠️ Lowcost đang BẬT nhưng đang đọc cache từ lần chạy "
                    f"trước (không rõ provider) — tick 'Buộc chạy lại' để có kết quả "
                    f"thật sự sinh bởi model tự host"
                )
        logger.info(f"[{topic}] Sử dụng kết quả có sẵn từ answer.txt & chunks.txt")
        cached_trajectories = load_trajectory_texts(trajectory_file)
        results = []
        for q in questions:
            num = q["num"]
            results.append(
                {
                    "num": num,
                    "question": q["text"],
                    "response": cached_answers.get(num, ""),
                    "reference": references.get(num, ""),
                    "used_chunks": cached_chunks.get(num, []),
                    "trajectory": cached_trajectories.get(num),
                }
            )
        return results, True

    if not index_path.exists():
        raise FileNotFoundError(f"Không tìm thấy index: {index_path}")

    if not force_rerun:
        missing_nums = [q["num"] for q in questions if q["num"] not in cached_answers]
        reason = (
            f"thiếu {len(missing_nums)}/{len(questions)} câu trong cache ({missing_nums[:10]})"
            if missing_nums
            else f"số lượng câu trong cache ({len(cached_answers)}) < tổng số câu hỏi ({len(questions)})"
        )
        logger.warning(
            f"[{topic}] Không dùng được cache do: {reason} (dù không tick 'Buộc chạy lại'). Bắt đầu chạy lại Agent..."
        )
        if progress_callback:
            progress_callback(
                f"[{topic}] ⚠️ Không dùng được cache ({reason}) — Đang khởi tạo CRAG Agent..."
            )
    elif progress_callback:
        progress_callback(f"[{topic}] Đang khởi tạo CRAG Agent...")

    ra = load_ra(index_path)
    set_retrieval_augmentation(ra)
    # Benchmark closed-book: reference.txt được viết trên giả định agent CHỈ
    # dùng data/src/pldc.md. Nếu web_search được bind trong lúc eval, agent có
    # thể trả lời đúng-trong-thực-tế nhưng mâu thuẫn với reference (vd. tra ra
    # thời hiệu khởi kiện dân sự thật trong khi reference nói "tài liệu không
    # đề cập"), làm sai lệch điểm accuracy/faithfulness. Chạy thật (main.py)
    # vẫn giữ mặc định bật.
    set_web_search_enabled(False)
    set_generate_provider("local" if lowcost else "gemini")
    graph = build_graph()
    recorder = TrajectoryRecorder(
        graph, tags=["evaluate", topic], metadata={"topic": topic}
    )

    results = []
    new_answers = {}
    new_chunks = {}
    new_generate_states: dict[int, dict] = {}
    new_trajectories: dict[int, Trajectory] = {}
    total = len(questions)

    # Tích lũy token Gemini XUYÊN SUỐT cả benchmark — mỗi traj.token_usage chỉ
    # là của RIÊNG câu hỏi đó (LLMModel.reset_usage() chạy đầu mỗi
    # TrajectoryRecorder.arun()), nên phải cộng dồn thủ công ở đây mới theo
    # dõi được tổng chi phí thật của cả lần chạy topic này.
    gemini_running_total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }

    for idx, q in enumerate(questions):
        num = q["num"]
        q_text = q["text"]
        ref = references.get(num, "")

        if progress_callback:
            progress_callback(
                f"[{topic}] Đang chạy Agent sinh câu trả lời {num}/{total}..."
            )

        logger.info(f"[{topic}] Agent đang trả lời câu {num}/{total}: {q_text[:70]}...")

        try:
            state_res, traj = recorder.run(
                make_initial_state(q_text),
                run_name=f"{topic}:Q{num}",
                metadata={
                    "topic": topic,
                    "question_num": num,
                },
            )
            answer = (state_res.get("generation") or "").strip()
            used_docs = state_res.get("documents", [])
            generate_state = {
                "facet_sufficiency": state_res.get("facet_sufficiency") or [],
                "document_priorities": state_res.get("document_priorities") or [],
            }
        except Exception as e:
            logger.error(f"[{topic}] Lỗi khi chạy Agent câu {num}: {e}")
            answer = f"[LỖI AGENT] {e}"
            used_docs = []
            generate_state = {"facet_sufficiency": [], "document_priorities": []}
            traj = Trajectory(query=q_text, started_at=datetime.now(), error=str(e))

        new_answers[num] = answer
        new_chunks[num] = used_docs
        new_generate_states[num] = generate_state
        new_trajectories[num] = traj

        q_usage = traj.token_usage or {}
        q_gemini_tokens = q_usage.get(
            "gemini_total_tokens", q_usage.get("total_tokens", 0)
        )
        q_cost_usd = q_usage.get("total_cost_usd", 0.0)
        gemini_running_total["input_tokens"] += q_usage.get(
            "gemini_input_tokens", q_usage.get("input_tokens", 0)
        )
        gemini_running_total["output_tokens"] += q_usage.get(
            "gemini_output_tokens", q_usage.get("output_tokens", 0)
        )
        gemini_running_total["total_tokens"] += q_gemini_tokens
        gemini_running_total["cost_usd"] += q_cost_usd

        usage_msg = (
            f"[{topic}] Câu {num}/{total}: Gemini +{q_gemini_tokens:,} token (~${q_cost_usd:.4f}) "
            f"| Tích lũy: {gemini_running_total['total_tokens']:,} token "
            f"(~${gemini_running_total['cost_usd']:.4f} USD)"
        )
        logger.info(usage_msg)
        if progress_callback:
            progress_callback(usage_msg)

        results.append(
            {
                "num": num,
                "question": q_text,
                "response": answer,
                "reference": ref,
                "used_chunks": used_docs,
                "trajectory": traj.render_text(),
            }
        )

    save_answers(answer_file, new_answers)
    save_chunks(chunks_file, new_chunks)
    save_generate_state(generate_state_file, new_generate_states)
    save_trajectories(trajectory_file, new_trajectories, topic)
    logger.info(
        f"[{topic}] Đã lưu câu trả lời vào {answer_file}, chunks vào {chunks_file}, "
        f"generate state vào {generate_state_file}, trajectory vào {trajectory_file}"
    )
    logger.info(
        f"[{topic}] Tổng Gemini token cả lần chạy: "
        f"in={gemini_running_total['input_tokens']:,} out={gemini_running_total['output_tokens']:,} "
        f"tổng={gemini_running_total['total_tokens']:,} "
        f"| Chi phí ước tính: ${gemini_running_total['cost_usd']:.4f} USD "
        f"(~{gemini_running_total['cost_usd'] * 25400:.0f} VNĐ)"
    )

    return results, False


def regenerate_answers_for_topic(
    topic: str,
    progress_callback=None,
    lowcost: bool = False,
) -> list[dict]:
    """
    Chạy lại DUY NHẤT generate_node trên bộ chunk đã có sẵn trong chunks.txt,
    KHÔNG chạy retrieve/grade/context_evaluator và không cần nạp index.

    Dùng để lặp nhanh khi sửa GENERATE_SYSTEM_PROMPT hoặc đổi model/provider
    của riêng bước tổng hợp: một lần chạy đầy đủ tốn ~10 phút và tiêu tốn
    ~2/3 chi phí cho các node retrieval, trong khi phần thực sự cần đo lại
    chỉ là bước generate (~1/4 thời gian, ~1/3 chi phí của run đầy đủ). Giữ
    nguyên chunk cũng làm phép so sánh SẠCH hơn: mọi khác biệt điểm số đến từ
    prompt/model của generate chứ không lẫn nhiễu do retrieval khác đi giữa
    hai lần chạy.

    Ghi đè answer.txt (câu trả lời mới) nhưng KHÔNG đụng chunks.txt hay
    trajectory.txt — chunk là đầu vào cố định, còn trajectory cũ vẫn mô tả
    đúng cách bộ chunk đó được tìm ra.

    Args:
        lowcost: True → generate_node dùng model tự host thay Gemini.

    Returns:
        results: cùng shape với get_answers_and_chunks_for_topic (để
            run_evaluation_for_topic dùng lại nguyên si đường chấm điểm).
    """
    from slidegen_rag.agent.models.llm import LLMModel
    from slidegen_rag.agent.nodes.generate_node import (
        generate_node,
        set_generate_provider,
    )
    from slidegen_rag.config import LangSmithConfig

    question_file = DATA_TEST_DIR / topic / "question.txt"
    reference_file = DATA_TEST_DIR / topic / "reference.txt"
    answer_file = DATA_TEST_DIR / topic / "answer.txt"
    chunks_file = DATA_TEST_DIR / topic / "chunks.txt"
    trajectory_file = DATA_TEST_DIR / topic / LangSmithConfig.TRAJECTORY_FILENAME
    generate_state_file = DATA_TEST_DIR / topic / GENERATE_STATE_FILENAME

    questions = load_questions(question_file)
    references = load_references(reference_file)
    cached_chunks = load_chunks(chunks_file)
    cached_states = load_generate_state(generate_state_file)
    cached_trajectories = load_trajectory_texts(trajectory_file)

    if not questions:
        raise ValueError(f"Không tìm thấy câu hỏi nào trong {question_file}")

    missing = [q["num"] for q in questions if not cached_chunks.get(q["num"])]
    if len(missing) == len(questions):
        raise FileNotFoundError(
            f"[{topic}] chunks.txt không có chunk nào — chế độ 'chỉ chạy lại "
            f"generate' cần một lần chạy đầy đủ trước đó để có sẵn chunk."
        )
    if missing:
        logger.warning(
            f"[{topic}] {len(missing)} câu không có chunk trong chunks.txt "
            f"({missing[:10]}) — vẫn chạy generate với ngữ cảnh rỗng cho các câu này"
        )

    if not cached_states:
        # DỪNG HẲN thay vì chỉ cảnh báo: thiếu facet_sufficiency thì
        # _format_sufficiency_note báo "ngữ cảnh ĐẦY ĐỦ, không cần rào đón" cho
        # MỌI câu, trong khi run đầy đủ gần nhất có 20/30 câu context_sufficient
        # =False. Điểm chấm ra vẫn trông như so sánh được nhưng thực chất đo hai
        # điều kiện khác nhau — đã thực sự dẫn tới kết luận sai một lần (run
        # 20260915_162029: Q11 rơi 9.75 -> 2.0 vì được bảo ngữ cảnh đã đầy đủ
        # nên bịa Điều 124/126, trong khi bản 8.56 rào đón đúng và đạt 9.75).
        # Số liệu gây hiểu nhầm nguy hiểm hơn là không có số liệu.
        raise FileNotFoundError(
            f"[{topic}] Thiếu {GENERATE_STATE_FILENAME} — chế độ 'chỉ chạy lại "
            f"generate' sẽ chạy như thể MỌI câu đều có ngữ cảnh đầy đủ, nên điểm "
            f"không so sánh được với run đầy đủ (đặc biệt là các câu mà đáp án "
            f"đúng chính là 'tài liệu không đề cập'). Hãy chạy đầy đủ 1 lần — nó "
            f"sinh ra file này — rồi mới dùng chế độ này để lặp nhanh."
        )

    set_generate_provider("local" if lowcost else "gemini")
    LLMModel.reset_usage()

    results = []
    new_answers = {}
    total = len(questions)

    for idx, q in enumerate(questions, 1):
        num = q["num"]
        q_text = q["text"]
        docs = cached_chunks.get(num, [])
        state = cached_states.get(num, {})
        priorities = state.get("document_priorities") or []

        if progress_callback:
            progress_callback(
                f"[{topic}] Đang sinh lại câu trả lời {idx}/{total} "
                f"({len(docs)} chunk có sẵn)..."
            )

        try:
            out = asyncio.run(
                generate_node(
                    {
                        "query": q_text,
                        "documents": docs,
                        # Lệch độ dài so với documents (chunks.txt sửa tay, hoặc
                        # cache cũ) sẽ khiến build_context bỏ qua priority và
                        # quay về cắt theo thứ tự — chấp nhận được, nhưng ghi log
                        # để không âm thầm so sánh nhầm 2 chế độ cắt khác nhau.
                        "document_priorities": priorities,
                        "facet_sufficiency": state.get("facet_sufficiency") or [],
                    }
                )
            )
            answer = (out.get("generation") or "").strip()
        except Exception as e:
            logger.error(f"[{topic}] Lỗi khi sinh lại câu {num}: {e}")
            answer = f"[LỖI GENERATE] {e}"

        if priorities and len(priorities) != len(docs):
            logger.warning(
                f"[{topic}] Câu {num}: document_priorities ({len(priorities)}) lệch "
                f"số chunk ({len(docs)}) — build_context bỏ qua priority, cắt theo "
                f"thứ tự chunk thay vì theo chất lượng"
            )

        new_answers[num] = answer
        results.append(
            {
                "num": num,
                "question": q_text,
                "response": answer,
                "reference": references.get(num, ""),
                "used_chunks": docs,
                "trajectory": cached_trajectories.get(num),
            }
        )

    save_answers(answer_file, new_answers)

    usage = LLMModel.get_usage_summary()
    cost = usage.get("total_cost_usd", 0.0)
    logger.info(
        f"[{topic}] Đã sinh lại {total} câu trả lời (chỉ generate) và ghi vào "
        f"{answer_file} | Gemini token: in={usage.get('gemini_input_tokens', 0):,} "
        f"out={usage.get('gemini_output_tokens', 0):,} | Chi phí ước tính: "
        f"${cost:.4f} USD (~{cost * 25400:.0f} VNĐ)"
    )

    return results


def _parse_trajectory_summary(
    traj_text: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Trích 3 chỉ số ngắn gọn từ text trajectory (dùng chung cho cả câu vừa
    chạy lẫn câu nạp từ cache — cả hai đều ở dạng text render sẵn, nên chỉ
    cần parse text thay vì phải giữ lại Trajectory object).

    "Vòng lặp retrieve: N/MAX" giữ nguyên ý nghĩa MAX qua các facet (xem
    Trajectory.retrieve_count trong agent/observability/trajectory.py) —
    KHÔNG đổi format dòng này để không vỡ script/CSV cũ đang parse nó.
    "Vòng lặp theo facet: label1=n1, label2=n2, ..." là dòng MỚI, chỉ xuất
    hiện khi trajectory có >= 1 facet (mọi trajectory mới đều có).

    Returns:
        (traj_loops: "2/3" hoặc None,
         traj_tools: "semantic_search x1, ..." hoặc None,
         traj_facet_loops: "label1=n1, label2=n2, ..." hoặc None)
    """
    if not traj_text:
        return None, None, None
    loop_match = re.search(r"Vòng lặp retrieve:\s*(\d+/\d+)", traj_text)
    tools_match = re.search(r"Tool đã gọi\s*:\s*(.+)", traj_text)
    facet_loops_match = re.search(r"Vòng lặp theo facet:\s*(.+)", traj_text)
    traj_loops = loop_match.group(1) if loop_match else None
    traj_tools = tools_match.group(1).strip() if tools_match else None
    traj_facet_loops = facet_loops_match.group(1).strip() if facet_loops_match else None
    return traj_loops, traj_tools, traj_facet_loops


def run_evaluation_for_topic(
    topic: str,
    force_rerun: bool = False,
    progress_callback=None,
    lowcost: bool = False,
    regenerate_only: bool = False,
) -> tuple[pd.DataFrame, dict, bool]:
    """
    Thực hiện đánh giá toàn diện một chủ đề.
    Nếu một câu hỏi bị lỗi (không đánh giá được), câu đó sẽ được đánh dấu FAILED
    và KHÔNG ĐƯỢC TÍNH VÀO ĐIỂM TỔNG để tránh làm lệch kết quả.

    Args:
        lowcost: True → generate_node dùng model tự host thay Gemini (xem
            get_answers_and_chunks_for_topic). Ghi lại vào summary["generate_provider"]
            để về sau biết file summary/score nào ứng với provider nào.
        regenerate_only: True → KHÔNG chạy agent, chỉ chạy lại generate_node
            trên chunk có sẵn rồi chấm (xem regenerate_answers_for_topic).
            Bỏ qua `force_rerun` vì chế độ này luôn sinh lại câu trả lời.

    Returns:
        (detail_df, summary_dict, was_cached)
    """
    if regenerate_only:
        results = regenerate_answers_for_topic(
            topic, progress_callback=progress_callback, lowcost=lowcost
        )
        was_cached = False
    else:
        results, was_cached = get_answers_and_chunks_for_topic(
            topic,
            force_rerun=force_rerun,
            progress_callback=progress_callback,
            lowcost=lowcost,
        )

    primary_llm, fallback_llm = create_evaluator_llm()
    if not primary_llm and not fallback_llm:
        raise RuntimeError(
            "Không có LLM Judge nào khả dụng. Vui lòng kiểm tra LLM_BASE_URL hoặc GEMINI_API_KEY trong .env"
        )

    rows = []
    total = len(results)

    for idx, item in enumerate(results, 1):
        num = item["num"]
        q_text = item["question"]
        resp = item["response"]
        ref = item["reference"]
        chunks = item["used_chunks"]
        traj_text = item.get("trajectory")
        traj_loops, traj_tools, traj_facet_loops = _parse_trajectory_summary(traj_text)

        if progress_callback:
            progress_callback(f"[{topic}] LLM Judge đang chấm câu {idx}/{total}...")

        logger.info(f"[{topic}] Đang chấm câu {idx}/{total}...")

        eval_res = asyncio.run(
            evaluate_single_answer_with_judge(
                primary_llm, fallback_llm, q_text, ref, resp, chunks
            )
        )

        coverage = calculate_keyword_coverage(ref, chunks)
        coverage_strict = calculate_keyword_coverage_strict(ref, chunks)
        # answer_coverage: cùng công thức strict nhưng đo trên CÂU TRẢ LỜI thay vì
        # chunk — tách được 3 loại lỗi mà coverage chunk đơn thuần không phân biệt
        # nổi (xem docstring calculate_answer_coverage).
        answer_coverage = calculate_answer_coverage(ref, resp)
        utilization = (
            round(answer_coverage / coverage_strict * 100.0, 1)
            if coverage_strict > 0
            else 0.0
        )

        if eval_res is not None:
            is_success = True
            acc_val = round(eval_res.accuracy, 2)
            comp_val = round(eval_res.completeness, 2)
            rel_val = round(eval_res.relevance, 2)
            faith_val = round(eval_res.faithfulness, 2)
            total_score = round((acc_val + comp_val + rel_val + faith_val) / 4.0, 2)
            feedback_val = eval_res.feedback
        else:
            is_success = False
            acc_val = None
            comp_val = None
            rel_val = None
            faith_val = None
            total_score = None
            feedback_val = (
                "[LỖI ĐÁNH GIÁ] Không nhận được phản hồi hợp lệ từ LLM Judge."
            )

        rows.append(
            {
                "STT": num,
                "topic": topic,
                "question": q_text,
                "response": resp,
                "reference": ref,
                "accuracy": acc_val,
                "completeness": comp_val,
                "relevance": rel_val,
                "faithfulness": faith_val,
                "total_score": total_score,
                "coverage_pct": round(coverage, 1),
                "coverage_pct_strict": round(coverage_strict, 1),
                "answer_coverage_pct": round(answer_coverage, 1),
                "utilization_pct": utilization,
                "chunks_count": len(chunks),
                "feedback": feedback_val,
                "is_success": is_success,
                "used_chunks": chunks,
                "traj_loops": traj_loops,
                "traj_tools": traj_tools,
                "traj_facet_loops": traj_facet_loops,
                "traj_text": traj_text
                or "(chưa có trajectory — cache cũ trước khi bật tính năng này)",
            }
        )

    detail_df = pd.DataFrame(rows)

    valid_df = detail_df[detail_df["is_success"] == True]
    success_count = len(valid_df)
    failed_count = len(detail_df) - success_count

    if not valid_df.empty:
        summary = {
            "accuracy": float(valid_df["accuracy"].astype(float).mean()),
            "completeness": float(valid_df["completeness"].astype(float).mean()),
            "relevance": float(valid_df["relevance"].astype(float).mean()),
            "faithfulness": float(valid_df["faithfulness"].astype(float).mean()),
            "total_score": float(valid_df["total_score"].astype(float).mean()),
            "coverage_pct": float(valid_df["coverage_pct"].astype(float).mean()),
            "coverage_pct_strict": float(
                valid_df["coverage_pct_strict"].astype(float).mean()
            ),
            "answer_coverage_pct": float(
                valid_df["answer_coverage_pct"].astype(float).mean()
            ),
            "utilization_pct": float(valid_df["utilization_pct"].astype(float).mean()),
            "avg_chunks": float(detail_df["chunks_count"].mean()),
            "total_questions": len(detail_df),
            "success_count": success_count,
            "failed_count": failed_count,
            "generate_provider": "local" if lowcost else "gemini",
            # Chunk tái dùng từ lần chạy đầy đủ trước → coverage_pct/
            # coverage_pct_strict/avg_chunks KHÔNG phải số đo mới, chỉ
            # accuracy/completeness/relevance/faithfulness/answer_coverage_pct
            # mới phản ánh thay đổi ở bước generate.
            "regenerate_only": regenerate_only,
        }
    else:
        summary = {
            "accuracy": 0.0,
            "completeness": 0.0,
            "relevance": 0.0,
            "faithfulness": 0.0,
            "total_score": 0.0,
            "coverage_pct": 0.0,
            "coverage_pct_strict": 0.0,
            "answer_coverage_pct": 0.0,
            "utilization_pct": 0.0,
            "avg_chunks": 0.0,
            "total_questions": len(detail_df),
            "success_count": 0,
            "failed_count": failed_count,
            "generate_provider": "local" if lowcost else "gemini",
            "regenerate_only": regenerate_only,
        }

    return detail_df, summary, was_cached
