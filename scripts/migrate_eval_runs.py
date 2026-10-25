"""
migrate_eval_runs.py — chuyển layout đánh giá cũ (`scores_<ts>.csv`/
`summary_<ts>.json` nằm PHẲNG trong `data/test/<topic>/`, bị ghi đè mỗi lần
`answer.txt`/`chunks.txt`/`trajectory.txt` chạy lại) sang layout tự chứa
`data/test/<topic>/runs/<ts>/` (xem plan5.md Phần C).

KHÔNG dùng LLM, KHÔNG chạm index — thuần thao tác file, an toàn chạy nhiều
lần (idempotent).

Các bước cho MỖI topic:
    1. Ghép cặp `scores_<ts>.csv` <-> `summary_<ts>.json` theo timestamp
       trong tên file. Cặp lẻ (chỉ có 1 trong 2 file) VẪN được migrate,
       đánh dấu `artifacts_complete=false` trong `run_config.json`.
    2. Tạo `runs/<ts>/`, MOVE (giữ nguyên byte) `scores_<ts>.csv` ->
       `runs/<ts>/scores.csv`, `summary_<ts>.json` -> `runs/<ts>/summary.json`.
    3. Sinh `runs/<ts>/run_config.json` với `migrated: true`, `mode: "unknown"`,
       `config_fingerprint: null` (không bao giờ dùng làm cache — xem
       evaluate/run_config.py). `generate_provider` lấy từ summary.json nếu có
       (summary trước 2026-09-14 không có key này -> null).
    4. Gán `answer.txt`/`chunks.txt`/`generate_state.json`/`trajectory.txt`
       (đang nằm PHẲNG, bị các lần chạy sau đè lên nhau) cho run nào THẬT SỰ
       sinh ra chúng, bằng BẰNG CHỨNG chứ không đoán: so cột `response` của
       từng `scores.csv` (từ ts MỚI -> CŨ) với `answer.txt` hiện tại (đọc qua
       `load_answers`), theo từng câu đã `.strip()`. Run nào khớp >= 95% số
       câu thì COPY cả 4 file, đánh `artifacts_verified=true`. Không run nào
       khớp -> gán cho run MỚI NHẤT, `artifacts_verified=false`. Các run còn
       lại: `artifacts="missing"` + một dòng trong `MIGRATION_NOTES.md`.
    5. Ghi `runs/.migration_manifest.json`: liệt kê MỌI thao tác move thật sự
       ({src, dst, sha1, size}) kèm thời điểm — dùng cho `--rollback`.

`baseline_crag/` KHÔNG bị đụng vào (không có `summary.json`, không so sánh
được bằng máy — xem BASELINE_NOTES.md của chính nó).

Usage:
    uv run python scripts/migrate_eval_runs.py --dry-run
    uv run python scripts/migrate_eval_runs.py
    uv run python scripts/migrate_eval_runs.py --topic pldc_v2
    uv run python scripts/migrate_eval_runs.py --verify
    uv run python scripts/migrate_eval_runs.py --rollback
    uv run python scripts/migrate_eval_runs.py --force   # re-migrate ts đã có runs/<ts>/
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_TS_RE = re.compile(r"^(\d{8}_\d{6})$")
_MANIFEST_NAME = ".migration_manifest.json"
_NOTES_NAME = "MIGRATION_NOTES.md"
_MATCH_THRESHOLD = 0.95


class MigrationError(Exception):
    """Trạng thái không an toàn để migrate/rollback tự động — dừng lại thay
    vì đoán, đúng tinh thần 'thà không có số còn hơn số gây hiểu nhầm'."""


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _discover_pairs(topic_dir: Path) -> dict[str, dict[str, Path]]:
    """ts -> {"scores": Path, "summary": Path} — 1 trong 2 khoá có thể vắng
    (cặp lẻ)."""
    pairs: dict[str, dict[str, Path]] = {}
    for p in sorted(topic_dir.glob("scores_*.csv")):
        m = re.match(r"^scores_(\d{8}_\d{6})\.csv$", p.name)
        if m:
            pairs.setdefault(m.group(1), {})["scores"] = p
    for p in sorted(topic_dir.glob("summary_*.json")):
        m = re.match(r"^summary_(\d{8}_\d{6})\.json$", p.name)
        if m:
            pairs.setdefault(m.group(1), {})["summary"] = p
    return pairs


def _load_summary_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Không đọc được {path}: {e}")
        return {}


def _read_scores_responses(csv_path: Path) -> dict[int, str]:
    """{STT: response.strip()} từ 1 file scores_<ts>.csv — dùng pandas vì
    response có thể chứa newline/dấu phẩy đã được CSV-escape sẵn."""
    import pandas as pd

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        logger.warning(f"Không đọc được {csv_path}: {e}")
        return {}
    if "STT" not in df.columns or "response" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        try:
            stt = int(row["STT"])
        except (ValueError, TypeError):
            continue
        resp = row.get("response")
        out[stt] = str(resp).strip() if isinstance(resp, str) else ""
    return out


def _match_fraction(scores_answers: dict[int, str], flat_answers: dict[int, str]) -> float:
    if not scores_answers:
        return 0.0
    matched = sum(
        1
        for stt, text in scores_answers.items()
        if stt in flat_answers and flat_answers[stt].strip() == text
    )
    return matched / len(scores_answers)


def _build_migrated_run_config(
    ts: str, topic: str, summary_data: dict, *, artifacts_complete: bool
) -> dict:
    from slidegen_rag.evaluate.run_config import RUN_CONFIG_SCHEMA_VERSION

    return {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "run_id": ts,
        "topic": topic,
        "started_at": None,
        "finished_at": None,
        "mode": "unknown",
        "generate_provider": summary_data.get("generate_provider"),
        "lowcost": None,
        "max_turns": None,
        "web_search_enabled": None,
        "max_retrieve": None,
        "models": {
            "gemini_model": None,
            "local_base_url": None,
            "judge_primary": None,
            "judge_fallback": None,
        },
        "index": {"path": None, "source_sha1": None, "pkl_sha1": None},
        "dataset": {
            "questions_sha1": None,
            "references_sha1": None,
            "total_questions": summary_data.get("total_questions"),
        },
        "git_sha": None,
        "git_dirty": None,
        "source_run_id": None,
        "migrated": True,
        "artifacts_complete": artifacts_complete,
        "config_fingerprint": None,
    }


def _load_manifest(topic_runs_dir: Path) -> list[dict]:
    p = topic_runs_dir / _MANIFEST_NAME
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_manifest(topic_runs_dir: Path, manifest: list[dict]) -> None:
    p = topic_runs_dir / _MANIFEST_NAME
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _move_tracked(
    src: Path, dst: Path, manifest: list[dict], *, dry_run: bool
) -> None:
    if dry_run:
        logger.info(f"  [dry-run] move {src} -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    sha1 = _sha1_file(src)
    size = src.stat().st_size
    shutil.move(str(src), str(dst))
    manifest.append(
        {
            "src": str(src),
            "dst": str(dst),
            "sha1": sha1,
            "size": size,
            "moved_at": datetime.now().isoformat(),
        }
    )


def migrate_one_topic(
    topic_dir: Path, *, dry_run: bool = False, force: bool = False
) -> bool:
    """Migrate toàn bộ cặp `scores_<ts>.csv`/`summary_<ts>.json` phẳng của
    MỘT topic. Trả True nếu có ít nhất 1 ts thực sự được migrate (hoặc sẽ
    được, ở chế độ dry-run); False nếu không có gì để làm."""
    from slidegen_rag.evaluate.runs import RunPaths, runs_dir, write_latest_pointer

    topic = topic_dir.name
    pairs = _discover_pairs(topic_dir)
    if not pairs:
        logger.info(f"[{topic}] Không có scores_*.csv/summary_*.json phẳng — bỏ qua")
        return False

    rdir = runs_dir(topic)
    manifest = _load_manifest(rdir)
    did_something = False

    flat_answer = topic_dir / "answer.txt"
    from slidegen_rag.evaluate.io import load_answers

    flat_answers = load_answers(flat_answer) if flat_answer.exists() else {}

    # Từ MỚI -> CŨ: ts nào có scores.csv khớp >=95% answer.txt hiện tại
    # thắng — được gán cả 4 artifact hiện đang nằm phẳng.
    ordered_ts = sorted(pairs.keys(), reverse=True)
    winner_ts = None
    for ts in ordered_ts:
        scores_path = pairs[ts].get("scores")
        if scores_path is None:
            continue
        frac = _match_fraction(_read_scores_responses(scores_path), flat_answers)
        if frac >= _MATCH_THRESHOLD:
            winner_ts = ts
            break
    artifacts_verified = winner_ts is not None
    if winner_ts is None and ordered_ts:
        winner_ts = ordered_ts[0]  # fallback: run mới nhất, verified=false

    notes: list[str] = []

    for ts in ordered_ts:
        if not _TS_RE.match(ts):
            continue
        run_root = rdir / ts
        if (run_root / "summary.json").exists() and not force:
            logger.info(f"[{topic}] {ts} đã migrate trước đó — bỏ qua (idempotent)")
            continue

        did_something = True
        scores_path = pairs[ts].get("scores")
        summary_path = pairs[ts].get("summary")
        artifacts_complete = scores_path is not None and summary_path is not None
        logger.info(
            f"[{topic}] Migrate {ts} "
            f"(scores={'có' if scores_path else 'THIẾU'}, "
            f"summary={'có' if summary_path else 'THIẾU'})"
        )

        rp = RunPaths(topic=topic, run_id=ts, root=run_root)
        if not dry_run:
            run_root.mkdir(parents=True, exist_ok=True)

        if scores_path is not None:
            _move_tracked(scores_path, rp.scores, manifest, dry_run=dry_run)
        if summary_path is not None:
            _move_tracked(summary_path, rp.summary, manifest, dry_run=dry_run)

        summary_data = _load_summary_json(rp.summary) if not dry_run and summary_path else {}
        cfg = _build_migrated_run_config(
            ts, topic, summary_data, artifacts_complete=artifacts_complete
        )

        is_winner = ts == winner_ts
        if is_winner:
            cfg["artifacts_verified"] = artifacts_verified
            if not dry_run:
                for name in (
                    "answer.txt",
                    "chunks.txt",
                    "generate_state.json",
                    "trajectory.txt",
                ):
                    src = topic_dir / name
                    if src.exists():
                        shutil.copy2(src, run_root / name)
            else:
                logger.info(f"  [dry-run] copy answer/chunks/generate_state/trajectory -> {run_root}")
        else:
            cfg["artifacts_verified"] = False
            cfg["artifacts"] = "missing"
            notes.append(
                f"- Run `{ts}`: chỉ còn scores/summary — answer.txt/chunks.txt/"
                f"trajectory.txt của nó đã bị các lần chạy SAU ghi đè (không "
                f"khớp answer.txt hiện tại khi migrate)."
            )

        if not dry_run:
            with open(rp.run_config, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

    if notes and not dry_run:
        notes_path = rdir / _NOTES_NAME
        header = (
            f"# Ghi chú migration — {topic}\n\n"
            f"Sinh tự động bởi `scripts/migrate_eval_runs.py` "
            f"({datetime.now().isoformat()}). Các run dưới đây CHỈ còn "
            f"scores.csv/summary.json (điểm chất lượng) — answer.txt/"
            f"chunks.txt/trajectory.txt của chúng không còn khôi phục được.\n\n"
        )
        rdir.mkdir(parents=True, exist_ok=True)
        notes_path.write_text(header + "\n".join(notes) + "\n", encoding="utf-8")

    if did_something and not dry_run:
        _save_manifest(rdir, manifest)
        if winner_ts is not None:
            write_latest_pointer(topic, winner_ts)

    return did_something


def verify_topic(topic_dir: Path) -> bool:
    """Kiểm sha1 mọi entry trong manifest của topic này vẫn khớp file hiện
    tại — KHÔNG thay đổi gì. Trả True nếu mọi thứ khớp."""
    from slidegen_rag.evaluate.runs import runs_dir

    rdir = runs_dir(topic_dir.name)
    manifest = _load_manifest(rdir)
    if not manifest:
        return True
    ok = True
    for entry in manifest:
        dst = Path(entry["dst"])
        if not dst.exists():
            logger.error(f"[verify] THIẾU: {dst} (đáng lẽ tồn tại theo manifest)")
            ok = False
            continue
        actual = _sha1_file(dst)
        if actual != entry["sha1"]:
            logger.error(f"[verify] SHA1 LỆCH: {dst} (đã đổi kể từ lúc migrate)")
            ok = False
    if ok:
        logger.info(f"[verify] {topic_dir.name}: {len(manifest)} entry khớp sha1")
    return ok


def rollback_topic(topic_dir: Path, *, dry_run: bool = False) -> bool:
    """Đọc manifest, kiểm sha1 rồi move ngược — dừng lại (không move gì cả
    trong topic này) nếu BẤT KỲ entry nào lệch sha1, để không phá dữ liệu
    mới hơn migration bằng nhầm lẫn."""
    from slidegen_rag.evaluate.runs import runs_dir

    topic = topic_dir.name
    rdir = runs_dir(topic)
    manifest = _load_manifest(rdir)
    if not manifest:
        logger.info(f"[{topic}] Không có manifest — không có gì để rollback")
        return False

    for entry in manifest:
        dst = Path(entry["dst"])
        if not dst.exists():
            raise MigrationError(
                f"[{topic}] Rollback dừng: {dst} không còn tồn tại (đã bị xoá/di "
                f"chuyển sau migration) — không an toàn để tự động xử lý."
            )
        if _sha1_file(dst) != entry["sha1"]:
            raise MigrationError(
                f"[{topic}] Rollback dừng: {dst} đã bị SỬA kể từ lúc migrate "
                f"(sha1 lệch) — có thể một run mới đã ghi đè lên run_id này. "
                f"Không tự động rollback đè lên dữ liệu mới hơn."
            )

    for entry in manifest:
        src = Path(entry["src"])
        dst = Path(entry["dst"])
        if dry_run:
            logger.info(f"  [dry-run] move ngược {dst} -> {src}")
            continue
        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dst), str(src))

    if not dry_run:
        run_ids = {Path(e["dst"]).parent.name for e in manifest}
        for run_id in run_ids:
            run_root = rdir / run_id
            if run_root.exists():
                shutil.rmtree(run_root, ignore_errors=True)
        manifest_path = rdir / _MANIFEST_NAME
        manifest_path.unlink(missing_ok=True)
        notes_path = rdir / _NOTES_NAME
        notes_path.unlink(missing_ok=True)

    logger.info(f"[{topic}] Rollback {'(dry-run) ' if dry_run else ''}hoàn tất: {len(manifest)} file")
    return True


def main() -> None:
    from slidegen_rag.evaluate.settings import DATA_TEST_DIR

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in ra kế hoạch, không đổi gì.")
    parser.add_argument("--topic", type=str, default=None, help="Chỉ migrate 1 topic.")
    parser.add_argument("--rollback", action="store_true", help="Hoàn tác migration (đọc manifest).")
    parser.add_argument("--verify", action="store_true", help="Kiểm sha1 manifest, không đổi gì.")
    parser.add_argument("--force", action="store_true", help="Re-migrate ts đã có runs/<ts>/summary.json.")
    args = parser.parse_args()

    if not DATA_TEST_DIR.exists():
        logger.warning(f"Không tìm thấy {DATA_TEST_DIR}")
        return

    topic_dirs = (
        [DATA_TEST_DIR / args.topic]
        if args.topic
        else sorted(p for p in DATA_TEST_DIR.iterdir() if p.is_dir())
    )
    topic_dirs = [d for d in topic_dirs if d.is_dir()]

    if args.rollback:
        rolled = 0
        for d in topic_dirs:
            try:
                if rollback_topic(d, dry_run=args.dry_run):
                    rolled += 1
            except MigrationError as e:
                logger.error(f"[ERROR] {e}")
        logger.info(f"Hoàn tất rollback: {rolled} topic")
        return

    if args.verify:
        all_ok = all(verify_topic(d) for d in topic_dirs)
        logger.info("Verify: OK" if all_ok else "Verify: CÓ LỖI (xem log ở trên)")
        sys.exit(0 if all_ok else 1)

    migrated, skipped = 0, 0
    for d in topic_dirs:
        try:
            if migrate_one_topic(d, dry_run=args.dry_run, force=args.force):
                migrated += 1
            else:
                skipped += 1
        except MigrationError as e:
            logger.error(f"[ERROR] {e}")
            skipped += 1

    logger.info(
        f"Hoàn tất{' (dry-run)' if args.dry_run else ''}: {migrated} topic migrated, {skipped} skipped"
    )


if __name__ == "__main__":
    main()
