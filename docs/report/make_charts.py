"""Sinh biểu đồ cho báo cáo từ các file summary_*.json trong data/test/."""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def load(topic, ts):
    with open(os.path.join(ROOT, "data", "test", topic, f"summary_{ts}.json"), encoding="utf-8") as f:
        return json.load(f)


# Naive RAG: kết quả nhóm tự chạy.
NAIVE = {"accuracy": 5.8, "completeness": 4.9, "relevance": 6.5, "faithfulness": 7.9, "total_score": 6.3}

systems = [
    ("Naive RAG", NAIVE),
    ("RAPTOR\n+ CRAG", load("pldc_raptor", "20260914_232437")),
    ("Cây heading\n+ CRAG", load("pldc", "20260914_232716")),
    ("Agent\nmulti-facet", load("pldc_v2", "20260915_192417")),
]
crit = [("accuracy", "Accuracy"), ("completeness", "Completeness"), ("relevance", "Relevance"),
        ("faithfulness", "Faithfulness"), ("total_score", "Tổng")]
colors = ["#B0B7C3", "#E0A458", "#5B8DB8", "#2E4A7D"]

# 1. So sánh các hệ thống theo tiêu chí
fig, ax = plt.subplots(figsize=(9, 4.6))
w = 0.2
x = np.arange(len(crit))
for i, (name, s) in enumerate(systems):
    vals = [s[k] for k, _ in crit]
    bars = ax.bar(x + (i - 1.5) * w, vals, w, label=name.replace("\n", " "), color=colors[i],
                  edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(x, [c for _, c in crit])
ax.set_ylim(0, 10.5)
ax.set_ylabel("Điểm LLM-judge (0–10)")
ax.legend(ncol=2, frameon=False, loc="upper left", fontsize=9)
ax.set_title("So sánh chất lượng câu trả lời (30 câu hỏi)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "chart_compare.png"), dpi=200)
plt.close(fig)

# 2. Tiến trình tổng điểm qua các run pldc_v2
runs = []
for p in sorted(glob.glob(os.path.join(ROOT, "data/test/pldc_v2/summary_*.json"))):
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("topic") == "pldc_v2" and d.get("total_score", 0) > 5:
        runs.append((d["timestamp"], d["total_score"]))
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(range(len(runs)), [r[1] for r in runs], marker="o", color="#2E4A7D")
ax.axhline(7.91, ls="--", color="#E0A458", lw=1)
ax.text(0.1, 7.95, "CRAG trên cây heading v1 (7.91)", fontsize=8, color="#B07A2A")
ax.set_xticks(range(len(runs)), [r[0][4:8] + "\n" + r[0][9:13] for r in runs], fontsize=7)
ax.set_ylabel("Tổng điểm")
ax.set_xlabel("Run (MMDD / HHMM)")
ax.set_title("Tiến trình cải thiện điểm tổng qua các lần chạy (pldc_v2)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "chart_progress.png"), dpi=200)
plt.close(fig)

# 3. Chi phí/độ trễ: baseline CRAG legacy vs agent (số trong README / BASELINE_NOTES)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.6))
for ax, (title, vals, unit) in zip(axs, [
    ("Độ trễ / câu (s)", [38.6, 63.0, 19.1], ""),
    ("Số lần gọi LLM / câu", [65.8, None, None], ""),
    ("Token / câu", [44802, None, None], ""),
]):
    pass
labels = ["CRAG legacy", "Agent\n(generate local)", "Agent\n(generate Gemini)"]
axs[0].bar(labels, [38.6, 63.0, 19.1], color=["#E0A458", "#2E4A7D", "#5B8DB8"])
for i, v in enumerate([38.6, 63.0, 19.1]):
    axs[0].text(i, v + 1, f"{v}", ha="center", fontsize=9)
axs[0].set_title("Độ trễ trung bình / câu (s)", fontsize=10)
axs[1].bar(["CRAG legacy"], [65.8], color="#E0A458")
axs[1].text(0, 67, "65.8", ha="center", fontsize=9)
axs[1].set_title("Số lần gọi LLM / câu\n(93% là grader)", fontsize=10)
axs[2].bar(["CRAG legacy"], [44802], color="#E0A458")
axs[2].text(0, 45500, "44,802", ha="center", fontsize=9)
axs[2].set_title("Token / câu", fontsize=10)
for ax in axs[:1]:
    ax.tick_params(axis="x", labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "chart_cost.png"), dpi=200)
plt.close(fig)
print("naive total", NAIVE["total_score"], "runs", len(runs))
