"""
Gộp và khử trùng lặp document giữa các vòng retrieve.

Tách khỏi tool_node.py vì grade_documents_node và collect_facets_node đều cần —
trước đây hai node đó import HÀM PRIVATE `_merge_documents_with_ids` xuyên qua
tool_node, khiến người đọc tưởng chúng phụ thuộc vào node thực thi tool trong
khi thực chất chỉ dùng chung một tiện ích thuần tuý.
"""


def merge_documents_with_ids(
    existing_docs: list[str],
    existing_ids: list,
    new_docs: list[str],
    new_ids: list,
) -> tuple[list[str], list]:
    """Gộp danh sách document và giữ `node_id` đi kèm từng document (1:1),
    để grade_documents_node duy trì `document_node_ids` song song với
    `documents` qua các vòng — dùng bởi ConversationManager (mục 7 trong
    plan) để biết node nào thực sự được trích dẫn ở lượt hỏi này.

    Hoàn toàn generic trên tham số "id" (chỉ zip 1:1 và dedup theo doc) —
    grade_documents_node và collect_facets_node gọi hàm này 2 LẦN với CÙNG
    `existing_docs`/`new_docs` (1 lần cho document_node_ids, 1 lần cho
    document_priorities) để giữ 2 danh sách metadata song song luôn khớp
    index mà không cần viết lại logic merge/dedup riêng."""
    seen = set(existing_docs)
    merged_docs = list(existing_docs)
    merged_ids = list(existing_ids)
    for doc, node_id in zip(new_docs, new_ids):
        if doc not in seen:
            seen.add(doc)
            merged_docs.append(doc)
            merged_ids.append(node_id)
    return merged_docs, merged_ids
