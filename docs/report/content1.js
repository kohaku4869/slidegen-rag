const L = require("./lib");
const { p, h1, h2, bullet, num, code, fig, table } = L;
const C = [];

// ---------- 1. Giới thiệu ----------
C.push(h1("CHƯƠNG 1. GIỚI THIỆU"));
C.push(h2("1.1. Bối cảnh và bài toán"));
C.push(p("Các mô hình ngôn ngữ lớn (LLM) trả lời tốt các câu hỏi tổng quát nhưng không biết nội dung riêng của một tài liệu cụ thể và dễ bịa thông tin. Retrieval-Augmented Generation (RAG) khắc phục điều này bằng cách tìm các đoạn liên quan trong tài liệu rồi đưa vào ngữ cảnh của LLM. Với các tài liệu dài và có cấu trúc như giáo trình, luật hay báo cáo kỹ thuật, cách làm phổ biến (naive RAG: cắt văn bản thành các đoạn có kích thước cố định, nhúng vector, lấy top-k đoạn gần nhất) bộc lộ nhiều hạn chế."));
C.push(bullet("Cắt cố định phá vỡ cấu trúc chương/mục, một khái niệm bị chia vào nhiều đoạn rời nhau."));
C.push(bullet("Câu hỏi tổng hợp hoặc so sánh nhiều khái niệm cần thông tin ở nhiều vị trí khác nhau, một lần tìm top-k không đủ."));
C.push(bullet("Không có cơ chế tự kiểm tra: nếu đoạn tìm được sai hoặc thiếu thì LLM vẫn phải trả lời."));
C.push(bullet("Không có trí nhớ hội thoại nên câu hỏi nối tiếp như “còn trường hợp thứ hai thì sao?” không thể tìm kiếm được."));
C.push(p("Đề tài **HRAG (Hierarchical Agentic RAG)** xây dựng một hệ thống hỏi đáp trên tài liệu dài theo hướng khác: lập chỉ mục dạng cây phân cấp theo heading, dùng một agent LangGraph tự quyết định công cụ tìm kiếm, tự chấm điểm và tự đánh giá độ đủ của ngữ cảnh, đồng thời có backend, cơ sở dữ liệu, bộ nhớ hội thoại và giao diện web hoàn chỉnh."));
C.push(h2("1.2. Mục tiêu"));
C.push(bullet("Xây dựng chỉ mục cây phân cấp giữ nguyên cấu trúc tài liệu và cho phép truy ngược về dòng nguồn."));
C.push(bullet("Xây dựng agent truy hồi có sửa lỗi (Corrective RAG) với phân rã câu hỏi thành nhiều khía cạnh chạy song song."));
C.push(bullet("Xây dựng bộ đánh giá tự động bằng LLM-as-judge và so sánh với các phương pháp nền (naive RAG, RAPTOR)."));
C.push(bullet("Xây dựng tầng services dùng chung, cơ sở dữ liệu PostgreSQL, bộ nhớ ngắn hạn cho hội thoại nhiều lượt và giao diện Streamlit."));
C.push(h2("1.3. Phạm vi và dữ liệu"));
C.push(p("Dữ liệu thử nghiệm là giáo trình “Pháp luật đại cương” (khoảng 534 nghìn ký tự, 5.491 dòng). Bộ đánh giá gồm 30 câu hỏi tiếng Việt kèm đáp án tham chiếu. Hệ thống tập trung vào hỏi đáp có trích dẫn trên tài liệu dài."));
C.push(h2("1.4. Phân công công việc"));
C.push(...table("Phân công công việc của nhóm", ["Thành viên", "Phụ trách"], [
  ["Đinh Tuấn Quang", "Agent, truy hồi thông tin và đánh giá hệ thống"],
  ["Nguyễn Duy Long", "Backend, cơ sở dữ liệu và bộ nhớ hội thoại"],
  ["Lê Anh Tú", "Giao diện người dùng Streamlit"],
], [3000, 6360]));
C.push(h2("1.5. Khai báo sử dụng công cụ AI"));
C.push(p("Trong quá trình thực hiện bài tập, nhóm có sử dụng công cụ AI là trợ lý lập trình Claude Code (Anthropic). Công cụ này được dùng cho các nhiệm vụ sau:"));
C.push(bullet("Khảo sát mã nguồn của dự án để tổng hợp kiến trúc, luồng xử lý và danh sách thư viện phục vụ viết báo cáo."));
C.push(bullet("Vẽ các sơ đồ minh hoạ (kiến trúc, đồ thị agent, quy trình lập chỉ mục, sơ đồ quan hệ cơ sở dữ liệu) và biểu đồ so sánh kết quả."));
C.push(bullet("Chạy ứng dụng và chụp ảnh màn hình giao diện để đưa vào báo cáo."));
C.push(bullet("Soạn thảo và định dạng văn bản báo cáo, xuất file Word và PDF."));
C.push(p("Các nhiệm vụ trên đều do thành viên nhóm giao và kiểm tra lại. Số liệu đánh giá trong chương 6 là kết quả các lần chạy thực nghiệm của nhóm, AI chỉ tổng hợp và trình bày. Ngoài ra, các mô hình ngôn ngữ như Gemini hay DeepSeek xuất hiện trong báo cáo là thành phần của hệ thống được xây dựng, không thuộc phạm vi công cụ hỗ trợ khai báo ở mục này."));

// ---------- 2. Cơ sở lý thuyết ----------
C.push(h1("CHƯƠNG 2. CƠ SỞ LÝ THUYẾT"));
C.push(h2("2.1. Naive RAG"));
C.push(p("Naive RAG gồm hai giai đoạn. Giai đoạn lập chỉ mục cắt tài liệu thành các đoạn (chunk) có độ dài cố định, tính vector nhúng cho từng đoạn bằng mô hình embedding và lưu vào kho vector. Giai đoạn truy vấn nhúng câu hỏi, lấy k đoạn có độ tương đồng cosine cao nhất rồi đưa chúng cùng câu hỏi vào LLM để sinh câu trả lời. Ưu điểm là đơn giản, nhanh; nhược điểm là mất cấu trúc, chỉ tìm một lần và không kiểm tra chất lượng kết quả."));
C.push(h2("2.2. RAPTOR"));
C.push(p("RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) xây cây tóm tắt: các đoạn 100 token được gom cụm bằng Gaussian Mixture Model sau khi giảm chiều bằng UMAP, mỗi cụm được LLM tóm tắt thành nút cha, lặp lại cho đến khi đủ số tầng. Truy vấn tìm trên toàn bộ các tầng (collapsed tree). Cách này giúp trả lời câu hỏi cần cái nhìn tổng thể, nhưng cụm là cụm theo ngữ nghĩa, không theo cấu trúc tài liệu, nên nút không gắn được với vị trí dòng trong văn bản gốc."));
C.push(h2("2.3. Corrective RAG (CRAG)"));
C.push(p("CRAG bổ sung bước đánh giá sau truy hồi: một bộ chấm điểm (grader) loại các đoạn không liên quan, và nếu ngữ cảnh chưa đủ thì hệ thống truy hồi lại với truy vấn khác. Trong dự án, ý tưởng này được mở rộng thành agent có vòng lặp và nhiều công cụ."));
C.push(h2("2.4. Agentic RAG và LangGraph"));
C.push(p("Agentic RAG giao cho LLM quyền chọn công cụ và tham số tìm kiếm ở mỗi vòng. Dự án dùng LangGraph, thư viện mô hình hoá luồng agent thành đồ thị trạng thái (StateGraph). Cơ chế `Send` cho phép rẽ nhánh động và chạy song song nhiều đồ thị con, được dùng để xử lý các khía cạnh của câu hỏi cùng lúc."));
C.push(h2("2.5. LLM-as-judge"));
C.push(p("Chất lượng câu trả lời được chấm bằng một LLM so với đáp án tham chiếu theo bốn tiêu chí độc lập (Accuracy, Completeness, Relevance, Faithfulness) trên thang 0–10. Mỗi tiêu chí dùng một lời nhắc riêng, chạy song song để tránh hiệu ứng “halo” (điểm tiêu chí này kéo theo tiêu chí khác)."));

// ---------- 3. Kiến trúc ----------
C.push(h1("CHƯƠNG 3. KIẾN TRÚC TỔNG THỂ"));
C.push(h2("3.1. Các thành phần"));
C.push(p("Hệ thống chia thành bốn lớp. Lớp giao diện gồm Streamlit và dashboard Gradio. Lớp backend là tầng dịch vụ dùng chung (hỏi đáp, lập chỉ mục, hàng đợi job, bộ nhớ đệm cây). Lớp lõi RAG gồm agent LangGraph, chỉ mục cây và ConversationManager. Lớp lưu trữ gồm PostgreSQL (danh mục, phiên chat, tin nhắn) và các file trên đĩa (Markdown nguồn, chỉ mục dạng pickle)."));
C.push(...fig("diag_arch.png", "Kiến trúc tổng thể của hệ thống", 600));
C.push(p("Điểm thiết kế đáng chú ý là tầng dịch vụ không phụ thuộc Streamlit nên giao diện, dòng lệnh và dashboard cùng gọi chung một mã nguồn nghiệp vụ. Streamlit gọi trực tiếp services trong cùng tiến trình, không đi qua HTTP. Repo có kèm một lớp REST API (FastAPI) mỏng bọc các services này nhưng giao diện không dùng đến nên báo cáo không đi sâu."));
C.push(h2("3.2. Cấu trúc mã nguồn"));
C.push(p("Cấu trúc thư mục chính của dự án:", { noIndent: true, para: { keepNext: true } }));
C.push(...code(`slidegen-rag/
├── src/slidegen_rag/
│   ├── agent/        # đồ thị agent, node, công cụ tìm kiếm
│   ├── tree/         # xây chỉ mục cây và truy hồi
│   ├── evaluate/     # chạy đánh giá, judge, dashboard Gradio
│   ├── services/     # tầng backend dùng chung
│   ├── memory/       # mô hình dữ liệu, repository, bộ nhớ hội thoại
│   └── config.py     # cấu hình
├── ui/               # giao diện Streamlit
├── migrations/       # Alembic, cấu trúc cơ sở dữ liệu
├── tests/            # kiểm thử
├── data/             # tài liệu nguồn, chỉ mục, kết quả đánh giá
├── docs/             # tài liệu, báo cáo
└── pyproject.toml    # khai báo phụ thuộc`));
C.push(...table("Các phần chính của dự án", ["Phần", "Nội dung"], [
  ["Agent", "Đồ thị agent, các node, công cụ tìm kiếm, theo dõi quỹ đạo"],
  ["Cây chỉ mục", "Xây chỉ mục cây và truy hồi trên cây"],
  ["Đánh giá", "Chạy đánh giá, judge, tính chỉ số, so sánh các lần chạy, dashboard Gradio"],
  ["Dịch vụ (backend)", "Hỏi đáp, lập chỉ mục, hàng đợi job, bộ nhớ đệm, đọc tài liệu, trích dẫn"],
  ["Bộ nhớ và cơ sở dữ liệu", "Mô hình dữ liệu, repository, migration"],
  ["Giao diện", "Ứng dụng Streamlit"],
], [3300, 6060]));
C.push(h2("3.3. Luồng một lượt hỏi đáp"));
C.push(num("Người dùng chọn tài liệu trong phạm vi (scope) của phiên chat và gửi câu hỏi."));
C.push(num("Dịch vụ hỏi đáp nạp các cây chỉ mục của tài liệu trong scope và gộp thành một cây tìm kiếm (nếu hơn 3 tài liệu thì lọc trước còn 3)."));
C.push(num("ConversationManager viết lại câu hỏi nối tiếp thành câu hỏi độc lập dựa trên lịch sử."));
C.push(num("Agent phân rã câu hỏi, truy hồi có sửa lỗi theo từng khía cạnh rồi sinh câu trả lời."));
C.push(num("Trích dẫn được trích ra, lượt hỏi được ghi vào PostgreSQL và lịch sử được nén nếu cần."));

// ---------- 4. Indexing ----------
C.push(h1("CHƯƠNG 4. LẬP CHỈ MỤC: CÂY HEADING PHÂN CẤP"));
C.push(p("~Phần do Đinh Tuấn Quang phụ trách.~", { noIndent: true }));
C.push(h2("4.1. Ý tưởng"));
C.push(p("Thay vì cắt văn bản theo độ dài hay gom cụm, dự án dùng chính cấu trúc heading của tài liệu Markdown làm cấu trúc cây. Nút lá là nguyên văn các mục (không cắt, không viết lại, chỉ được nhúng vector). Nút cha là bản tóm tắt do LLM sinh từ các nút con, tính từ dưới lên theo từng tầng. Mỗi nút giữ lại đường dẫn heading và số dòng đầu và dòng cuối trong tài liệu nguồn, nhờ đó có thể trích dẫn chính xác và các công cụ “đi xuống / đi sang bên” trong cây hoạt động được."));
C.push(...fig("diag_tree.png", "Cấu trúc cây chỉ mục của giáo trình (320 nút, 5 tầng)", 420));
C.push(h2("4.2. Quy trình xây cây"));
C.push(...fig("diag_ingest.png", "Quy trình nạp và lập chỉ mục tài liệu", 600));
C.push(bullet("**Phân tích heading**: độ sâu lấy theo đánh số (ví dụ 1.1.2.1), từ khoá chương ép về cấp 1, loại bỏ nhiễu dấu đầu dòng do OCR; nếu có nhiều gốc thì thêm một siêu gốc."));
C.push(bullet("**Nhúng vector**: mô hình BAAI/bge-m3 (1024 chiều). Ưu tiên gọi dịch vụ embedding từ xa, nếu lỗi thì dùng sentence-transformers cục bộ."));
C.push(bullet("**Tóm tắt từ dưới lên**: gom theo từng tầng và chạy theo lô; độ dài tóm tắt tối đa 2000 token. Mô hình tóm tắt mặc định là deepseek-v4-flash, dự phòng gemini-2.5-flash."));
C.push(bullet("**Lưu trữ**: toàn bộ cây (văn bản, vector, quan hệ cha–con) được lưu thành file pickle; không dùng cơ sở dữ liệu vector, tìm kiếm là cosine brute-force trên ma trận đã chuẩn hoá."));
C.push(h2("4.3. Chi phí lập chỉ mục"));
C.push(...table("Chi phí xây chỉ mục cho giáo trình Pháp luật đại cương", ["Chỉ số", "Giá trị"], [
  ["Kích thước tài liệu", "534 nghìn ký tự, 5.491 dòng"],
  ["Số nút / số tầng", "320 nút / 5 tầng (216 lá, 61, 26, 16, 1)"],
  ["Thời gian phân tích heading", "7 ms"],
  ["Số lần gọi LLM tóm tắt", "103"],
  ["Số vector nhúng", "320"],
  ["Tổng thời gian", "khoảng 5 phút trở lên"],
], [4300, 5060]));
C.push(h2("4.4. So sánh với RAPTOR"));
C.push(p("Dự án ban đầu dùng RAPTOR (chunk 100 token, GMM + UMAP, tối đa 5 tầng) nhưng đã thay bằng cây heading vì hàm `split_text` của RAPTOR làm mất vị trí dòng, khiến các công cụ truy hồi theo cấu trúc không thể hoạt động. Kết quả đánh giá ở Chương 6 cũng cho thấy cây heading cho điểm cao hơn rõ rệt: 7,89 so với 6,93 khi dùng cùng quy trình CRAG."));
C.push(...table("So sánh các kiểu chỉ mục", ["Tiêu chí", "Naive RAG", "RAPTOR", "Cây heading (dự án)"], [
  ["Đơn vị lưu", "Đoạn cố định", "Đoạn 100 token + tóm tắt theo cụm", "Mục nguyên văn + tóm tắt theo cấu trúc"],
  ["Cấu trúc", "Phẳng", "Cây theo cụm ngữ nghĩa", "Cây theo heading"],
  ["Truy ngược dòng nguồn", "Hạn chế", "Không", "Có (số dòng đầu/cuối)"],
  ["Đi lên/xuống/sang bên", "Không", "Không", "Có (expand_node, find_in_subtree)"],
  ["Chi phí lập chỉ mục", "Thấp", "Cao (41 MB pickle)", "Trung bình (4,5 MB pickle)"],
], [2000, 2000, 2600, 2760]));

// ---------- 5. Agent ----------
C.push(h1("CHƯƠNG 5. AGENT TRUY HỒI"));
C.push(p("~Phần do Đinh Tuấn Quang phụ trách.~", { noIndent: true }));
C.push(h2("5.1. Đồ thị cha: phân rã và gộp khía cạnh"));
C.push(p("Đồ thị cha có bốn bước: `decompose_facets` → fan-out `Send` → `facet_subgraph` (chạy song song N bản) → `collect_facets` → `generate`. Node phân rã chia câu hỏi thành tối đa 4 khía cạnh. Nếu câu hỏi chứa dấu hiệu so sánh (“so sánh”, “phân biệt”…) mà chỉ ra một khía cạnh thì hệ thống hỏi LLM thêm một lần để tách; khi lỗi thì lùi về một khía cạnh duy nhất."));
C.push(...fig("diag_parent.png", "Đồ thị cha của agent", 380));
C.push(h2("5.2. Đồ thị con: vòng truy hồi có sửa lỗi"));
C.push(p("Mỗi khía cạnh chạy một vòng lặp riêng với ngân sách riêng: `retrieve` (LLM chọn công cụ và tham số) → `tool` (thực thi song song) → `grade_documents` (lọc theo mức liên quan) → `context_evaluator` (đủ chưa?). Vòng lặp kết thúc khi ngữ cảnh đủ, khi hết `MAX_RETRIEVE = 5` vòng, hoặc khi từ vòng 3 trở đi hai vòng liên tiếp không thu thêm đoạn mới nào (no-progress guard)."));
C.push(...fig("diag_facet.png", "Vòng truy hồi có sửa lỗi của mỗi khía cạnh", 600));
C.push(...table("Vai trò của các node", ["Node", "Quyết định", "Ghi vào state"], [
  ["`decompose_facets`", "Câu hỏi có nhiều khía cạnh không, mỗi câu hỏi con là gì", "facet_queries"],
  ["`retrieve`", "Gọi công cụ nào, tham số gì; loại cặp (tool, query) đã thử", "tool_calls"],
  ["`tool`", "Thực thi song song, tách thành đoạn, khử trùng giữa các vòng", "pending_documents, search_history, node_digest"],
  ["`grade_documents`", "Đoạn nào giữ lại; đoạn nào đúng chủ đề nhưng chưa đủ (cần drill-down)", "documents, document_priorities, drill_candidates"],
  ["`context_evaluator`", "Ngữ cảnh đã đủ chưa, nếu chưa thì cần tìm gì", "context_sufficient, missing_aspects, feedback"],
  ["`collect_facets`", "Gộp và khử trùng tài liệu giữa các khía cạnh", "documents, facet_sufficiency"],
  ["`generate`", "Sinh câu trả lời trong giới hạn 32.000 ký tự ngữ cảnh", "generation"],
], [2300, 4300, 2760]));
C.push(h2("5.3. Các công cụ tìm kiếm"));
C.push(...table("Công cụ của agent", ["Công cụ", "Chức năng"], [
  ["`semantic_search`", "Cosine trên mọi tầng của cây (collapsed tree), top_k = 3, tối đa 4000 token"],
  ["`keyword_search`", "BM25 (rank_bm25) trên mọi nút"],
  ["`expand_node`", "Đi tới nút con, nút cha hoặc nút anh em của một nút"],
  ["`find_in_subtree`", "Tìm BM25 hoặc ngữ nghĩa trong phạm vi một nhánh"],
  ["`web_search`", "Tìm web bằng Tavily (chỉ bật khi có khoá API; tắt khi đánh giá)"],
], [2800, 6560]));
C.push(p("LLM của node `retrieve` được ép gọi ít nhất một công cụ mỗi vòng (`tool_choice=\"any\"`). Mỗi bước giới hạn 6 lệnh gọi công cụ, sau khi từng ghi nhận một lần chạy sinh tới 517 lệnh gọi trong một bước."));
C.push(h2("5.4. Chấm điểm đoạn và đánh giá ngữ cảnh"));
C.push(bullet("**Grader** chạy bất đồng bộ, 16 lệnh gọi đồng thời, ba nhãn: `relevant` (ưu tiên 2), `relevant_incomplete` (1), `irrelevant` (0). Nếu tất cả bị loại, hệ thống vẫn giữ 5 đoạn đầu (hard floor)."));
C.push(bullet("Các đoạn `relevant_incomplete` trở thành **ứng viên drill-down** (cần chi tiết, cần anh em, cần ngữ cảnh), tối đa 10, giúp vòng sau đi xuống đúng nhánh."));
C.push(bullet("**Context evaluator** trả về SUFFICIENT/MISSING/FEEDBACK; phản hồi này cùng lịch sử tìm kiếm là đầu vào để LLM viết truy vấn mới (không có node viết lại truy vấn riêng)."));
C.push(bullet("**Ngân sách ngữ cảnh**: 32.000 ký tự, lấp theo mức ưu tiên rồi đến độ mới."));
C.push(h2("5.5. Mô hình ngôn ngữ và giám sát"));
C.push(...table("Mô hình sử dụng", ["Vai trò", "Mô hình"], [
  ["Retrieve, decompose, context_evaluator, contextualize", "Gemini (gemini-3.5-flash-lite)"],
  ["Grader (max_tokens = 24)", "Mô hình tự host qua endpoint tương thích OpenAI"],
  ["Generate", "Gemini hoặc mô hình tự host (chuyển đổi được, tự lùi về Gemini khi lỗi)"],
  ["Tóm tắt khi lập chỉ mục", "deepseek-v4-flash, dự phòng gemini-2.5-flash"],
  ["Embedding", "BAAI/bge-m3"],
], [4700, 4660]));
C.push(p("Dự án ghi lại quỹ đạo từng node (thời gian, công cụ gọi, số vòng), đồng thời hỗ trợ LangSmith để theo dõi lời gọi LLM. `TokenTracker` tính số token và chi phí."));

// ---------- 6. Evaluation ----------
C.push(h1("CHƯƠNG 6. ĐÁNH GIÁ"));
C.push(p("~Phần do Đinh Tuấn Quang phụ trách.~", { noIndent: true }));
C.push(h2("6.1. Phương pháp"));
C.push(p("Bộ dữ liệu gồm 30 câu hỏi pháp luật tiếng Việt và đáp án tham chiếu. Mỗi câu trả lời được chấm bằng bốn lời nhắc judge độc lập, mỗi tiêu chí thang 0–10; điểm tổng là trung bình bốn tiêu chí. Judge chạy ở nhiệt độ 0, ưu tiên mô hình tự host, dự phòng Gemini."));
C.push(...table("Các chỉ số đánh giá", ["Chỉ số", "Ý nghĩa"], [
  ["Accuracy", "Đúng so với đáp án mẫu"],
  ["Completeness", "Bao quát các ý cốt lõi"],
  ["Relevance", "Bám sát câu hỏi, không lan man"],
  ["Faithfulness", "Chỉ dựa vào ngữ cảnh đã truy hồi, không bịa"],
  ["Chunk coverage", "% từ khoá của đáp án mẫu xuất hiện trong các đoạn truy hồi"],
  ["Answer coverage", "% từ khoá của đáp án mẫu xuất hiện trong câu trả lời"],
  ["Utilization", "Answer coverage chia cho chunk coverage (strict)"],
], [2600, 6760]));
C.push(p("Độ nhiễu của judge khoảng ±0,06 (chấm lại cùng một bộ câu trả lời bốn lần cho 8,03 / 8,03 / 8,05 / 8,09), vì vậy chênh lệch dưới khoảng 0,1 được coi là nhiễu."));
C.push(h2("6.2. Các hệ thống so sánh"));
C.push(...table("Các hệ thống được so sánh", ["Hệ thống", "Chỉ mục", "Truy hồi / sinh", "Nguồn số liệu"], [
  ["Naive RAG", "Đoạn cố định, nhúng bge-m3", "Top-k một lần rồi sinh", "Nhóm tự chạy trên cùng 30 câu hỏi và cùng judge"],
  ["RAPTOR + CRAG", "GMM + UMAP, đoạn 100 token", "CRAG 5 node (gemini-2.5-flash)", "Lần chạy 14/09/2026"],
  ["Cây heading + CRAG", "Cây heading v1", "CRAG 5 node", "Lần chạy 14/09/2026"],
  ["Agent multi-facet (dự án)", "Cây heading v2", "Phân rã + vòng truy hồi song song", "Lần chạy 15/09/2026"],
], [2200, 2400, 2500, 2260]));
C.push(h2("6.3. Kết quả"));
C.push(...table("Điểm trung bình trên 30 câu hỏi", ["Hệ thống", "Acc", "Comp", "Rel", "Faith", "Tổng"], [
  ["Naive RAG", "5,8", "4,9", "6,5", "7,9", "6,3"],
  ["RAPTOR + CRAG", "6,38", "5,47", "7,08", "8,80", "6,93"],
  ["Cây heading + CRAG", "7,27", "6,92", "8,10", "9,27", "7,89"],
  ["**Agent multi-facet**", "**8,22**", "**8,38**", "**8,42**", "**9,15**", "**8,54**"],
], [3260, 1220, 1220, 1220, 1220, 1200]));
C.push(p("Số liệu Naive RAG do nhóm tự chạy trên cùng bộ 30 câu hỏi và cùng judge; các dòng còn lại lấy từ kết quả các lần chạy đã lưu của dự án.", { noIndent: true }));
C.push(...fig("chart_compare.png", "So sánh điểm LLM-judge giữa các hệ thống", 580));
C.push(p("Cây heading kết hợp CRAG đã tăng khoảng 0,96 điểm so với RAPTOR, và agent multi-facet tiếp tục tăng thêm khoảng 0,65 điểm, lớn hơn nhiều mức nhiễu ±0,06. Cải thiện lớn nhất nằm ở Completeness (5,47 → 8,38): việc phân rã khía cạnh và drill-down giúp thu đủ các ý của câu hỏi tổng hợp. Faithfulness ở mọi hệ thống đều cao (8,8–9,3) vì câu trả lời được sinh từ ngữ cảnh truy hồi."));
C.push(...table("Chỉ số chẩn đoán của lần chạy agent multi-facet", ["Chỉ số", "Giá trị"], [
  ["Chunk coverage", "81,1% (strict 79,0%)"],
  ["Answer coverage", "73,5%"],
  ["Utilization", "94,0%"],
  ["Số đoạn truy hồi trung bình / câu", "10,0"],
  ["Câu hoàn thành", "30/30"],
  ["Chi phí và thời gian toàn bộ run", "khoảng 0,27 USD, 31,5 phút"],
], [5000, 4360]));
C.push(h2("6.4. Tiến trình cải thiện"));
C.push(p("Toàn bộ các lần chạy đều được giữ lại trong thư mục kết quả, kể cả những lần điểm giảm. Điểm tổng tăng từ khoảng 7,7 (CRAG vòng lặp đơn) lên 8,06 khi thêm phân rã đa khía cạnh, và đạt đỉnh 8,56 khi sinh câu trả lời bằng Gemini."));
C.push(...fig("chart_progress.png", "Điểm tổng qua các lần chạy", 580));
C.push(p("Phân tích theo số khía cạnh cho thấy các câu 2 khía cạnh có điểm trung bình 8,49 (n = 13), cao hơn câu 1 khía cạnh là 8,05 (n = 16)."));
C.push(h2("6.5. Chi phí và độ trễ"));
C.push(...table("Độ trễ theo nhà cung cấp sinh câu trả lời", ["", "Run 8,54 (sinh bằng mô hình tự host)", "Run 8,56 (sinh bằng Gemini)"], [
  ["Trung bình / câu", "63,0 s", "19,1 s"],
  ["Trung vị", "62,6 s", "20,1 s"],
  ["Nhanh nhất / chậm nhất", "22,5 s / 100,8 s", "5,6 s / 56,0 s"],
  ["Tổng 30 câu", "31,5 phút", "9,6 phút"],
], [2800, 3500, 3060]));
C.push(...fig("chart_cost.png", "Độ trễ và chi phí so với CRAG legacy", 580));
C.push(p("Vòng truy hồi có sửa lỗi rất rẻ: toàn bộ khoảng 200 vòng retrieve + tool + grade + evaluate chỉ tốn khoảng 14 giây cho mỗi câu. Thời gian chủ yếu nằm ở bước sinh câu trả lời (76% ở run dùng mô hình tự host). Đổi nhà cung cấp sinh làm độ trễ giảm 3 lần nhưng điểm chỉ lệch 0,02 (trong ngưỡng nhiễu), nghĩa là độ trễ và chất lượng là hai núm điều chỉnh độc lập. Trung bình agent dùng 3,9/5 vòng, 20/30 câu dùng hết ngân sách vòng."));
C.push(p("Để so sánh, baseline CRAG legacy dùng trung bình 44.802 token và 65,8 lệnh gọi LLM mỗi câu (93% là lệnh gọi của grader), và 18/30 câu (60%) hết số vòng truy hồi mà vẫn chưa đủ ngữ cảnh."));
module.exports = C;
