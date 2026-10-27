const fs = require("fs");
const L = require("./lib");
const { p, h1, h2, bullet, code, fig, table } = L;
const C = [];

// ---------- 7. Backend ----------
C.push(h1("CHƯƠNG 7. BACKEND"));
C.push(p("~Phần do Nguyễn Duy Long phụ trách.~", { noIndent: true }));
C.push(p("Backend là một tầng dịch vụ (services) viết thuần Python, tách khỏi giao diện để Streamlit, dòng lệnh và dashboard đánh giá cùng dùng chung. Tầng này gồm các thành phần sau:"));
C.push(bullet("**Dịch vụ hỏi đáp**: thực hiện trọn một lượt chat, từ nạp lịch sử, viết lại câu hỏi, chạy agent đến lưu câu trả lời và trích dẫn."));
C.push(bullet("**Dịch vụ lập chỉ mục và hàng đợi job**: chạy việc lập chỉ mục ở luồng nền bằng `ThreadPoolExecutor`, có theo dõi tiến độ và cho phép huỷ."));
C.push(bullet("**Bộ nhớ đệm cây và lọc tài liệu**: giữ cây chỉ mục đã nạp để dùng lại; khi có nhiều tài liệu thì chọn trước vài tài liệu liên quan nhất."));
C.push(bullet("**Đọc tài liệu**: chuyển các định dạng md, txt, docx và pdf về Markdown trước khi lập chỉ mục."));
C.push(bullet("**Xử lý lỗi**: dùng các ngoại lệ riêng của dự án để giao diện báo lỗi rõ ràng, ví dụ chưa có tài liệu sẵn sàng hay mất kết nối cơ sở dữ liệu."));
C.push(p("Hạn chế hiện tại: chưa có đăng nhập nhiều người dùng, câu trả lời chưa được truyền theo dòng (streaming) và hàng đợi job chỉ nằm trong bộ nhớ."));

// ---------- 8. Database ----------
C.push(h1("CHƯƠNG 8. CƠ SỞ DỮ LIỆU"));
C.push(p("~Phần do Nguyễn Duy Long phụ trách.~", { noIndent: true }));
C.push(p("Dữ liệu quan hệ lưu trong PostgreSQL 18, truy cập bất đồng bộ bằng SQLAlchemy 2 và asyncpg; cấu trúc bảng được quản lý bằng Alembic. Có 9 bảng, quan hệ giữa chúng được mô tả ở Hình 9."));
C.push(...fig("diag_erd.png", "Sơ đồ quan hệ thực thể (ERD)", 600, 780));
C.push(...table("Các nhóm bảng chính", ["Nhóm", "Nội dung"], [
  ["Người dùng và project", "Chủ sở hữu và không gian làm việc, có xoá mềm"],
  ["Tài liệu và chỉ mục", "Thông tin tài liệu và từng lần lập chỉ mục"],
  ["Phiên chat", "Phiên hội thoại và phạm vi tài liệu được dùng"],
  ["Bộ nhớ và tin nhắn", "Tin nhắn, trích dẫn và tóm tắt lịch sử"],
], [3000, 6360]));
C.push(p("File Markdown nguồn và chỉ mục cây (dạng pickle) được lưu trên đĩa theo từng project; vector không dùng cơ sở dữ liệu vector riêng. Truy cập dữ liệu đi qua các interface (`typing.Protocol`) có hai bản cài đặt: PostgreSQL và trong bộ nhớ, nhờ đó chạy được dòng lệnh khi không có cơ sở dữ liệu."));

// ---------- 9. Memory ----------
C.push(h1("CHƯƠNG 9. BỘ NHỚ HỘI THOẠI"));
C.push(p("~Phần do Nguyễn Duy Long phụ trách.~", { noIndent: true }));
C.push(p("Bộ nhớ ngắn hạn do lớp `ConversationManager` đảm nhiệm, lịch sử được lưu trong PostgreSQL và nạp lại trước mỗi lượt. Các cơ chế chính:"));
C.push(...fig("diag_memory.png", "Luồng bộ nhớ trong một lượt hỏi đáp", 600));
C.push(bullet("**Cửa sổ trượt**: giữ 5 lượt hỏi đáp gần nhất."));
C.push(bullet("**Nén lịch sử**: khi hội thoại dài, các tin cũ được LLM tóm tắt lại để không phình ngữ cảnh."));
C.push(bullet("**Viết lại câu hỏi**: câu hỏi nối tiếp (“Vậy còn…?”) được viết lại thành câu hỏi đầy đủ trước khi đưa vào agent."));
C.push(bullet("**Gợi ý truy hồi**: các đoạn đã trích dẫn ở câu trước giúp câu sau tìm đúng vùng tài liệu nhanh hơn."));
C.push(bullet("**Chế độ dự phòng**: nếu cơ sở dữ liệu lỗi, hội thoại tiếp tục trong bộ nhớ thay vì dừng hẳn."));

// ---------- 10. Frontend ----------
C.push(h1("CHƯƠNG 10. GIAO DIỆN STREAMLIT"));
C.push(p("~Phần do Lê Anh Tú phụ trách.~", { noIndent: true }));
C.push(h2("10.1. Các chức năng chính"));
C.push(p("Giao diện xây bằng Streamlit, gọi trực tiếp tầng services trong cùng tiến trình. Các chức năng chính gồm quản lý project, upload tài liệu và theo dõi tiến độ lập chỉ mục, xem cây chỉ mục, chat có trích dẫn và trang cài đặt. Tiến trình của agent được hiển thị theo từng bước ngay trong lúc chạy."));
C.push(...fig("ui_projects.png", "Trang danh sách project", 560));
C.push(...fig("ui_ingest_0.png", "Tài liệu đang được lập chỉ mục", 560));
C.push(...fig("ui_document_tree.png", "Xem cây chỉ mục của tài liệu", 560));
C.push(...fig("ui_chat_answer.png", "Câu trả lời kèm các chip trích dẫn", 560));
C.push(...fig("ui_chat_citation.png", "Popover của một trích dẫn", 560));
C.push(...fig("ui_chat_details.png", "Chi tiết lượt chạy: thời gian, số vòng, token, chi phí và quỹ đạo", 560));
C.push(h2("10.2. Cách giao diện tương tác với các phần khác"));
C.push(p("Giao diện không chạy logic nghiệp vụ mà chỉ gọi sang các phần khác của dự án. Hình 16 mô tả một lượt hỏi đáp và bảng bên dưới tóm tắt các điểm giao tiếp."));
C.push(...fig("diag_ui.png", "Giao diện tương tác với tầng services, agent và cơ sở dữ liệu trong một lượt chat", 560));
C.push(...table("Các điểm giao tiếp của giao diện", ["Thành phần", "Cách giao diện tương tác"], [
  ["Tầng services (backend)", "Gọi trực tiếp các hàm trong cùng tiến trình, không đi qua HTTP"],
  ["Agent", "Chạy ở luồng nền vì agent là bất đồng bộ còn Streamlit chạy đồng bộ; mỗi bước hoàn thành được đẩy vào hàng đợi để giao diện hiển thị dần"],
  ["Lập chỉ mục", "Giao diện tạo job nền, một đoạn giao diện tự làm mới mỗi giây để cập nhật thanh tiến độ và nút huỷ"],
  ["PostgreSQL và bộ nhớ", "Danh sách project, tài liệu, phiên chat và lịch sử được đọc, ghi qua tầng services; mở lại phiên là nạp lại hội thoại"],
  ["Cây chỉ mục", "Trích dẫn trong câu trả lời liên kết tới nút trong cây và dòng trong tài liệu nguồn"],
  ["Trạng thái trang", "Mã project, tài liệu và phiên được lưu trên địa chỉ trang nên tải lại trang không mất vị trí"],
], [2800, 6560]));
C.push(p("Ngoài Streamlit, dự án có dashboard Gradio để chạy và so sánh các lần đánh giá chất lượng (chương 6)."));

// ---------- 11. Thư viện ----------
C.push(h1("CHƯƠNG 11. CÔNG NGHỆ VÀ THƯ VIỆN"));
C.push(p("Dự án viết bằng Python 3.12, quản lý môi trường và phụ thuộc bằng `uv`. Bảng dưới liệt kê các thư viện chính cùng phiên bản đang dùng."));
C.push(...table("Thư viện chính", ["Nhóm", "Thư viện (phiên bản)", "Vai trò"], [
  ["Cơ sở dữ liệu", "sqlalchemy 2.0.52, asyncpg 0.31.0, alembic 1.20.0", "ORM bất đồng bộ, driver, migration"],
  ["Agent", "langgraph 1.2.10, langchain 1.3.14, langchain-google-genai 4.3.4, langchain-openai 1.4.3, langsmith 0.10.12", "Đồ thị agent, gọi LLM, theo dõi"],
  ["Truy hồi", "sentence-transformers 5.6.1, torch 2.13.0, rank-bm25 0.2.2, tiktoken, numpy, scipy, tavily", "Embedding bge-m3, BM25, cosine, web search"],
  ["Giao diện", "streamlit 1.64.0, gradio 6.21.0, matplotlib", "UI chính, dashboard đánh giá, biểu đồ"],
  ["Tài liệu", "pypdf 6.21.2, python-docx 1.2.0", "Đọc pdf, docx"],
  ["Kiểm thử", "pytest 9.1.1, pytest-asyncio 1.4.0, httpx 0.28.1, ruff 0.16.6", "Test, kiểm tra mã, định dạng"],
], [1700, 4600, 3060]));

// ---------- 13. Kết luận ----------
C.push(h1("CHƯƠNG 12. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN"));
C.push(h2("12.1. Kết quả đạt được"));
C.push(bullet("Chỉ mục cây heading giữ cấu trúc tài liệu và cho phép trích dẫn đến từng dòng nguồn."));
C.push(bullet("Agent đa khía cạnh với vòng truy hồi có sửa lỗi đạt 8,54/10 trên 30 câu hỏi, cao hơn CRAG trên cây heading (7,89) và CRAG trên RAPTOR (6,93)."));
C.push(bullet("Hệ thống hoàn chỉnh: tầng services, PostgreSQL với migration, bộ nhớ hội thoại có nén lịch sử, giao diện Streamlit và dashboard đánh giá."));
C.push(h2("12.2. Hạn chế"));
C.push(bullet("Thử nghiệm mới trên một tài liệu tiếng Việt và 30 câu hỏi."));
C.push(bullet("Chat chưa streaming; chưa xác thực; job lập chỉ mục chỉ nằm trong bộ nhớ."));
C.push(bullet("Chưa có bộ nhớ dài hạn."));
C.push(h2("12.3. Hướng phát triển"));
C.push(bullet("Streaming câu trả lời qua SSE; hàng đợi job bền vững; xác thực nhiều người dùng."));
C.push(bullet("Dùng pgvector để lưu vector thay vì pickle; bộ nhớ dài hạn theo người dùng."));
C.push(bullet("Mở rộng sang sinh nội dung từ các câu trả lời có trích dẫn, ví dụ tạo slide tóm tắt."));
C.push(bullet("Mở rộng bộ đánh giá sang nhiều tài liệu và nhiều lĩnh vực."));

module.exports = C;
