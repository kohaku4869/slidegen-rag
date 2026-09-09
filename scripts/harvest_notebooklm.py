import asyncio
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Dict

try:
    from playwright_stealth import Stealth
    HAS_STEALTH = True
except ImportError:
    HAS_STEALTH = False

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_QUESTION_FILE = ROOT_DIR / "data" / "test" / "pldc" / "question.txt"
DEFAULT_REFERENCE_FILE = ROOT_DIR / "data" / "test" / "pldc" / "reference.txt"
DEFAULT_NOTEBOOK_URL = "https://notebooklm.google.com/notebook/dc53252a-4dc2-4fd7-a7f1-e56a7f60eb0d"
USER_DATA_DIR = ROOT_DIR / "benchmarks" / ".notebooklm_browser_data"


def load_questions(file_path: Path) -> List[Dict[str, str]]:
    """Đọc danh sách câu hỏi từ file question.txt."""
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file câu hỏi: {file_path}")

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
            q_num = match.group(1)
            questions.append({"num": int(q_num), "full_text": block, "clean_text": match.group(2).strip()})
        else:
            questions.append({"num": len(questions) + 1, "full_text": block, "clean_text": block})

    return questions


def is_valid_response(text: str) -> bool:
    """Kiểm tra xem câu trả lời có hợp lệ và đầy đủ không."""
    if not text:
        return False
    t = text.strip()
    if len(t) < 80:
        return False
    if re.search(r"^[\w\.-]+@[\w\.-]+\.\w+$", t):
        return False
    if re.search(r"^(?:đã tìm kiếm trong các nguồn|thoughts|expand_more|\s)+$", t, re.IGNORECASE):
        return False
    if "lỗi hiển thị kỹ thuật" in t.lower() or "ký tự đại diện" in t.lower():
        return False
    if "không thể trả lời câu hỏi này" in t.lower():
        return False
    return True


def get_answered_numbers(reference_file: Path) -> set:
    """Kiểm tra các câu hỏi đã được trả lời HỢP LỆ trong file reference.txt."""
    if not reference_file.exists():
        return set()

    with open(reference_file, "r", encoding="utf-8") as f:
        content = f.read()

    answered = set()
    blocks = re.split(r"(?:^|\n)(?=-\s*\d+\.)", content)
    for block in blocks:
        match = re.match(r"^-\s*(\d+)\.\s*(.+)$", block, re.DOTALL)
        if match:
            num = int(match.group(1))
            ans = match.group(2).strip()
            if is_valid_response(ans):
                answered.add(num)
    return answered


def format_clean_answer(text: str) -> str:
    """
    Xử lý câu trả lời:
    - Loại bỏ các dấu xuống dòng liên tiếp thừa (\n\n\n... -> \n\n)
    - Loại bỏ khoảng trắng thừa ở cuối dòng
    """
    if not text:
        return ""
    # Xóa khoảng trắng thừa ở cuối mỗi dòng
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Rút gọn 3 dấu xuống dòng trở lên thành tối đa 2 (\n\n)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_dom_text(text: str) -> str:
    """Loại bỏ các thành phần rác nếu fallback trích xuất DOM."""
    text = re.sub(
        r"^(?:Thoughts\s*\n\s*expand_more\s*\n|expand_more\s*\n|Đã tìm kiếm trong các nguồn của bạn\.\.\.\s*\n\s*expand_more\s*\n)",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"^Đã tìm kiếm trong các nguồn của bạn\.\.\.\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\d+\s*\n\s*\.\s*", " ", text)
    # Loại bỏ các nhãn button UI NotebookLM nếu dính vào
    text = re.sub(r"\n\s*(?:keep_pin|Lưu vào ghi chú|copy_all|thumb_up|thumb_down)\s*", "", text)
    return format_clean_answer(text)


async def close_any_modal_dialogs(page):
    """Đóng bất kỳ modal / dialog nào vô tình mở."""
    try:
        await page.keyboard.press("Escape")
    except Exception:
        pass


async def wait_for_notebooklm_ready(page):
    """Đợi giao diện NotebookLM sẵn sàng."""
    await close_any_modal_dialogs(page)
    try:
        await page.wait_for_selector(
            "textarea:not([disabled]), div[contenteditable='true'], input[placeholder*='câu hỏi']",
            timeout=30000
        )
        await asyncio.sleep(3.0)
    except Exception:
        await asyncio.sleep(3.0)


async def copy_response_via_button(page) -> str:
    """
    Click nút Copy nằm trong bong bóng tin nhắn của bot tại cột Cuộc trò chuyện.
    """
    await close_any_modal_dialogs(page)

    clicked = await page.evaluate("""() => {
        const topHeader = document.querySelector('header, nav, .app-bar, mat-toolbar');
        const dialogs = Array.from(document.querySelectorAll('mat-dialog-container, [role="dialog"]'));

        const buttons = Array.from(document.querySelectorAll('button'));
        const chatCopyButtons = buttons.filter(b => {
            if (topHeader && topHeader.contains(b)) return false;
            for (const d of dialogs) {
                if (d.contains(b)) return false;
            }

            const label = (b.getAttribute('aria-label') || b.getAttribute('mattooltip') || b.getAttribute('data-tooltip') || '').toLowerCase();
            const btnText = (b.innerText || '').toLowerCase();
            const icon = b.querySelector('mat-icon, i');
            const iconText = icon ? (icon.innerText || '').toLowerCase() : '';

            if (btnText.includes('sổ ghi chú') || btnText.includes('notebook') || label.includes('notebook')) {
                return false;
            }

            const isCopyBtn = label.includes('copy') || label.includes('sao chép') || 
                              iconText.includes('content_copy') || iconText.includes('copy');

            return isCopyBtn;
        });

        if (chatCopyButtons.length > 0) {
            const target = chatCopyButtons[chatCopyButtons.length - 1];
            target.scrollIntoView({ behavior: 'smooth', block: 'center' });
            target.click();
            return true;
        }
        return false;
    }""")

    if clicked:
        await asyncio.sleep(0.6)
        try:
            clipboard_text = await page.evaluate("async () => await navigator.clipboard.readText()")
            if is_valid_response(clipboard_text):
                return clipboard_text.strip()
        except Exception:
            pass

    # Fallback DOM text
    dom_text = await page.evaluate("""() => {
        const topHeader = document.querySelector('header, nav, .app-bar, mat-toolbar');
        const selectors = [
            '.model-response',
            '[data-message-author="bot"]',
            '.chat-message-content',
            '.rendered-markdown',
            'chat-message'
        ];
        for (const s of selectors) {
            const elements = Array.from(document.querySelectorAll(s))
                .filter(el => !(topHeader && topHeader.contains(el)));
            if (elements.length > 0) {
                const last = elements[elements.length - 1];
                return last.innerText || last.textContent || "";
            }
        }
        return "";
    }""")

    return clean_dom_text(dom_text)


async def wait_for_response_with_delay(
    page, wait_seconds: int = 60, previous_text: str = ""
) -> str:
    """
    Tự động đợi đủ 60 giây để NotebookLM sinh trọn vẹn câu trả lời rồi lấy kết quả.
    """
    for remaining in range(wait_seconds, 0, -5):
        print(f"    [~] Đang đợi NotebookLM sinh câu trả lời ({remaining}s còn lại)...", end="\r", flush=True)
        await asyncio.sleep(5)
    print("    [~] Đã đợi đủ 60s. Đang kiểm tra và sao chép câu trả lời...            ")

    # Kiểm tra xem có đang streaming không
    for _ in range(15):
        is_streaming = await page.evaluate("""() => {
            const stopBtns = Array.from(document.querySelectorAll('button'))
                .filter(b => /stop|dừng/i.test(b.getAttribute('aria-label') || b.innerText || ''));
            return stopBtns.length > 0;
        }""")
        if not is_streaming:
            break
        await asyncio.sleep(1.0)

    # Click nút Copy
    await asyncio.sleep(0.8)
    result = await copy_response_via_button(page)
    if is_valid_response(result) and result != previous_text:
        return result

    # Fallback trích xuất DOM
    dom_text = await page.evaluate("""() => {
        const topHeader = document.querySelector('header, nav, .app-bar, mat-toolbar');
        const selectors = [
            '.model-response',
            '[data-message-author="bot"]',
            '.chat-message-content',
            '.rendered-markdown',
            'chat-message'
        ];
        for (const s of selectors) {
            const elements = Array.from(document.querySelectorAll(s))
                .filter(el => !(topHeader && topHeader.contains(el)));
            if (elements.length > 0) {
                const last = elements[elements.length - 1];
                return last.innerText || last.textContent || "";
            }
        }
        return "";
    }""")
    return clean_dom_text(dom_text)


async def find_and_fill_chat_input(page, text: str) -> bool:
    """Tìm ô nhập chat trong NotebookLM, xóa sạch nội dung cũ và gõ câu hỏi mới."""
    await close_any_modal_dialogs(page)

    input_selectors = [
        "textarea[placeholder*='Đặt câu hỏi']",
        "textarea[placeholder*='Start typing']",
        "textarea[placeholder*='Bắt đầu nhập']",
        "textarea[placeholder*='Ask']",
        "textarea[placeholder*='Hỏi']",
        "textarea",
        "div[contenteditable='true']",
        "[role='textbox']",
        "input[type='text']"
    ]

    for selector in input_selectors:
        try:
            elem = page.locator(selector).last
            if await elem.is_visible(timeout=2000):
                await elem.click()
                await asyncio.sleep(0.3)
                await page.keyboard.press("Control+A")
                await page.keyboard.press("Backspace")
                await asyncio.sleep(0.2)
                await page.keyboard.insert_text(text)
                await asyncio.sleep(0.5)
                await page.keyboard.press("Enter")
                return True
        except Exception:
            continue

    # Fallback evaluate
    success = await page.evaluate("""(text) => {
        const textareas = document.querySelectorAll('textarea, [contenteditable="true"]');
        if (textareas.length > 0) {
            const target = textareas[textareas.length - 1];
            target.focus();
            if (target.tagName.toLowerCase() === 'textarea') {
                target.value = text;
                target.dispatchEvent(new Event('input', { bubbles: true }));
                target.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true }));
            } else {
                target.innerText = text;
                target.dispatchEvent(new Event('input', { bubbles: true }));
            }
            return true;
        }
        return false;
    }""", text)

    return success


async def main():
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("\n[!] Lỗi: Chưa cài đặt thư viện 'playwright'.")
        print("Vui lòng chạy: uv add playwright && uv run playwright install chromium\n")
        return

    print("=" * 68)
    print("  TOOL AUTO ĐIỀN QUESTION VÀ LẤY REFERENCE TỪ NOTEBOOKLM (STEALTH)")
    print("=" * 68)

    # 1. Đọc danh sách câu hỏi
    questions = load_questions(DEFAULT_QUESTION_FILE)
    print(f"[+] Đã tải {len(questions)} câu hỏi từ: {DEFAULT_QUESTION_FILE}")

    # 2. Kiểm tra các câu đã hoàn thành trước đó (để resume)
    answered = get_answered_numbers(DEFAULT_REFERENCE_FILE)
    if answered:
        print(f"[i] Tìm thấy {len(answered)} câu đã có HỢP LỆ trong {DEFAULT_REFERENCE_FILE.name}: {sorted(list(answered))}")

    # 3. Lựa chọn chế độ kết nối trình duyệt
    print("\nChọn cách khởi chạy trình duyệt:")
    print("  1. Mở Chrome/Chromium với Stealth + Profile (Lưu phiên tại benchmarks/.notebooklm_browser_data) [Khuyên dùng]")
    print("  2. Kết nối tới Chrome đang mở sẵn qua cổng Remote Debugging (localhost:9222)")
    
    mode_choice = input("Nhập lựa chọn (1 hoặc 2, mặc định 1): ").strip() or "1"

    async with async_playwright() as p:
        browser = None
        context = None

        if mode_choice == "2":
            print("\n[+] Đang kết nối tới Chrome tại http://localhost:9222...")
            try:
                browser = await p.chromium.connect_over_cdp("http://localhost:9222")
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                page = context.pages[0] if context.pages else await context.new_page()
            except Exception as e:
                print(f"[!] Không thể kết nối CDP: {e}")
                print("Chuyển sang chế độ 1 (Mở trình duyệt có lưu profile)...")
                mode_choice = "1"

        if mode_choice == "1":
            USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
            print(f"\n[+] Khởi chạy trình duyệt với profile tại: {USER_DATA_DIR}")
            browser_args = [
                "--start-maximized",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
            ]
            
            try:
                context = await p.chromium.launch_persistent_context(
                    user_data_dir=str(USER_DATA_DIR),
                    headless=False,
                    channel="chrome",
                    permissions=["clipboard-read", "clipboard-write"],
                    args=browser_args,
                    no_viewport=True,
                )
            except Exception:
                context = await p.chromium.launch_persistent_context(
                    user_data_dir=str(USER_DATA_DIR),
                    headless=False,
                    permissions=["clipboard-read", "clipboard-write"],
                    args=browser_args,
                    no_viewport=True,
                )
            page = context.pages[0] if context.pages else await context.new_page()

        if HAS_STEALTH:
            try:
                stealth = Stealth()
                await stealth.apply_stealth_async(page)
                print("[+] Đã kích hoạt Playwright Stealth (Anti-detection).")
            except Exception as e:
                print(f"[i] Không thể áp dụng stealth: {e}")

        try:
            await context.grant_permissions(["clipboard-read", "clipboard-write"], origin="https://notebooklm.google.com")
        except Exception:
            pass

        print("\n" + "-" * 68)
        print("[HƯỚNG DẪN]:")
        print("1. Trình duyệt đã mở. Hãy đăng nhập tài khoản Google (nếu chưa).")
        print(f"2. Trình duyệt sẽ tự mở Notebook PLDC: {DEFAULT_NOTEBOOK_URL}")
        print("3. Đảm bảo giao diện đã sẵn sàng ở khung chat của Notebook.")
        print("-" * 68)

        current_url = page.url
        if "dc53252a-4dc2-4fd7-a7f1-e56a7f60eb0d" not in current_url:
            await page.goto(DEFAULT_NOTEBOOK_URL, wait_until="domcontentloaded")
            await wait_for_notebooklm_ready(page)

        input("\n>>> Nhấn [ENTER] tại đây sau khi đã sẵn sàng ở khung chat NotebookLM để BẮT ĐẦU tự động cào...")

        total_questions = len(questions)
        previous_response_text = ""

        for idx, q in enumerate(questions, start=1):
            q_num = q["num"]
            q_text = q["clean_text"]

            if q_num in answered:
                print(f"[{idx}/{total_questions}] Đã có câu {q_num} HỢP LỆ trong file reference -> Bỏ qua.")
                continue

            # Random delay 2-4s giữa các câu hỏi
            delay_sec = random.uniform(2.0, 4.0)
            print(f"\n[~] Nghỉ {delay_sec:.1f}s trước khi gửi câu tiếp theo...")
            await asyncio.sleep(delay_sec)

            print(f"\n[{idx}/{total_questions}] Đang gửi Câu {q_num}: {q_text[:75]}...")

            # Điền câu hỏi và gửi
            sent = await find_and_fill_chat_input(page, q_text)
            if not sent:
                print(f"[!] Không tìm thấy ô nhập tin nhắn! Hãy kiểm tra lại màn hình trình duyệt.")
                input("Nhấn Enter để thử lại...")
                await find_and_fill_chat_input(page, q_text)

            # Đợi đủ 60 giây để NotebookLM sinh xong câu trả lời
            response_text = await wait_for_response_with_delay(
                page, wait_seconds=60, previous_text=previous_response_text
            )

            if not is_valid_response(response_text):
                if not response_text:
                    response_text = "[Không lấy được phản hồi từ NotebookLM]"

            previous_response_text = response_text

            # Xử lý làm sạch và chuẩn hóa các dấu xuống dòng liên tiếp thừa
            cleaned_resp = format_clean_answer(response_text)

            # Ghi vào reference.txt theo định dạng -<num>. <câu trả lời>
            formatted_entry = f"-{q_num}. {cleaned_resp}\n\n"
            
            with open(DEFAULT_REFERENCE_FILE, "a", encoding="utf-8") as f:
                f.write(formatted_entry)

            answered.add(q_num)
            print(f"    [V] -> Đã lưu câu {q_num} vào {DEFAULT_REFERENCE_FILE.name} ({len(cleaned_resp)} ký tự)")

        print("\n" + "=" * 68)
        print(f"[V] HOÀN TẤT TOÀN BỘ 30 CÂU HỎI!")
        print(f"    File kết quả: {DEFAULT_REFERENCE_FILE}")
        print("=" * 68)

        keep_open = input("\nBạn có muốn giữ trình duyệt mở không? (y/n, mặc định y): ").strip().lower()
        if keep_open == 'n':
            await context.close()


if __name__ == "__main__":
    asyncio.run(main())
