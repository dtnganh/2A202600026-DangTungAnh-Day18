"""Convert scanned PDF pages to Markdown using OpenAI vision OCR."""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

import pypdfium2 as pdfium

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OPENAI_API_KEY, OPENAI_MODEL


def _require_client():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Điền key thật vào .env trước khi chạy OCR.")
    from openai import OpenAI

    return OpenAI(api_key=OPENAI_API_KEY)


def _page_to_data_url(page, scale: float) -> str:
    image = page.render(scale=scale).to_pil().convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _ocr_page(client, data_url: str, page_number: int, total_pages: int) -> str:
    for attempt in range(1, 8):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0,
                max_tokens=2600,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là OCR engine cho tài liệu tiếng Việt. Hãy đọc ảnh trang PDF và "
                            "chuyển toàn bộ nội dung nhìn thấy sang Markdown. Giữ thứ tự đọc, giữ số liệu, "
                            "mã biểu mẫu, điều/khoản/mục. Nếu có bảng, biểu diễn bằng Markdown table hoặc "
                            "các dòng văn bản rõ ràng. Không giải thích, không thêm nội dung ngoài ảnh."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"OCR trang {page_number}/{total_pages} sang Markdown."},
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                        ],
                    },
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            message = str(exc)
            is_rate_limit = "rate_limit" in message.lower() or "429" in message
            if not is_rate_limit or attempt == 7:
                raise
            wait_seconds = min(10 + attempt * 5, 45)
            print(f"Rate limit at page {page_number}; retrying in {wait_seconds}s (attempt {attempt}/7)")
            time.sleep(wait_seconds)
    raise RuntimeError(f"OpenAI OCR failed for page {page_number}")


def pdf_to_markdown(input_pdf: Path, output_md: Path, scale: float = 2.2) -> None:
    client = _require_client()
    document = pdfium.PdfDocument(str(input_pdf))
    total_pages = len(document)
    rendered_pages = []
    cache_dir = output_md.parent / ".openai_ocr_cache" / input_pdf.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    for page_index in range(total_pages):
        page_number = page_index + 1
        cache_path = cache_dir / f"page_{page_number:03d}.md"
        if cache_path.exists() and cache_path.read_text(encoding="utf-8").strip():
            print(f"OCR {input_pdf.name}: page {page_number}/{total_pages} (cached)")
            page_md = cache_path.read_text(encoding="utf-8")
        else:
            print(f"OCR {input_pdf.name}: page {page_number}/{total_pages}")
            data_url = _page_to_data_url(document[page_index], scale=scale)
            page_md = _ocr_page(client, data_url, page_number, total_pages)
            cache_path.write_text(page_md, encoding="utf-8")
        rendered_pages.append(f"## Trang {page_number}\n\n{page_md}".strip())
        output_md.write_text(
            f"# OCR bằng OpenAI từ {input_pdf.name}\n\n" + "\n\n".join(rendered_pages) + "\n",
            encoding="utf-8",
        )

    output_md.write_text(
        f"# OCR bằng OpenAI từ {input_pdf.name}\n\n" + "\n\n".join(rendered_pages) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("output_md", type=Path)
    parser.add_argument("--scale", type=float, default=2.2)
    args = parser.parse_args()
    pdf_to_markdown(args.input_pdf, args.output_md, scale=args.scale)


if __name__ == "__main__":
    main()
