# Group Report - Lab 18: Production RAG

**Nhóm:** Làm cá nhân  
**Ngày:** 04/05/2026

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Đặng Tùng Anh | M1: Chunking local | Có | 13/13 |
| Đặng Tùng Anh | M2: Hybrid Search local | Có | 5/5 |
| Đặng Tùng Anh | M3: OpenAI API Reranking | Có | 5/5 |
| Đặng Tùng Anh | M4: Local Evaluation | Có | 4/4 |
| Đặng Tùng Anh | M5: OpenAI API Enrichment | Có | 10/10 |

## Kết quả RAGAS

| Metric | Naive | Production | Delta |
|--------|------:|-----------:|------:|
| Faithfulness | 1.0000 | 0.9005 | -0.0995 |
| Answer Relevancy | 0.6477 | 0.7548 | +0.1071 |
| Context Precision | 0.7946 | 0.8090 | +0.0144 |
| Context Recall | 0.7686 | 0.9381 | +0.1695 |

## Key Findings

1. **Biggest improvement:** Production pipeline cải thiện Answer Relevancy từ 0.6477 lên 0.7548 và Context Recall từ 0.7686 lên 0.9381 nhờ OpenAI API cho enrichment, reranking và answer generation.
2. **Biggest challenge:** PDF gốc gần như không có text layer nên MarkItDown/pypdf không đủ. Hướng hợp lý là OCR bằng OpenAI Vision API để tạo Markdown trước khi đưa vào RAG.
3. **Surprise finding:** Baseline có Faithfulness cao vì answer extractive lấy nguyên context, nhưng production hữu ích hơn vì trả lời tự nhiên hơn và lấy đủ context liên quan hơn.

## Presentation Notes (5 phút)

1. Kiến trúc: local cho chunking/search/evaluation, OpenAI API cho OCR/rerank/enrichment/generate answer.
2. RAGAS scores: Production đạt Faithfulness 0.9005, Answer Relevancy 0.7548, Context Precision 0.8090, Context Recall 0.9381.
3. Biggest win: hệ thống chạy end-to-end trên tài liệu OCR thật, có cache OCR/enrichment để lần chạy sau nhanh hơn.
4. Case study: câu hỏi "Kỳ tính thuế..." thất bại vì thông tin ở header biểu mẫu chưa được OCR/chunk đủ tốt.
5. Next optimization: OCR form/table chuyên biệt hơn, tách table row/header chunk nhỏ và thêm reranking cấp câu hoặc cấp dòng bảng.
