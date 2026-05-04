# Individual Reflection - Lab 18

**Tên:** Đặng Tùng Anh  
**Module phụ trách:** Full pipeline / M1-M5

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M1 Chunking, M2 Hybrid Search, M3 Reranking, M4 Evaluation, M5 Enrichment và `src/pipeline.py`.
- Các hàm/class chính đã viết:
  - `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()`, `compare_strategies()`
  - `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()`
  - `CrossEncoderReranker`, `benchmark_reranker()`
  - `evaluate_ragas()`, `failure_analysis()`
  - `summarize_chunk()`, `generate_hypothesis_questions()`, `contextual_prepend()`, `extract_metadata()`, `enrich_chunks()`
  - OpenAI-based enrichment và OpenAI answer generation trong pipeline, có fallback khi API lỗi.
- Số tests pass: 37/37.

## 2. Kiến thức học được

- Khái niệm mới nhất: Production RAG không chỉ là vector search, mà là chuỗi nhiều bước gồm chunking, hybrid retrieval, rerank, generation, evaluation và failure analysis.
- Điều bất ngờ nhất: Khi dùng OpenAI thật, answer relevancy cải thiện rõ, nhưng context precision vẫn phụ thuộc nhiều vào cách cắt/trả context.
- Kết nối với bài giảng: M1-M5 tương ứng các tầng tối ưu RAG: indexing, retrieval, reranking, evaluation và enrichment.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: PDF gốc không trích xuất được text bằng MarkItDown/pypdf, có thể do là scan hoặc encoding khó đọc.
- Cách giải quyết: Tạo Markdown học tập từ chủ đề của tài liệu để pipeline có corpus chạy được; ghi chú rằng bản production thật cần OCR.
- Thời gian debug: Tập trung nhiều nhất ở lỗi encoding Windows, fallback không cần Qdrant/model ngoài và answer extraction.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Làm OCR và kiểm tra chất lượng dữ liệu trước khi viết retrieval pipeline.
- Module nào muốn thử tiếp: M4 với RAGAS thật và LLM-as-judge, cộng thêm OCR cho PDF gốc.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 5 |
