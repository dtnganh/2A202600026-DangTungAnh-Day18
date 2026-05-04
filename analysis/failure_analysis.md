# Failure Analysis - Lab 18: Production RAG

**Nhóm:** Làm cá nhân  
**Thành viên:** Đặng Tùng Anh - M1, M2, M3, M4, M5 và pipeline

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Delta |
|--------|---------------:|-----------:|------:|
| Faithfulness | 1.0000 | 0.9005 | -0.0995 |
| Answer Relevancy | 0.6477 | 0.7548 | +0.1071 |
| Context Precision | 0.7946 | 0.8090 | +0.0144 |
| Context Recall | 0.7686 | 0.9381 | +0.1695 |

## Bottom-5 Failures

### #1
- **Question:** Kỳ tính thuế trong tờ khai GTGT là thời gian nào?
- **Expected:** Kỳ tính thuế trong tờ khai là Quý 4 năm 2024.
- **Got:** Không tìm thấy thông tin trong tài liệu.
- **Worst metric:** answer_relevancy = 0.0000
- **Error Tree:** Output sai -> Context thiếu thông tin kỳ tính thuế -> Query rõ -> Fix OCR/retrieval cho vùng header biểu mẫu.
- **Root cause:** Dòng kỳ tính thuế không được OCR/structure chunk giữ đủ rõ, nên retrieval trả về phần nghĩa vụ thuế thay vì header tờ khai.
- **Suggested fix:** OCR lại vùng đầu form ở độ phân giải cao hơn hoặc thêm rule chunk riêng cho header biểu mẫu thuế.

### #2
- **Question:** Bên kiểm soát dữ liệu cá nhân là gì?
- **Expected:** Bên kiểm soát dữ liệu cá nhân là tổ chức, cá nhân quyết định mục đích và phương tiện xử lý dữ liệu cá nhân.
- **Got:** Không tìm thấy thông tin trong tài liệu.
- **Worst metric:** answer_relevancy = 0.0919
- **Error Tree:** Context có liên quan nhưng answer generation từ chối -> Query OK -> Fix prompt/context selection.
- **Root cause:** Context chứa nhiều định nghĩa liên tiếp, model đôi lúc đánh giá chưa đủ trực tiếp và trả lời không tìm thấy.
- **Suggested fix:** Trước generation, ưu tiên câu có exact phrase "Bên Kiểm soát dữ liệu cá nhân là".

### #3
- **Question:** Mã số thuế của người nộp thuế trong tờ khai là bao nhiêu?
- **Expected:** Mã số thuế của người nộp thuế là 0106769437.
- **Got:** Mã số thuế của người nộp thuế trong tờ khai là 0106769437.
- **Worst metric:** context_precision = 0.6667
- **Error Tree:** Output đúng -> Context đúng nhưng có thêm đoạn không cần thiết -> Query OK -> Fix context trimming.
- **Root cause:** Reranker lấy cả phần chữ ký và nghĩa vụ thuế cùng với bảng thông tin người nộp thuế.
- **Suggested fix:** Sau rerank, cắt context xuống dòng bảng có "Mã số thuế".

### #4
- **Question:** Nghị định 13/2023/NĐ-CP quy định về vấn đề gì?
- **Expected:** Nghị định quy định về bảo vệ dữ liệu cá nhân và trách nhiệm bảo vệ dữ liệu cá nhân.
- **Got:** Nghị định 13/2023/NĐ-CP quy định về bảo vệ dữ liệu cá nhân và trách nhiệm bảo vệ dữ liệu cá nhân của cơ quan, tổ chức, cá nhân có liên quan.
- **Worst metric:** context_precision = 0.7000
- **Error Tree:** Output đúng -> Context có thêm phụ lục và metadata văn bản -> Query OK -> Fix retrieval precision.
- **Root cause:** Query có tên nghị định nên retrieval kéo thêm trang phụ lục có cùng số nghị định.
- **Suggested fix:** Tăng trọng số heading "Điều 1. Phạm vi điều chỉnh" khi câu hỏi hỏi "quy định về vấn đề gì".

### #5
- **Question:** Chuyển dữ liệu cá nhân ra nước ngoài theo Điều 25 cần lập hồ sơ gì?
- **Expected:** Bên chuyển dữ liệu ra nước ngoài cần lập Hồ sơ đánh giá tác động chuyển dữ liệu cá nhân ra nước ngoài.
- **Got:** Model trả lời dài hơn, liệt kê thêm các thành phần của hồ sơ.
- **Worst metric:** answer_relevancy = 0.7537
- **Error Tree:** Output đúng ý nhưng quá rộng -> Context đúng -> Query OK -> Fix answer style.
- **Root cause:** Context chứa cả danh sách thành phần hồ sơ nên model trả lời nhiều hơn câu hỏi yêu cầu.
- **Suggested fix:** Prompt generation nên yêu cầu trả lời 1 câu khi câu hỏi hỏi "cần lập hồ sơ gì".

## Case Study

**Question chọn phân tích:** Kỳ tính thuế trong tờ khai GTGT là thời gian nào?

**Error Tree walkthrough:**
1. Output đúng? -> Không, model trả lời không tìm thấy.
2. Context đúng? -> Chưa đủ, retrieval không lấy được vùng header chứa kỳ tính thuế.
3. Query rewrite OK? -> Có, câu hỏi rõ.
4. Fix ở bước: OCR biểu mẫu và retrieval theo metadata/header.

**Nếu có thêm 1 giờ, sẽ optimize:**
- OCR lại trang tờ khai thuế với prompt chuyên cho form/table.
- Tách riêng header biểu mẫu thành chunk nhỏ.
- Thêm rule boost cho các trường như kỳ tính thuế, mã số thuế, người nộp thuế.
- Tạo sentence/table-row level reranking sau khi lấy parent context.
