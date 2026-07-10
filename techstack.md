# Tech Stack — TVT/NER Y khoa tiếng Việt (Structure-Aware Few-Shot Extraction)

> Đi kèm `phuong_an_5_structure_aware_extraction.md`. Ở những chỗ chưa chắc lựa chọn nào tối ưu nhất trong điều kiện tài nguyên hạn chế, file này liệt kê **nhiều option** kèm khuyến nghị tạm thời — cần tự đo thử trên máy thật trước khi chốt.

## 0. Ràng buộc cần nhớ khi chọn công cụ

- Model tự host ≤ **9B tham số**, **không dùng API ngoài** (ưu tiên mọi thứ chạy offline, kể cả tra cứu mã chuẩn)
- Phần cứng: Windows + **RTX 3050Ti 4GB VRAM** (local) + **Colab free** + **Kaggle free** (backup, VRAM lớn hơn nhưng có giới hạn giờ chạy)
- → Ưu tiên thư viện **nhẹ, pure-Python hoặc chạy CPU được**, tránh dựng thêm server (Elasticsearch, vector DB nặng...) không cần thiết cho quy mô bài toán (100 file)

## 1. Ngôn ngữ & môi trường nền

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Ngôn ngữ | Python 3.10+ | |
| Môi trường phát triển | Jupyter Notebook (local) + Google Colab + Kaggle Notebook | Local để code/debug nhanh, Colab/Kaggle để chạy batch cần GPU mạnh hơn |
| Quản lý package | `venv` hoặc `conda` (local) | Colab/Kaggle dùng `pip install` trực tiếp trong cell |

## 2. Bước 1 — Structural Parser

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Tách block theo header | `re` (regex, built-in Python) | Không cần AI, không cần thư viện ngoài |

## 3. Bước 2 — Few-shot LLM Extraction

### 3.1. Model LLM ≤9B, hỗ trợ tiếng Việt (đang phân vân, liệt kê option)

| Option | Đặc điểm | Rủi ro/Ưu điểm |
|---|---|---|
| **Qwen2.5-7B-Instruct** (GGUF) | Đa ngôn ngữ tốt, cộng đồng lớn, nhiều bản quantize sẵn | Khuyến nghị thử đầu tiên — cân bằng chất lượng/tốc độ, hỗ trợ tiếng Việt khá ổn |
| **SeaLLM-7B** | Được tune riêng cho tiếng Đông Nam Á (có tiếng Việt) | Có thể hiểu ngữ cảnh y khoa tiếng Việt tốt hơn model đa ngôn ngữ chung, nhưng ít phổ biến hơn → cần tự test |
| **Vistral-7B-Chat** | Fine-tune từ Mistral riêng cho tiếng Việt | Tiếng Việt tự nhiên tốt, nhưng chưa rõ độ mạnh khi theo instruction phức tạp (JSON schema + nhiều nhãn) |
| **PhoGPT-7.5B** | Model tiếng Việt "gốc" (train từ đầu, không phải fine-tune từ model Anh) | Tiếng Việt tự nhiên, nhưng khả năng theo prompt phức tạp/multi-task có thể yếu hơn các model instruct hiện đại hơn |

→ **Khuyến nghị**: thử Qwen2.5-7B-Instruct trước (bản GGUF Q4_K_M, ~4-5GB) vì hệ sinh thái công cụ hỗ trợ (constrained decoding, quantize) đầy đủ nhất; test thêm SeaLLM/Vistral trên 7 sample nếu kết quả chưa đạt.

### 3.2. Inference engine

| Option | Ghi chú |
|---|---|
| **llama.cpp** (qua `llama-cpp-python`) | Khuyến nghị chính — nhẹ, chạy tốt trên 4GB VRAM với model quantize GGUF, **hỗ trợ GBNF grammar** để ép JSON Schema (cần cho 7.3) |
| Ollama | Dễ cài đặt hơn (wrap llama.cpp), phù hợp nếu ưu tiên tốc độ setup hơn tùy biến sâu |
| vLLM | Throughput cao hơn nhưng cần nhiều VRAM hơn — chỉ phù hợp khi chạy trên Colab/Kaggle GPU lớn, không dùng được cho local 4GB |

### 3.3. Constrained decoding (ép JSON Schema)

| Option | Ghi chú |
|---|---|
| **GBNF grammar** (llama.cpp built-in) | Khuyến nghị nếu dùng llama.cpp — tích hợp sẵn, không cần cài thêm |
| `Outlines` (thư viện Python) | Thay thế nếu chuyển sang chạy qua HuggingFace `transformers`/vLLM thay vì llama.cpp |

## 4. Bước 3 — Position Resolver

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Fuzzy matching cơ bản | `difflib.SequenceMatcher` (built-in) | Đơn giản, đủ dùng cho case dễ |
| Fuzzy matching có tùy chỉnh cost | **`rapidfuzz`** | Khuyến nghị — nhanh hơn `difflib` nhiều lần, cho phép custom cost (cần thiết để set "chi phí chèn khoảng trắng" thấp hơn "chi phí sửa ký tự" như mô tả ở 7.4) |
| Tính edit-distance thuần | `python-Levenshtein` | Có thể dùng thay `rapidfuzz` nếu chỉ cần đo khoảng cách, không cần custom cost linh hoạt |

## 5. Bước 4 — NegEx Backstop

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Rule engine phủ định | Tự viết bằng Python (regex + danh sách trigger) | Khuyến nghị — rule đã thiết kế riêng cho tiếng Việt, không có framework tiếng Việt sẵn nào tốt hơn |
| Tham khảo kiến trúc (không dùng trực tiếp) | `pyConTextNLP` | Framework gốc tiếng Anh cho negation/context detection — có thể tham khảo cấu trúc luật (pre/post-negation, pseudo-negation, termination) để tổ chức code, nhưng cần tự port toàn bộ trigger sang tiếng Việt |

## 6. Bước 5 — Candidate Linking

### 6.1. Tầng 0 — Từ điển viết tắt

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Lưu trữ mapping | File JSON/CSV tự xây | Đơn giản, dễ mở rộng thủ công |
| Giải nghĩa khi đa nghĩa | Regex kiểm tra từ khóa lân cận | Rule-based, không cần model riêng |
| *(Tùy chọn, nếu dư tài nguyên)* | Model BERT y khoa (ClinicalBERT/ViHealthBERT/ViPubMedDeBERTa) | Rủi ro: tốn thêm VRAM khi đã chạy LLM 9B — chỉ thử nếu đo thấy còn dư ngân sách (mục 7.8) |

### 6.2. Tầng 1 — Từ điển ánh xạ (anchor)

| Nguồn dữ liệu | Ghi chú |
|---|---|
| Danh mục thuốc lưu hành (Cục Quản lý Dược - Bộ Y tế) | Cần tìm bản tải về được (file Excel/PDF công khai trên trang Cục QLD) |
| DrugBank (Open Data) | Chú ý giấy phép sử dụng — bản đầy đủ có thể yêu cầu license cho mục đích thương mại, kiểm tra điều khoản trước khi dùng |
| Bảng ICD-10 tiếng Việt (bản dịch chính thức Bộ Y tế/WHO) | Cần tìm bản .csv/.xlsx công khai (hoặc tự số hóa từ tài liệu PDF nếu không có sẵn dạng bảng) |
| Lưu trữ để tra cứu | **SQLite** hoặc bảng `pandas` load từ CSV — không cần DB server |

### 6.3. Tầng 2 — Hybrid retrieval (cho ICD-10)

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Lexical search (BM25) | **`rank_bm25`** (pure Python) | Khuyến nghị — nhẹ, `pip install` là chạy được, không cần Elasticsearch |
| Embedding ngữ nghĩa | Xem bảng 6.3.1 bên dưới | |
| Vector index/so khớp | **`faiss-cpu`** nếu tập ICD-10 lớn (hàng chục ngàn mã); hoặc chỉ cần **numpy cosine similarity** nếu tập mã dùng thực tế nhỏ hơn vài ngàn | Khuyến nghị bắt đầu bằng numpy thuần — đơn giản, đủ nhanh cho quy mô bài toán này, chỉ chuyển sang FAISS nếu đo thấy chậm |
| Gộp điểm (fusion) | Tự cài đặt **Reciprocal Rank Fusion (RRF)** bằng Python thuần | Vài dòng code, không cần thư viện riêng |

#### 6.3.1. Model embedding đa ngôn ngữ (đang phân vân, liệt kê option)

| Option | Đặc điểm | Rủi ro/Ưu điểm |
|---|---|---|
| **SapBERT multilingual** (`cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` trên HuggingFace) | Huấn luyện chuyên cho việc nối thuật ngữ y khoa đồng nghĩa liên ngôn ngữ (dựa trên XLM-R, ~270M tham số) | Khuyến nghị thử trước — chuyên biệt nhất cho đúng bài toán, nhẹ hơn nhiều so với LLM chính nên chạy tuần tự được trên 4GB VRAM; cần tự kiểm tra chất lượng thực tế với tiếng Việt (model gốc train chủ yếu trên dữ liệu UMLS, độ phủ tiếng Việt chưa chắc cao) |
| `multilingual-e5-base` hoặc `-large` (qua `sentence-transformers`) | Embedding đa ngôn ngữ tổng quát, không chuyên y khoa | Phương án dự phòng nếu SapBERT multilingual cho kết quả kém trên tiếng Việt — dễ dùng, cộng đồng lớn |
| LaBSE | Embedding đa ngôn ngữ tổng quát khác, ổn định | Tương tự e5, dùng làm phương án so sánh chéo |

→ **Khuyến nghị**: thử SapBERT multilingual trước trên 7 sample; nếu chất lượng không hơn rõ rệt so với `multilingual-e5-base`, dùng e5 cho đơn giản (ít rủi ro tương thích thư viện hơn).

### 6.4. Tầng 2 (mở rộng, tùy chọn) — Cross-Encoder rerank

| Option | Ghi chú |
|---|---|
| **xMEN** (github.com/hpi-dhc/xmen) | Công cụ chuyên cho cross-lingual medical entity normalization — đúng bài toán nhất, nhưng cần đo thời gian/VRAM thực tế trước khi thêm vào pipeline chính thức (mục 7.8) |
| Cross-encoder tổng quát qua `sentence-transformers` (VD nhóm model `cross-encoder/mmarco-*`) | Nhẹ hơn, dễ tích hợp hơn xMEN nhưng không chuyên y khoa — phương án dự phòng nếu xMEN quá nặng hoặc khó cài trên Windows |

**Lưu ý:** đây là bước **tùy chọn**, chỉ thêm nếu đo ngân sách compute (7.8) còn dư; bỏ qua bước này không ảnh hưởng tới việc pipeline chạy được, chỉ ảnh hưởng độ chính xác biên.

### 6.5. Tầng 2' — Riêng cho THUỐC (RxNorm)

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Trích xuất trường có cấu trúc | Dùng lại LLM ở Bước 2 (không cần model riêng) | Chỉ cần mở rộng JSON Schema đã ép ở 7.3 để có thêm field brand/ingredient/strength/form |
| Dữ liệu RxNorm để so khớp | **RxNorm Full Monthly Release** (tải file từ NLM, cần đăng ký tài khoản UMLS miễn phí) | Bắt buộc tải về dùng **offline** — không gọi API RxNav sống, để tránh vướng ràng buộc "không dùng API ngoài" |
| Lưu trữ/so khớp | SQLite (RxNorm cung cấp sẵn schema dạng file `.RRF`, có thể import thẳng vào SQLite) | Có sẵn script import mẫu từ NLM, không cần tự viết parser từ đầu |

### 6.6. Tầng 3 — Fuzzy matching cuối

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| So khớp lỗi chính tả nhẹ còn sót | `rapidfuzz` (dùng lại thư viện ở Bước 3) | Không cần thêm thư viện mới |

## 7. Bước 7.7 — Tự kiểm định / Validation

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Tính WER (`text_score`) | **`jiwer`** | Thư viện chuyên tính Word Error Rate, `pip install jiwer` |
| Tính Jaccard (`assertions`, `candidates`) | Tự viết bằng `set` thuần Python | Công thức đơn giản, không cần thư viện riêng |

## 8. Đóng gói output

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| Ghi JSON | `json` (built-in), `ensure_ascii=False` | Giữ dấu tiếng Việt khi ghi file, như đã nêu ở mục 1 |
| Kiểm tra schema trước khi nộp | `jsonschema` (thư viện) | Validate field bắt buộc, `type` đúng 5 giá trị, trước khi đóng gói |
| Đóng gói `.zip` | `zipfile` (built-in) | Không cần công cụ ngoài |

## 9. Tổng hợp: cài đặt nhanh (pip install)

```bash
pip install llama-cpp-python rank_bm25 rapidfuzz sentence-transformers \
            faiss-cpu jiwer jsonschema pandas --break-system-packages
```

> Lưu ý: `llama-cpp-python` cần build với cờ hỗ trợ CUDA nếu muốn chạy GPU trên RTX 3050Ti (`CMAKE_ARGS="-DGGML_CUDA=on"` khi cài) — nếu không, mặc định sẽ chạy CPU (chậm hơn nhiều cho model 7-9B).

## 10. Những điểm còn cần tự đo thử trước khi chốt cuối cùng

- SapBERT multilingual vs `multilingual-e5-base` — chưa rõ cái nào cho kết quả tốt hơn thực tế trên thuật ngữ y khoa tiếng Việt, cần test trên 7 sample.
- Có thêm bước Cross-Encoder rerank (xMEN) hay không — phụ thuộc ngân sách compute đo được ở 7.8.
- Model LLM chính (Qwen2.5-7B vs SeaLLM-7B vs Vistral-7B) — chưa có cơ sở để chọn chắc chắn, cần test prompt thật trên 7 sample trước khi quyết định dùng model nào cho cả 100 file.
- Nguồn dữ liệu danh mục thuốc Bộ Y tế/ICD-10 tiếng Việt — cần xác nhận có tải được bản dùng được cho máy tính (dạng bảng, không phải chỉ PDF quét) trước khi thiết kế Tầng 1.
