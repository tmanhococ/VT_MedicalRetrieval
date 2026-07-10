# TVT/NER Y khoa tiếng Việt — Phương án tiếp cận: Structure-Aware Few-Shot Extraction

## 1. Ngữ cảnh đề bài

Bài toán yêu cầu xây dựng hệ thống AI xử lý văn bản y khoa tự do (free-form clinical text) — ghi chú bác sĩ, giấy xuất viện, kết quả xét nghiệm, hồ sơ EHR — để:

1. **Phát hiện** các khái niệm y tế xuất hiện trong văn bản
2. **Phân loại** khái niệm theo 5 nhãn: `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`, `CHẨN_ĐOÁN`, `THUỐC`
3. **Chuẩn hóa** (candidate mapping): CHẨN_ĐOÁN → mã ICD-10, THUỐC → mã RxNorm
4. **Suy luận ngữ cảnh** (assertions) cho CHẨN_ĐOÁN, THUỐC, TRIỆU_CHỨNG: `isNegated`, `isFamily`, `isHistorical`

Đây là bài toán nền tảng cho chuyển đổi số y tế — giúp dữ liệu lâm sàng phi cấu trúc có thể liên thông, khai thác cho chẩn đoán, nghiên cứu dịch tễ và ứng dụng AI y khoa.

### Input/Output

**Input:** 1 đoạn văn bản y khoa tự do (.txt), có thể chứa nhiều khái niệm đồng thời, viết tắt, ký hiệu chuyên ngành.

**Output:** danh sách dictionary, mỗi dict gồm:

```json
{
  "text": "cụm từ được xác định là khái niệm y tế",
  "position": [start, end],
  "type": "TRIỆU_CHỨNG | TÊN_XÉT_NGHIỆM | KẾT_QUẢ_XÉT_NGHIỆM | CHẨN_ĐOÁN | THUỐC",
  "assertions": ["isNegated" | "isFamily" | "isHistorical"],
  "candidates": ["mã ICD-10 hoặc RxNorm"]
}
```

Lưu ý:
- `position` tính theo **ký tự** (0 đến n-1), không phải theo token
- `assertions` chỉ áp dụng cho CHẨN_ĐOÁN, THUỐC, TRIỆU_CHỨNG
- `candidates` chỉ áp dụng cho CHẨN_ĐOÁN (ICD-10), THUỐC (RxNorm)

### Định dạng file nộp bài

Bài nộp là 1 file `output.zip`, giải nén ra thư mục `output/` chứa đúng **100 file `.json`**, đặt tên theo số thứ tự bản ghi input tương ứng (`1.json` … `100.json`, không phải `1.txt.json`):

```
output/
    ├── 1.json     # Nhãn của bản ghi 1
    ├── 2.json     # Nhãn của bản ghi 2
    ├── …
    └── 100.json
```

Mỗi file `N.json` chứa **1 danh sách (array)** các entity dict theo đúng schema ở mục trên, ví dụ:

```json
[
  {
    "text": "sốt cao",
    "position": [102, 109],
    "type": "TRIỆU_CHỨNG",
    "assertions": [],
    "candidates": []
  },
  {
    "text": "viêm phổi thùy",
    "position": [220, 234],
    "type": "CHẨN_ĐOÁN",
    "assertions": ["isHistorical"],
    "candidates": ["J18.1"]
  }
]
```

Các ràng buộc kỹ thuật cần tuân thủ khi sinh output để tránh bị chấm 0 điểm oan do lỗi định dạng (không phải lỗi mô hình):

- **Số lượng file phải khớp tuyệt đối** với 100 file input — thiếu file nào, entity của file đó coi như toàn bộ bị tính là miss (0 điểm cả 3 thành phần).
- **File rỗng vẫn phải tồn tại** dưới dạng `[]` (mảng rỗng hợp lệ), không được bỏ trống file hoặc để lỗi JSON — nếu văn bản không phát hiện được entity nào.
- `assertions` và `candidates`: nếu không áp dụng (VD entity `TÊN_XÉT_NGHIỆM`/`KẾT_QUẢ_XÉT_NGHIỆM` không có assertion, hoặc TRIỆU_CHỨNG không có candidate) → để **mảng rỗng `[]`**, không để `null` hay bỏ field, để tránh lỗi parse khi chấm điểm tự động.
- Encoding UTF-8 khi ghi file, đảm bảo giữ nguyên dấu tiếng Việt (không escape thành `\uXXXX` gây khó đọc, dù cả hai dạng thường vẫn parse JSON hợp lệ — ưu tiên `ensure_ascii=False` khi dùng `json.dump` trong Python để dễ kiểm tra thủ công).
- Nên có 1 bước **validate toàn bộ output.zip trước khi nộp**: parse lại cả 100 file bằng script riêng, kiểm tra đủ field bắt buộc, `position` nằm trong độ dài văn bản gốc, `type` thuộc đúng 5 giá trị cho phép — để bắt lỗi định dạng trước khi mất điểm oan.

### Ràng buộc tài nguyên

- Nếu dùng giải pháp LLM/agent tự host: model tối đa **9B tham số**, **không dùng API ngoài**
- Phần cứng thực tế: Windows + RTX 3050Ti (4GB VRAM) + Colab free + Kaggle free
- Tập test: 100 file `.txt`, **không có nhãn** — chỉ có input, không có ground truth để train

## 2. Cách chấm điểm (quan trọng — quyết định chiến lược đầu tư thời gian)

```
final_score = 0.3 · text_score + 0.3 · assertions_score + 0.4 · candidates_score
```

| Thành phần | Trọng số | Metric | Ý nghĩa |
|---|---|---|---|
| `text` | 30% | Word Error Rate (WER) | Đo độ chính xác khoanh vùng + nội dung cụm từ |
| `assertions` | 30% | Jaccard similarity | Đo độ đúng của tập ngữ cảnh (phủ định/tiền sử/người nhà) |
| `candidates` | 40% | Jaccard similarity (có trọng số theo độ dài ground truth) | Đo độ đúng của mã chuẩn hóa ICD/RxNorm |

**Luật phạt quan trọng:** đoán đúng `text` nhưng sai `type` → bị tính là 2 lỗi riêng biệt (thiếu 1 entity đúng + thừa 1 entity sai), cả 2 đều 0 điểm.

**Hệ quả cho chiến lược:** `candidates` (entity linking) đáng đầu tư nhiều thời gian nhất; nhầm lẫn `type` giữa các nhãn gần nghĩa (đặc biệt TRIỆU_CHỨNG vs CHẨN_ĐOÁN) nguy hiểm hơn nhiều so với lỗi biên `position` nhẹ.

## 3. Vấn đề cốt lõi: không có dữ liệu training, không có thời gian gán nhãn

Đề bài không cấp sẵn dữ liệu có nhãn — chỉ có 100 file input thô để nộp bài. Câu "*thí sinh cần sử dụng các giải pháp nằm ngoài lời giải chính để tạo thêm dữ liệu*" gợi ý phải tự tạo training set, nhưng trong điều kiện thực tế của dự án này, **không có đủ thời gian để gán nhãn thủ công hoặc bootstrapping dữ liệu quy mô lớn**.

→ Cần 1 hướng tiếp cận **không phụ thuộc fine-tuning trên dữ liệu có nhãn**, mà vẫn đạt độ chính xác xấp xỉ hoặc tốt hơn, bằng cách khai thác tối đa các nguồn tín hiệu "miễn phí" khác.

## 4. Phát hiện quan trọng từ dữ liệu mẫu thực tế (7 sample: 4, 14, 18, 70, 88, 93, 99.txt)

Khác với câu ví dụ tự do liền mạch trong đề bài gốc, dữ liệu thật quan sát được có **cấu trúc bán template rất mạnh**:

```
1. Tiền sử bệnh / Tiền sử bệnh nội khoa
   Các bệnh mãn tính: ...
   Tiền sử phẫu thuật / thủ thuật: ...
   Thuốc đang dùng trước khi nhập viện: [bullet list]
2. Bệnh sử hiện tại
   Lý do nhập viện: ...
   Triệu chứng hiện tại: [bullet list]
   Đặc điểm triệu chứng: ...
   Các sự kiện trước khi nhập viện: [bullet list]
3. Đánh giá tại bệnh viện
   Kết quả xét nghiệm: [bullet list dạng "tên_xn: giá trị"]
   Chẩn đoán: ...
```

### Các tín hiệu cấu trúc cụ thể quan sát được

| Header/pattern quan sát được | Suy luận gần như chắc chắn |
|---|---|
| Mục "Thuốc đang dùng trước khi nhập viện" / "Thuốc trước khi nhập viện" | Các bullet bên dưới → `THUỐC` + `isHistorical` |
| Mục "Triệu chứng hiện tại" | Các bullet bên dưới → `TRIỆU_CHỨNG`, không có assertion |
| Mục "Kết quả xét nghiệm" dạng `"ast 421"`, `"alt 336"` | Regex `(chữ) (số)` → tách `TÊN_XÉT_NGHIỆM` + `KẾT_QUẢ_XÉT_NGHIỆM` |
| Mục "Chẩn đoán" / "Chẩn đoán sơ bộ" | → `CHẨN_ĐOÁN` |
| Cụm "Không có", "âm tính", "Chưa từng" (file 70) | Trigger phủ định (NegEx) hoạt động tốt trên dữ liệu thật |

### Nhiễu cần xử lý ở mức câu chữ

- **Lỗi dính chữ** do OCR/copy-paste: `"Dùngmethadonekéo dài"`, `"Lý do vào việni"`, `"2. Tiền sử bệnh hiện tạiBệnh nhân..."` → cần bước chèn lại khoảng trắng trước khi xử lý tiếp
- **Code-switching tiếng Anh xen tiếng Việt**: `"taking 10 pills at a time"` (file 70) → PhoBERT thuần Việt xử lý kém phần này, cần cách tiếp cận chấp nhận đa ngôn ngữ

### Kết luận từ phát hiện này

Cấu trúc văn bản (header, bullet, vị trí đoạn) **gánh phần lớn "việc học ngữ cảnh"** mà lẽ ra fine-tuning NER phải đảm nhiệm. Có thể khai thác bằng regex/heuristic (miễn phí, không cần dữ liệu training) thay vì huấn luyện model.

## 5. Phương án đề xuất: Structure-Aware Few-Shot Extraction

### Sơ đồ pipeline

```
Bước 1 — Structural Parser (regex/heuristic thuần, không cần AI)
  Input thô → tách thành các "block" theo header quan sát được
  → mỗi block gắn "prior label" (gợi ý type + assertion mặc định)

Bước 2 — Few-shot LLM extraction (LLM ≤9B, self-host, KHÔNG fine-tune)
  Mỗi block đưa vào LLM kèm:
    - 1-2 ví dụ mẫu chuẩn (ví dụ đề bài + ví dụ tự soạn)
    - "prior label" từ Bước 1 làm gợi ý ngữ cảnh trong prompt
    - Ép output theo JSON Schema bằng constrained decoding (GBNF/Outlines)
      → loại bỏ hoàn toàn rủi ro JSON hỏng/không parse được
  → LLM trả JSON: text + type + assertion nháp (chưa cần position)

Bước 3 — Position resolver (thuần thuật toán, không cần AI)
  Fuzzy string matching để tìm lại vị trí ký tự chính xác của "text"
  trong input gốc (bù trừ lỗi chính tả/dính chữ, xử lý riêng lỗi
  "thiếu khoảng trắng" khác với lỗi chính tả thông thường)

Bước 4 — Rule-based NegEx chạy backstop/verify
  Chạy song song rule NegEx (pre/post-negation, pseudo-negation, termination)
  → nếu mâu thuẫn với assertion LLM đoán, rule ghi đè
    (rule đáng tin hơn cho pattern tường minh: "không có X", "âm tính X")

Bước 5 — Candidate linking (retrieval, không cần training)
  5a. Tầng ánh xạ từ điển: tên Việt/biệt dược/viết tắt → hoạt chất chuẩn
      (dùng danh mục thuốc Bộ Y tế / DrugBank Việt hoá, ICD-10 bản dịch
      tiếng Việt chính thức) làm anchor trước khi retrieval
  5b. Hybrid retrieval trên danh mục ICD-10/RxNorm:
      BM25 (lexical, bắt khớp từ khóa/viết tắt chính xác)
      + Embedding search đa ngôn ngữ (semantic, bắt đồng nghĩa/diễn đạt khác)
      → kết hợp điểm số (ensemble/rerank)
  5c. Fuzzy matching bổ sung cho ca lỗi chính tả nhẹ còn sót
```

### Vai trò từng thành phần

| Bước | Công cụ | Cần training data? | Cần GPU mạnh? |
|---|---|---|---|
| 1. Structural Parser | Regex/Python thuần | Không | Không |
| 2. Few-shot extraction | LLM ≤9B, prompt-only, constrained decoding | Không | Có (inference) |
| 3. Position resolver | Fuzzy string matching (thuật toán) | Không | Không |
| 4. NegEx backstop | Rule-based Python thuần | Không | Không |
| 5. Candidate linking | Từ điển ánh xạ + BM25 + Embedding + fuzzy | Không (dùng danh mục/model có sẵn) | Nhẹ |

## 6. Lý do chọn phương án này thay vì fine-tune

1. **Không cần dữ liệu training có nhãn** — giải quyết đúng nút thắt lớn nhất: không có thời gian gán nhãn.

2. **Cấu trúc văn bản thay thế phần lớn việc "học ngữ cảnh"** mà fine-tuning NER lẽ ra phải đảm nhiệm — tín hiệu từ header mạnh hơn nhiều so với việc suy luận thuần từ câu chữ.

3. **Giảm rủi ro chất lượng so với hướng bootstrapping dữ liệu tự sinh** — nếu tự sinh silver-data bằng LLM rồi fine-tune, chất lượng nhãn không đảm bảo; fine-tune trên nhãn nhiễu có thể cho kết quả **tệ hơn** few-shot LLM có prompt tốt + structural prior mạnh.

4. **Tiết kiệm thời gian cho các việc quyết định điểm số nhiều hơn** — không tốn công sưu tầm nguồn thô, train/debug model, tốn quota Colab/Kaggle cho training. Thời gian dồn vào:
   - Candidate linking (chiếm 40% điểm số)
   - Position resolver + xử lý nhiễu chính tả/dính chữ (ảnh hưởng trực tiếp đến `text_score`)

5. **Rule-based NegEx đã được xác nhận phù hợp trên dữ liệu thật** — các trigger phủ định ("không có", "âm tính", "chưa từng") xuất hiện đúng như thiết kế ban đầu, không cần thay đổi lớn.

## 7. Cách triển khai chi tiết

### 7.1. Structural Parser

- Xây danh sách header pattern đã quan sát được (và biến thể viết tắt/dính chữ): `"Tiền sử bệnh"`, `"Tiền sử bệnh nội khoa"`, `"Bệnh sử hiện tại"`, `"Đánh giá tại bệnh viện"`, `"Thuốc...trước khi nhập viện"`, `"Triệu chứng hiện tại"`, `"Kết quả xét nghiệm"`, `"Chẩn đoán"`, v.v.
- Dùng regex để dò header ở đầu dòng (chấp nhận số thứ tự `"1."`, `"2."`, dấu gạch đầu dòng)
- Mỗi block sau khi tách được gắn:
  - `section_type`: loại mục (tiền sử / hiện tại / đánh giá)
  - `default_type_hint`: gợi ý loại khái niệm (VD block "Thuốc..." → hint THUỐC)
  - `default_assertion_hint`: gợi ý assertion mặc định (VD block tiền sử → hint isHistorical)
- **Fallback bắt buộc**: nếu không khớp header nào → coi cả đoạn là 1 "unstructured block", không gán prior, để LLM tự xử lý hoàn toàn theo ngữ nghĩa câu văn (đề phòng tập test có văn bản dạng free-text thuần như ví dụ gốc trong đề)

### 7.2. Tiền xử lý nhiễu

- Viết heuristic phát hiện & tách chữ dính (VD dựa vào chuyển hoa→thường bất thường, hoặc dùng dictionary tiếng Việt để dò điểm ngắt hợp lý) trước khi đưa vào LLM
- Chuẩn hóa khoảng trắng, giữ nguyên bản gốc riêng để dùng cho bước resolve position (không sửa trực tiếp lên bản gốc)

### 7.3. Few-shot prompt

- Chọn 2-3 ví dụ mẫu bao phủ đủ 5 loại nhãn + cả 3 loại assertion (dùng chính ví dụ mẫu trong đề bài + tự soạn thêm 1-2 ví dụ có đủ isNegated/isFamily)
- Prompt cần nêu rõ: định nghĩa 5 nhãn, định nghĩa 3 loại assertion, gợi ý từ `default_type_hint`/`default_assertion_hint` của block hiện tại, yêu cầu output JSON có cấu trúc cố định
- Thêm ít nhất 1-2 **"hard example"** trong few-shot: câu có cả TRIỆU_CHỨNG và CHẨN_ĐOÁN xuất hiện gần nhau/dễ nhầm (VD: "ho kéo dài" là triệu chứng vs "viêm phế quản mãn" là chẩn đoán trong cùng câu), kèm chú thích ngắn trong prompt về ranh giới phân biệt 2 nhãn này — vì đây là cặp nhãn bị phạt kép nhiều nhất
- Yêu cầu tường minh trong prompt: **trích xuất `text` nguyên văn (verbatim)**, giữ nguyên lỗi chính tả/dấu câu/khoảng trắng của văn bản gốc, không tự sửa hay chuẩn hóa — giảm bớt (nhưng không loại bỏ hoàn toàn) hiện tượng LLM viết lại cụm từ
- **Ép JSON Schema bằng constrained decoding** (VD: GBNF grammar trong `llama.cpp`, hoặc thư viện `Outlines`) thay vì chỉ dặn suông trong prompt — với model ≤9B, đặc biệt khi phải sinh tiếng Việt có dấu, xác suất JSON lỗi (thiếu ngoặc, escape sai) nếu chỉ dựa vào prompt là không nhỏ; ép cứng schema đảm bảo 100% output parse được, không tốn thời gian debug parser ở bước sau
- Test prompt trên chính 7 sample đã có để hiệu chỉnh trước khi chạy full 100 file

### 7.4. Position resolver

- **Nguyên nhân gốc rễ cần lường trước**: LLM được train để sinh văn bản "sạch", nên có xu hướng tự thêm khoảng trắng/sửa dấu câu khi copy lại cụm từ từ input nhiễu (dính chữ) — đây là hạn chế bản chất của LLM, không phải lỗi thuật toán match. Vì vậy pipeline vẫn cần luôn đối chiếu lại với bản gốc thay vì tin thẳng text LLM sinh ra.
- Dùng fuzzy string matching (VD `difflib`, hoặc edit-distance có giới hạn) để tìm chuỗi `text` LLM trả về trong văn bản gốc (đã tính điểm bắt đầu/kết thúc theo ký tự)
- **Xử lý riêng lỗi "dính chữ"**: đây là lỗi *thiếu khoảng trắng*, khác bản chất với lỗi chính tả (thay/thêm/xóa ký tự) — nên dùng chi phí (cost) chèn khoảng trắng thấp hơn hẳn so với chi phí sửa ký tự khi tính edit-distance, để thuật toán match "khoan dung" đúng chỗ cần khoan dung
- Nếu match với độ tin cậy thấp (khả năng LLM viết lại khác nhiều so với gốc) → ưu tiên cắt lại theo đúng chuỗi gốc thay vì tin theo bản LLM sinh ra, để tránh lệch `text_score`
- **Không giao việc tính `position` cho LLM tự sinh**, kể cả khi thử truyền kèm chỉ số ký tự vào input prompt: model ≤9B nói chung xử lý tiếng Việt có dấu (ký tự Unicode tổ hợp) kém trong việc đếm/tính toán vị trí chính xác, rủi ro lệch `position` do LLM tự "đếm sai" còn cao hơn rủi ro của fuzzy matching thuần thuật toán — giữ nguyên cách tiếp cận match lại bằng thuật toán ở bước này là lựa chọn an toàn hơn

### 7.5. NegEx backstop

- Áp dụng bộ rule đã thiết kế (pre-negation, post-negation, pseudo-negation, termination trigger tiếng Việt) trên toàn văn bản, độc lập với LLM
- So sánh với assertion LLM trả về theo từng entity → nếu rule phát hiện trigger phủ định/tiền sử mạnh mà LLM bỏ sót, ưu tiên kết quả rule (đáng tin hơn cho pattern tường minh)

### 7.6. Candidate linking

**Vấn đề cần giải quyết trước khi retrieval**: RxNorm là hệ mã xây cho tên thuốc/hoạt chất tiếng Anh (thị trường Mỹ); ICD-10 gốc cũng bằng tiếng Anh. Văn bản input lại là tiếng Việt, có thể chứa tên biệt dược Việt hóa, tên thương mại địa phương, hoặc viết tắt. Nếu chạy embedding đa ngôn ngữ trực tiếp trên chuỗi tiếng Việt thô để so khớp sang mã tiếng Anh, tỷ lệ map sai sẽ cao — đây là rủi ro lớn nhất của 40% điểm số, cần một tầng trung gian trước khi retrieval, không chỉ dựa vào chọn embedding tốt hơn:

- **Tầng 0 — Giải nghĩa viết tắt (mới bổ sung)**: trước khi tra mã, mở rộng các viết tắt hay gặp trong bệnh án (VD "AH" → "ảo thanh/ảo giác thính giác", "COPD", "MICU"...) bằng một **từ điển viết tắt y khoa tự xây** (rule-based, tra cứu ngữ cảnh câu lân cận để chọn nghĩa đúng khi viết tắt đa nghĩa). Đây là bản rút gọn, ít tốn tài nguyên hơn của cách dùng model BERT chuyên khoa (ClinicalBERT/ViHealthBERT) để giải nghĩa viết tắt — cùng mục tiêu nhưng không cần thêm 1 model nặng chạy trên máy vốn đã hạn chế VRAM. Nếu còn dư tài nguyên/thời gian, có thể thử nghiệm thêm hướng dùng model chuyên khoa cho các viết tắt mà từ điển rule-based không phủ được.
- **Tầng 1 — Từ điển ánh xạ (anchor layer)**:
  - THUỐC: xây/dùng bảng ánh xạ tên biệt dược Việt/viết tắt → hoạt chất chuẩn quốc tế (INN) → mã RxNorm, tham khảo danh mục thuốc lưu hành của Bộ Y tế hoặc DrugBank
  - CHẨN_ĐOÁN: dùng bảng **ICD-10 bản dịch tiếng Việt chính thức** (Bộ Y tế/WHO) làm anchor thay vì chỉ so khớp tiếng Việt ↔ tiếng Anh qua embedding
  - Nếu entity khớp thẳng vào từ điển anchor → nhận mã ngay, không cần qua bước retrieval mờ ở dưới
- **Tầng 2 — Hybrid retrieval** cho các ca không khớp thẳng từ điển (áp dụng chủ yếu cho CHẨN_ĐOÁN/ICD-10, vì RxNorm xử lý riêng ở Tầng 2':
  - **BM25 (lexical search)**: bắt các trường hợp khớp từ khóa/viết tắt gần đúng chính xác về mặt ký tự
  - **Embedding search đa ngôn ngữ chuyên biệt y khoa (semantic search)**: ưu tiên dùng embedding huấn luyện riêng cho việc nối thuật ngữ y khoa đồng nghĩa liên ngôn ngữ (kiểu SapBERT — dựa trên XLM-RoBERTa, ~270M tham số, nhẹ hơn nhiều so với LLM chính 9B nên vẫn chạy được tuần tự trên 4GB VRAM) thay vì embedding đa ngôn ngữ tổng quát; loại embedding này bắt tốt các trường hợp diễn đạt khác hẳn từ vựng gốc (VD lỗi dịch thô "các băng nhóm oligoclonal" vẫn kéo được về đúng khái niệm) mà BM25 chắc chắn bỏ lỡ
  - **Kết hợp điểm số 2 nguồn bằng Reciprocal Rank Fusion (RRF)** thay vì "ensemble" chung chung: gộp theo nghịch đảo thứ hạng của mỗi nguồn ($s(d) = \frac{1}{k+r_{BM25}(d)} + \frac{1}{k+r_{embedding}(d)}$, $k$≈60) — công thức đơn giản, không cần chuẩn hóa 2 loại điểm số khác bản chất (điểm BM25 vs cosine similarity) về cùng thang đo, dễ cài đặt và rẻ về compute
  - **(Tùy chọn, nếu dư thời gian/tài nguyên) Cross-Encoder rerank** cho danh sách Top-K sau RRF: model đọc đồng thời cụm từ tiếng Việt và mô tả mã tiếng Anh để xếp hạng lại — chính xác hơn Bi-encoder/RRF thuần nhưng tốn thêm 1 lượt inference cho mỗi ứng viên, cần đo thử ngân sách compute (mục 7.8) trước khi quyết định thêm bước này, tránh vượt quota Colab/Kaggle
- **Tầng 2' — Riêng cho THUỐC (RxNorm), tách khỏi luồng embedding/BM25 ở trên**: vì sai lệch hoạt chất/hàm lượng nguy hiểm hơn nhiều so với sai một mã chẩn đoán gần nghĩa, nên xử lý bằng cách trích xuất có cấu trúc thay vì tìm kiếm ngữ nghĩa mờ:
  - Dùng chính LLM đã có trong pipeline (Bước 2) để xuất thêm các trường có cấu trúc cho mỗi entity THUỐC: tên biệt dược, hoạt chất, hàm lượng, dạng bào chế (ép theo JSON Schema như đã làm ở 7.3)
  - So khớp cứng (deterministic) các trường này với **bộ dữ liệu RxNorm tải về máy, dùng offline** (RxNorm cung cấp file dữ liệu đầy đủ dùng miễn phí) — không gọi API RxNorm qua mạng, để tránh vướng ràng buộc "không dùng API ngoài" và tránh phụ thuộc kết nối mạng khi chạy trên Colab/Kaggle
  - Nếu LLM trả về thuốc đa thành phần (VD thuốc phối hợp 2 hoạt chất) → gộp đúng các hoạt chất trước khi so khớp, tránh map nhầm sang mã của thuốc đơn thành phần
- **Tầng 3 — Fuzzy matching** bổ sung cuối cùng cho ca lỗi chính tả nhẹ còn sót lại sau các tầng trên
- Cân nhắc thêm bước rerank có xét số liều lượng/hàm lượng (đặc biệt với THUỐC) để tránh nhầm mã giữa các hàm lượng khác nhau của cùng hoạt chất

### 7.7. Tự kiểm định trước khi chạy full 100 file (không có ground truth)

Vì tập test không có nhãn, cần tự tạo một vòng kiểm định nội bộ để biết pipeline đang yếu ở đâu trước khi tốn thời gian/quota chạy hết 100 file:

- Gán nhãn thủ công (tay) cho chính 7 sample đã có (4, 14, 18, 70, 88, 93, 99) theo đúng schema output — khối lượng nhỏ, khả thi trong thời gian có hạn
- Viết script đo thử **đúng công thức chấm điểm đề bài** (WER cho `text`, Jaccard cho `assertions`, Jaccard có trọng số cho `candidates`) trên 7 sample này, chạy pipeline và so với nhãn tay
- Dùng kết quả đo để xác định bước nào đang kéo điểm xuống nhiều nhất (VD: `candidates` thấp do từ điển ánh xạ chưa đủ → ưu tiên bổ sung từ điển; hay `text` thấp do position resolver còn lệch → ưu tiên chỉnh fuzzy matching) — tránh phân bổ thời gian theo cảm tính
- Lặp lại vòng hiệu chỉnh này trước khi chạy chính thức trên 100 file, để tránh phải chạy lại nhiều lần (tốn quota Colab/Kaggle free)

### 7.8. Ước lượng ngân sách tính toán (compute budget)

- Đo thời gian inference thực tế trên phần cứng sẵn có (RTX 3050Ti 4GB, model đã quantize VD GGUF Q4) cho 1 block, rồi nhân với tổng số block ước tính trên 100 file (không phải nhân theo số file, vì mỗi file có thể tách thành nhiều block)
- Tính thêm chi phí của các bước gọi LLM lặp lại nếu có (VD: gọi lại khi JSON lỗi trước khi có constrained decoding, hoặc bước rerank) — dù ở phương án này JSON lỗi gần như được loại bỏ nhờ constrained decoding, vẫn nên có ngân sách dự phòng
- So sánh tổng thời gian ước tính với giới hạn quota thực tế của Colab/Kaggle free (số giờ GPU liên tục, khả năng bị ngắt phiên) để quyết định có cần chia nhỏ batch chạy nhiều phiên hay không — nên đo và tính sớm, tránh phát hiện ra vấn đề khi đã gần hết thời gian làm bài

## 8. Các lưu ý quan trọng

- **Rủi ro phụ thuộc cấu trúc**: 7 sample hiện có khá nhất quán về template, nhưng tập test 100 file có thể chứa văn bản không theo cấu trúc này (gần với dạng free-text thuần trong ví dụ gốc đề bài). Structural Parser bắt buộc phải có fallback an toàn, không được "gãy" khi gặp văn bản lạ.

- **Chi phí inference LLM tính theo số block, không phải số file** — cần đo thử tốc độ thực tế trên phần cứng sẵn có (3050Ti/Colab/Kaggle) để ước lượng tổng thời gian chạy hết 100 file, tránh vượt quota hoặc quá thời gian cho phép.

- **Chất lượng few-shot prompt quan trọng hơn nhiều so với thông thường** — vì đây là thành phần thay thế cho việc model "học" từ dữ liệu, không có hàng trăm ví dụ để bù trừ sai sót như khi fine-tune.

- **Sai `type` bị phạt kép (0 điểm x2)** — ưu tiên kiểm tra kỹ các cặp nhãn dễ nhầm (đặc biệt TRIỆU_CHỨNG vs CHẨN_ĐOÁN) trong ví dụ few-shot và trong bước review.

- **Position resolver là nơi dễ mất điểm oan nhất** nếu không cẩn thận — LLM có thể viết lại cụm từ hơi khác bản gốc (thừa/thiếu dấu cách, chuẩn hóa chữ hoa/thường), cần luôn ưu tiên khớp lại với bản gốc thay vì dùng thẳng text LLM sinh ra. Không nên "sửa lỗi" này bằng cách để LLM tự tính `position` (kể cả khi truyền index vào prompt) — model nhỏ dễ đếm sai vị trí ký tự tiếng Việt có dấu hơn là fuzzy matching thuần thuật toán bị lệch.

- **Từ điển ánh xạ Việt–chuẩn quốc tế là nền tảng của `candidates` (40% điểm)** — retrieval (embedding/BM25) chỉ nên là lớp xử lý ngoại lệ cho ca không khớp thẳng từ điển, không phải lớp xử lý chính; nếu bỏ qua tầng từ điển ánh xạ tên biệt dược Việt/ICD-10 tiếng Việt, dù retrieval có tốt đến đâu vẫn dễ map sai vì lệch ngôn ngữ gốc của hệ mã.

- **Làm rõ trước ràng buộc "không dùng API ngoài"**: xác nhận với ban tổ chức liệu ràng buộc này áp dụng cho toàn bộ pipeline hay chỉ áp dụng lúc chấm điểm cuối cùng — ảnh hưởng đến việc có thể dùng LLM mạnh hơn cho một số bước thử nghiệm/hiệu chỉnh offline hay không.

- **Ưu tiên đầu tư thời gian theo đúng trọng số điểm**: candidates (40%) > text/assertions (30% mỗi loại) — nên dành phần lớn thời gian còn lại cho việc tinh chỉnh candidate linking (đặc biệt từ điển ánh xạ) và xử lý nhiễu ảnh hưởng `text_score`, thay vì cố tối ưu thêm phần đã đủ tốt.

- **Checklist trước khi nộp `output.zip`**: đủ 100 file, tên file đúng số thứ tự, không có file JSON lỗi cú pháp, `assertions`/`candidates` luôn là mảng (không `null`), `position` nằm trong độ dài văn bản gốc tương ứng, `type` chỉ nhận đúng 5 giá trị cho phép — nên chạy 1 script validate độc lập, tách biệt với pipeline sinh output, để bắt lỗi định dạng trước khi nộp.
