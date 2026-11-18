# Báo cáo Day 09 — Nguyễn Việt Anh, 2A202603095

## 1. Kết luận trước

Kết luận là **Hold**. Có 224 ảnh đầu vào, 19 ảnh bị loại và 205 ảnh được xuất. Sau khi sửa tự động và duyệt mô phỏng, không còn lỗi bắt buộc nhưng vẫn có **38/205 ảnh (18,54%)** còn cảnh báo chưa được duyệt, vượt ngưỡng 10% nên kết luận là Hold. Dataset này được dùng làm bản nháp để train so sánh raw với clean, chưa đủ điều kiện phát hành.

## 2. Lỗi và phân việc

Số liệu lấy từ `02_error_list_v1`, trên 224 ảnh đầu vào và 205 ảnh sau intake. Một ảnh có thể có nhiều lỗi hoặc cảnh báo.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| `CORRUPT_IMAGE` | 2 ảnh | Máy — intake | Ảnh hỏng, không đọc được |
| `BAD_SIZE` | 3 ảnh | Máy — intake | Ảnh nhỏ hơn quy định |
| `DUP_IMAGE_EXACT` | 8 ảnh | Máy — intake | File ảnh trùng từng byte |
| `DUP_IMAGE_NEAR` | 2 ảnh | Máy — intake | Ảnh gần trùng theo ngưỡng đã chọn |
| `LEAK_TEST` | 4 ảnh | Máy — intake | Trùng ảnh test/val, gây rò rỉ dữ liệu |
| `MISSING_LABEL_FILE` | 6 ảnh | Người | Máy không thể tự tạo nhãn đúng |
| `BAD_FORMAT` | 3 dòng | Máy — autofix | Xóa dòng sai định dạng |
| `BAD_CLASS_ID` | 6 dòng | Máy — autofix | Xóa dòng có ID lớp không hợp lệ |
| `BOX_ZERO` | 4 dòng | Máy — autofix | Xóa box có chiều rộng hoặc cao không dương |
| `DUP_BOX` | 7 dòng | Máy — autofix | Xóa box lặp ở dòng sau |
| `BOX_OUT_OF_IMAGE` | 8 dòng | Máy — autofix | Cắt box về trong ảnh |
| `BAD_CLASS_NAME` | 2 lượt báo trong YAML | Máy — autofix | Sửa tên và thứ tự lớp theo cấu hình |
| `EMPTY_LABEL` | 12 ảnh | Người | Kiểm tra ảnh nền hay bị thiếu nhãn |
| `BOX_HUGE` | 3 dòng | Người | Box lớn chưa chắc là sai |
| `BOX_TINY` | 6 dòng | Người | Box nhỏ cần xem có đúng vật thể không |
| `CLASS_RARE` | 4 cảnh báo cấp tập dữ liệu | Người | Kiểm tra phân bố lớp và ID bất thường |
| `POSSIBLE_MISSING` | 53 cảnh báo | Người | Xác minh vật thể có bị bỏ sót không |
| `POSSIBLE_LOOSE_BOX` | 28 cảnh báo | Người | Kiểm tra box có ôm sát vật thể không |
| `POSSIBLE_WRONG_CLASS` | 18 cảnh báo | Người | Xác minh lớp đúng theo nội dung ảnh |
| `POSSIBLE_FALSE_POS` | 29 cảnh báo | Người | Kiểm tra box có gán vào vật không cần nhãn không |

- Pipeline tính điểm ưu tiên cho từng ảnh bằng hàm `build_queue()` trong `pipeline/steps/review_queue.py`: cộng điểm ở dòng 14, 18, 21 rồi xếp giảm dần ở dòng 24. Vì vậy, ảnh `img_0020` đứng đầu với điểm cao nhất là **11,524**.

- Với ngân sách 80 ảnh, người duyệt mô phỏng xem được **94 cảnh báo ngữ nghĩa**.

## 3. Calibration (2 ngưỡng)

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
| --- | --- | --- | --- | --- |
| `duplicates.phash_dist` | 2 | 4 | 100% / 50% → 100% / 66,7% | Bắt thêm ảnh gần trùng, hạn chế loại nhầm ảnh tốt. |
| `second_opinion.conf` | 0.25 | 0.6 | Thiếu nhãn: 50% / 100% → 57% / 100%; nhầm lớp: 56% / 83% → 62% / 83% | Giảm báo động giả, giữ khả năng bắt thiếu nhãn và nhầm lớp trên calib. |

- Ngưỡng 16 đạt precision 100% trên calib nhưng báo nhầm hai cặp `0008/0011` và `0113/0177` trên raw. Tôi chọn ngưỡng 4, chấp nhận bỏ sót cặp `0074/0084` để tránh loại nhầm ảnh tốt.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được

- **Gate chặn đúng:** Khi chạy với `--budget 0`, còn 6 lỗi thiếu file nhãn nên pipeline kết luận **Rework**, mã thoát **20** và không xuất dataset (`Test-Path` trả `False`).
- **Bỏ sót ảnh gần trùng:** Ngưỡng 4 không bắt được cặp `img_0074/img_0084`, dù cùng nội dung và chỉ khác kích thước.
- **Bỏ sót nhãn thừa:** Tôi quan sát thấy động cơ ở đuôi thuyền trong `img_0002` bị gán `person`, nhưng checker không cảnh báo và ảnh không được đưa vào lượt duyệt.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)

Train YOLO26n trên RTX 4050, 20 epoch, seed 0 và 1, ảnh 640, batch 8. Bảng lấy từ `submission/results/comparison.md`:

| biến thể | #ảnh train | số seed | map50 (TB ± độ lệch) | map50_95 (TB ± độ lệch) | precision (TB ± độ lệch) | recall (TB ± độ lệch) |
|---|---|---|---|---|---|---|
| raw | 224 | 2 | 0.698 ± 0.008 | 0.485 ± 0.021 | 0.792 ± 0.082 | 0.587 ± 0.014 |
| clean | 205 | 2 | 0.710 ± 0.002 | 0.510 ± 0.000 | 0.796 ± 0.051 | 0.610 ± 0.020 |

- **Kết quả:** mAP50 của clean tăng **0.011** so với raw, lớn hơn độ lệch chuẩn giữa seed của raw (**0.008**) và clean (**0.002**). Kết quả bước đầu nghiêng về clean, nhưng mới có 2 seed nên chưa đủ để khẳng định cải thiện ổn định.
- **Giả thuyết 1:** Sửa nhãn và loại ảnh không đạt có thể giúp clean học tốt hơn. Cách kiểm: train thêm biến thể chỉ sửa tự động để so với clean có người duyệt.
- **Giả thuyết 2:** Raw có ảnh trùng test nên kết quả có thể bị ảnh hưởng bởi rò rỉ. Cách kiểm: đối chiếu các ảnh `LEAK_TEST` và thử đánh giá cả hai model trên cùng tập test đã loại ảnh rò rỉ.
- **Việc tiếp theo:** Chạy thêm seed và đánh giá trên tập test độc lập lớn hơn. Hai giả thuyết trên chưa được chứng minh và các cách kiểm đề xuất chưa được thực hiện.

## 6. Quyết định và ghi chú AI

- **Quyết định cuối là Hold.** Để thành Release, cần không còn lỗi bắt buộc và tỷ lệ ảnh còn cảnh báo chưa duyệt dưới 10%. Hiện điều kiện thứ hai chưa đạt.
- **Prompt và kết quả:** Tôi yêu cầu AI đọc docstring và test để triển khai từng module, làm rõ đầu vào, quy tắc kiểm, kết quả trả về và phạm vi được sửa. AI viết code từng module và giải thích cách hoạt động.
- **Cách kiểm:** Tôi xem phần giải thích, chạy test từng module, toàn bộ 79 test, các ca bổ sung trên dữ liệu tạm và calibration; đồng thời xem ảnh để đối chiếu.
- **Điểm AI còn thiếu:** AI đề xuất thử `phash_dist=16` dựa trên calib, nhưng khi xem raw tôi phát hiện hai cặp bị báo trùng nhầm. Tôi chọn ngưỡng 4 và kiểm tra lại, chấp nhận bỏ sót một cặp gần trùng.

**Câu hỏi phụ (tùy chọn — CVAT)**

- **CVAT:** Lọc `width < 20` thấy 9 box: `img_0020` có 3, `img_0030` có 3, `img_0053` có 1, `img_0135` có 2. Các box này không bị báo `BOX_TINY` ở báo cáo cuối vì pipeline dùng ngưỡng cạnh ngắn dưới 12 px.
- `can_review` đánh dấu box cần người xem lại, tương tự hàng đợi duyệt của pipeline.

- Ảnh minh chứng: [3 box khớp bộ lọc](reports/cvat_img_0020_boxes.png), [điều kiện Width < 20](reports/cvat_img_0020_filter.png).

**Vấn đề khi chạy kiểm tra bài nộp trên Windows**

- **Lỗi gặp phải:** `check_submission.py` báo 4 lỗi dù 79 test đều đạt. Hai lỗi do đường dẫn Windows dùng dấu `\` không khớp dấu `/` trong danh sách file bảo vệ, khiến `calibrate.py` bị nhận nhầm là code không được đọc đáp án calib. Hai lỗi còn lại do script train tính hash manifest trên byte gốc CRLF, còn checker đổi CRLF thành LF trước khi tính hash.
- **Cách sửa thử trong `scripts/check_submission.py`:**
  - Dòng 74: thay `str(p.relative_to(ROOT)) not in prot` bằng `p.relative_to(ROOT).as_posix() not in prot` để thống nhất dấu đường dẫn.
  - Dòng 133: thay `if r.get("manifest_sha1") != sha1(mf):` bằng `if r.get("manifest_sha1") != hashlib.sha1(mf.read_bytes()).hexdigest():` để tính hash giống script train.
- **Kết quả:** Sau khi sửa thử, chạy kiểm tra được **79 passed, 0 lỗi, 0 cảnh báo**. Sau đó đã khôi phục checker về đúng bản gốc và xác nhận hash không thay đổi; lệnh kiểm tra gốc vẫn báo 4 lỗi trên. Vấn đề cần được mentor xác nhận, không coi bản sửa thử là kết quả đạt của checker gốc.
