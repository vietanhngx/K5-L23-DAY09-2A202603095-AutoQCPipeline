"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def run(ds, cfg, preds=None):
    limits = cfg["second_opinion"]
    if preds is None:
        preds = get_preds(ds, cfg, "checker")
    issues = []
    for image in ds.image_files():
        stem = image.stem
        if stem not in preds or not ds.label_path(stem).exists():
            continue
        size = ds.image_size(stem)
        if size is None:
            continue
        width, height = size

        # Giữ số dòng gốc cho FALSE_POS; bỏ dòng hỏng và ID ngoài danh sách.
        labels = []
        for line, text in ds.read_label_lines(stem) or []:
            try:
                box = parse_yolo_line(text)
            except (ValueError, OverflowError):
                continue
            if 0 <= box[0] < len(ds.classes):
                labels.append((line, box))

        predictions = preds[stem]
        for class_id, confidence, x, y, w, h in predictions:
            if confidence < limits["conf"] or w * width * h * height < limits["min_area_px"]:
                continue
            candidates = []
            for _, label in labels:
                overlap = iou_xywh((x, y, w, h), label[1:])
                if overlap >= limits["iou_missing"]:
                    candidates.append((label[0], overlap))

            if not candidates:
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_MISSING", stem,
                                    f"model thấy lớp {class_id} nhưng không có box nhãn khớp",
                                    extra={"conf": confidence}))
                continue

            same_class = [overlap for label_class, overlap in candidates if label_class == class_id]
            if same_class:
                best = max(same_class)
                if best < limits["iou_loose"]:
                    issues.append(Issue("second_opinion", WARN, "POSSIBLE_LOOSE_BOX", stem,
                                        f"box cùng lớp {class_id} khớp chưa sát: IoU={best:.4f}",
                                        extra={"conf": confidence, "iou": best}))
            else:
                best = max(overlap for _, overlap in candidates)
                if best >= limits["iou_match"]:
                    issues.append(Issue("second_opinion", WARN, "POSSIBLE_WRONG_CLASS", stem,
                                        f"model dự đoán lớp {class_id}, box nhãn khớp vị trí nhưng khác lớp",
                                        extra={"conf": confidence, "iou": best}))

        # FALSE_POS dùng ngưỡng riêng và không lọc diện tích dự đoán.
        fp_predictions = [prediction for prediction in predictions if prediction[1] >= limits["fp_conf"]]
        for line, label in labels:
            best = max((iou_xywh(label[1:], prediction[2:]) for prediction in fp_predictions), default=0.0)
            if not any(iou_xywh(label[1:], prediction[2:]) >= limits["iou_missing"]
                       for prediction in fp_predictions):
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_FALSE_POS", stem,
                                    "box nhãn không khớp dự đoán nào đủ ngưỡng tin cậy",
                                    line=line, extra={"iou": best}))
    return issues
