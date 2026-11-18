"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, ROOT, parse_yolo_line


def fix(ds, issues, cfg):
    if ds.root.resolve() == (ROOT / "data" / "raw").resolve():
        raise ValueError("Chỉ sửa bản sao làm việc, không sửa data/raw")

    errors = [issue for issue in issues if issue.severity == ERROR]
    fixes = []

    def record(image, line, action, before, after):
        fixes.append(dict(image=image, line=line, action=action, before=before, after=after))

    # Loại ảnh trước để không sửa các dòng nhãn của ảnh đã bị loại.
    removed = {issue.image for issue in errors
               if issue.code.startswith("DUP_IMAGE_")
               or issue.code in {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}}
    for stem in sorted(removed):
        image = ds.image_path(stem)
        label = ds.label_path(stem)
        existing = [p for p in (image, label) if p is not None and p.exists()]
        if existing:
            before = ", ".join(p.name for p in existing)
            for path in existing:
                path.unlink()
            record(stem, None, "remove_image", before, "(đã loại ảnh và nhãn)")

    if any(issue.code == "BAD_CLASS_NAME" for issue in errors):
        path = ds.yaml_path
        before = path.read_text(encoding="utf-8") if path.exists() else ""
        data = yaml.safe_load(before) or {}
        names = dict(enumerate(cfg["classes"]))
        if data.get("names") != names:
            data["names"] = names
            after = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
            path.write_text(after, encoding="utf-8")
            record("data.yaml", None, "rename_classes", before, after)

    needs_label = {issue.image for issue in errors if issue.code == "MISSING_LABEL_FILE"}
    for stem in sorted(needs_label - removed):
        if ds.image_path(stem) is not None and not ds.label_path(stem).exists():
            record(stem, None, "needs_label", "(không có file)", "chuyển người gán")

    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
    grouped = defaultdict(lambda: defaultdict(set))
    for issue in errors:
        if issue.image not in removed and issue.line is not None:
            if issue.code in drop_codes or issue.code == "BOX_OUT_OF_IMAGE":
                grouped[issue.image][issue.line].add(issue.code)

    for stem, line_codes in sorted(grouped.items()):
        path = ds.label_path(stem)
        if not path.exists():
            continue
        before_file = path.read_text(encoding="utf-8", errors="replace")
        output = []
        changed = False
        # Duyệt toàn bộ dòng gốc, kể cả dòng trống; chỉ ghi file một lần.
        for number, text in enumerate(before_file.splitlines(), 1):
            codes = line_codes.get(number, set())
            if codes & drop_codes:
                record(stem, number, "drop_line", text, "")
                changed = True
                continue
            if "BOX_OUT_OF_IMAGE" in codes:
                try:
                    class_id, x, y, w, h = parse_yolo_line(text)
                except (ValueError, OverflowError):
                    record(stem, number, "drop_line", text, "")
                    changed = True
                    continue
                left, right = max(0.0, x - w / 2), min(1.0, x + w / 2)
                top, bottom = max(0.0, y - h / 2), min(1.0, y + h / 2)
                if right <= left or bottom <= top:
                    record(stem, number, "drop_line", text, "")
                    changed = True
                    continue
                if left != x - w / 2 or right != x + w / 2 or top != y - h / 2 or bottom != y + h / 2:
                    coords = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
                    after = f"{class_id} " + " ".join(f"{value:.12g}" for value in coords)
                    record(stem, number, "clip_box", text, after)
                    text = after
                    changed = True
            output.append(text)
        if changed:
            path.write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")
    return fixes
