import os
from importlib.metadata import version as pkg_version
from typing import Any

import numpy as np
from paddleocr import PaddleOCR

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _paddleocr_major_version() -> int:
    try:
        return int(pkg_version("paddleocr").split(".")[0])
    except Exception:
        return 2


def _box_extremes(box: Any) -> tuple[float, float]:
    if box is None:
        return 0.0, 0.0
    points = np.asarray(box, dtype=float)
    if points.ndim == 1 and points.size >= 2:
        return float(points[0]), float(points[1])
    if points.ndim == 2 and points.shape[1] >= 2:
        return float(points[:, 0].min()), float(points[:, 1].min())
    return 0.0, 0.0


class PaddleOCRProcessor:
    """Wrapper around PaddleOCR with support for 2.x and 3.x APIs."""

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = False,
        show_log: bool = False,
        use_gpu: bool = False,
    ):
        self._major_version = _paddleocr_major_version()

        if self._major_version >= 3:
            # PaddleOCR 3.x removed show_log/use_gpu from the constructor.
            self.ocr = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=use_angle_cls,
            )
        else:
            self.ocr = PaddleOCR(
                lang=lang,
                use_angle_cls=use_angle_cls,
                show_log=show_log,
                use_gpu=use_gpu,
            )

    def _run_ocr(self, np_image: np.ndarray) -> list[Any]:
        if self._major_version >= 3:
            return self.ocr.predict(np_image)
        return self.ocr.ocr(np_image, det=True, rec=True)

    def _iter_items(self, results: list[Any]) -> list[dict[str, Any]]:
        if not results:
            return []

        if self._major_version >= 3:
            page = results[0]
            rec_texts = page["rec_texts"]
            rec_polys = page["rec_polys"]
            items: list[dict[str, Any]] = []
            for text, poly in zip(rec_texts, rec_polys):
                text = str(text).strip()
                if not text:
                    continue
                x_min, y_min = _box_extremes(poly)
                items.append({"text": text, "x_min": x_min, "y_min": y_min})
            return items

        items = []
        for result in results:
            if not result:
                continue
            for box, (text, _confidence) in result:
                text = str(text).strip()
                if not text:
                    continue
                x_min, y_min = _box_extremes(box)
                items.append({"text": text, "x_min": x_min, "y_min": y_min})
        return items

    def _layout_from_items(self, items: list[dict[str, Any]], threshold: int) -> str:
        items.sort(key=lambda item: item["y_min"])

        lines: list[list[dict[str, Any]]] = []
        current_line: list[dict[str, Any]] = []
        current_y: float | None = None

        for item in items:
            if current_y is None:
                current_line = [item]
                current_y = item["y_min"]
                continue

            if abs(item["y_min"] - current_y) <= threshold:
                current_line.append(item)
            else:
                lines.append(current_line)
                current_line = [item]
                current_y = item["y_min"]

        if current_line:
            lines.append(current_line)

        output_lines: list[str] = []
        for line in lines:
            line.sort(key=lambda item: item["x_min"])
            line_text = ""
            prev_x: float | None = None

            for item in line:
                if prev_x is not None:
                    gap = max(0, int((item["x_min"] - prev_x) / 10))
                    line_text += " " * gap
                line_text += item["text"]
                prev_x = item["x_min"] + len(item["text"]) * 7

            output_lines.append(line_text)

        return "\n".join(output_lines)

    def extract_text_from_pil(self, pil_image) -> str:
        np_image = np.array(pil_image)
        items = self._iter_items(self._run_ocr(np_image))
        return "\n".join(item["text"] for item in items)

    def extract_text_layout_from_pil(self, pil_image, threshold: int) -> str:
        np_image = np.array(pil_image)
        items = self._iter_items(self._run_ocr(np_image))
        return self._layout_from_items(items, threshold)
