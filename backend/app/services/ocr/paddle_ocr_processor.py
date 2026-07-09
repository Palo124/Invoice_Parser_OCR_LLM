import os
from importlib.metadata import version as pkg_version
from typing import Any

import numpy as np
from paddleocr import PaddleOCR

from app.config import settings
from app.services.ocr.layout_reconstruction import layout_text_from_items

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _paddleocr_major_version() -> int:
    try:
        return int(pkg_version("paddleocr").split(".")[0])
    except Exception:
        return 2


def _box_bounds(box: Any) -> tuple[float, float, float, float]:
    if box is None:
        return 0.0, 0.0, 0.0, 0.0
    points = np.asarray(box, dtype=float)
    if points.ndim == 1 and points.size >= 2:
        x = float(points[0])
        y = float(points[1])
        return x, y, x, y
    if points.ndim == 2 and points.shape[1] >= 2:
        x_min = float(points[:, 0].min())
        y_min = float(points[:, 1].min())
        x_max = float(points[:, 0].max())
        y_max = float(points[:, 1].max())
        return x_min, y_min, x_max, y_max
    return 0.0, 0.0, 0.0, 0.0


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
                x_min, y_min, x_max, _y_max = _box_bounds(poly)
                items.append({"text": text, "x_min": x_min, "y_min": y_min, "x_max": x_max})
            return items

        items = []
        for result in results:
            if not result:
                continue
            for box, (text, _confidence) in result:
                text = str(text).strip()
                if not text:
                    continue
                x_min, y_min, x_max, _y_max = _box_bounds(box)
                items.append({"text": text, "x_min": x_min, "y_min": y_min, "x_max": x_max})
        return items

    def extract_text_from_pil(self, pil_image) -> str:
        np_image = np.array(pil_image)
        items = self._iter_items(self._run_ocr(np_image))
        return "\n".join(item["text"] for item in items)

    def extract_text_layout_from_pil(self, pil_image, threshold: int) -> str:
        np_image = np.array(pil_image)
        items = self._iter_items(self._run_ocr(np_image))
        return layout_text_from_items(
            items,
            line_threshold=threshold,
            page_width=pil_image.size[0],
            column_split_enabled=settings.ocr_column_split_enabled,
            column_gap_ratio=settings.ocr_column_gap_ratio,
            char_width=settings.ocr_layout_char_width,
            space_divisor=settings.ocr_layout_space_divisor,
            blank_line_y_multiplier=settings.ocr_layout_blank_line_multiplier,
        )
