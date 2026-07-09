def _item_x_max(item: dict, char_width: float) -> float:
    if item.get("x_max") is not None:
        return float(item["x_max"])
    return float(item["x_min"]) + len(str(item["text"])) * char_width


def _spaces_for_gap(gap_pixels: float, space_divisor: float) -> int:
    if gap_pixels <= 4:
        return 1 if gap_pixels > 0 else 0
    return max(1, int(round(gap_pixels / space_divisor)))


def _format_line(
    line_items: list[dict],
    *,
    page_width: int,
    column_split_enabled: bool,
    column_gap_ratio: float,
    char_width: float,
    space_divisor: float,
) -> str:
    if not line_items:
        return ""

    sorted_items = sorted(line_items, key=lambda item: item["x_min"])
    column_threshold = page_width * column_gap_ratio if page_width > 0 else 0
    parts: list[str] = []
    prev_x_max: float | None = None

    for item in sorted_items:
        x_min = float(item["x_min"])
        text = str(item["text"])

        if prev_x_max is not None:
            gap = x_min - prev_x_max
            if column_split_enabled and column_threshold > 0 and gap >= column_threshold:
                parts.append(" | ")
            else:
                parts.append(" " * _spaces_for_gap(gap, space_divisor))

        parts.append(text)
        prev_x_max = _item_x_max(item, char_width)

    return "".join(parts)


def layout_text_from_items(
    items: list[dict],
    *,
    line_threshold: int,
    page_width: int,
    column_split_enabled: bool = True,
    column_gap_ratio: float = 0.18,
    char_width: float = 8.0,
    space_divisor: float = 8.0,
    blank_line_y_multiplier: float = 2.5,
) -> str:
    """Rebuild page text from OCR word boxes with horizontal and vertical spacing."""
    if not items:
        return ""

    sorted_items = sorted(items, key=lambda item: (item["y_min"], item["x_min"]))
    line_groups: list[dict] = []
    current_items: list[dict] = []
    current_y: float | None = None

    for item in sorted_items:
        y_min = float(item["y_min"])
        if current_y is None:
            current_items = [item]
            current_y = y_min
            continue

        if abs(y_min - current_y) <= line_threshold:
            current_items.append(item)
        else:
            line_groups.append({"y_min": current_y, "items": current_items})
            current_items = [item]
            current_y = y_min

    if current_items:
        line_groups.append({"y_min": current_y, "items": current_items})

    output_lines: list[str] = []
    previous_y: float | None = None

    for group in line_groups:
        y_min = float(group["y_min"])
        if previous_y is not None:
            y_gap = y_min - previous_y
            extra_breaks = int(y_gap // max(line_threshold * blank_line_y_multiplier, 1)) - 1
            if extra_breaks > 0:
                output_lines.extend([""] * extra_breaks)

        output_lines.append(
            _format_line(
                group["items"],
                page_width=page_width,
                column_split_enabled=column_split_enabled,
                column_gap_ratio=column_gap_ratio,
                char_width=char_width,
                space_divisor=space_divisor,
            )
        )
        previous_y = y_min

    return "\n".join(output_lines)
