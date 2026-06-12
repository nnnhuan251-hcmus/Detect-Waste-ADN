from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2

from waste_detection.utils.io import IOUtils


class BoxDrawer:
    """
    Vẽ bounding boxes, class labels và confidence lên ảnh.
    """

    @staticmethod
    def draw_predictions(
        image_path: str | Path,
        predictions: List[Dict[str, Any]],
        save_path: str | Path,
    ) -> Path:
        image_path = Path(image_path)
        save_path = Path(save_path)

        image = IOUtils.load_image_bgr(image_path)

        for prediction in predictions:
            xyxy = prediction["xyxy"]
            class_name = prediction.get("class_name", "unknown")
            final_confidence = prediction.get("final_confidence", None)

            x1, y1, x2, y2 = map(int, xyxy)

            display_text = class_name

            if final_confidence is not None:
                display_text = f"{class_name} {final_confidence:.2f}"

            BoxDrawer._draw_one_box(
                image=image,
                xyxy=(x1, y1, x2, y2),
                text=display_text,
            )

        IOUtils.save_image_bgr(save_path, image)

        return save_path

    @staticmethod
    def _draw_one_box(
        image,
        xyxy: tuple[int, int, int, int],
        text: str,
    ) -> None:
        x1, y1, x2, y2 = xyxy

        color = (0, 255, 0)
        text_color = (0, 0, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        text_bg_x1 = x1
        text_bg_y1 = max(0, y1 - text_height - 8)
        text_bg_x2 = x1 + text_width + 6
        text_bg_y2 = y1

        cv2.rectangle(
            image,
            (text_bg_x1, text_bg_y1),
            (text_bg_x2, text_bg_y2),
            color,
            thickness=-1,
        )

        cv2.putText(
            image,
            text,
            (x1 + 3, max(text_height + 3, y1 - 5)),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
