from __future__ import annotations

from typing import List


class BBoxUtils:
    """
    Utility functions for xyxy bounding boxes.
    """

    @staticmethod
    def clip_xyxy(
        box: List[float],
        image_width: int,
        image_height: int,
    ) -> List[float]:
        x1, y1, x2, y2 = map(float, box)

        x1 = max(0.0, min(x1, float(image_width)))
        y1 = max(0.0, min(y1, float(image_height)))
        x2 = max(0.0, min(x2, float(image_width)))
        y2 = max(0.0, min(y2, float(image_height)))

        return [x1, y1, x2, y2]

    @staticmethod
    def normalize_xyxy(
        box: List[float],
        image_width: int,
        image_height: int,
    ) -> List[float]:
        x1, y1, x2, y2 = BBoxUtils.clip_xyxy(
            box=box,
            image_width=image_width,
            image_height=image_height,
        )

        return [
            x1 / float(image_width),
            y1 / float(image_height),
            x2 / float(image_width),
            y2 / float(image_height),
        ]

    @staticmethod
    def denormalize_xyxy(
        box: List[float],
        image_width: int,
        image_height: int,
    ) -> List[float]:
        x1, y1, x2, y2 = map(float, box)

        denormalized = [
            x1 * float(image_width),
            y1 * float(image_height),
            x2 * float(image_width),
            y2 * float(image_height),
        ]

        return BBoxUtils.clip_xyxy(
            box=denormalized,
            image_width=image_width,
            image_height=image_height,
        )

    @staticmethod
    def flip_xyxy_horizontal(
        box: List[float],
        image_width: int,
    ) -> List[float]:
        x1, y1, x2, y2 = map(float, box)

        flipped_x1 = float(image_width) - x2
        flipped_x2 = float(image_width) - x1

        return [flipped_x1, y1, flipped_x2, y2]
