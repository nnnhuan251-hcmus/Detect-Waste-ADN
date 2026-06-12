from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np


logger = logging.getLogger("IOUtils")


class IOUtils:
    """
    Low-level I/O utilities dùng chung cho toàn repo.

    Nguyên tắc:
    - Không chứa logic nghiệp vụ như map label, split, crop, convert YOLO.
    - Chỉ xử lý đọc/ghi/copy/tạo thư mục/lưu ảnh/lưu JSON/CSV.
    - Các domain class như YoloConverter, CropDatasetBuilder, TacoLabelMapper
      được phép gọi IOUtils, nhưng IOUtils không được biết domain đó là gì.
    """

    # ------------------------------------------------------------------
    # Directory / filesystem
    # ------------------------------------------------------------------
    @staticmethod
    def ensure_dir(path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def copy_file(
        source_path: str | Path,
        dest_path: str | Path,
        overwrite: bool = True,
    ) -> Path:
        source_path = Path(source_path)
        dest_path = Path(dest_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file không tồn tại: {source_path}")

        if not source_path.is_file():
            raise ValueError(f"Source path không phải file: {source_path}")

        if dest_path.exists() and not overwrite:
            logger.info("File đã tồn tại, bỏ qua copy: %s", dest_path)
            return dest_path

        IOUtils.ensure_dir(dest_path.parent)
        shutil.copy2(source_path, dest_path)

        logger.debug("Copied file: %s -> %s", source_path, dest_path)

        return dest_path

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    @staticmethod
    def load_json(source_path: str | Path) -> Any:
        source_path = Path(source_path)

        logger.info("Đang nạp JSON từ: %s", source_path)

        if not source_path.exists():
            logger.error("Không tìm thấy file: %s", source_path)
            raise FileNotFoundError(f"File không tồn tại: {source_path}")

        if not source_path.is_file():
            logger.error("Đường dẫn không phải file: %s", source_path)
            raise ValueError(f"Đường dẫn không phải file: {source_path}")

        try:
            with source_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            file_size_mb = source_path.stat().st_size / (1024 * 1024)

            logger.info(
                "Nạp JSON thành công: %s | Dung lượng: %.2f MB",
                source_path,
                file_size_mb,
            )

            return data

        except json.JSONDecodeError:
            logger.exception("File JSON bị lỗi cấu trúc: %s", source_path)
            raise

    @staticmethod
    def save_json(
        dest_path: str | Path,
        data: Any,
        indent: int = 2,
    ) -> Any:
        dest_path = Path(dest_path)
        IOUtils.ensure_dir(dest_path.parent)

        logger.info("Chuẩn bị lưu JSON tại: %s", dest_path)

        temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

        try:
            with temp_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, indent=indent, ensure_ascii=False)

            temp_path.replace(dest_path)

            logger.info("Lưu JSON thành công: %s", dest_path)
            return data

        except Exception:
            logger.exception("Thất bại khi lưu JSON: %s", dest_path)

            if temp_path.exists():
                temp_path.unlink()

            raise

    @staticmethod
    def save_coco_json(
        dest_path: str | Path,
        dataset: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Lưu dataset COCO với thứ tự key chuẩn.
        """
        dest_path = Path(dest_path)

        data_dict = {
            "info": dataset.get("info", {}),
            "licenses": dataset.get("licenses", []),
            "images": dataset.get("images", []),
            "annotations": dataset.get("annotations", []),
            "categories": dataset.get("categories", []),
        }

        IOUtils.save_json(dest_path=dest_path, data=data_dict, indent=2)

        logger.info(
            "Lưu COCO JSON thành công: %s | %d images, %d annotations, %d categories",
            dest_path,
            len(data_dict["images"]),
            len(data_dict["annotations"]),
            len(data_dict["categories"]),
        )

        return data_dict

    # ------------------------------------------------------------------
    # Text / CSV
    # ------------------------------------------------------------------
    @staticmethod
    def write_text(
        dest_path: str | Path,
        content: str,
        encoding: str = "utf-8",
    ) -> Path:
        dest_path = Path(dest_path)
        IOUtils.ensure_dir(dest_path.parent)

        with dest_path.open("w", encoding=encoding) as file:
            file.write(content)

        logger.debug("Đã ghi text file: %s", dest_path)

        return dest_path

    @staticmethod
    def write_csv_dicts(
        dest_path: str | Path,
        rows: List[Dict[str, Any]],
        fieldnames: Sequence[str] | None = None,
    ) -> Path:
        dest_path = Path(dest_path)
        IOUtils.ensure_dir(dest_path.parent)

        if not rows:
            logger.warning("Không có dòng CSV để lưu. Tạo file rỗng: %s", dest_path)
            dest_path.touch()
            return dest_path

        if fieldnames is None:
            fieldnames = list(rows[0].keys())

        with dest_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Đã lưu CSV: %s | rows=%d", dest_path, len(rows))

        return dest_path

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------
    @staticmethod
    def load_image_bgr(image_path: str | Path) -> np.ndarray:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

        if not image_path.is_file():
            raise ValueError(f"Đường dẫn ảnh không phải file: {image_path}")

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"OpenCV không đọc được ảnh: {image_path}")

        return image

    @staticmethod
    def load_image_rgb(image_path: str | Path) -> np.ndarray:
        image_bgr = IOUtils.load_image_bgr(image_path)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def save_image_bgr(
        dest_path: str | Path,
        image_bgr: np.ndarray,
        jpeg_quality: int | None = None,
    ) -> Path:
        dest_path = Path(dest_path)
        IOUtils.ensure_dir(dest_path.parent)

        params: List[int] = []

        suffix = dest_path.suffix.lower()

        if jpeg_quality is not None and suffix in {".jpg", ".jpeg"}:
            jpeg_quality = int(jpeg_quality)
            jpeg_quality = max(1, min(100, jpeg_quality))
            params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

        success = cv2.imwrite(str(dest_path), image_bgr, params)

        if not success:
            raise ValueError(f"Không thể lưu ảnh tại: {dest_path}")

        return dest_path
