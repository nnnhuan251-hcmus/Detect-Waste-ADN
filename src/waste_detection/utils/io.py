import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("IOUtils")

class IOUtils:
    """
    Công cụ đọc/ghi file JSON và COCO JSON.

    Nguyên tắc:
    - Không hard-code đường dẫn.
    - Luôn tạo thư mục cha trước khi lưu.
    - Ghi file theo kiểu atomic save để tránh hỏng file nếu runtime bị ngắt.
    """

    @staticmethod
    def load_json(source_path: Path) -> Any:
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
    def save_json(dest_path: Path, data: Any, indent: int = 2) -> Any:
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

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
    def save_coco_json(dest_path: Path, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lưu dataset COCO với thứ tự key chuẩn.

        Các key được giữ:
        - info
        - licenses
        - images
        - annotations
        - categories
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
