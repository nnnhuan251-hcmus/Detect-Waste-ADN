import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


logger = logging.getLogger("ImageResolver")


@dataclass
class ImageResolutionReport:
    total_images: int = 0
    found_images: int = 0
    missing_images: int = 0
    missing_files: List[str] = field(default_factory=list)
    resolved_paths: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_images": self.total_images,
            "found_images": self.found_images,
            "missing_images": self.missing_images,
            "missing_files": self.missing_files,
            "resolved_paths": self.resolved_paths,
        }


class ImageResolver:
    """
    Resolve đường dẫn ảnh trong TACO.

    Hỗ trợ 2 trường hợp:

    1. file_name có batch prefix:
       batch_1/000006.jpg

    2. file_name không có batch prefix:
       000006.jpg

    Trong trường hợp 2, resolver sẽ dò trong batch_1 đến batch_15.
    """

    def __init__(
        self,
        image_root: str | Path,
        batch_dirs: List[str] | None = None,
        allowed_extensions: List[str] | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.batch_dirs = batch_dirs or [f"batch_{index}" for index in range(1, 16)]
        self.allowed_extensions = allowed_extensions or [
            ".jpg",
            ".jpeg",
            ".png",
            ".JPG",
            ".JPEG",
            ".PNG",
        ]

    def resolve(self, file_name: str) -> Path | None:
        file_name_path = Path(file_name)

        direct_path = self.image_root / file_name_path

        if direct_path.exists() and direct_path.is_file():
            return direct_path

        basename = file_name_path.name

        root_candidate = self.image_root / basename
        if root_candidate.exists() and root_candidate.is_file():
            return root_candidate

        for batch_dir in self.batch_dirs:
            candidate = self.image_root / batch_dir / basename

            if candidate.exists() and candidate.is_file():
                return candidate

        stem = file_name_path.stem

        for batch_dir in self.batch_dirs:
            for extension in self.allowed_extensions:
                candidate = self.image_root / batch_dir / f"{stem}{extension}"

                if candidate.exists() and candidate.is_file():
                    return candidate

        return None

    def build_report(self, dataset: Dict[str, Any]) -> ImageResolutionReport:
        images = dataset.get("images", [])

        report = ImageResolutionReport(total_images=len(images))

        for image in images:
            file_name = image.get("file_name")

            if not file_name:
                report.missing_images += 1
                report.missing_files.append("<missing_file_name>")
                continue

            resolved_path = self.resolve(file_name)

            if resolved_path is None:
                report.missing_images += 1
                report.missing_files.append(file_name)
            else:
                report.found_images += 1
                report.resolved_paths[file_name] = str(resolved_path)

        logger.info(
            "Image resolving xong: found=%d, missing=%d, total=%d.",
            report.found_images,
            report.missing_images,
            report.total_images,
        )

        if report.missing_images > 0:
            logger.warning("Có %d ảnh không tìm thấy.", report.missing_images)

            for missing_file in report.missing_files[:20]:
                logger.warning("Missing image: %s", missing_file)

            if report.missing_images > 20:
                logger.warning(
                    "Còn %d ảnh missing khác chưa hiển thị.",
                    report.missing_images - 20,
                )

        return report
