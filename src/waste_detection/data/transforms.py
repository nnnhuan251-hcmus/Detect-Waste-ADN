from __future__ import annotations

from typing import Callable

from PIL import Image, ImageOps
from torchvision import transforms


class SquarePad:
    """
    Pad ảnh thành hình vuông trước khi resize.

    Lý do:
    - Crop từ bbox thường là hình chữ nhật.
    - EfficientNet-B0 yêu cầu input 224x224.
    - Resize trực tiếp chữ nhật -> vuông sẽ làm méo vật thể.
    - Padding giữ nguyên aspect ratio tốt hơn.
    """

    def __init__(self, fill: int | tuple[int, int, int] = 0) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size

        if width == height:
            return image

        max_side = max(width, height)

        pad_left = (max_side - width) // 2
        pad_right = max_side - width - pad_left
        pad_top = (max_side - height) // 2
        pad_bottom = max_side - height - pad_top

        padding = (pad_left, pad_top, pad_right, pad_bottom)

        return ImageOps.expand(image, border=padding, fill=self.fill)


class ClassifierTransformFactory:
    """
    Factory tạo transform cho EfficientNet-B0 crop classifier.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    @staticmethod
    def build_train_transform(
        image_size: int = 224,
        augmentation_level: str = "minimal",
    ) -> Callable:
        transform_list = [
            SquarePad(fill=0),
            transforms.Resize((image_size, image_size)),
        ]

        if augmentation_level == "strong":
            transform_list.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.2,
                                contrast=0.2,
                                saturation=0.2,
                                hue=0.05,
                            )
                        ],
                        p=0.7,
                    ),
                    transforms.RandomApply(
                        [
                            transforms.RandomChoice(
                                [
                                    transforms.RandomRotation((90, 90)),
                                    transforms.RandomRotation((180, 180)),
                                ]
                            )
                        ],
                        p=0.5,
                    ),
                ]
            )

        elif augmentation_level == "minimal":
            pass

        else:
            raise ValueError(
                f"augmentation_level không hợp lệ: {augmentation_level}. "
                "Chỉ hỗ trợ: minimal, strong."
            )

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=ClassifierTransformFactory.IMAGENET_MEAN,
                    std=ClassifierTransformFactory.IMAGENET_STD,
                ),
            ]
        )

        return transforms.Compose(transform_list)

    @staticmethod
    def build_eval_transform(image_size: int = 224) -> Callable:
        return transforms.Compose(
            [
                SquarePad(fill=0),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=ClassifierTransformFactory.IMAGENET_MEAN,
                    std=ClassifierTransformFactory.IMAGENET_STD,
                ),
            ]
        )
