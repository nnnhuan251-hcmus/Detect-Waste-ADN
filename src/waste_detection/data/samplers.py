from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict

import torch
from torch.utils.data import WeightedRandomSampler


logger = logging.getLogger("SamplerFactory")


class SamplerFactory:
    """
    Factory tạo sampler cho crop classifier.

    Hiện hỗ trợ:
    - weighted_random: sample ảnh theo trọng số nghịch đảo tần suất class.
    """

    @staticmethod
    def build_classification_sampler(
        dataset,
        sampler_config: Dict[str, Any] | None,
        seed: int = 42,
    ):
        sampler_config = sampler_config or {}

        enabled = bool(sampler_config.get("enabled", False))

        if not enabled:
            logger.info("Balanced sampler disabled.")
            return None

        sampler_name = str(sampler_config.get("name", "weighted_random")).lower()

        if sampler_name != "weighted_random":
            raise ValueError(
                f"Unsupported sampler: {sampler_name}. "
                "Hiện chỉ hỗ trợ: weighted_random."
            )

        if not hasattr(dataset, "samples"):
            raise AttributeError(
                "Dataset không có attribute `samples`. "
                "Weighted sampler cần dataset.samples dạng List[(path, label)]."
            )

        labels = [int(label) for _, label in dataset.samples]

        if not labels:
            raise ValueError("Không thể tạo sampler vì dataset không có sample.")

        label_counts = Counter(labels)

        sample_weights = []

        for label in labels:
            count = label_counts[label]

            if count <= 0:
                sample_weights.append(0.0)
            else:
                sample_weights.append(1.0 / float(count))

        sample_weights_tensor = torch.DoubleTensor(sample_weights)

        replacement = bool(sampler_config.get("replacement", True))
        num_samples_multiplier = float(sampler_config.get("num_samples_multiplier", 1.0))
        num_samples = int(len(sample_weights_tensor) * num_samples_multiplier)

        if num_samples <= 0:
            raise ValueError("sampler.num_samples phải > 0.")

        generator = torch.Generator()
        generator.manual_seed(seed)

        logger.info(
            "Created WeightedRandomSampler | samples=%d | replacement=%s | class_counts=%s",
            num_samples,
            replacement,
            dict(label_counts),
        )

        return WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )
