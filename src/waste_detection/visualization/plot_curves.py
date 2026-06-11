from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


class CurvePlotter:
    """
    Vẽ training curves từ history.
    """

    @staticmethod
    def plot_metric_curve(
        history: List[Dict],
        train_key: str,
        val_key: str,
        ylabel: str,
        title: str,
        save_path: str | Path,
    ) -> Path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        epochs = [record.get("epoch", index + 1) for index, record in enumerate(history)]
        train_values = [record.get(train_key) for record in history]
        val_values = [record.get(val_key) for record in history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_values, label=train_key)
        plt.plot(epochs, val_values, label=val_key)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        return save_path

    @staticmethod
    def plot_loss_curve(
        history: List[Dict],
        save_path: str | Path,
    ) -> Path:
        return CurvePlotter.plot_metric_curve(
            history=history,
            train_key="train_loss",
            val_key="val_loss",
            ylabel="Loss",
            title="Training and Validation Loss",
            save_path=save_path,
        )

    @staticmethod
    def plot_f1_curve(
        history: List[Dict],
        save_path: str | Path,
    ) -> Path:
        return CurvePlotter.plot_metric_curve(
            history=history,
            train_key="train_macro_f1",
            val_key="val_macro_f1",
            ylabel="Macro F1",
            title="Training and Validation Macro F1",
            save_path=save_path,
        )
