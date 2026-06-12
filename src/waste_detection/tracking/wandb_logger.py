from __future__ import annotations

import csv
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


logger = logging.getLogger("WandbLogger")


def to_serializable(value: Any) -> Any:
    """
    Convert Python objects into JSON/W&B-friendly values.
    Handles Path, dataclass, dict, list, tuple, set, and simple objects.
    """
    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (str, int, float, bool)):
        return value

    if is_dataclass(value):
        return to_serializable(asdict(value))

    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]

    if hasattr(value, "__dict__"):
        return to_serializable(vars(value))

    return str(value)


class WandbLogger:
    """
    Safe wrapper for Weights & Biases.

    Goals:
    - Keep training runnable when W&B is disabled.
    - Avoid spreading wandb.init / wandb.log across the project.
    - Support Kaggle offline logging.
    - Support model/checkpoint artifacts.
    """

    def __init__(
        self,
        enabled: bool,
        project: str,
        run_name: str,
        config: Dict[str, Any] | None = None,
        group: str | None = None,
        tags: Iterable[str] | None = None,
        entity: str | None = None,
        mode: str | None = None,
        job_type: str | None = None,
        notes: str | None = None,
        output_dir: str | Path | None = None,
        log_code: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.project = project
        self.run_name = run_name
        self.config = to_serializable(config or {})
        self.group = group
        self.tags = list(tags or [])
        self.entity = entity
        self.mode = mode
        self.job_type = job_type
        self.notes = notes
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.log_code = log_code

        self.run = None
        self._wandb = None

    def __enter__(self) -> "WandbLogger":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.finish()

    def start(self):
        if self.mode == "disabled":
            self.enabled = False

        if not self.enabled:
            logger.info("W&B disabled.")
            return None

        try:
            import wandb
        except ImportError:
            logger.warning(
                "wandb chưa được cài. Bỏ qua experiment tracking. "
                "Hãy thêm `wandb` vào requirements.txt nếu muốn dùng W&B."
            )
            self.enabled = False
            return None

        self._wandb = wandb

        wandb_dir = None
        if self.output_dir is not None:
            wandb_dir = self.output_dir / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)

        init_kwargs = {
            "project": self.project,
            "name": self.run_name,
            "config": self.config,
            "group": self.group,
            "tags": self.tags,
            "job_type": self.job_type,
            "notes": self.notes,
        }

        if self.entity:
            init_kwargs["entity"] = self.entity

        if self.mode:
            init_kwargs["mode"] = self.mode

        if wandb_dir is not None:
            init_kwargs["dir"] = str(wandb_dir)

        self.run = wandb.init(**init_kwargs)

        if self.log_code and self.run is not None:
            try:
                self.run.log_code(".")
            except Exception as exc:
                logger.warning("Không thể log source code lên W&B: %s", exc)

        logger.info("W&B run started: %s", self.run_name)
        return self.run

    def log(self, data: Dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self.run is not None:
            self.run.log(to_serializable(data), step=step)

    def log_summary(self, data: Dict[str, Any]) -> None:
        if not (self.enabled and self.run is not None):
            return

        for key, value in to_serializable(data).items():
            self.run.summary[key] = value

    def watch_model(
        self,
        model: Any,
        log: str = "gradients",
        log_freq: int = 100,
    ) -> None:
        if not (self.enabled and self.run is not None and self._wandb is not None):
            return

        try:
            self._wandb.watch(model, log=log, log_freq=log_freq)
        except Exception as exc:
            logger.warning("Không thể watch model bằng W&B: %s", exc)

    def log_artifact(
        self,
        file_path: str | Path,
        artifact_name: str,
        artifact_type: str,
        aliases: Sequence[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        if not (self.enabled and self.run is not None and self._wandb is not None):
            return

        file_path = Path(file_path)

        if not file_path.exists():
            logger.warning("Artifact không tồn tại, bỏ qua: %s", file_path)
            return

        artifact = self._wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=to_serializable(metadata or {}),
        )
        artifact.add_file(str(file_path))
        self.run.log_artifact(artifact, aliases=list(aliases or []))

    def log_csv_history(
        self,
        csv_path: str | Path,
        prefix: str = "",
        step_column: str = "epoch",
    ) -> None:
        """
        Log a CSV history file to W&B row by row.

        Useful for Ultralytics results.csv.
        """
        if not (self.enabled and self.run is not None):
            return

        csv_path = Path(csv_path)

        if not csv_path.exists():
            logger.warning("Không tìm thấy CSV history để log W&B: %s", csv_path)
            return

        with csv_path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row_index, row in enumerate(reader):
                cleaned_row: Dict[str, Any] = {}

                for raw_key, raw_value in row.items():
                    key = f"{prefix}{raw_key.strip()}"

                    try:
                        value: Any = float(str(raw_value).strip())
                    except ValueError:
                        value = str(raw_value).strip()

                    cleaned_row[key] = value

                step = row_index + 1
                if step_column in row:
                    try:
                        step = int(float(str(row[step_column]).strip())) + 1
                    except ValueError:
                        step = row_index + 1

                self.log(cleaned_row, step=step)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()
            logger.info("W&B run finished.")

        self.run = None
        self._wandb = None
