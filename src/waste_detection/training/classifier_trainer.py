from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from waste_detection.training.trainer_base import TrainerBase
from waste_detection.models.efficientnet_classifier import EfficientNetB0Classifier
from waste_detection.training.losses import LossFactory
from waste_detection.training.schedulers import SchedulerFactory
from waste_detection.utils.io import IOUtils


logger = logging.getLogger("ClassifierTrainer")


class ClassifierTrainer(TrainerBase):
    """
    Training loop cho EfficientNet-B0 classifier.

    Có:
    - forward
    - loss
    - backward
    - optimizer step
    - validation
    - macro F1
    - checkpoint
    - early stopping
    - mixed precision
    - optional W&B logging
    - freeze/unfreeze strategy
    """

    def __init__(
        self,
        model: EfficientNetB0Classifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: List[str],
        experiment_config: Dict,
        output_dir: str | Path,
        device: torch.device,
        class_weights: torch.Tensor | None = None,
        wandb_run=None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.experiment_config = experiment_config
        self.output_dir = Path(output_dir)
        self.device = device
        self.class_weights = class_weights
        self.wandb_run = wandb_run

        IOUtils.ensure_dir(self.output_dir)

        self.training_config = experiment_config.get("training", {})
        self.optimizer_config = experiment_config.get("optimizer", {})
        self.loss_config = experiment_config.get("loss", {})
        self.scheduler_config = experiment_config.get("scheduler", {})
        self.fine_tuning_config = experiment_config.get("fine_tuning", {})
        self.gradient_config = experiment_config.get("gradient_control", {})
        self.checkpoint_config = experiment_config.get("checkpoint", {})

        self.epochs = int(self.training_config.get("epochs", 50))
        early_stopping_enabled = bool(
            self.training_config.get("early_stopping", True)
        )
        
        self.patience = (
            int(self.training_config.get("patience", 10))
            if early_stopping_enabled
            else 0
        )
        
        self.mixed_precision = bool(self.training_config.get("mixed_precision", True))

        self.criterion = LossFactory.build_classification_loss(
            loss_config=self.loss_config,
            class_weights=self.class_weights,
            device=self.device,
        )

        self.optimizer = self._build_optimizer()

        self.scheduler = SchedulerFactory.build_scheduler(
            optimizer=self.optimizer,
            scheduler_config=self.scheduler_config,
            total_epochs=self.epochs,
        )

        self.scaler = GradScaler("cuda", enabled=self.mixed_precision and device.type == "cuda")

        self.best_metric = float("-inf")
        self.early_stop_counter = 0
        self.history = []

    def train(self) -> List[Dict]:
        logger.info("Bắt đầu huấn luyện EfficientNet-B0 classifier.")

        for epoch in range(self.epochs):
            self._apply_fine_tuning_schedule(epoch)

            train_metrics = self._train_one_epoch(epoch)
            val_metrics = self._validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_record = {
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }

            self.history.append(epoch_record)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                "val_acc=%.4f | val_macro_f1=%.4f | lr=%.8f",
                epoch + 1,
                self.epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
                current_lr,
            )

            if self.wandb_run is not None:
                self.wandb_run.log(epoch_record, step=epoch + 1)

            improved = self._save_if_best(val_metrics)

            if improved:
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            if self.patience > 0 and self.early_stop_counter >= self.patience:
                logger.info(
                    "Early stopping tại epoch %d vì metric không cải thiện trong %d epoch.",
                    epoch + 1,
                    self.patience,
                )
                break

        IOUtils.save_json(
            self.output_dir / "training_history.json",
            self.history,
        )

        logger.info("Huấn luyện classifier hoàn tất.")

        return self.history

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        y_true = []
        y_pred = []

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Train classifier epoch {epoch + 1}",
        )

        for batch in progress_bar:
            inputs, labels = batch[:2]
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.mixed_precision and self.device.type == "cuda"):
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            if bool(self.gradient_config.get("gradient_clipping", False)):
                self.scaler.unscale_(self.optimizer)
                max_norm = float(self.gradient_config.get("max_norm", 10.0))
                clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            predictions = torch.argmax(logits.detach(), dim=1)

            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

        return self._compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            loss=total_loss / max(1, len(self.train_loader)),
        )

    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                desc=f"Validate classifier epoch {epoch + 1}",
            )

            for batch in progress_bar:
                inputs, labels = batch[:2]
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(
                    enabled=self.mixed_precision and self.device.type == "cuda"
                ):
                    logits = self.model(inputs)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=1)

                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(predictions.cpu().tolist())

        return self._compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            loss=total_loss / max(1, len(self.val_loader)),
        )

    def validate(self) -> Dict[str, float]:
        """
        Public validation method theo TrainerBase.
        """
        return self._validate(epoch=-1)

    def _compute_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        loss: float,
    ) -> Dict[str, float]:
        labels = list(range(len(self.class_names)))

        return {
            "loss": float(loss),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "macro_precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    def _build_optimizer(self):
        optimizer_name = self.optimizer_config.get("name", "adamw").lower()
        lr = float(self.optimizer_config.get("lr", 1e-4))
        weight_decay = float(self.optimizer_config.get("weight_decay", 5e-4))

        model_params = list(self.model.parameters())

        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                model_params,
                lr=lr,
                weight_decay=weight_decay,
            )

        if optimizer_name == "adam":
            return torch.optim.Adam(
                model_params,
                lr=lr,
                weight_decay=weight_decay,
            )

        if optimizer_name == "sgd":
            return torch.optim.SGD(
                model_params,
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )

        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _apply_fine_tuning_schedule(self, epoch: int) -> None:
        train_mode = self.fine_tuning_config.get("train_mode", "end_to_end")

        if train_mode == "end_to_end":
            return

        if train_mode != "staged_finetuning":
            return

        freeze_epochs = int(self.fine_tuning_config.get("freeze_epochs", 0))
        freeze_backbone = bool(self.fine_tuning_config.get("freeze_backbone", False))

        if freeze_epochs <= 0:
            if epoch == 0:
                logger.info(
                    "Fine-tuning schedule | freeze_epochs<=0, không freeze backbone."
                )
            return

        if epoch == 0 and freeze_backbone:
            self.model.freeze_backbone()
            logger.info(
                "Fine-tuning schedule | epoch=%d | action=freeze_backbone",
                epoch + 1,
            )
            return

        if epoch == freeze_epochs:
            self.model.unfreeze_last_blocks(num_blocks=2)
            logger.info(
                "Fine-tuning schedule | epoch=%d | action=unfreeze_last_blocks",
                epoch + 1,
            )
            return

        if epoch == freeze_epochs + 5:
            self.model.unfreeze_backbone()
            logger.info(
                "Fine-tuning schedule | epoch=%d | action=unfreeze_full_backbone",
                epoch + 1,
            )
            return

    def _save_if_best(self, val_metrics: Dict[str, float]) -> bool:
        monitor_metric = self.checkpoint_config.get("monitor_classifier", "val_macro_f1")
        metric_key = monitor_metric.replace("val_", "")

        current_metric = val_metrics.get(metric_key)

        if current_metric is None:
            current_metric = val_metrics["macro_f1"]

        if current_metric <= self.best_metric:
            return False

        self.best_metric = current_metric

        checkpoint_path = self.output_dir / "best.pth"

        self.model.save_checkpoint(
            checkpoint_path,
            class_names=self.class_names,
            extra={
                "best_metric": self.best_metric,
                "experiment_config": self.experiment_config,
            },
        )

        logger.info(
            "Đã lưu best checkpoint: %s | best_metric=%.4f",
            checkpoint_path,
            self.best_metric,
        )

        return True
