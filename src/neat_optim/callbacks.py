"""Keras callbacks for NEAT optimizer diagnostics."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import keras


@keras.saving.register_keras_serializable(package="neat_optim")
class NEATDiagnosticsCallback(keras.callbacks.Callback):
    """Collect and optionally log `NEAT.diagnostic_snapshot()` metrics."""

    def __init__(
        self,
        log_dir: str | None = None,
        reset_each_epoch: bool = True,
        prefix: str = "neat",
    ) -> None:
        super().__init__()
        self.log_dir = log_dir
        self.reset_each_epoch = reset_each_epoch
        self.prefix = prefix.strip("/")
        self.history: list[dict[str, float]] = []
        self._writer: Any | None = None

    def _optimizer(self):
        optimizer = getattr(self.model, "optimizer", None)
        if optimizer is None or not hasattr(optimizer, "diagnostic_snapshot"):
            raise TypeError(
                "NEATDiagnosticsCallback requires a compiled model using "
                "`neat_optim.NEAT` or an optimizer with `diagnostic_snapshot()`."
            )
        return optimizer

    def _summary_writer(self):
        if self.log_dir is None:
            return None
        if self._writer is None:
            tensorflow = import_module("tensorflow")
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self._writer = tensorflow.summary.create_file_writer(self.log_dir)
        return self._writer

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        optimizer = self._optimizer()
        snapshot = {
            key: float(value)
            for key, value in optimizer.diagnostic_snapshot().items()
        }
        self.history.append(snapshot)

        writer = self._summary_writer()
        if writer is not None:
            tensorflow = import_module("tensorflow")
            with writer.as_default():
                for key, value in snapshot.items():
                    tensorflow.summary.scalar(f"{self.prefix}/{key}", value, step=epoch)
                writer.flush()

        if self.reset_each_epoch and hasattr(optimizer, "reset_diagnostics"):
            optimizer.reset_diagnostics()

    def get_config(self) -> dict[str, Any]:
        return {
            "log_dir": self.log_dir,
            "reset_each_epoch": self.reset_each_epoch,
            "prefix": self.prefix,
        }
