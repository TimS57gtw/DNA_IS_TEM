# YOLOv5 loggers, wandb/comet-free
"""Minimal GenericLogger: CSV results + optional TensorBoard, no wandb/comet.

This vendored yolov5 copy shipped without utils/loggers at all; this is a
stripped-down replacement covering only what segment/train.py actually
calls (update_params, log_images, log_metrics, log_model).
"""
import shutil
from pathlib import Path

from utils.general import LOGGER

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class GenericLogger:
    """Console + CSV + optional TensorBoard logger, no wandb/comet dependency."""

    def __init__(self, opt, console_logger=LOGGER, include=("tb",)):
        self.save_dir = Path(opt.save_dir)
        self.include = include
        self.console_logger = console_logger
        self.csv = self.save_dir / "results.csv"
        self.tb = None
        if "tb" in self.include and SummaryWriter is not None:
            self.console_logger.info(
                f"TensorBoard: run 'tensorboard --logdir {self.save_dir.parent}' to view."
            )
            self.tb = SummaryWriter(str(self.save_dir))

    def log_metrics(self, metrics, epoch):
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1
        s = "" if self.csv.exists() else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")
        with open(self.csv, "a") as f:
            f.write(s + ("%23.5g," * n % tuple([epoch] + vals)).rstrip(",") + "\n")

        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, epoch)

    def log_images(self, files, name="Images", epoch=0):
        files = [Path(f) for f in (files if isinstance(files, (list, tuple)) else [files])]
        files = [f for f in files if f.exists()]
        if self.tb:
            import cv2
            for f in files:
                img = cv2.imread(str(f))
                if img is not None:
                    self.tb.add_image(f"{name}/{f.stem}", img[..., ::-1].copy(), epoch, dataformats="HWC")

    def log_model(self, model_path, epoch=0, metadata=None):
        self.console_logger.info(f"Saved model checkpoint: {model_path}")

    def update_params(self, params):
        self.console_logger.info(f"Updated params: {params}")
