"""Training loop for both stages:

  stage 1  teacher / baseline   kd.enabled = false   -> CE + triplet
  stage 2  student with KD      kd.enabled = true    -> CE + triplet + DistillLoss(teacher)

Designed for Colab: writes last.pth every epoch and auto-resumes from it.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn

from ..config import save_config
from ..losses import CrossEntropyLabelSmooth, DistillLoss, TripletLoss
from ..utils import load_model_weights, save_checkpoint
from .evaluator import evaluate


def build_scheduler(optimizer, cfg_train):
    """Linear warmup then step decay (Bag-of-Tricks schedule)."""
    warmup = cfg_train.get("warmup_epochs", 10)
    milestones = cfg_train.get("milestones", [40, 70])
    gamma = cfg_train.get("gamma", 0.1)

    def factor(epoch: int) -> float:
        if epoch < warmup:
            return 0.1 + 0.9 * epoch / warmup
        return gamma ** sum(epoch >= m for m in milestones)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class Trainer:
    def __init__(
        self,
        cfg,
        student: nn.Module,
        train_loader,
        test_loader,
        num_query: int,
        device,
        logger,
        teacher: nn.Module | None = None,
    ):
        self.cfg, self.device, self.logger = cfg, device, logger
        self.model = student.to(device)
        self.train_loader, self.test_loader, self.num_query = train_loader, test_loader, num_query
        self.out_dir = Path(cfg.output_dir)

        self.ce = CrossEntropyLabelSmooth(cfg.loss.get("label_smooth", 0.1))
        self.tri = TripletLoss(cfg.loss.get("triplet_margin", 0.3))
        self.w_ce = cfg.loss.get("ce_weight", 1.0)
        self.w_tri = cfg.loss.get("triplet_weight", 1.0)

        self.teacher = None
        self.kd = None
        params = list(self.model.parameters())
        if cfg.kd.get("enabled", False):
            if teacher is None:
                raise ValueError("kd.enabled=true but no teacher model was given")
            self.teacher = teacher.to(device).eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.kd = DistillLoss(cfg.kd, self.model.feat_dim, self.teacher.feat_dim).to(device)
            params += list(self.kd.parameters())  # learnable projector, if any

        t = cfg.train
        self.optimizer = torch.optim.Adam(params, lr=t.lr, weight_decay=t.weight_decay)
        self.scheduler = build_scheduler(self.optimizer, t)
        self.use_amp = t.get("amp", True) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.start_epoch, self.best_map = 0, 0.0

    # ---------------------------------------------------------------- checkpoint
    def _state(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "kd": self.kd.state_dict() if self.kd is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_map": self.best_map,
        }

    def try_resume(self) -> None:
        ckpt = self.out_dir / "last.pth"
        if not (self.cfg.train.get("resume", True) and ckpt.exists()):
            return
        s = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(s["model"])
        if self.kd is not None and s.get("kd"):
            self.kd.load_state_dict(s["kd"])
        self.optimizer.load_state_dict(s["optimizer"])
        self.scheduler.load_state_dict(s["scheduler"])
        self.scaler.load_state_dict(s["scaler"])
        self.start_epoch, self.best_map = s["epoch"] + 1, s["best_map"]
        self.logger.info(f"Resumed from {ckpt} at epoch {self.start_epoch}")

    # ---------------------------------------------------------------- train
    def _teacher_forward(self, imgs: torch.Tensor) -> dict:
        with torch.no_grad():
            out = self.teacher(imgs)  # eval mode -> logits are None
            out["logits"] = self.teacher.classifier(out["bn_feat"])
        return out

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        meters: dict[str, float] = {}
        n = 0
        log_every = self.cfg.train.get("log_period", 50)
        for it, (imgs, pids, _) in enumerate(self.train_loader):
            imgs = imgs.to(self.device, non_blocking=True)
            pids = pids.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.use_amp):
                out = self.model(imgs)
                losses = {
                    "ce": self.w_ce * self.ce(out["logits"], pids),
                    "tri": self.w_tri * self.tri(out["feat"], pids),
                }
                if self.kd is not None:
                    t_out = self._teacher_forward(imgs)
                    losses.update(self.kd(out, t_out))
                total = sum(losses.values())

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            n += 1
            for k, v in losses.items():
                meters[k] = meters.get(k, 0.0) + v.item()
            meters["total"] = meters.get("total", 0.0) + total.item()
            if (it + 1) % log_every == 0:
                avg = " ".join(f"{k}={v / n:.3f}" for k, v in meters.items())
                self.logger.info(f"ep {epoch + 1} it {it + 1}/{len(self.train_loader)} {avg}")
        return {k: v / max(n, 1) for k, v in meters.items()}

    def fit(self) -> float:
        save_config(self.cfg, self.out_dir / "config.yaml")
        self.try_resume()
        t = self.cfg.train
        for epoch in range(self.start_epoch, t.epochs):
            tic = time.time()
            stats = self.train_one_epoch(epoch)
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"Epoch {epoch + 1}/{t.epochs} done in {time.time() - tic:.0f}s | lr {lr:.2e} | "
                + " ".join(f"{k}={v:.3f}" for k, v in stats.items())
            )
            if (epoch + 1) % t.get("eval_period", 10) == 0 or epoch + 1 == t.epochs:
                r = evaluate(self.model, self.test_loader, self.num_query, self.device)
                self.logger.info(
                    f"  mAP {r['mAP']:.1%} | R1 {r['rank1']:.1%} | R5 {r['rank5']:.1%} | R10 {r['rank10']:.1%}"
                )
                if r["mAP"] > self.best_map:
                    self.best_map = r["mAP"]
                    save_checkpoint(
                        {"model": self.model.state_dict(), "epoch": epoch, "metrics": r},
                        self.out_dir / "best.pth",
                    )
            save_checkpoint(self._state(epoch), self.out_dir / "last.pth")
        self.logger.info(f"Best mAP {self.best_map:.1%}")
        return self.best_map


def load_teacher(cfg, num_classes: int):
    from ..models import build_model

    teacher = build_model(cfg.teacher, num_classes)
    load_model_weights(teacher, cfg.teacher.checkpoint)
    return teacher
