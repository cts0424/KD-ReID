"""Training loop for both stages:

  baseline / teacher    kd.enabled = false   -> CE + triplet
  student with KD       kd.enabled = true    -> CE + triplet + sum_i w_i * DistillLoss_i(teacher_i)

Teachers come from `cfg.teacher` (one) or `cfg.teachers` (a list, e.g. big teacher + assistant).
Each teacher entry may carry `weight` and a `kd` block that overrides the global `cfg.kd` for
that teacher only.

Designed for Colab: writes last.pth every epoch and auto-resumes from it.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn

from ..config import Config, _merge, save_config
from ..losses import CrossEntropyLabelSmooth, DistillLoss, TripletLoss
from ..utils import load_model_weights, save_checkpoint
from .evaluator import evaluate

_TEACHER_ONLY_KEYS = {"checkpoint", "weight", "kd"}


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


def teacher_specs(cfg) -> list[Config]:
    """Normalise `teacher:` / `teachers:` into a list of teacher configs."""
    if cfg.get("teachers"):
        return [Config.wrap(dict(t)) for t in cfg.teachers]
    if cfg.get("teacher"):
        return [cfg.teacher]
    return []


def load_teachers(cfg, num_classes: int) -> list[nn.Module]:
    from ..models import build_model

    teachers = []
    for spec in teacher_specs(cfg):
        model_cfg = Config({k: v for k, v in spec.items() if k not in _TEACHER_ONLY_KEYS})
        model_cfg.setdefault("pretrained", False)  # weights come from the checkpoint
        t = build_model(model_cfg, num_classes)
        load_model_weights(t, spec.checkpoint)
        teachers.append(t)
    return teachers


def load_teacher(cfg, num_classes: int):
    """Backward-compatible single-teacher loader."""
    return load_teachers(cfg, num_classes)[0]


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
        teacher: nn.Module | list[nn.Module] | None = None,
    ):
        self.cfg, self.device, self.logger = cfg, device, logger
        self.model = student.to(device)
        self.train_loader, self.test_loader, self.num_query = train_loader, test_loader, num_query
        self.out_dir = Path(cfg.output_dir)

        self.ce = CrossEntropyLabelSmooth(cfg.loss.get("label_smooth", 0.1))
        self.tri = TripletLoss(cfg.loss.get("triplet_margin", 0.3))
        self.w_ce = cfg.loss.get("ce_weight", 1.0)
        self.w_tri = cfg.loss.get("triplet_weight", 1.0)

        self.teachers: list[nn.Module] = []
        self.teacher_w: list[float] = []
        self.kds = nn.ModuleList()
        params = list(self.model.parameters())
        if cfg.kd.get("enabled", False):
            if teacher is None:
                raise ValueError("kd.enabled=true but no teacher model was given")
            teachers = teacher if isinstance(teacher, list | tuple) else [teacher]
            specs = teacher_specs(cfg) or [Config()]
            if len(specs) != len(teachers):
                raise ValueError(f"{len(teachers)} teacher models but {len(specs)} teacher configs")
            for t, spec in zip(teachers, specs, strict=True):
                t = t.to(device).eval()
                for p in t.parameters():
                    p.requires_grad_(False)
                kd_cfg = Config.wrap(_merge(cfg.kd.to_dict(), dict(spec.get("kd") or {})))
                self.teachers.append(t)
                self.teacher_w.append(float(spec.get("weight", 1.0)))
                self.kds.append(DistillLoss(kd_cfg, self.model.feat_dim, t.feat_dim))
            self.kds.to(device)
            params += list(self.kds.parameters())  # learnable projectors, if any
        # kept for code that expects a single teacher
        self.teacher = self.teachers[0] if self.teachers else None
        self.kd = self.kds[0] if len(self.kds) else None

        t = cfg.train
        self.optimizer = torch.optim.Adam(params, lr=t.lr, weight_decay=t.weight_decay)
        self.scheduler = build_scheduler(self.optimizer, t)
        self.use_amp = t.get("amp", True) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.freeze_epochs = int(t.get("freeze_backbone_epochs", 0) or 0)
        self.start_epoch, self.best_map = 0, 0.0

    # ---------------------------------------------------------------- checkpoint
    def _state(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "kd": self.kds.state_dict() if len(self.kds) else None,
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
        kd_state = s.get("kd")
        if len(self.kds) and kd_state:
            if not any(k.split(".")[0].isdigit() for k in kd_state):  # old single-teacher format
                kd_state = {f"0.{k}": v for k, v in kd_state.items()}
            self.kds.load_state_dict(kd_state)
        self.optimizer.load_state_dict(s["optimizer"])
        self.scheduler.load_state_dict(s["scheduler"])
        self.scaler.load_state_dict(s["scaler"])
        self.start_epoch, self.best_map = s["epoch"] + 1, s["best_map"]
        self.logger.info(f"Resumed from {ckpt} at epoch {self.start_epoch}")

    # ---------------------------------------------------------------- train
    @staticmethod
    def _teacher_forward(teacher: nn.Module, imgs: torch.Tensor, need_logits: bool) -> dict:
        with torch.no_grad():
            out = teacher(imgs)  # eval mode -> logits are None
            if need_logits:
                out["logits"] = teacher.classifier(out["bn_feat"])
        return out

    def _kd_terms(self, out: dict, imgs: torch.Tensor, pids: torch.Tensor) -> dict:
        terms: dict[str, torch.Tensor] = {}
        multi = len(self.teachers) > 1
        for i, (t, w, kd) in enumerate(zip(self.teachers, self.teacher_w, self.kds, strict=True)):
            t_out = self._teacher_forward(t, imgs, kd.needs_logits)
            for k, v in kd(out, t_out, pids).items():
                terms[f"t{i}_{k}" if multi else k] = w * v
        return terms

    def _set_backbone_frozen(self, frozen: bool) -> None:
        if not hasattr(self.model, "backbone_parameters"):
            return
        for p in self.model.backbone_parameters():
            p.requires_grad_(not frozen)
        if frozen:  # also keep BN statistics fixed, as torchreid does
            self.model.backbone.eval()
            if getattr(self.model, "embed", None) is not None:
                self.model.embed.eval()

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        frozen = epoch < self.freeze_epochs
        self._set_backbone_frozen(frozen)
        meters: dict[str, float] = {}
        n, data_time = 0, 0.0
        log_every = self.cfg.train.get("log_period", 50)
        tic = time.time()
        for it, (imgs, pids, _) in enumerate(self.train_loader):
            data_time += time.time() - tic
            imgs = imgs.to(self.device, non_blocking=True)
            pids = pids.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.use_amp):
                out = self.model(imgs)
                losses = {
                    "ce": self.w_ce * self.ce(out["logits"], pids),
                    "tri": self.w_tri * self.tri(out["feat"], pids),
                }
                if self.teachers:
                    losses.update(self._kd_terms(out, imgs, pids))
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
            tic = time.time()
        if frozen:
            self._set_backbone_frozen(False)
        stats = {k: v / max(n, 1) for k, v in meters.items()}
        stats["data_s"] = data_time
        return stats

    def fit(self) -> float:
        save_config(self.cfg, self.out_dir / "config.yaml")
        self.try_resume()
        t = self.cfg.train
        last = None
        if self.freeze_epochs and self.start_epoch < self.freeze_epochs:
            self.logger.info(f"Backbone frozen for the first {self.freeze_epochs} epochs")
        for epoch in range(self.start_epoch, t.epochs):
            tic = time.time()
            stats = self.train_one_epoch(epoch)
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            data_s = stats.pop("data_s")
            self.logger.info(
                f"Epoch {epoch + 1}/{t.epochs} done in {time.time() - tic:.0f}s "
                f"(data {data_s:.0f}s) | lr {lr:.2e} | "
                + " ".join(f"{k}={v:.3f}" for k, v in stats.items())
            )
            if (epoch + 1) % t.get("eval_period", 10) == 0 or epoch + 1 == t.epochs:
                r = evaluate(self.model, self.test_loader, self.num_query, self.device)
                last = (epoch + 1, r)
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
        if last is not None:
            ep, r = last
            self.logger.info(
                f"Final (epoch {ep}) mAP {r['mAP']:.1%} | R1 {r['rank1']:.1%}  <- report this"
            )
        self.logger.info(f"Best mAP {self.best_map:.1%}")
        return self.best_map
