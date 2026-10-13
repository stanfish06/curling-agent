import argparse
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .physics import Physics

N_STONES = 16


class ShotDataset(Dataset):
    def __init__(self, root: Path):
        self.files = sorted(Path(root).rglob("*.npz"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        d = np.load(self.files[i])
        return {k: torch.as_tensor(d[k]) for k in ("t", "pos", "vel", "spin", "active")}


def collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    T = max(len(x["t"]) for x in items)
    out = {k: torch.stack([x[k] for x in items]) for k in ("spin", "active")}
    out["t"] = torch.stack(
        [
            torch.nn.functional.pad(x["t"], (0, T - len(x["t"])), value=float("nan"))
            for x in items
        ]
    )
    for k in ("pos", "vel"):
        out[k] = torch.stack(
            [
                torch.nn.functional.pad(
                    x[k], (0, 0, 0, 0, 0, T - len(x[k])), value=float("nan")
                )
                for x in items
            ]
        )
    return out


class PhysicsTrainer(L.LightningModule):
    def __init__(self, lr: float = 0.01, substeps: int = 10):
        super().__init__()
        self.physics = Physics()
        self.lr = lr
        self.substeps = substeps

    def rollout(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        t, pos, vel = batch["t"], batch["pos"], batch["vel"]
        frame_dt = torch.nanmean(t[:, 1:] - t[:, :-1]).item()
        pos0 = torch.nan_to_num(pos[:, 0])
        vel0 = torch.nan_to_num(vel[:, 0])
        _, _, _, _, frames = self.physics(
            pos0,
            vel0,
            batch["spin"],
            batch["active"],
            dt=frame_dt / self.substeps,
            max_steps=(t.shape[1] - 1) * self.substeps,
            record_every=self.substeps,
        )
        frames = frames.transpose(0, 1)[:, : t.shape[1]]
        pad = t.shape[1] - frames.shape[1]
        return torch.cat([frames, frames[:, -1:].expand(-1, pad, -1, -1)], 1)

    def training_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        pred = self.rollout(batch)
        obs = batch["pos"]
        mask = ~torch.isnan(obs).any(-1)
        err = ((pred - torch.nan_to_num(obs)) ** 2).sum(-1)
        loss = (err * mask).sum() / mask.sum().clamp(min=1)
        self.log("loss", loss, prog_bar=True)
        for k in self.physics.log:
            self.log(k, getattr(self.physics, k))
        self.log("restitution", self.physics.restitution)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train(
    root: Path, out: Path, epochs: int = 50, batch_size: int = 8, lr: float = 0.01
) -> Path:
    loader = DataLoader(
        ShotDataset(root), batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    model = PhysicsTrainer(lr=lr)
    trainer = L.Trainer(max_epochs=epochs, default_root_dir=out, log_every_n_steps=1)
    trainer.fit(model, loader)
    ckpt = out / "physics.ckpt"
    trainer.save_checkpoint(ckpt)
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/physics"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()
    print(train(args.root, args.out, args.epochs, args.batch_size, args.lr))
