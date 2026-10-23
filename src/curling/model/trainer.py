import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from .physics import Physics, Surface
from .policy import MAX_SCORE, PolicyValueNet

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
    def __init__(
        self, lr: float = 0.01, substeps: int = 10, surface: dict | None = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.physics = Physics(surface=Surface(surface))
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

    def errors(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.rollout(batch)
        obs = batch["pos"]
        mask = ~torch.isnan(obs).any(-1)
        err = ((pred - torch.nan_to_num(obs)) ** 2).sum(-1)
        last = mask.shape[1] - 1 - mask.flip(1).float().argmax(1)
        final = err.gather(1, last[:, None]).squeeze(1).sqrt() * mask.any(1)
        return (err * mask).sum() / mask.sum().clamp(min=1), final.sum() / mask.any(
            1
        ).sum().clamp(min=1)

    def training_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        loss, final = self.errors(batch)
        self.log("loss", loss, prog_bar=True)
        self.log("final_m", final, prog_bar=True)
        for k in self.physics.log:
            self.log(k, getattr(self.physics, k))
        self.log("restitution", self.physics.restitution)
        for k, p in self.physics.surface.named_parameters():
            for i, v in enumerate(p.flatten()) if p.requires_grad else ():
                self.log(f"surface.{k}.{i}", v)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], _: int) -> None:
        loss, final = self.errors(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_final_m", final, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train(
    root: Path,
    out: Path,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 0.01,
    val: float = 0.25,
    surface: dict | None = None,
) -> Path:
    data = ShotDataset(root)
    n_val = int(len(data.files) * val)
    val_data, train_data = ShotDataset(root), ShotDataset(root)
    val_data.files, train_data.files = data.files[:n_val], data.files[n_val:]
    loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate)
    model = PhysicsTrainer(lr=lr, surface=surface)
    trainer = L.Trainer(max_epochs=epochs, default_root_dir=out, log_every_n_steps=1)
    print("before:", trainer.validate(model, val_loader)[0] if n_val else {})
    trainer.fit(model, loader, val_loader if n_val else None)
    if n_val:
        print("after:", trainer.validate(model, val_loader)[0])
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
    ap.add_argument("--val", type=float, default=0.25)
    ap.add_argument("--surface", type=Path)
    args = ap.parse_args()
    surface = yaml.safe_load(args.surface.read_text()) if args.surface else None
    print(
        train(
            args.root,
            args.out,
            args.epochs,
            args.batch_size,
            args.lr,
            args.val,
            surface,
        )
    )


K_CANDIDATES = 16
TEMPERATURE = 0.5


def utilities(net: PolicyValueNet, stones, context) -> torch.Tensor:
    _, value = net(stones, context, torch.zeros(stones.shape[0], 1, 3), stones[:, None])
    return value.softmax(-1) @ torch.arange(
        -MAX_SCORE, MAX_SCORE + 1, dtype=torch.float32
    )


def choose(net: PolicyValueNet | None, obs, acts, nxt, temperature: float) -> int:
    if net is None:
        return int(np.random.randint(len(acts)))
    stones, context = (torch.as_tensor(x)[None] for x in obs)
    with torch.no_grad():
        logits, _ = net(
            stones,
            context,
            torch.as_tensor(acts, dtype=torch.float32)[None],
            torch.as_tensor(nxt)[None],
        )
    return int(torch.multinomial((logits[0] / temperature).softmax(-1), 1))


def play_end(
    state_dict,
    seed: int,
    k: int = K_CANDIDATES,
    temperature: float = 1.0,
    opponent=None,
    policy_team: int = 0,
) -> list[dict]:
    from curling.sim import End, observation

    torch.set_num_threads(1)
    net = None
    if state_dict is not None:
        net = PolicyValueNet()
        net.load_state_dict(state_dict)
        net.eval()
    end = End(seed=seed)
    out = []
    while not end.done:
        obs = end.observe()
        acts, npos, nact = end.candidates(k)
        nxt = np.stack(
            [observation(p, a, min(end.shot + 1, 15))[0] for p, a in zip(npos, nact)]
        )
        nctx = np.stack(
            [observation(p, a, min(end.shot + 1, 15))[1] for p, a in zip(npos, nact)]
        )
        mover = (
            net
            if opponent is None or end.team == policy_team
            else (None if opponent == "random" else opponent)
        )
        j = choose(mover, obs, acts, nxt, temperature)
        out.append(
            {
                "stones": obs[0],
                "context": obs[1],
                "actions": acts.astype(np.float32),
                "next_stones": nxt,
                "next_context": nctx,
                "chosen": j,
                "sign": 1.0 if end.team == 0 else -1.0,
            }
        )
        (_, _), score, _ = end.step(acts[j])
    stones, context = end.observe()
    for t in out:
        t["score"] = score
    out.append({"stones": stones, "context": context, "score": score, "terminal": True})
    return out


def collate_policy(items: list[dict]) -> dict[str, torch.Tensor]:
    f = lambda key, dtype=torch.float32: torch.stack(
        [torch.as_tensor(x[key], dtype=dtype) for x in items]
    )
    out = {
        k: f(k)
        for k in ("stones", "context", "actions", "next_stones", "next_context", "sign")
    }
    out["chosen"] = f("chosen", torch.long)
    out["score"] = f("score", torch.long) + MAX_SCORE
    return out


class PolicyTrainer(L.LightningModule):
    def __init__(self, lr: float = 1e-3, temperature: float = TEMPERATURE):
        super().__init__()
        self.net = PolicyValueNet()
        self.lr = lr
        self.temperature = temperature

    def training_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        logits, value = self.net(
            batch["stones"], batch["context"], batch["actions"], batch["next_stones"]
        )
        value_loss = torch.nn.functional.cross_entropy(value, batch["score"])
        k = batch["actions"].shape[1]
        with torch.no_grad():
            u = utilities(
                self.net,
                batch["next_stones"].flatten(0, 1),
                batch["next_context"].flatten(0, 1),
            )
            target = (
                u.unflatten(0, (-1, k)) * batch["sign"][:, None] / self.temperature
            ).softmax(-1)
        policy_loss = -(target * logits.log_softmax(-1)).sum(-1).mean()
        self.log("value_loss", value_loss, prog_bar=True)
        self.log("policy_loss", policy_loss, prog_bar=True)
        return value_loss + policy_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)


def generate(
    state_dict,
    n_ends: int,
    seed: int,
    workers: int,
    k: int,
    temperature: float,
    opponent=None,
    policy_team: int = 0,
) -> list[dict]:
    with ProcessPoolExecutor(workers) as ex:
        ends = list(
            ex.map(
                play_end,
                [state_dict] * n_ends,
                range(seed, seed + n_ends),
                [k] * n_ends,
                [temperature] * n_ends,
                [opponent] * n_ends,
                [policy_team] * n_ends,
            )
        )
    return [t for e in ends for t in e]


def evaluate(state_dict, n_ends: int, seed: int, workers: int, k: int) -> float:
    half = n_ends // 2
    first = generate(state_dict, half, seed, workers, k, 1e-3, "random", 0)
    second = generate(state_dict, half, seed + half, workers, k, 1e-3, "random", 1)
    scores = [e["score"] for e in first if e.get("terminal")] + [
        -e["score"] for e in second if e.get("terminal")
    ]
    return float(np.mean(scores))


def train_policy(
    out: Path,
    rounds: int = 10,
    ends: int = 64,
    epochs: int = 5,
    k: int = K_CANDIDATES,
    workers: int = 8,
    lr: float = 1e-3,
    temperature: float = TEMPERATURE,
    sample_temperature: float = 1.0,
    buffer_rounds: int = 4,
    eval_ends: int = 128,
    mixed: bool = False,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    model = PolicyTrainer(lr=lr, temperature=temperature)
    buffer: list[dict] = []
    state = None
    for r in range(rounds):
        if mixed:
            new = (
                generate(state, ends // 2, r * ends, workers, k, sample_temperature)
                + generate(
                    state,
                    ends // 4,
                    r * ends + ends // 2,
                    workers,
                    k,
                    sample_temperature,
                    "random",
                    0,
                )
                + generate(
                    state,
                    ends // 4,
                    r * ends + 3 * ends // 4,
                    workers,
                    k,
                    sample_temperature,
                    "random",
                    1,
                )
            )
        else:
            new = generate(state, ends, r * ends, workers, k, sample_temperature)
        buffer += [t for t in new if not t.get("terminal")]
        buffer = buffer[-ends * buffer_rounds * 16 :]
        loader = DataLoader(
            buffer, batch_size=64, shuffle=True, collate_fn=collate_policy
        )
        trainer = L.Trainer(
            max_epochs=epochs,
            default_root_dir=out,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model, loader)
        state = {k_: v.cpu() for k_, v in model.net.state_dict().items()}
        score = evaluate(state, eval_ends, 10_000 + r * eval_ends, workers, k)
        m = trainer.callback_metrics
        print(
            f"round {r}: buffer {len(buffer)} shots, value_loss {float(m['value_loss']):.3f}, "
            f"policy_loss {float(m['policy_loss']):.3f}, mean end score vs random, "
            f"sides alternated {score:+.2f}",
            flush=True,
        )
        torch.save(state, out / "policy.pt")
    return out / "policy.pt"


def policy_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/policy"))
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--ends", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--k", type=int, default=K_CANDIDATES)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--sample-temperature", type=float, default=1.0)
    ap.add_argument("--buffer-rounds", type=int, default=4)
    ap.add_argument("--eval-ends", type=int, default=128)
    ap.add_argument("--mixed", action="store_true")
    args = ap.parse_args()
    print(
        train_policy(
            args.out,
            args.rounds,
            args.ends,
            args.epochs,
            args.k,
            args.workers,
            args.lr,
            args.temperature,
            args.sample_temperature,
            args.buffer_rounds,
            args.eval_ends,
            args.mixed,
        )
    )
