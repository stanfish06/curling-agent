import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .physics import FCV1, HANDLE_SPIN, TEE_LINE_Y, Physics, Surface, release
from .segment import N_STONES, handle_sign, smooth_track, stitch_tracks
from .trainer import PhysicsTrainer, collate

CHASKA = {"tee_px": (165.8, 787.2), "m_per_px": 0.01218, "y_min_px": 560}
MIN_SPEED = 0.15
MIN_ENTRY_SPEED = 0.5
MAX_SPEEDUP = 1.15
MAX_TURN = np.deg2rad(30)
MAX_KINK = np.deg2rad(10)
KINK_FRAMES = 15
MIN_FRAMES = 20
LINK_GAP = 45
LINK_RADIUS = 0.25
TILT_SPEC = {"height": [{"tilt": {"x": 0.0, "y": 0.0, "learn": True}}]}
SCHEMA = "curling.estimation/1"
VIEW_Y = 35.6
SYNTHETIC_TRUTH = {"height": [{"tilt": {"x": 1.5e-4, "y": 0.0}}]}
SYNTHETIC_NOISE = 0.02


def window_tracks(path: Path, calib: dict = CHASKA) -> tuple[np.ndarray, float, float]:
    d = np.load(path)
    px, fps, start = d["px"].copy(), float(d["fps"]), float(d["start"])
    px[px[..., 1] < calib["y_min_px"]] = np.nan
    px = stitch_tracks(px)
    tx, ty = calib["tee_px"]
    m = calib["m_per_px"]
    z = np.stack([-(px[..., 0] - tx) * m, TEE_LINE_Y + (px[..., 1] - ty) * m], -1)
    return z, fps, start


def free_slide(xs: np.ndarray) -> bool:
    v = xs[:, 2:]
    speed = np.linalg.norm(v, axis=1)
    heading = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    local = np.abs(heading[KINK_FRAMES:] - heading[:-KINK_FRAMES]).max()
    return (
        speed[0] >= MIN_ENTRY_SPEED
        and speed.max() <= MAX_SPEEDUP * speed[0]
        and abs(heading[-1] - heading[0]) <= MAX_TURN
        and local <= MAX_KINK
    )


def runs_of(xs: np.ndarray, seen: np.ndarray) -> list[tuple[int, int]]:
    moving = np.linalg.norm(xs[:, 2:], axis=1) > MIN_SPEED
    idx = np.flatnonzero(seen)
    moving[: idx[0]] = moving[idx[-1] + 1 :] = False
    edges = np.flatnonzero(np.diff(np.r_[0, moving.astype(int), 0]))
    runs = []
    for a, b in zip(edges[::2], edges[1::2]):
        inside = idx[(idx >= a) & (idx < b)]
        if len(inside) >= MIN_FRAMES:
            runs.append((int(inside[0]), int(inside[-1]) + 1))
    return runs


def link_moving(z: np.ndarray, fps: float) -> np.ndarray:
    z = z.copy()
    seen = ~np.isnan(z[..., 0])
    first = np.where(seen.any(0), seen.argmax(0), -1)
    for k in range(z.shape[1]):
        while True:
            if seen[:, k].sum() < MIN_FRAMES:
                break
            xs = smooth_track(z[:, k], dt=1 / fps)
            last = np.flatnonzero(seen[:, k])[-1]
            if np.linalg.norm(xs[last, 2:]) < MIN_SPEED:
                break
            best = None
            for j in range(z.shape[1]):
                gap = first[j] - last
                if j == k or first[j] < 0 or gap <= 0 or gap > LINK_GAP:
                    continue
                pred = xs[last, :2] + xs[last, 2:] * gap / fps
                d = np.linalg.norm(z[first[j], j] - pred)
                if d < LINK_RADIUS * (1 + gap / 15) and (best is None or d < best[0]):
                    best = (d, j)
            if best is None:
                break
            j = best[1]
            z[seen[:, j], k] = z[seen[:, j], j]
            z[:, j] = np.nan
            seen[:, k] |= seen[:, j]
            seen[:, j] = False
            first[j] = -1
    return z


def shots_from_window(path: Path, calib: dict = CHASKA) -> list[dict]:
    z, fps, start = window_tracks(path, calib)
    z = link_moving(z, fps)
    seen = ~np.isnan(z[..., 0])
    shots = []
    for k in range(z.shape[1]):
        if seen[:, k].sum() < MIN_FRAMES:
            continue
        xs = smooth_track(z[:, k], dt=1 / fps)
        for a, b in runs_of(xs, seen[:, k]):
            if not free_slide(xs[a:b]):
                continue
            others = [j for j in np.flatnonzero(seen[a]) if j != k][: N_STONES - 1]
            ids = [k, *others]
            pos = np.full((b - a, N_STONES, 2), np.nan)
            vel = np.full((b - a, N_STONES, 2), np.nan)
            pos[:, 0] = np.where(seen[a:b, k, None], xs[a:b, :2], np.nan)
            vel[:, 0] = xs[a:b, 2:]
            for i, j in enumerate(others, 1):
                pos[:, i] = np.where(seen[a:b, j, None], z[a:b, j], np.nan)
                vel[:, i] = 0.0
            spin = np.zeros(N_STONES)
            spin[0] = HANDLE_SPIN * handle_sign(xs[a:b])
            active = np.zeros(N_STONES, bool)
            active[: len(ids)] = True
            shots.append(
                {
                    "t": np.arange(b - a) / fps,
                    "pos": pos,
                    "vel": vel,
                    "spin": spin,
                    "active": active,
                    "window": Path(path).stem,
                    "frame0": a,
                    "time": start + a / fps,
                }
            )
    return shots


def shots_from_dir(root: Path, calib: dict = CHASKA) -> list[dict]:
    return [
        s for p in sorted(Path(root).glob("*.npz")) for s in shots_from_window(p, calib)
    ]


def to_batch(shots: list[dict]) -> dict[str, torch.Tensor]:
    keys = ("t", "pos", "vel", "spin", "active")
    return collate(
        [
            {
                k: torch.as_tensor(
                    s[k], dtype=torch.float32 if k != "active" else torch.bool
                )
                for k in keys
            }
            for s in shots
        ]
    )


def synthetic_shots(n: int, seed: int = 0, fps: float = 30.0) -> list[dict]:
    rng = np.random.default_rng(seed)
    truth = Physics({**FCV1, "friction_a": FCV1["friction_a"] * 0.9})
    truth.surface = Surface(SYNTHETIC_TRUTH)
    shots = []
    for i in range(n):
        speed = torch.tensor([float(rng.uniform(2.3, 2.5))])
        handle = torch.tensor([float(rng.choice([-1.0, 1.0]))])
        angle = torch.tensor([float(np.pi / 2 - handle * rng.uniform(0.03, 0.07))])
        vel, spin = release(speed, angle, handle)
        active = torch.ones(1, 1, dtype=torch.bool)
        with torch.no_grad():
            frames = truth(
                torch.zeros(1, 1, 2),
                vel[None],
                spin[None],
                active,
                0.001,
                60000,
                int(1000 / fps),
            )[4]
        z = frames[:, 0, 0].numpy()
        z = z[z[:, 1] > VIEW_Y] + rng.normal(
            0, SYNTHETIC_NOISE, (len(z[z[:, 1] > VIEW_Y]), 2)
        )
        xs = smooth_track(z, dt=1 / fps)
        pos = np.full((len(z), N_STONES, 2), np.nan)
        vel_ = np.full((len(z), N_STONES, 2), np.nan)
        pos[:, 0], vel_[:, 0] = z, xs[:, 2:]
        spins = np.zeros(N_STONES)
        spins[0] = float(spin)
        act = np.zeros(N_STONES, bool)
        act[0] = True
        shots.append(
            {
                "t": np.arange(len(z)) / fps,
                "pos": pos,
                "vel": vel_,
                "spin": spins,
                "active": act,
                "window": f"synthetic_{i:02d}",
                "frame0": 0,
                "time": float(i * 60),
            }
        )
    return shots


class Estimator:
    def __init__(
        self,
        surface: dict | None = None,
        lr: float = 0.01,
        steps: int = 5,
        buffer: int = 8,
        substeps: int = 10,
    ):
        self.model = PhysicsTrainer(
            lr, substeps, TILT_SPEC if surface is None else surface
        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr)
        self.steps, self.buffer = steps, buffer
        self.shots: list[dict] = []
        self.history: list[dict] = []
        self.records: list[dict] = []

    def state(self) -> dict[str, float]:
        p = self.model.physics
        out = {k: float(getattr(p, k)) for k in p.log}
        out["restitution"] = float(p.restitution)
        for name, m in zip(("tilt_x", "tilt_y"), (0, 1)):
            tilt = p.surface.fields["height"][0]
            out[name] = float(tilt.slope[m] * tilt.unit)
        return out

    def predict(self, shot: dict) -> np.ndarray:
        with torch.no_grad():
            return self.model.rollout(to_batch([shot]))[0].numpy()

    def observe(self, shot: dict) -> dict[str, float]:
        before = self.predict(shot)[:, 0]
        self.shots.append(shot)
        batch = to_batch(self.shots[-self.buffer :])
        loss = final = torch.zeros(())
        for _ in range(self.steps):
            self.opt.zero_grad()
            loss, final = self.model.errors(batch)
            loss.backward()
            self.opt.step()
        rec = {"loss": loss.item(), "final_m": final.item(), **self.state()}
        self.history.append(rec)
        others = shot["pos"][0, 1:][shot["active"][1:]]
        self.records.append(
            {
                "index": len(self.history),
                "window": shot.get("window", ""),
                "time": float(shot.get("time", 0.0)),
                "spin": float(shot["spin"][0]),
                "t": shot["t"].round(4).tolist(),
                "observed": [
                    None if np.isnan(p[0]) else p.round(4).tolist()
                    for p in shot["pos"][:, 0]
                ],
                "others": others.round(4).tolist(),
                "before": before.round(4).tolist(),
                "after": self.predict(shot)[:, 0].round(4).tolist(),
                "estimate": rec,
            }
        )
        return rec

    def record(self, source: str, calib: dict | None = None) -> dict:
        defaults = {**FCV1, "restitution": 1.0, "tilt_x": 0.0, "tilt_y": 0.0}
        return {
            "schema": SCHEMA,
            "source": source,
            "frame": {"tee_y": TEE_LINE_Y, "view_y": VIEW_Y, "units": "m"},
            "calib": calib,
            "defaults": defaults,
            "shots": self.records,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/estimation.json"))
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(4)
    shots = (
        shots_from_dir(args.windows)
        if args.windows
        else synthetic_shots(args.synthetic)
    )
    est = Estimator(steps=args.steps)
    for s in shots:
        r = est.observe(s)
        print(
            s["window"],
            f"loss={r['loss']:.4f}",
            f"tilt=({r['tilt_x']:+.1e}, {r['tilt_y']:+.1e})",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            est.record(
                "windows" if args.windows else "synthetic",
                CHASKA if args.windows else None,
            )
        )
    )
    print(args.out, "shots", len(shots))
