from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .physics import HANDLE_SPIN, TEE_LINE_Y

N_STONES = 16
SAM3_ID = "facebook/sam3"
TEXT_PROMPT = "curling stone"


def constant_velocity(dt: float, sigma_a: float, sigma_z: float):
    F = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    G = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
    Q = sigma_a**2 * G @ G.T
    R = sigma_z**2 * np.eye(2)
    return F, H, Q, R


def filter_track(
    z: np.ndarray, dt: float = 1 / 30, sigma_a: float = 50.0, sigma_z: float = 3.0
):
    F, H, Q, R = constant_velocity(dt, sigma_a, sigma_z)
    n = len(z)
    xs, Ps, xp, Pp = (
        np.zeros((n, 4)),
        np.zeros((n, 4, 4)),
        np.zeros((n, 4)),
        np.zeros((n, 4, 4)),
    )
    first = np.flatnonzero(~np.isnan(z[:, 0]))[0]
    x = np.array([z[first, 0], z[first, 1], 0, 0])
    P = np.diag([sigma_z**2, sigma_z**2, 1e4, 1e4])
    for i in range(n):
        if i > first:
            x, P = F @ x, F @ P @ F.T + Q
        xp[i], Pp[i] = x, P
        if not np.isnan(z[i, 0]):
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z[i] - H @ x)
            P = (np.eye(4) - K @ H) @ P
        xs[i], Ps[i] = x, P
    return xs, Ps, xp, Pp, F


def smooth_track(
    z: np.ndarray, dt: float = 1 / 30, sigma_a: float = 50.0, sigma_z: float = 3.0
):
    xs, Ps, xp, Pp, F = filter_track(z, dt, sigma_a, sigma_z)
    out = xs.copy()
    for i in range(len(z) - 2, -1, -1):
        C = Ps[i] @ F.T @ np.linalg.pinv(Pp[i + 1])
        out[i] = xs[i] + C @ (out[i + 1] - xp[i + 1])
    return out


def extract_frames(video: Path, out_dir: Path, start: float, end: float) -> float:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    for i in range(int((end - start) * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), frame)
    cap.release()
    return fps


def track(
    frames_dir: Path,
    prompt: str = TEXT_PROMPT,
    model_id: str = SAM3_ID,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    model = Sam3VideoModel.from_pretrained(model_id).to(device)
    processor = Sam3VideoProcessor.from_pretrained(model_id)
    frames = [
        np.asarray(Image.open(f).convert("RGB"))
        for f in sorted(Path(frames_dir).glob("*.jpg"))
    ]
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
    )
    session = processor.add_text_prompt(inference_session=session, text=prompt)
    per_frame = {}
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(inference_session=session):
            res = processor.postprocess_outputs(session, out)
            per_frame[out.frame_idx] = (
                res["object_ids"].tolist(),
                res["masks"].cpu().numpy(),
            )
    ids = sorted({i for ids, _ in per_frame.values() for i in ids})
    centroids = np.full((len(frames), len(ids), 2), np.nan)
    for f, (obj_ids, masks) in per_frame.items():
        for obj, m in zip(obj_ids, masks):
            ys, xs = np.nonzero(m)
            if len(xs):
                centroids[f, ids.index(obj)] = xs.mean(), ys.mean()
    return centroids, np.array(ids)


def to_sheet(
    px: np.ndarray, tee_px: tuple[float, float], m_per_px: float
) -> np.ndarray:
    x = (px[..., 0] - tee_px[0]) * m_per_px
    y = TEE_LINE_Y - (px[..., 1] - tee_px[1]) * m_per_px
    return np.stack([x, y], -1)


def handle_sign(xs: np.ndarray) -> float:
    v0, v1 = xs[0, 2:], xs[-1, 2:]
    return float(np.sign(v0[0] * v1[1] - v0[1] * v1[0])) or 1.0


def segment(
    video: Path,
    start: float,
    end: float,
    tee_px: tuple[float, float],
    m_per_px: float,
    out: Path,
    thrown: int = 0,
    device: str = "cpu",
) -> Path:
    frames_dir = out.with_suffix("") / "frames"
    fps = extract_frames(video, frames_dir, start, end)
    px, ids = track(frames_dir, device=device)
    z = to_sheet(px, tee_px, m_per_px)
    T, K = z.shape[:2]
    pos = np.full((T, N_STONES, 2), np.nan)
    vel = np.full((T, N_STONES, 2), np.nan)
    spin = np.zeros(N_STONES)
    active = np.zeros(N_STONES, bool)
    for k in range(K):
        xs = smooth_track(z[:, k], dt=1 / fps)
        pos[:, k], vel[:, k] = xs[:, :2], xs[:, 2:]
        active[k] = True
    spin[thrown] = HANDLE_SPIN * handle_sign(smooth_track(z[:, thrown], dt=1 / fps))
    t = np.arange(T) / fps
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, t=t, pos=pos, vel=vel, spin=spin, active=active, px=px, ids=ids)
    return out
