from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .physics import BACK_LINE_Y, HANDLE_SPIN, STONE_RADIUS, TEE_LINE_Y

N_STONES = 16
SAM3_ID = "facebook/sam3"
TEXT_PROMPT = "round object"
STONE_SIZE = (250, 650)
MIN_ROUND = 0.93
MAX_STEP_PX = 15.0
MAX_GAP_SCALE = 3
MIN_OBS = 10


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


def extract_frames(
    video: Path, out_dir: Path, start: float, end: float, step: int = 1
) -> float:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    for i in range(int((end - start) * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            cv2.imwrite(str(out_dir / f"{i // step:05d}.jpg"), frame)
    cap.release()
    return fps / step


PRUNE_FRAMES = 30
PRUNE_COND = 8
REMOVE_AFTER = 30
HIDDEN_AFTER = 60


class LazyFrames:
    def __init__(self, paths: list[Path], processor, cache: int = 4):
        self.paths, self.processor, self.size, self.cache = paths, processor, cache, {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        if i not in self.cache:
            if len(self.cache) >= self.size:
                self.cache.pop(min(self.cache))
            frame = np.asarray(Image.open(self.paths[i]).convert("RGB"))
            self.cache[i] = self.processor.video_processor(
                videos=[frame], return_tensors="pt"
            ).pixel_values_videos[0][0]
        return self.cache[i]


def prune_session(session, frame_idx: int, keep: int, keep_cond: int):
    for d in session.output_dict_per_obj.values():
        nc = d["non_cond_frame_outputs"]
        for f in [f for f in nc if f < frame_idx - keep]:
            del nc[f]
        cd = d["cond_frame_outputs"]
        for f in sorted(cd)[:-keep_cond]:
            del cd[f]
    for d in session.mask_inputs_per_obj.values():
        for f in [f for f in d if f < frame_idx - keep]:
            del d[f]
    for d in (session.suppressed_obj_ids, session.obj_id_to_tracker_score_frame_wise):
        for f in [f for f in d if f < frame_idx - keep]:
            del d[f]
    for d in (session.unmatched_frame_inds, session.overlap_pair_to_frame_inds):
        for v in d.values():
            del v[:-8]


def remove_unmatched(
    session,
    unmatched: dict,
    limit: int,
    hidden: dict | None = None,
    visible: set | None = None,
    hidden_limit: int = HIDDEN_AFTER,
):
    for obj_id in list(session.obj_ids):
        alive = session.trk_keep_alive.get(obj_id, 1)
        unmatched[obj_id] = unmatched.get(obj_id, 0) + 1 if alive <= 0 else 0
        gone = unmatched[obj_id] > limit
        if hidden is not None and visible is not None:
            hidden[obj_id] = hidden.get(obj_id, 0) + 1 if obj_id not in visible else 0
            gone = gone or hidden[obj_id] > hidden_limit
        if gone and len(session.obj_ids) > 1:
            pid = session.obj_id_to_prompt_id.get(obj_id)
            session.removed_obj_ids.add(obj_id)
            session.hotstart_removed_obj_ids.add(obj_id)
            session.obj_id_to_score[obj_id] = -1e4
            session.obj_id_to_last_occluded.pop(obj_id, None)
            session.remove_object(obj_id)
            if pid is not None:
                session.obj_id_to_prompt_id[obj_id] = pid
            unmatched.pop(obj_id)
            if hidden is not None:
                hidden.pop(obj_id, None)


def track(
    frames_dir: Path,
    prompt: str = TEXT_PROMPT,
    model_id: str = SAM3_ID,
    device: str = "cpu",
    stone_size: tuple[int, int] = STONE_SIZE,
    sam3: dict | None = None,
    boxes=None,
    prune: int | None = PRUNE_FRAMES,
    remove_after: int | None = REMOVE_AFTER,
    hidden_after: int = HIDDEN_AFTER,
    progress: int = 0,
    checkpoint: Path | None = None,
    start: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    import pickle
    import resource

    from transformers import Sam3VideoModel, Sam3VideoProcessor
    from transformers.models.sam3_video.modeling_sam3_video import nms_masks

    model = Sam3VideoModel.from_pretrained(model_id, **(sam3 or {})).to(device)
    processor = Sam3VideoProcessor.from_pretrained(model_id)
    all_paths = sorted(Path(frames_dir).glob("*.jpg"))
    paths = all_paths[start:]
    W, H = Image.open(paths[0]).size
    session = processor.init_video_session(
        inference_device=device,
        inference_state_device="cpu",
        processing_device="cpu",
        video_storage_device="cpu",
    )
    session.processed_frames = LazyFrames(paths, processor)
    session.video_height, session.video_width = H, W
    session = processor.add_text_prompt(inference_session=session, text=prompt)
    state = {"frame": 0}
    det_one = model._det_track_one_frame

    def det_one_frame(inference_session, frame_idx, *a, **k):
        state["frame"] = frame_idx
        return det_one(inference_session, frame_idx, *a, **k)

    model._det_track_one_frame = det_one_frame

    def run_detection(inference_session, vision_embeds):
        g = state["frame"] + start
        bx = boxes[g] if boxes is not None and g < len(boxes) else []
        kw = {}
        if bx:
            b = torch.tensor(bx, dtype=torch.float32, device=model.device)
            kw["input_boxes"] = torch.stack(
                [
                    (b[:, 0] + b[:, 2]) / 2 / W,
                    (b[:, 1] + b[:, 3]) / 2 / H,
                    (b[:, 2] - b[:, 0]) / W,
                    (b[:, 3] - b[:, 1]) / H,
                ],
                -1,
            )[None]
            kw["input_boxes_labels"] = torch.ones(
                1, len(bx), dtype=torch.long, device=model.device
            )
        dets = {}
        for pid in inference_session.prompts:
            if pid not in inference_session.prompt_embeddings:
                inference_session.prompt_embeddings[pid] = (
                    model.detector_model.get_text_features(
                        input_ids=inference_session.prompt_input_ids[pid],
                        attention_mask=inference_session.prompt_attention_masks[pid],
                        return_dict=True,
                    )
                )
            o = model.detector_model(
                vision_embeds=vision_embeds,
                text_embeds=inference_session.prompt_embeddings[pid],
                attention_mask=inference_session.prompt_attention_masks[pid],
                **kw,
            )
            probs = o.pred_logits.sigmoid() * o.presence_logits.sigmoid()
            if model.det_nms_thresh > 0.0:
                keep = nms_masks(
                    pred_probs=probs[0],
                    pred_masks=o.pred_masks[0],
                    prob_threshold=model.score_threshold_detection,
                    iou_threshold=model.det_nms_thresh,
                )
                probs[0][~keep] = 0.0
            idx = torch.where(probs > model.score_threshold_detection)
            dets[pid] = {
                "bbox": o.pred_boxes[idx[0], idx[1]],
                "mask": o.pred_masks[idx[0], idx[1]],
                "scores": probs[idx[0], idx[1]],
            }
        return dets

    def filtered_detection(inference_session, vision_embeds):
        dets = run_detection(inference_session, vision_embeds)
        for d in dets.values():
            m = d["mask"]
            area = (m > 0).flatten(1).sum(1) * (H * W / (m.shape[-1] * m.shape[-2]))
            keep = (area >= stone_size[0]) & (area <= stone_size[1])
            for k in d:
                d[k] = d[k][keep]
        return dets

    model.run_detection = filtered_detection
    found = {}
    unmatched = {}
    hidden = {}
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(inference_session=session):
            f = out.frame_idx
            res = processor.postprocess_outputs(session, out)
            g = f + start
            for obj, m in zip(res["object_ids"].tolist(), res["masks"].cpu().numpy()):
                c = round_mask(m, stone_size)
                if c is not None:
                    found.setdefault(obj, {})[g] = c
            if remove_after is not None:
                remove_unmatched(
                    session,
                    unmatched,
                    remove_after,
                    hidden,
                    set(res["object_ids"].tolist()),
                    hidden_after,
                )
            if prune is not None:
                prune_session(session, f, prune, PRUNE_COND)
            if progress and g % progress == 0:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
                print(
                    f"frame {g} objects {len(res['object_ids'])} session {len(session.obj_ids)} tracks {len(found)} maxrss {rss:.1f} GB",
                    flush=True,
                )
                if checkpoint is not None:
                    with open(checkpoint, "wb") as fh:
                        pickle.dump(found, fh)
    ids = sorted(found)
    centroids = np.full((len(all_paths), len(ids), 2), np.nan)
    for k, obj in enumerate(ids):
        for f, c in found[obj].items():
            centroids[f, k] = c
    return centroids, np.array(ids)


def round_mask(
    m: np.ndarray, stone_size: tuple[int, int] = STONE_SIZE
) -> tuple[float, float] | None:
    ys, xs = np.nonzero(m)
    if not stone_size[0] <= len(xs) <= stone_size[1]:
        return None
    cx, cy = xs.mean(), ys.mean()
    inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= len(xs) / np.pi
    if inside.mean() < MIN_ROUND:
        return None
    return cx, cy


def to_sheet(
    px: np.ndarray, tee_px: tuple[float, float], m_per_px: float
) -> np.ndarray:
    x = (px[..., 0] - tee_px[0]) * m_per_px
    y = TEE_LINE_Y - (px[..., 1] - tee_px[1]) * m_per_px
    return np.stack([x, y], -1)


def handle_sign(xs: np.ndarray) -> float:
    v0, v1 = xs[0, 2:], xs[-1, 2:]
    return float(np.sign(v0[0] * v1[1] - v0[1] * v1[0])) or 1.0


STITCH_GAP = 90
STITCH_RADIUS = 20.0
DUPLICATE_RADIUS = 10.0


def merge_duplicates(px: np.ndarray, radius: float = DUPLICATE_RADIUS) -> np.ndarray:
    seen = ~np.isnan(px[..., 0])
    order = np.argsort(-seen.sum(0))
    alive = np.ones(px.shape[1], bool)
    for a, i in enumerate(order):
        for j in order[a + 1 :]:
            if not (alive[i] and alive[j]):
                continue
            both = seen[:, i] & seen[:, j]
            if both.sum() < max(MIN_OBS, 1):
                continue
            d = np.linalg.norm(px[both, i] - px[both, j], axis=1)
            if np.median(d) <= radius:
                fill = seen[:, j] & ~seen[:, i]
                px[fill, i] = px[fill, j]
                seen[:, i] |= fill
                alive[j] = False
    return px[:, alive]


def split_jumps(
    px: np.ndarray, max_step: float = MAX_STEP_PX, min_obs: int = MIN_OBS
) -> np.ndarray:
    cols = []
    for k in range(px.shape[1]):
        fr = np.flatnonzero(~np.isnan(px[:, k, 0]))
        if len(fr) == 0:
            continue
        d = np.linalg.norm(np.diff(px[fr, k], axis=0), axis=1)
        allowed = max_step * np.minimum(np.diff(fr), MAX_GAP_SCALE)
        for seg in np.split(fr, np.flatnonzero(d > allowed) + 1):
            if len(seg) >= min_obs:
                col = np.full((px.shape[0], 2), np.nan)
                col[seg] = px[seg, k]
                cols.append(col)
    return np.stack(cols, 1) if cols else px[:, :0]


COLOR_HALF = 8
MIN_SAT = 90
STONE_SAT_FRAC = 0.5
YELLOW_HUE = (15, 40)
RED_HUE = (10, 165)


def piece_colors(px: np.ndarray, frames: np.ndarray) -> np.ndarray:
    import cv2

    out = np.zeros(px.shape[1], int)
    for k in range(px.shape[1]):
        votes = np.zeros(3, int)
        for t in np.flatnonzero(~np.isnan(px[:, k, 0])):
            x, y = px[t, k].astype(int)
            crop = frames[
                t,
                max(y - COLOR_HALF, 0) : y + COLOR_HALF + 1,
                max(x - COLOR_HALF, 0) : x + COLOR_HALF + 1,
            ]
            hsv = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_RGB2HSV).reshape(
                -1, 3
            )
            sat = hsv[:, 1] > MIN_SAT
            if sat.mean() < STONE_SAT_FRAC:
                votes[0] += 1
                continue
            h = np.median(hsv[sat, 0])
            if YELLOW_HUE[0] <= h <= YELLOW_HUE[1]:
                votes[2] += 1
            elif h <= RED_HUE[0] or h >= RED_HUE[1]:
                votes[1] += 1
            else:
                votes[0] += 1
        out[k] = int(np.argmax(votes))
    return out


DARK_V = 60
DARK_RADIUS = 13
MAX_DARK_FRAC = 0.1


def dark_fraction(px: np.ndarray, frames: np.ndarray) -> np.ndarray:
    H, W = frames.shape[1:3]
    out = np.zeros(px.shape[1])
    for k in range(px.shape[1]):
        d = []
        for t in np.flatnonzero(~np.isnan(px[:, k, 0])):
            x, y = px[t, k]
            y0, y1 = max(int(y) - DARK_RADIUS, 0), min(int(y) + DARK_RADIUS + 1, H)
            x0, x1 = max(int(x) - DARK_RADIUS, 0), min(int(x) + DARK_RADIUS + 1, W)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            disk = (yy - y) ** 2 + (xx - x) ** 2 <= DARK_RADIUS**2
            d.append((frames[t, y0:y1, x0:x1].max(-1)[disk] < DARK_V).mean())
        out[k] = np.median(d) if d else 0.0
    return out


def stitch_tracks(
    px: np.ndarray,
    max_gap: int = STITCH_GAP,
    radius: float = STITCH_RADIUS,
    frames: np.ndarray | None = None,
) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    if frames is not None:
        px = px[:, dark_fraction(px, frames) <= MAX_DARK_FRAC]
    px = merge_duplicates(split_jumps(px))
    n = px.shape[1]
    if n == 0:
        return px
    seen = ~np.isnan(px[..., 0])
    first = seen.argmax(0)
    last = px.shape[0] - 1 - seen[::-1].argmax(0)
    ends, starts = px[last, np.arange(n)], px[first, np.arange(n)]
    gap = first[None, :] - last[:, None]
    dist = np.linalg.norm(ends[:, None] - starts[None, :], axis=-1)
    ok = (gap > 0) & (gap <= max_gap) & (dist <= radius)
    if frames is not None:
        c = piece_colors(px, frames)
        ok &= (c[:, None] == c[None, :]) & (c[:, None] > 0)
    big = 1e9
    cost = np.concatenate(
        [
            np.where(ok, dist**2, big),
            np.where(np.eye(n, dtype=bool), (1.05 * radius) ** 2, big),
        ],
        1,
    )
    rows, cols = linear_sum_assignment(cost)
    nxt = {int(i): int(j) for i, j in zip(rows, cols) if j < n and ok[i, j]}
    alive = np.ones(n, bool)
    for i in set(range(n)) - set(nxt.values()):
        j = i
        while j in nxt:
            j = nxt[j]
            px[first[j] : last[j] + 1, i] = px[first[j] : last[j] + 1, j]
            px[:, j] = np.nan
            alive[j] = False
    return px[:, alive]


MIN_SPEED = 0.15
MIN_SHOT_FRAMES = 30
MIN_ENTRY_SPEED = 0.5
HANDLE_FRAMES = 45
PAD_FRAMES = 15


def meta_tokens(r: dict) -> list[str]:
    text = (r.get("meta") or {}).get("text", [])
    return [t.strip() for entry in text for t in entry.split(",") if t.strip()]


def handle_tags(metas: list[list[str]]) -> list[list[tuple[int, float]]]:
    out = []
    for m in metas:
        tags = []
        for t in m:
            name, _, frame = t.partition("@")
            if name in ("cw", "ccw", "thrown"):
                sign = {"cw": -1.0, "ccw": 1.0, "thrown": 0.0}[name]
                tags.append((int(frame) if frame else 0, sign))
        out.append(sorted(tags))
    return out


def tagged_handle(tags: list[tuple[int, float]], start: int, end: int) -> float | None:
    before = [s for f, s in tags if f <= start]
    inside = [s for f, s in tags if start < f < end]
    if inside:
        return inside[0]
    return before[-1] if before else None


def split_shots(
    z: np.ndarray,
    fps: float,
    out_dir: Path,
    prefix: str = "shot",
    handles: list[list[tuple[int, float]]] | None = None,
) -> list[Path]:
    z = np.where((z[..., 1:2] < BACK_LINE_Y + STONE_RADIUS), z, np.nan)
    T, K = z.shape[:2]
    seen = ~np.isnan(z[..., 0])
    keep = seen.sum(0) >= MIN_SHOT_FRAMES
    z = z[:, keep]
    handles = [h for h, k in zip(handles, keep) if k] if handles else None
    K = z.shape[1]
    seen = ~np.isnan(z[..., 0])
    xs = np.stack([smooth_track(z[:, k], dt=1 / fps) for k in range(K)], 1)
    moving = ((np.linalg.norm(xs[..., 2:], axis=-1) > MIN_SPEED) & seen).any(1)
    edges = np.flatnonzero(np.diff(np.r_[0, moving.astype(int), 0]))
    files = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
        b = min(b + PAD_FRAMES, T)
        tagged_ids = [
            k
            for k in range(K)
            if handles
            and seen[a:b, k].any()
            and tagged_handle(handles[k], a, b) is not None
        ]
        if not tagged_ids:
            continue
        thrown_id = tagged_ids[0]
        a = int(np.flatnonzero(seen[:, thrown_id] & (np.arange(T) >= a))[0])
        if b - a < MIN_SHOT_FRAMES:
            continue
        if np.nanmax(np.linalg.norm(xs[a:b, thrown_id, 2:], axis=-1)) < MIN_ENTRY_SPEED:
            continue
        ids = np.flatnonzero(seen[a])[:N_STONES]
        thrown = int(np.flatnonzero(ids == thrown_id)[0])
        pos = np.full((b - a, N_STONES, 2), np.nan)
        vel = np.full((b - a, N_STONES, 2), np.nan)
        pos[:, : len(ids)] = np.where(seen[a:b, ids, None], xs[a:b, ids, :2], np.nan)
        vel[:, : len(ids)] = xs[a:b, ids, 2:]
        spin = np.zeros(N_STONES)
        tagged = tagged_handle(handles[thrown_id], a, b) if handles else None
        spin[thrown] = HANDLE_SPIN * (
            tagged
            if tagged
            else handle_sign(xs[a : min(b, a + HANDLE_FRAMES), thrown_id])
        )
        active = np.zeros(N_STONES, bool)
        active[: len(ids)] = True
        files.append(out_dir / f"{prefix}{i:03d}.npz")
        np.savez(
            files[-1],
            t=np.arange(b - a) / fps,
            pos=pos,
            vel=vel,
            spin=spin,
            active=active,
            frame0=a,
        )
    return files


def segment(
    video: Path,
    start: float,
    end: float,
    tee_px: tuple[float, float],
    m_per_px: float,
    out_dir: Path,
    device: str = "cpu",
    stone_size: tuple[int, int] = STONE_SIZE,
    sam3: dict | None = None,
) -> list[Path]:
    frames_dir = out_dir / "frames"
    fps = extract_frames(video, frames_dir, start, end)
    px, ids = track(frames_dir, device=device, stone_size=stone_size, sam3=sam3)
    np.savez(out_dir / "tracks.npz", px=px, ids=ids, fps=fps, start=start)
    z = to_sheet(stitch_tracks(px), tee_px, m_per_px)
    return split_shots(z, fps, out_dir / "shots", prefix=f"{int(start)}_")


KEYFRAME_TOL = 1.5
BOX_PX = 2.4 * STONE_RADIUS


def keyframes(
    xy: np.ndarray, frames: np.ndarray, tol: float = KEYFRAME_TOL
) -> list[int]:
    keep = [0]
    i = 0
    while i < len(frames) - 1:
        j = i + 1
        while j + 1 < len(frames):
            w = (frames[i + 1 : j + 1] - frames[i]) / (frames[j + 1] - frames[i])
            line = xy[i] + w[:, None] * (xy[j + 1] - xy[i])
            if np.abs(xy[i + 1 : j + 1] - line).max() > tol:
                break
            j += 1
        keep.append(j)
        i = j
    return keep


def to_label_studio(
    px: np.ndarray,
    fps: float,
    size: tuple[int, int],
    video: str,
    labels: list[str],
    m_per_px: float,
    out: Path,
    rotate: bool = False,
) -> Path:
    import json

    W, H = size
    if rotate:
        px = np.stack([H - px[..., 1], px[..., 0]], -1)
        W, H = H, W
    T, K = px.shape[:2]
    box = BOX_PX / m_per_px
    result = []
    for k in range(K):
        seq = []
        fr = np.flatnonzero(~np.isnan(px[:, k, 0]))
        for seg in np.split(fr, np.flatnonzero(np.diff(fr) > 1) + 1):
            xy = px[seg, k]
            keep = keyframes(xy, seg)
            for n, i in enumerate(keep):
                seq.append(
                    {
                        "frame": int(seg[i]) + 1,
                        "x": float((xy[i, 0] - box / 2) / W * 100),
                        "y": float((xy[i, 1] - box / 2) / H * 100),
                        "width": float(box / W * 100),
                        "height": float(box / H * 100),
                        "enabled": n < len(keep) - 1,
                        "rotation": 0,
                        "time": float(seg[i] / fps),
                    }
                )
        if seq:
            result.append(
                {
                    "id": f"t{k}",
                    "type": "videorectangle",
                    "from_name": "box",
                    "to_name": "video",
                    "origin": "manual",
                    "value": {
                        "framesCount": T,
                        "duration": T / fps,
                        "sequence": seq,
                        "labels": [labels[k]],
                    },
                }
            )
    task = {
        "data": {"video": video},
        "predictions": [{"model_version": "sam3", "result": result}],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([task]))
    return out


def from_label_studio(
    path: Path,
    size: tuple[int, int],
    rotate: bool = False,
    meta: str | None = None,
) -> tuple[np.ndarray, list[str], list[list[str]]]:
    import json

    W, H = size
    if rotate:
        W, H = H, W
    task = json.loads(Path(path).read_text())[0]
    ann = task.get("annotations") or task["predictions"]
    boxes = [r for r in ann[0]["result"] if r["type"] == "videorectangle"]
    result = [r for r in boxes if meta is None or meta in meta_tokens(r)]
    links = [r for r in ann[0]["result"] if r["type"] == "relation"]
    T = int(boxes[0]["value"]["framesCount"]) if boxes else 0
    px = np.full((T, len(result), 2), np.nan)
    labels, metas = [], []
    for k, r in enumerate(result):
        v = r["value"]
        labels.append((v.get("labels") or [""])[0])
        metas.append(meta_tokens(r))
        seq = sorted(v["sequence"], key=lambda s: s["frame"])
        for a, b in zip(seq, seq[1:] + [None]):
            fa = a["frame"] - 1
            ca = np.array([a["x"] + a["width"] / 2, a["y"] + a["height"] / 2])
            if b is None or not a["enabled"]:
                px[fa, k] = ca
                continue
            fb = b["frame"] - 1
            cb = np.array([b["x"] + b["width"] / 2, b["y"] + b["height"] / 2])
            w = np.arange(fb - fa + 1) / max(fb - fa, 1)
            px[fa : fb + 1, k] = ca + w[:, None] * (cb - ca)
    group = list(range(len(result)))
    ids = {r["id"]: k for k, r in enumerate(result)}
    for r in links:
        if r["from_id"] in ids and r["to_id"] in ids:
            a, b = sorted((group[ids[r["from_id"]]], group[ids[r["to_id"]]]))
            group = [a if g == b else g for g in group]
    keep = sorted(set(group))
    merged = np.full((T, len(keep), 2), np.nan)
    for j, g in enumerate(keep):
        for k in [k for k in range(len(result)) if group[k] == g]:
            fill = np.isnan(merged[:, j, 0])
            merged[fill, j] = px[fill, k]
    px = merged
    labels = [labels[g] for g in keep]
    metas = [
        sorted({t for k in range(len(result)) if group[k] == g for t in metas[k]})
        for g in keep
    ]
    px *= np.array([W, H]) / 100
    if rotate:
        px = np.stack([px[..., 1], W - px[..., 0]], -1)
    return px, labels, metas
