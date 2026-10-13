from pathlib import Path

import numpy as np


def crop_video(video: Path, out: Path, x: int, y: int, w: int, h: int):
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(
        filename=str(out), fourcc=fourcc, fps=fps, frameSize=(w, h)
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        roi = frame[y : y + h, x : x + w]
        out_writer.write(roi)

    cap.release()
    out_writer.release()


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
