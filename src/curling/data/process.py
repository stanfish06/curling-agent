from pathlib import Path


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
