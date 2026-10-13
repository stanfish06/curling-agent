import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .datasets import download_youtube_async
from .process import crop_video

ap = argparse.ArgumentParser()
ap.add_argument("--download", default=False, action="store_true")
ap.add_argument("--process", default=False, action="store_true")
MAX_DOWNLOAD_CONCURRENT = 5


async def cli(args):
    from importlib.resources import files

    import yaml

    src_path = Path(files("curling"))
    out_path = src_path.parent.parent / "data"
    text = (src_path / "data" / "yt_catalog.yaml").read_text()
    datasets = yaml.safe_load(text) or {}
    if args.download:
        sem = asyncio.Semaphore(MAX_DOWNLOAD_CONCURRENT)

        async def limited(info: dict):
            async with sem:
                return await download_youtube_async(info)

        results = await asyncio.gather(
            *(limited({"out": out_path / k, **v}) for k, v in datasets.items()),
            return_exceptions=True,
        )
        for k, r in zip(datasets, results):
            if isinstance(r, Exception):
                print(f"{k}: FAILED: {r}")

    if args.process:
        need_process = [k for k, v in datasets.items() if "process" in v]
        crop_jobs = []
        for k in need_process:
            if "crop" in datasets[k]["process"]:
                x, y, w, h = datasets[k]["process"]["crop"]
                crop_jobs.append(
                    {
                        "video": out_path / k / "raw.mp4",
                        "out": out_path / k / "processed.mp4",
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                    }
                )
        with ProcessPoolExecutor() as exec:
            futures = [exec.submit(crop_video, **job) for job in crop_jobs]
            for job, f in zip(crop_jobs, futures):
                print(f"{job['video']}\n  -> {job['out']}")
                f.result()


def main():
    args = ap.parse_args()
    asyncio.run(cli(args))
