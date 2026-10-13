import argparse
import asyncio
from pathlib import Path

from .datasets import download_youtube_async

ap = argparse.ArgumentParser()
ap.add_argument("--download", default=False, action="store_true")
MAX_DOWNLOAD_CONCURRENT = 5


async def cli(args):
    if args.download:
        from importlib.resources import files

        import yaml

        sem = asyncio.Semaphore(MAX_DOWNLOAD_CONCURRENT)

        src_path = Path(files("curling"))
        out_path = src_path.parent.parent / "data"
        text = (src_path / "data" / "yt_catalog.yaml").read_text()
        datasets = yaml.safe_load(text) or {}

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


def main():
    args = ap.parse_args()
    asyncio.run(cli(args))
