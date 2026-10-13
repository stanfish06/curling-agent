import asyncio
import functools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

MAX_ATTEMPTS = 5


def parse_time(t: str | None) -> float | None:
    if t is None:
        return None
    h, m, s = (float(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def download_youtube(
    url: str, out: Path, start_time: float | None = None, end_time: float | None = None
) -> int:
    import yt_dlp

    ydl_config = {
        "outtmpl": str(out.resolve() / "raw.%(ext)s"),
        "merge_output_format": "mp4",
        "format": "bestvideo+bestaudio/best",
        "retries": 10,
        "extractor_retries": 5,
        "socket_timeout": 30,
        "external_downloader_args": {
            "ffmpeg_i": [
                "-reconnect",
                "1",
                "-reconnect_streamed",
                "1",
                "-reconnect_delay_max",
                "30",
            ]
        },
    }
    if start_time is not None and end_time is not None:
        ydl_config["download_ranges"] = yt_dlp.utils.download_range_func(
            None, [(start_time, end_time)]
        )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_config) as ydl:
                return ydl.download(url)
        except yt_dlp.utils.DownloadError as e:
            if attempt == MAX_ATTEMPTS:
                raise
            delay = 10 * 2 ** (attempt - 1)
            print(f"{url}: attempt {attempt} failed ({e}), retrying in {delay}s")
            time.sleep(delay)
    return 1


async def download_youtube_async(info: dict) -> int:
    print(f"{info['title']}:\n  -> {info['out']}")
    info["out"].mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as exec:
        return await loop.run_in_executor(
            exec,
            functools.partial(
                download_youtube,
                url=info["url"],
                out=info["out"],
                start_time=parse_time(info.get("start_time")),
                end_time=parse_time(info.get("end_time")),
            ),
        )
