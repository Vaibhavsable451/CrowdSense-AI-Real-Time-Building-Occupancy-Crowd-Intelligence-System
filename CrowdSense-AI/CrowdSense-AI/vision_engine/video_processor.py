"""Phase 1 - Video ingestion: webcam, MP4 files or RTSP streams."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Union

import cv2
import numpy as np


@dataclass
class FrameInfo:
    index: int          # index of the frame in the source
    timestamp_s: float  # position in the video (seconds)
    frame: np.ndarray   # BGR image


class VideoSource:
    """Reads frames from a file path, webcam index (0, 1...) or RTSP URL.

    target_fps: process at most this many frames per second (skips the rest).
    max_width:  downscale frames wider than this (keeps aspect ratio).
    """

    def __init__(self, source: Union[int, str] = 0, target_fps: float = 10.0,
                 max_width: int = 960):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.source = source
        self.target_fps = target_fps
        self.max_width = max_width
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.is_live = isinstance(source, int) or str(source).startswith(("rtsp", "http"))
        self.step = max(1, round(self.src_fps / target_fps)) if target_fps else 1

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.max_width and w > self.max_width:
            scale = self.max_width / w
            frame = cv2.resize(frame, (self.max_width, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        return frame

    def frames(self) -> Iterator[FrameInfo]:
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if idx % self.step == 0:
                yield FrameInfo(idx, idx / self.src_fps, self._resize(frame))
            idx += 1

    @property
    def effective_fps(self) -> float:
        return self.src_fps / self.step

    def release(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
