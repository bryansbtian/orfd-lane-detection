"""H.264 decoding with source acquisition timestamps preserved through delayed output."""

from __future__ import annotations

import re
from fractions import Fraction

import av

NAL_START = re.compile(b"\x00\x00\x01(.)", re.DOTALL)
TIME_BASE = Fraction(1, 1_000_000_000)


class TimestampedDecoder:
    def __init__(self, threads: int):
        self.context = av.CodecContext.create("h264", "r")
        self.context.thread_count = threads
        self.context.thread_type = "SLICE"
        self.ready = False
        self.pending: dict[int, dict] = {}
        self.errors = 0
        self.prefix_packets_skipped = 0

    def reset(self):
        self.context.flush_buffers()
        self.ready = False
        self.pending.clear()

    def push(self, payload: bytes, metadata: dict):
        kinds = {match[1][0] & 31 for match in NAL_START.finditer(payload)}
        if not self.ready:
            if not {5, 7, 8}.issubset(kinds):
                self.prefix_packets_skipped += 1
                return []
            self.ready = True
        timestamp = metadata["image_header_ns"]
        if timestamp in self.pending:
            raise ValueError("Duplicate acquisition timestamp in video stream")
        self.pending[timestamp] = metadata
        packet = av.Packet(payload)
        packet.pts = timestamp
        packet.dts = timestamp
        packet.time_base = TIME_BASE
        try:
            frames = self.context.decode(packet)
        except av.error.InvalidDataError:
            self.errors += 1
            self.reset()
            return []
        return self._associate(frames)

    def flush(self):
        if not self.ready:
            return []
        return self._associate(self.context.decode(None))

    def _associate(self, frames):
        result = []
        for frame in frames:
            if frame.pts is None:
                raise ValueError("Decoded frame lacks a source PTS; timestamp guessing is disabled")
            frame_time_base = frame.time_base
            if frame_time_base is None:
                frame_time_base = TIME_BASE
            timestamp = int(Fraction(frame.pts) * frame_time_base / TIME_BASE)
            metadata = self.pending.pop(timestamp, None)
            if metadata is None:
                raise ValueError("Decoded frame PTS has no unique source message")
            if self.context.has_b_frames:
                raise ValueError(
                    "Reordered H.264 requires explicit source DTS; this recording is unsupported"
                )
            result.append((frame, metadata))
        return result
