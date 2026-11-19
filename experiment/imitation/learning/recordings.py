"""Bounded MCAP iteration directly inside ZIP archives, with embedded ROS schemas."""

from __future__ import annotations

import hashlib
import json
import zipfile
import zlib
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


def discover_recordings(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    sessions = []
    skipped = []
    seen = set()
    candidates = []
    loose_groups = defaultdict(list)
    for path in sorted(paths):
        if path.suffix.lower() == ".mcap":
            crc = 0
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024**2), b""):
                    crc = zlib.crc32(block, crc)
            loose_groups[(path.parent.resolve(), _session_prefix(path.name))].append(
                (path.name, path.stat().st_size, crc)
            )
            continue
        else:
            groups = defaultdict(list)
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.filename.lower().endswith(".mcap"):
                        group = str(Path(member.filename).parent / _session_prefix(member.filename))
                        groups[group].append((member.filename, member.file_size, member.CRC))
        if not groups:
            skipped.append(
                {
                    "source": path.name,
                    "reason": "No MCAP Members; Derived Data Is Not Another Session",
                }
            )
        candidates.extend((path, group, members) for group, members in groups.items())
    candidates.extend(
        (parent, prefix, members) for (parent, prefix), members in loose_groups.items()
    )
    for path, group, members in candidates:
        members.sort(key=lambda entry: _segment_number(entry[0]))
        signature = [(Path(name).name, size, crc) for name, size, crc in members]
        digest = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        if digest in seen:
            skipped.append({"source": path.name, "reason": "Duplicate Recording Members"})
            continue
        seen.add(digest)
        sessions.append(
            {
                "id": digest[:16],
                "source": str(path.resolve()),
                "group": group,
                "members": [entry[0] for entry in members],
                "signature": signature,
            }
        )
    return sessions, skipped


def _session_prefix(name: str):
    stem = Path(name).stem
    prefix, _, suffix = stem.rpartition("_")
    if suffix.isdigit():
        return prefix
    return stem


def _segment_number(name: str):
    tail = Path(name).stem.rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail)
    return 0


@contextmanager
def open_member(session: dict, member: str):
    source = Path(session["source"])
    if source.is_dir():
        with (source / member).open("rb") as stream:
            yield stream
    else:
        with zipfile.ZipFile(source) as archive, archive.open(member) as stream:
            yield stream


def messages(session: dict, topics: list[str] | None = None):
    from mcap.records import Channel, Chunk, Message, Schema
    from mcap.stream_reader import StreamReader, breakup_chunk
    from mcap_ros2.decoder import DecoderFactory

    for member in session["members"]:
        with open_member(session, member) as stream:
            reader = StreamReader(
                stream,
                emit_chunks=True,
                validate_crcs=True,
                record_size_limit=256 * 1024**2,
            )
            decoder = DecoderFactory()
            schemas = {}
            channels = {}
            for outer in reader.records:
                records = [outer]
                if isinstance(outer, Chunk):
                    # Compressed data must not bypass the bounded-record memory budget.
                    if outer.uncompressed_size > 256 * 1024**2:
                        raise ValueError("MCAP chunk exceeds the 256 MiB inspection limit")
                    records = breakup_chunk(outer, validate_crc=True)
                for record in records:
                    if isinstance(record, Schema):
                        schemas[record.id] = record
                    elif isinstance(record, Channel):
                        channels[record.id] = record
                    elif isinstance(record, Message):
                        channel = channels[record.channel_id]
                        if topics is not None and channel.topic not in topics:
                            continue
                        schema = schemas[channel.schema_id]
                        decode = decoder.decoder_for(channel.message_encoding, schema)
                        if decode is None:
                            raise ValueError("MCAP channel needs a supported CDR/ros2msg schema")
                        yield member, channel.topic, record, decode(record.data)


def header_ns(decoded) -> int:
    return int(decoded.header.stamp.sec) * 1_000_000_000 + int(decoded.header.stamp.nanosec)
