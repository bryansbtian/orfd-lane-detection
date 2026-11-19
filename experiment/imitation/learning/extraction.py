"""Build a compact, timestamp-audited steering-proxy dataset without unpacking MCAP files."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from experiment.imitation.config import ImitationConfig
from experiment.imitation.learning.recordings import discover_recordings, header_ns, messages
from experiment.imitation.learning.synchronization import (
    aligned_indices,
    steering_evidence,
    temporal_split,
)


def crop_resize(image: np.ndarray, settings: ImitationConfig) -> np.ndarray:
    height = image.shape[0]
    top = int(height * settings.crop_top)
    bottom = int(height * (1 - settings.crop_bottom))
    return cv2.resize(
        image[top:bottom],
        (settings.image_width, settings.image_height),
        interpolation=cv2.INTER_AREA,
    )


def read_telemetry(session: dict, settings: ImitationConfig):
    rows = []
    calibration = None
    gps_cells = set()
    for _, topic, message, decoded in messages(
        session, [settings.can_topic, settings.camera_info_topic]
    ):
        if topic == settings.camera_info_topic:
            candidate = {
                "width": decoded.width,
                "height": decoded.height,
                "k": list(decoded.k),
                "d": list(decoded.d),
                "model": decoded.distortion_model,
                "extrinsics_verified": False,
            }
            if calibration is None:
                calibration = candidate
            elif candidate != calibration:
                raise ValueError(
                    "Camera calibration changed within a recording; split or rectify it before extraction"
                )
            continue
        timestamp = header_ns(decoded)
        if timestamp <= 0 or rows and timestamp <= rows[-1]["header_ns"]:
            raise ValueError("CAN acquisition timestamps must be positive and strictly increasing")
        row = {
            "header_ns": timestamp,
            "log_ns": message.log_time,
            "raw_rwa": float(decoded.rwa),
            "speed_mps": float(decoded.spd_mps),
            "driverbraking": int(decoded.driverbraking),
            "mode_raw": int(decoded.cdcu_ctrl_mode),
            "yaw_rate": float(decoded.imu.imu.angular_velocity.z),
            "error": bool(decoded.error or decoded.spd_error),
        }
        rows.append(row)
        lat, lon = decoded.fix.latitude, decoded.fix.longitude
        if math.isfinite(lat) and math.isfinite(lon) and abs(lat) < 85:
            # Only overlap fractions leave memory; coordinates cannot identify a route in artifacts.
            x = math.radians(lon) * 6371000
            y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * 6371000
            gps_cells.add((int(x / settings.route_cell_m), int(y / settings.route_cell_m)))
    if not rows or calibration is None:
        raise ValueError("Recording needs CAN telemetry and camera calibration")
    evidence = steering_evidence(rows, settings)
    if not evidence["passed"]:
        raise ValueError(
            f"Steering proxy failed the configured dynamics plausibility check: {evidence}"
        )
    return rows, calibration, evidence, gps_cells


def build_dataset(paths: list[Path], output: Path, settings: ImitationConfig) -> dict:
    from experiment.imitation.learning.video import TimestampedDecoder

    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Dataset directory is not empty: {output}; select a new output directory")
    sessions, skipped = discover_recordings(paths)
    if not sessions:
        raise ValueError("No original MCAP recordings found")
    output.mkdir(parents=True, exist_ok=True)
    input_stats = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}
    all_telemetry = []
    for session in sessions:
        print(f"Reading Telemetry: {Path(session['source']).name}", flush=True)
        all_telemetry.append(read_telemetry(session, settings))
    order = sorted(range(len(sessions)), key=lambda index: all_telemetry[index][0][0]["header_ns"])
    sessions = [sessions[index] for index in order]
    all_telemetry = [all_telemetry[index] for index in order]
    first_calibration = all_telemetry[0][1]
    for telemetry in all_telemetry[1:]:
        if telemetry[1] != first_calibration:
            raise ValueError(
                "Recordings have different calibration; documented rectification is required before mixing them"
            )
    for prior, following in pairwise(all_telemetry):
        if prior[0][-1]["header_ns"] >= following[0][0]["header_ns"]:
            raise ValueError("Overlapping recordings must be deduplicated before extraction")
    counts = Counter()
    rejected = Counter()
    pixel_hashes = set()
    session_reports = []
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for session_index, (session, telemetry) in enumerate(zip(sessions, all_telemetry)):
            rows, calibration, evidence, _ = telemetry
            times = np.array([row["header_ns"] for row in rows], dtype=np.int64)
            first = int(times[0])
            duration = (int(times[-1]) - first) / 1e9
            decoder = TimestampedDecoder(settings.cpu_threads)
            last_timestamp = None
            next_sample = 0
            frames_decoded = 0
            local_counts = Counter()
            image_dir = output / "frames" / session["id"]
            image_dir.mkdir(parents=True, exist_ok=True)
            print(f"Decoding Video: {Path(session['source']).name}", flush=True)

            context = {
                "times": times,
                "rows": rows,
                "first": first,
                "duration": duration,
                "session_index": session_index,
                "image_dir": image_dir,
                "session": session,
                "local_counts": local_counts,
            }

            def consume(frame, metadata, context=context):
                nonlocal next_sample, frames_decoded
                times, rows = context["times"], context["rows"]
                first, duration = context["first"], context["duration"]
                session_index, image_dir = context["session_index"], context["image_dir"]
                session, local_counts = context["session"], context["local_counts"]
                frames_decoded += 1
                timestamp = metadata["image_header_ns"]
                if timestamp < next_sample:
                    return
                next_sample = timestamp + int(1e9 / settings.sample_hz)
                joined = aligned_indices(times, timestamp, settings)
                if joined is None:
                    rejected["timestamp_alignment"] += 1
                    return
                prior_index, target_index = joined
                prior, target = rows[prior_index], rows[target_index]
                if (
                    prior["error"]
                    or target["error"]
                    or not all(math.isfinite(target[key]) for key in ("raw_rwa", "speed_mps"))
                ):
                    rejected["invalid_telemetry"] += 1
                    return
                if (
                    not math.isfinite(prior["speed_mps"])
                    or abs(target["raw_rwa"]) > settings.max_angle_rad
                ):
                    rejected["invalid_range"] += 1
                    return
                elapsed = (timestamp - first) / 1e9
                split = temporal_split(elapsed, duration, settings)
                if len(sessions) >= 3:
                    split = "train"
                    if session_index == len(sessions) - 2:
                        split = "validation"
                    if session_index == len(sessions) - 1:
                        split = "test"
                    if (
                        elapsed < settings.split_guard_s
                        or duration - elapsed < settings.split_guard_s
                    ):
                        split = "discard"
                if split == "discard":
                    rejected["split_guard"] += 1
                    return
                image = crop_resize(frame.to_ndarray(format="bgr24"), settings)
                pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
                if pixel_hash in pixel_hashes:
                    rejected["duplicate_pixels"] += 1
                    return
                pixel_hashes.add(pixel_hash)
                name = image_dir / f"{timestamp}.jpg"
                if not cv2.imwrite(
                    str(name), image, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
                ):
                    raise OSError(f"Could not write dataset frame: {name}")
                record = {
                    **metadata,
                    "session_id": session["id"],
                    "split": split,
                    "frame": name.relative_to(output).as_posix(),
                    "pixel_sha256": pixel_hash,
                    "speed_mps": prior["speed_mps"],
                    "speed_header_ns": prior["header_ns"],
                    "steering_proxy_rad": target["raw_rwa"] * settings.label_scale,
                    "raw_rwa": target["raw_rwa"],
                    "target_header_ns": target["header_ns"],
                    "alignment_ms": (target["header_ns"] - timestamp) / 1e6,
                    "driverbraking_raw": target["driverbraking"],
                    "mode_raw": target["mode_raw"],
                    "target_kind": "measured_steering_proxy",
                    "human_control_confirmed": False,
                    "throttle_target": None,
                    "brake_intensity_target": None,
                }
                manifest.write(json.dumps(record, allow_nan=False) + "\n")
                counts[split] += 1
                local_counts[split] += 1

            for ordinal, (member, _, message, decoded) in enumerate(
                messages(session, [settings.image_topic])
            ):
                timestamp = header_ns(decoded)
                if decoded.format.lower() != "h264":
                    raise ValueError("Only timestamped Annex B H.264 image messages are supported")
                if timestamp <= 0 or last_timestamp is not None and timestamp <= last_timestamp:
                    raise ValueError("Image acquisition timestamps must be positive and increasing")
                if (
                    last_timestamp is not None
                    and timestamp - last_timestamp > settings.max_gap_ms * 1e6
                ):
                    rejected["camera_gaps"] += 1
                    decoder.reset()
                last_timestamp = timestamp
                metadata = {
                    "image_header_ns": timestamp,
                    "image_log_ns": message.log_time,
                    "source_mcap": Path(member).name,
                    "source_ordinal": ordinal,
                }
                for frame, frame_metadata in decoder.push(bytes(decoded.data), metadata):
                    consume(frame, frame_metadata)
            for frame, frame_metadata in decoder.flush():
                consume(frame, frame_metadata)
            if decoder.pending:
                raise ValueError("Some video messages did not map to output frames")
            session_reports.append(
                {
                    "id": session["id"],
                    "source": Path(session["source"]).name,
                    "member_signatures": session["signature"],
                    "first_header_ns": first,
                    "last_header_ns": int(times[-1]),
                    "duration_s": duration,
                    "can_messages": len(rows),
                    "frames_decoded": frames_decoded,
                    "saved_counts": dict(local_counts),
                    "decoder_errors": decoder.errors,
                    "prefix_packets_skipped": decoder.prefix_packets_skipped,
                    "calibration": calibration,
                    "steering_evidence": evidence,
                }
            )
            print(f"Session Saved: {dict(local_counts)}", flush=True)
    if not all(counts[split] > 0 for split in ("train", "validation", "test")):
        raise ValueError("Extraction did not produce three nonempty disjoint splits")
    overlap = []
    for first_index in range(len(sessions)):
        for second_index in range(first_index + 1, len(sessions)):
            first_cells, second_cells = (
                all_telemetry[first_index][3],
                all_telemetry[second_index][3],
            )
            fraction = 0.0
            if second_cells:
                fraction = len(first_cells & second_cells) / len(second_cells)
            overlap.append(
                {
                    "first_session": sessions[first_index]["id"],
                    "second_session": sessions[second_index]["id"],
                    "second_route_cell_overlap_fraction": fraction,
                }
            )
    for path in paths:
        if input_stats[str(path)] != (path.stat().st_size, path.stat().st_mtime_ns):
            raise OSError("Source recording changed during extraction")
    report = {
        "version": 1,
        "settings": asdict(settings),
        "counts": dict(counts),
        "rejected": dict(rejected),
        "sessions": session_reports,
        "skipped_archives": skipped,
        "route_overlap": overlap,
        "human_control_confirmed": False,
        "evaluation_scope": "Temporally Held-Out Recordings; Geographic Overlap Is Reported, Not Unseen-Road Generalization",
        "source_files_unchanged": True,
    }
    (output / "dataset_metadata.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report
