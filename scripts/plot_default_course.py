#!/usr/bin/env python3
"""Show The Full BeamNG Map Selected By The Default Driving Configuration."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offroad_autonomy.utils.config import load_config


def draw_course(config_path: Path, archive_path: Path | None, output: Path) -> None:
    cfg = load_config(config_path)
    if archive_path is None:
        archive_path = Path(cfg.beamng_home) / "content" / "levels" / f"{cfg.beamng_map}.zip"
    prefix = f"levels/{cfg.beamng_map}/"
    with ZipFile(archive_path) as archive:
        info = json.loads(archive.read(prefix + "info.json"))
        terrain = None
        for name in archive.namelist():
            if not name.startswith(prefix + "main/") or not name.endswith("items.level.json"):
                continue
            for line in archive.read(name).decode("utf-8-sig").splitlines():
                if '"TerrainBlock"' not in line:
                    continue
                candidate = json.loads(line)
                if candidate.get("minimapImage"):
                    terrain = candidate
                    break
            if terrain is not None:
                break
        if terrain is None:
            raise ValueError("No Terrain Minimap And Coordinate Metadata Found In The Map Archive")
        terrain_json = Path(terrain["terrainFile"]).with_suffix(".terrain.json").as_posix()
        terrain_data = json.loads(archive.read(terrain_json.lstrip("/")))
        image_bytes = archive.read(terrain["minimapImage"].lstrip("/"))

    # The terrain's own minimap binding supplies the world extent; an eyeballed
    # image registration could put the configured spawn on the wrong road.
    x_min, y_min = terrain["position"][:2]
    width = float(terrain_data["size"]) * float(terrain["squareSize"])
    extent = (x_min, x_min + width, y_min, y_min + width)
    spawn = cfg.map_spawns[cfg.beamng_map]["spawns"][cfg.beamng_spawn_index]["pos"]
    image = plt.imread(io.BytesIO(image_bytes), format="png")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    figure, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(image, extent=extent, origin="upper")
    ax.scatter(
        spawn[0],
        spawn[1],
        marker="*",
        s=280,
        color="#00dbff",
        edgecolor="#152c40",
        linewidth=1.3,
        zorder=5,
    )
    ax.annotate(
        f"Spawn {cfg.beamng_spawn_index}",
        xy=spawn[:2],
        xytext=(12, 12),
        textcoords="offset points",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#17617a",
            "alpha": 0.95,
        },
        fontsize=11,
        zorder=6,
    )
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_aspect("equal")
    ax.set_xlabel("BeamNG World X (m)")
    ax.set_ylabel("BeamNG World Y (m)")
    ax.set_title(f"{info['title']}\nFull Map Loaded By {config_path.name}", fontsize=17, pad=15)
    figure.text(
        0.5,
        0.025,
        f"Cyan Star: Default Spawn {cfg.beamng_spawn_index} | {cfg.beamng_vehicle.title()} | "
        f"XYZ: ({spawn[0]:.2f}, {spawn[1]:.2f}, {spawn[2]:.2f}) m\n"
        f"Built-In Minimap: {info['authors']} | World Extent From Terrain Metadata\n"
        "The Driving Configuration Specifies A Spawn; It Does Not Specify A Fixed Lap.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.085, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    provenance = {
        "map": cfg.beamng_map,
        "config": str(config_path),
        "archive": str(archive_path),
        "minimap_member": terrain["minimapImage"],
        "minimap_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "terrain_position": terrain["position"],
        "terrain_size_samples": terrain_data["size"],
        "square_size_m": terrain["squareSize"],
        "world_extent_xy_m": extent,
        "spawn_index": cfg.beamng_spawn_index,
        "spawn_xyz_m": spawn,
        "vehicle": cfg.beamng_vehicle,
        "fixed_lap_specified": False,
    }
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Full Map Written: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    parser.add_argument("--level-archive", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "results/r3/default_course_map.png")
    args = parser.parse_args()
    draw_course(args.config, args.level_archive, args.out)


if __name__ == "__main__":
    main()
