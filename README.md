# offroad_autonomy

End-to-end autonomous off-road driving pipeline for [BeamNG.tech](https://www.beamng.tech/).

## Architecture

```
BeamNG frame → preprocessing → perception → postprocessing → planning → control → BeamNG
```

| Module | Responsibility |
|---|---|
| `preprocessing` | Resize, normalize, optional CLAHE contrast enhancement |
| `perception` | YOLOE-26 road segmentation → binary traversable mask |
| `postprocessing` | EMA temporal smoothing, morphological cleanup |
| `planning` | Centerline extraction + Kalman-filter heading/curvature tracking |
| `control` | Stanley lateral controller + proportional speed control |
| `simulation` | BeamNG.tech session lifecycle, camera, vehicle I/O |

## Quick Start

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run the autonomy pipeline (requires BeamNG.tech running)
python scripts/run_main.py --config configs/default.yaml

# Run tests (no GPU or BeamNG required)
pytest tests/ -v
```

## Configuration

All tunable parameters live in `configs/default.yaml`:
- BeamNG home path, map, vehicle, spawn points
- Model weights path and confidence threshold
- Preprocessing, postprocessing, planning, and control gains

## Project Layout

```
├── pyproject.toml
├── configs/
│   └── default.yaml
├── scripts/
│   └── run_main.py
├── tests/
│   ├── test_types.py
│   └── test_pipeline.py
└── src/
    └── offroad_autonomy/
        ├── main.py               # Entry point + main loop
        ├── pipeline.py           # Stage orchestration
        ├── types.py              # Shared dataclasses
        ├── preprocessing/
        ├── perception/
        ├── postprocessing/
        ├── planning/
        ├── control/
        ├── simulation/
        └── utils/
```

## Legacy Scripts

The original standalone scripts remain at the repository root for reference and
development workflows (benchmarking, fine-tuning):

- `segment_road.py` — CLI segmentation runner
- `beamng_live_inference.py` — Side-by-side model comparison
- `benchmark.py` — Benchmark suite with pass/fail thresholds
- `finetune.py` — ORFD → YOLO dataset conversion and training

## License

MIT
