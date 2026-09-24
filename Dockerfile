# syntax=docker/dockerfile:1

# Ultralytics' Jetson image already carries the JetPack CUDA, cuDNN and
# TensorRT builds of torch, torchvision and OpenCV. Installing those from PyPI
# instead would pull CPU-only aarch64 wheels, so this project is installed
# with --no-deps on top of them. The tag matches the Ultralytics version the
# stack is tested against.
ARG BASE_IMAGE=ultralytics/ultralytics:8.4.15-jetson-jetpack6

FROM ${BASE_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/app/.config/Ultralytics

WORKDIR /app

RUN pip install "beamngpy==1.35"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY configs ./configs
COPY scripts ./scripts

# Non-root, but in the video group, which owns the Jetson GPU device nodes.
RUN useradd --create-home --groups video app \
    && mkdir -p /app/models /app/output "$YOLO_CONFIG_DIR" \
    && chown -R app:app /app
USER app

# Ultralytics resolves the YOLOE text encoder (mobileclip2_b.ts) from its
# weights dir; pointing that at the mounted models folder keeps the container
# from downloading it on every start.
RUN yolo settings weights_dir=/app/models

ENTRYPOINT ["offroad-autonomy"]
CMD ["--config", "configs/jetson.yaml"]


FROM runtime AS test

RUN pip install --user "pytest>=7.0"
COPY --chown=app:app tests ./tests
ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-q"]
