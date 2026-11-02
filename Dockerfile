# Supply version@sha256 references recorded in .env for reproducible builds.
ARG PYTHON_IMAGE=python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.5@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7
FROM ${UV_IMAGE} AS uv_binary
FROM ${PYTHON_IMAGE} AS bootstrap
COPY --from=uv_binary /uv /uvx /usr/local/bin/
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/workspace/.cache/uv \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace

FROM bootstrap AS runtime
# These apt operations run inside the image, never on the Ubuntu host.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg libegl1 libgl1 libgles2 libgl1-mesa-dri \
    libglib2.0-0 libgomp1 gcc libc6-dev linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/project
COPY pyproject.toml uv.lock .python-version ./
RUN UV_CACHE_DIR=/tmp/uv-cache uv sync --locked --no-dev --no-install-project --no-cache
# Register the EGL library supplied at runtime by NVIDIA Container Toolkit.
# This JSON does not install a driver into the image.
COPY configs/egl/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2
WORKDIR /workspace
USER 65532:65532
CMD ["sleep", "infinity"]
