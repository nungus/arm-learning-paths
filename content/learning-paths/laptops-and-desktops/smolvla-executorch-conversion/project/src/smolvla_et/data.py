"""Real LeRobot inputs for numerical validation.

This module deliberately does not depend on LeRobot's dataset classes.  It reads
the small amount of metadata required by the split exporter, and imports PyAV
and PyArrow only when :func:`load_real_sample` is called.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn.functional as F


STATE_KEY = "observation.state"
_NORMALIZATION_EPSILON = 1e-8


@dataclass(frozen=True)
class RealSample:
    """One model-ready observation from a LeRobot dataset.

    ``images`` has shape ``[1, cameras, 3, image_size, image_size]`` in the
    supplied camera order and values in ``[-1, 1]``. ``state`` has shape
    ``[1, padded_state_dim]``. ``task`` always ends in a newline, matching the
    SmolVLA task processor.
    """

    images: torch.Tensor
    state: torch.Tensor
    task: str


def preprocess_image(image: torch.Tensor, image_size: int) -> torch.Tensor:
    """Apply SmolVLA's resize, top/left padding, and image normalization.

    Args:
        image: A float ``[3, height, width]`` tensor whose values are in
            ``[0, 1]``.
        image_size: The square output height and width.

    Returns:
        A contiguous float32 tensor of shape ``[3, image_size, image_size]``.

    SmolVLA preserves aspect ratio, truncates each resized dimension to an
    integer, and places all padding on the top and left. Padding is applied as
    zero in ``[0, 1]`` space, so it becomes ``-1`` after normalization.
    """

    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected an image shaped [3, H, W], got {tuple(image.shape)}")
    if not image.is_floating_point():
        raise TypeError("image must be floating point with values in [0, 1]")

    image = image.to(dtype=torch.float32).unsqueeze(0)
    current_height, current_width = image.shape[-2:]
    ratio = max(current_width / image_size, current_height / image_size)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = F.interpolate(
        image,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )

    pad_height = max(0, image_size - resized_height)
    pad_width = max(0, image_size - resized_width)
    padded = F.pad(resized, (pad_width, 0, pad_height, 0), value=0.0)
    return (padded.squeeze(0) * 2.0 - 1.0).contiguous()


def normalize_and_pad_state(
    state: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    padded_state_dim: int,
) -> torch.Tensor:
    """Mean/std-normalize a state vector and zero-pad its final dimension."""

    state = torch.as_tensor(state, dtype=torch.float32).flatten()
    mean = torch.as_tensor(mean, dtype=torch.float32).flatten()
    std = torch.as_tensor(std, dtype=torch.float32).flatten()
    if state.shape != mean.shape or state.shape != std.shape:
        raise ValueError(
            "state and statistics must have the same shape, got "
            f"{tuple(state.shape)}, {tuple(mean.shape)}, and {tuple(std.shape)}"
        )
    if padded_state_dim < state.numel():
        raise ValueError(
            f"padded_state_dim={padded_state_dim} is smaller than the "
            f"state dimension {state.numel()}"
        )

    normalized = (state - mean) / (std + _NORMALIZATION_EPSILON)
    padded = torch.zeros((1, padded_state_dim), dtype=torch.float32)
    padded[0, : state.numel()] = normalized
    return padded


def _load_task(dataset_root: Path) -> str:
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PyArrow is required to load real SmolVLA data"
        ) from error

    task_path = dataset_root / "meta" / "tasks.parquet"
    if not task_path.is_file():
        raise FileNotFoundError(f"task metadata does not exist: {task_path}")
    table = parquet.read_table(task_path, columns=["task"])
    if table.num_rows == 0:
        raise ValueError(f"task metadata is empty: {task_path}")
    task = table.column("task")[0].as_py()
    if not isinstance(task, str) or not task:
        raise ValueError(f"first task is not a non-empty string: {task!r}")
    return task if task.endswith("\n") else task + "\n"


def _load_state_statistics(dataset_root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"dataset statistics do not exist: {stats_path}")
    with stats_path.open(encoding="utf-8") as stats_file:
        all_statistics = json.load(stats_file)
    try:
        state_statistics = all_statistics[STATE_KEY]
        mean = state_statistics["mean"]
        std = state_statistics["std"]
    except KeyError as error:
        raise ValueError(
            f"{stats_path} lacks {STATE_KEY!r} mean/std statistics"
        ) from error
    return (
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(std, dtype=torch.float32),
    )


def _load_states(dataset_root: Path, row_indices: tuple[int, ...]) -> list[torch.Tensor]:
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PyArrow is required to load real SmolVLA data"
        ) from error

    data_paths = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"no data Parquet files found under {dataset_root / 'data'}")

    results: list[torch.Tensor | None] = [None] * len(row_indices)
    next_result = 0
    first_row_in_file = 0
    for data_path in data_paths:
        parquet_file = parquet.ParquetFile(data_path)
        rows_in_file = parquet_file.metadata.num_rows
        end_row = first_row_in_file + rows_in_file
        if next_result < len(row_indices) and row_indices[next_result] < end_row:
            table = parquet_file.read(columns=[STATE_KEY])
            state_column = table.column(STATE_KEY)
            while (
                next_result < len(row_indices)
                and row_indices[next_result] < end_row
            ):
                local_row = row_indices[next_result] - first_row_in_file
                state = state_column[local_row].as_py()
                results[next_result] = torch.tensor(state, dtype=torch.float32)
                next_result += 1
        first_row_in_file = end_row
        if next_result == len(row_indices):
            break

    if next_result != len(row_indices):
        raise IndexError(
            f"requested dataset row {row_indices[next_result]}, but only "
            f"{first_row_in_file} rows are available"
        )
    return [state for state in results if state is not None]


def _camera_video_paths(dataset_root: Path, camera_name: str) -> list[Path]:
    candidates = (
        dataset_root / "videos" / camera_name,
        dataset_root / "videos" / f"observation.images.{camera_name}",
    )
    for camera_root in candidates:
        paths = sorted(camera_root.glob("chunk-*/file-*.mp4"))
        if paths:
            return paths
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"no videos found for camera {camera_name!r}; searched {searched}")


def _decode_camera_frames(
    video_paths: list[Path],
    row_indices: tuple[int, ...],
    image_size: int,
) -> list[torch.Tensor]:
    try:
        import av
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PyAV is required to decode real SmolVLA frames"
        ) from error

    frames: list[torch.Tensor] = []
    next_target = 0
    global_frame_index = 0
    for video_path in video_paths:
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                if global_frame_index == row_indices[next_target]:
                    rgb = frame.to_ndarray(format="rgb24")
                    image = (
                        torch.from_numpy(rgb)
                        .permute(2, 0, 1)
                        .to(dtype=torch.float32)
                        .div_(255.0)
                    )
                    frames.append(preprocess_image(image, image_size))
                    next_target += 1
                    if next_target == len(row_indices):
                        return frames
                global_frame_index += 1

    raise IndexError(
        f"requested video frame {row_indices[next_target]}, but only "
        f"{global_frame_index} frames are available"
    )


def load_real_sample(
    dataset_root: Path,
    camera_names: tuple[str, ...],
    image_size: int,
    padded_state_dim: int,
) -> RealSample:
    """Load the first synchronized, model-ready dataset observation.

    Frame zero is selected from every camera and from the Parquet observation
    rows. LeRobot writes one video frame per observation in dataset order, so
    walking each camera's sorted video shards by global frame ordinal keeps all
    inputs synchronized even when cameras have different shard boundaries.

    Args:
        dataset_root: Root of a LeRobot v3 dataset.
        camera_names: Camera feature names, in model input order. Both full
            names such as ``observation.images.overhead_cam`` and short names
            such as ``overhead_cam`` are accepted.
        image_size: Square SmolVLA input size.
        padded_state_dim: State width expected by the exported model.
    """

    dataset_root = Path(dataset_root).expanduser()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    if not camera_names:
        raise ValueError("camera_names must contain at least one camera")
    if len(set(camera_names)) != len(camera_names):
        raise ValueError(f"camera_names contains duplicates: {camera_names!r}")
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if padded_state_dim <= 0:
        raise ValueError(
            f"padded_state_dim must be positive, got {padded_state_dim}"
        )
    row_indices = (0,)
    task = _load_task(dataset_root)
    state_mean, state_std = _load_state_statistics(dataset_root)
    raw_states = _load_states(dataset_root, row_indices)

    frames_by_camera = [
        _decode_camera_frames(
            _camera_video_paths(dataset_root, camera_name),
            row_indices,
            image_size,
        )
        for camera_name in camera_names
    ]

    images = torch.stack(
        [camera_frames[0] for camera_frames in frames_by_camera], dim=0
    ).unsqueeze(0)
    state = normalize_and_pad_state(
        raw_states[0], state_mean, state_std, padded_state_dim
    )
    return RealSample(images=images.contiguous(), state=state, task=task)
