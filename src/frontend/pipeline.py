"""Frontend ingestion and promptable HPS model stubs."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VideoIngestionConfig:
    """Video ingestion contract for uncalibrated streams up to 120 Hz."""

    max_video_hz: int = 120


class FaceBlurrer:
    """On-device anonymization placeholder (face-blurring before transmission)."""

    def blur(self, frame: np.ndarray) -> np.ndarray:
        return frame.copy()


class PromptHMR:
    """PromptHMR placeholder.

    Spatial prompt fusion formula:
        T_fused = T_image + e_mask
    where e_box and e_mask are prompt embeddings aligned to image tokens.
    """

    def infer(self, image: np.ndarray) -> dict[str, np.ndarray]:
        batch = image.shape[0] if image.ndim == 4 else 1
        return {
            "pose_3d": np.zeros((batch, 24, 3), dtype=np.float32),
            "shape": np.zeros((batch, 10), dtype=np.float32),
        }


class PromptHMRVid:
    """Temporal PromptHMR-Vid placeholder returning world-grounded trajectories q(t)."""

    def infer_sequence(self, frames: np.ndarray) -> np.ndarray:
        # [time, dof] with 72 DoF as a mock full-body generalized coordinate vector.
        return np.zeros((frames.shape[0], 72), dtype=np.float32)
