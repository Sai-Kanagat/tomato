from pathlib import Path
import time

import numpy as np
import yaml

from src.biomarkers.kcf_estimator import CNNBiGRUAttentionKCF
from src.frontend.pipeline import FaceBlurrer, PromptHMR, PromptHMRVid
from src.kinetics.gait_dynamics import GaitDynamicsPipeline


ROOT = Path(__file__).resolve().parents[1]
FORCE_FEATURE_COUNT = 3


def test_stub_pipeline_latency_within_configured_budget():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    latency_budget_ms = float(config["performance"]["latency_ms_max"])

    frame = np.zeros((1, 16, 16, 3), dtype=np.float32)
    frames = np.zeros((2, 16, 16, 3), dtype=np.float32)

    t0 = time.perf_counter()
    blurred = FaceBlurrer().blur(frame)
    _ = PromptHMR().infer(blurred)
    q_t = PromptHMRVid().infer_sequence(frames)
    kinetics = GaitDynamicsPipeline().build_window_tensor()
    additional_features = np.zeros((q_t.shape[0], FORCE_FEATURE_COUNT), dtype=np.float32)
    _ = CNNBiGRUAttentionKCF().predict(np.hstack([q_t, additional_features]))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert kinetics.shape == (180, 32)
    assert elapsed_ms <= latency_budget_ms
