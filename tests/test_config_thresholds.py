from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_core_thresholds_match_spec():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    assert config["performance"]["latency_ms_max"] <= 16.6
    assert config["performance"]["joint_angle_rmse_deg_max"] <= 4.5
    assert config["performance"]["grf_error_percent_max"] <= 3.9
    assert config["performance"]["kcf_r2_min"] >= 0.95
    assert config["ingestion"]["max_video_hz"] >= 120
    assert config["synergy"]["k_min"] == 5
    assert config["synergy"]["k_max"] == 8
