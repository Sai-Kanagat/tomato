import numpy as np
import pytest

from src.kinetics.gait_dynamics import GaitDynamicsPipeline
from src.simulation.muscle_mimic import SynergyController


def test_gait_dynamics_window_shape():
    window = GaitDynamicsPipeline().build_window_tensor()
    assert window.shape == (180, 32)


def test_synergy_output_is_bounded_and_high_dimensional():
    controller = SynergyController(muscle_count=416, synergy_count=5)
    c_t = np.array([0.2, 0.4, 0.8, 1.2, -0.1], dtype=np.float32)

    activations = controller.reconstruct(c_t)

    assert activations.shape == (416,)
    assert np.all(activations >= 0.0)
    assert np.all(activations <= 1.0)


def test_synergy_controller_rejects_invalid_sizes():
    with pytest.raises(ValueError):
        SynergyController(muscle_count=416, synergy_count=0)
