#!/usr/bin/env python3
"""Regression checks for confidence calibration math and risk buckets."""

import sys

sys.path.insert(0, ".")

from backend.confidence_calibrator import ConfidenceCalibrator


def assert_close(a, b, tol=1e-6):
    assert abs(a - b) <= tol, f"Expected {b}, got {a}"


def run_tests():
    cal = ConfidenceCalibrator(domain="kaggle")

    # Formula check: calibrated = clip(raw * 1.3 - 0.08)
    expected_pairs = [
        (0.0, 0.0),
        (0.1, 0.05),
        (0.3, 0.31),
        (0.5, 0.57),
        (0.9, 1.0),
        (1.0, 1.0),
    ]

    for raw, expected in expected_pairs:
        calibrated = float(cal.calibrate_raw_confidence(raw))
        assert 0.0 <= calibrated <= 1.0, "Calibrated confidence out of [0,1]"
        assert_close(calibrated, expected)

    # Decision boundary check at threshold 0.3
    is_fake_low, _ = cal.get_decision(0.2)   # 0.18 calibrated -> REAL
    is_fake_edge, _ = cal.get_decision(0.3)  # 0.31 calibrated -> DEEPFAKE
    assert bool(is_fake_low) is False, "0.2 raw should be REAL"
    assert bool(is_fake_edge) is True, "0.3 raw should be DEEPFAKE"

    # Risk-level bucket checks
    confident_real = cal.get_metrics(0.0)
    uncertain_edge = cal.get_metrics(0.3)
    confident_fake = cal.get_metrics(0.9)

    assert confident_real["risk_level"] == "CONFIDENT"
    assert uncertain_edge["risk_level"] == "UNCERTAIN"
    assert confident_fake["risk_level"] == "CONFIDENT"

    print("All confidence calibrator tests passed")


if __name__ == "__main__":
    run_tests()
