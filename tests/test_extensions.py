import numpy as np
import pytest

from radiofisher.extensions import (
    frequency_noise_penalty,
    residual_power,
    validate_experiment_extensions,
    validate_residual_power,
    validate_volume_fraction,
)


FREQUENCIES = np.linspace(400.0, 800.0, 9)


def test_scalar_frequency_weight_is_broadcast():
    assert frequency_noise_penalty(lambda nu: 0.5, FREQUENCIES) == pytest.approx(2.0)


def test_exact_shape_weight_supports_both_modes():
    weights = np.linspace(0.5, 1.0, FREQUENCIES.size)
    weight_fn = lambda nu: weights

    assert frequency_noise_penalty(weight_fn, FREQUENCIES, "invvar") == pytest.approx(
        1.0 / np.mean(weights)
    )
    assert frequency_noise_penalty(weight_fn, FREQUENCIES, "fourier") == pytest.approx(
        np.mean(1.0 / weights)
    )


def test_nan_means_excised_and_all_nan_means_infinite_noise():
    weights = np.ones(FREQUENCIES.shape)
    weights[0] = np.nan
    assert frequency_noise_penalty(lambda nu: weights, FREQUENCIES) == 1.0
    assert np.isinf(
        frequency_noise_penalty(
            lambda nu: np.full(nu.shape, np.nan), FREQUENCIES
        )
    )


def test_wrong_frequency_weight_shape_is_rejected():
    with pytest.raises(ValueError, match="returned shape"):
        frequency_noise_penalty(lambda nu: np.ones(3), FREQUENCIES)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1, np.inf, -np.inf])
def test_invalid_frequency_weights_are_rejected(value):
    with pytest.raises(ValueError):
        frequency_noise_penalty(lambda nu: value, FREQUENCIES)


def test_invalid_frequency_mode_is_rejected():
    with pytest.raises(ValueError, match="noise_freq_mode"):
        frequency_noise_penalty(lambda nu: 1.0, FREQUENCIES, "average")


@pytest.mark.parametrize("value", [0.0, 0.25, 1.0])
def test_volume_fraction_accepts_closed_unit_interval(value):
    assert validate_volume_fraction(value) == value


@pytest.mark.parametrize("value", [True, np.bool_(False), [0.5]])
def test_volume_fraction_rejects_booleans_and_non_scalars(value):
    with pytest.raises(TypeError):
        validate_volume_fraction(value)


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf])
def test_volume_fraction_rejects_out_of_range_or_nonfinite_values(value):
    with pytest.raises(ValueError):
        validate_volume_fraction(value)


def test_experiment_extension_validation_fails_closed():
    with pytest.raises(TypeError, match="callable"):
        validate_experiment_extensions({"noise_freq_weight": [1.0]})
    with pytest.raises(ValueError, match="noise_freq_mode"):
        validate_experiment_extensions({"noise_freq_mode": "unknown"})


@pytest.mark.parametrize("value", [0.0, 0.25, np.float64(2.0)])
def test_residual_power_accepts_finite_nonnegative_scalars(value):
    assert validate_residual_power(value) == float(value)
    validate_experiment_extensions({"P_res": value})


@pytest.mark.parametrize("value", [True, np.bool_(False), [0.5], "0.5"])
def test_residual_power_rejects_non_real_scalars(value):
    with pytest.raises(TypeError, match="P_res"):
        validate_residual_power(value)
    with pytest.raises(TypeError, match="P_res"):
        validate_experiment_extensions({"P_res": value})


@pytest.mark.parametrize("value", [-0.01, np.nan, np.inf, -np.inf])
def test_residual_power_rejects_negative_or_nonfinite_scalars(value):
    with pytest.raises(ValueError, match="P_res"):
        validate_residual_power(value)
    with pytest.raises(ValueError, match="P_res"):
        validate_experiment_extensions({"P_res": value})


def test_residual_power_callable_broadcasts_to_the_fisher_grid():
    k = np.ones((2, 3))
    u = np.ones((2, 3))
    noise = np.full((2, 3), 4.0)
    signal = np.full((2, 3), 2.0)

    scalar = residual_power(lambda *_args: 0.5, k, u, noise, signal)
    rows = residual_power(
        lambda *_args: np.array([[1.0], [2.0]]), k, u, noise, signal
    )

    assert scalar.shape == k.shape
    assert np.all(scalar == 0.5)
    assert np.array_equal(rows, np.array([[1.0] * 3, [2.0] * 3]))


@pytest.mark.parametrize(
    "result,exception,match",
    [
        (np.ones(4), ValueError, "not broadcastable"),
        (np.array([[1.0, np.nan, 1.0]]), ValueError, "finite"),
        (np.array([[1.0, np.inf, 1.0]]), ValueError, "finite"),
        (np.array([[1.0, -0.1, 1.0]]), ValueError, "non-negative"),
        (np.array([[True, False, True]]), TypeError, "real numeric"),
        (np.array([[1.0 + 1.0j]]), TypeError, "real numeric"),
    ],
)
def test_residual_power_callable_output_fails_closed(result, exception, match):
    grid = np.ones((2, 3))
    with pytest.raises(exception, match=match):
        residual_power(lambda *_args: result, grid, grid, grid, grid)
