"""Validation and calculations for optional experiment extensions."""

from collections.abc import Mapping
from numbers import Real

import numpy as np

from .resources import validate_experiment_resources


NOISE_FREQUENCY_SAMPLES = 2049
DEFAULT_NOISE_FREQ_MODE = "invvar"
NOISE_FREQ_MODES = frozenset({"invvar", "fourier"})
MIN_VOLUME_FRACTION = 0.0
MAX_VOLUME_FRACTION = 1.0


def _finite_scalar(value, name):
    """Return *value* as a finite float, rejecting booleans and arrays."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("%s must be a real scalar" % name)
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("%s must be finite" % name)
    return result


def validate_volume_fraction(value):
    """Validate and return a surviving survey-volume fraction in ``[0, 1]``."""

    fraction = _finite_scalar(value, "vol_frac")
    if not MIN_VOLUME_FRACTION <= fraction <= MAX_VOLUME_FRACTION:
        raise ValueError("vol_frac must be between 0 and 1 inclusive")
    return fraction


def validate_residual_power(value):
    """Validate an additive residual-power specification.

    A scalar is the non-negative residual-to-thermal power ratio.  A callable
    cannot be evaluated until the Fisher grid is available, so this boundary
    only verifies that it is callable; :func:`residual_power` validates its
    returned values before they enter an integrand.
    """

    if callable(value):
        return value
    ratio = _finite_scalar(value, "P_res")
    if ratio < 0.0:
        raise ValueError("P_res must be non-negative")
    return ratio


def residual_power(value, k, u, noise_power, signal_power):
    """Evaluate and validate ``P_res`` on a Fisher integration grid.

    Callable results may be scalar or broadcastable to the common input grid.
    Every resulting power value must be finite and non-negative; NaNs and
    infinities are never interpreted as excision in this bias-response path.
    """

    target_shape = np.broadcast_shapes(
        np.shape(k), np.shape(u), np.shape(noise_power), np.shape(signal_power)
    )
    if callable(value):
        raw = value(k, u, noise_power, signal_power)
        raw_array = np.asarray(raw)
        if raw_array.dtype.kind == "b" or np.iscomplexobj(raw_array):
            raise TypeError("P_res callable must return real numeric power values")
        try:
            values = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "P_res callable must return real numeric power values"
            ) from exc
    else:
        ratio = validate_residual_power(value)
        values = ratio * np.asarray(noise_power, dtype=float)

    try:
        values = np.broadcast_to(values, target_shape)
    except ValueError as exc:
        raise ValueError(
            "P_res callable returned shape %s, which is not broadcastable to %s"
            % (np.shape(values), target_shape)
        ) from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("P_res must produce only finite power values")
    if np.any(values < 0.0):
        raise ValueError("P_res must produce non-negative power values")
    return values


def validate_experiment_extensions(expt):
    """Fail closed when optional experiment extension values are malformed.

    The function intentionally does not mutate the caller's experiment
    dictionary.  It is cheap enough to call at the public calculation
    boundaries.
    """

    if not isinstance(expt, Mapping):
        raise TypeError("expt must be a mapping")

    validate_experiment_resources(expt)

    mode = expt.get("noise_freq_mode", DEFAULT_NOISE_FREQ_MODE)
    if mode not in NOISE_FREQ_MODES:
        raise ValueError(
            "noise_freq_mode must be one of %s" % sorted(NOISE_FREQ_MODES)
        )

    if "noise_freq_weight" in expt and not callable(expt["noise_freq_weight"]):
        raise TypeError("noise_freq_weight must be callable")

    if "vol_frac" in expt:
        validate_volume_fraction(expt["vol_frac"])

    if "P_res" in expt:
        validate_residual_power(expt["P_res"])


def frequency_noise_penalty(
    weight_fn, frequencies_mhz, mode=DEFAULT_NOISE_FREQ_MODE
):
    """Return the thermal-noise penalty for frequency-dependent flagging.

    ``weight_fn`` may return a scalar (broadcast across the band) or an array
    with exactly the same shape as ``frequencies_mhz``.  Finite weights are
    surviving-time fractions and must be in ``(0, 1]``.  NaNs explicitly mark
    excised slices.  Infinite values are rejected instead of being silently
    interpreted as excision.  If every slice is excised, ``np.inf`` is
    returned.
    """

    if mode not in NOISE_FREQ_MODES:
        raise ValueError(
            "noise_freq_mode must be one of %s" % sorted(NOISE_FREQ_MODES)
        )
    if not callable(weight_fn):
        raise TypeError("noise_freq_weight must be callable")

    frequencies = np.asarray(frequencies_mhz, dtype=float)
    if frequencies.size == 0:
        raise ValueError("frequencies_mhz must not be empty")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies_mhz must contain only finite values")

    weights = np.asarray(weight_fn(frequencies), dtype=float)
    if weights.ndim == 0:
        weights = np.full(frequencies.shape, float(weights), dtype=float)
    elif weights.shape != frequencies.shape:
        raise ValueError(
            "noise_freq_weight returned shape %s; expected %s"
            % (weights.shape, frequencies.shape)
        )

    if np.any(np.isinf(weights)):
        raise ValueError("noise_freq_weight must not return infinite values")

    surviving = ~np.isnan(weights)
    if not np.any(surviving):
        return np.inf

    surviving_weights = weights[surviving]
    if np.any((surviving_weights <= 0.0) | (surviving_weights > 1.0)):
        raise ValueError(
            "finite noise_freq_weight values must be in the interval (0, 1]"
        )

    if mode == "fourier":
        return float(np.mean(1.0 / surviving_weights))
    return float(1.0 / np.mean(surviving_weights))
