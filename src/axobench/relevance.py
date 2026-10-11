"""IRon-default relevance function for imbalanced-regression metrics.

This reproduces the default relevance of the IRon R package (Ribeiro and
Moniz, 2020): ``phi.control(y)`` with method ``"extremes"``, ``extr.type =
"both"``, ``coef = 1.5`` and ``asym = TRUE``. Its control points are the lower
adjusted-boxplot fence (relevance 1), the median (relevance 0) and the upper
fence (relevance 1), joined by IRon's monotone piecewise-cubic Hermite
interpolation with zero slopes and held constant beyond the end points.

The fences follow ``robustbase::adjboxStats`` (Hubert and Vandervieren,
2008): Tukey's five-number summary and the medcouple ``mc`` skew the
whiskers to ``Q1 - coef exp(a mc) IQR`` and ``Q3 + coef exp(b mc) IQR`` with
``a = -4, b = 3`` for ``mc >= 0`` (mirrored for ``mc < 0``). The medcouple is
computed exactly with the O(n log n) algorithm of Johnson and Mizoguchi
(1978) as used by Brys, Hubert and Struyf (2004).
"""

from __future__ import annotations

import hashlib

import numpy as np


def fivenum(values: np.ndarray) -> np.ndarray:
    """Tukey's five-number summary, as R's ``fivenum``."""
    x = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    n = x.size
    if n == 0:
        raise ValueError("fivenum requires at least one value")
    n4 = np.floor((n + 3) / 2) / 2
    d = np.array([1, n4, (n + 1) / 2, n + 1 - n4, n], dtype=np.float64) - 1
    return 0.5 * (x[np.floor(d).astype(int)] + x[np.ceil(d).astype(int)])


def medcouple(values: np.ndarray) -> float:
    """Exact medcouple of a sample (Brys, Hubert and Struyf, 2004)."""
    x = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))[::-1]
    n = x.size
    if n < 3:
        return 0.0
    median = 0.5 * (x[(n - 1) // 2] + x[n // 2])
    z = x - median
    scale = 2.0 * np.max(np.abs(z))
    if scale == 0.0:
        return 0.0
    z = z / scale
    zplus = z[z >= 0.0]   # descending
    zminus = z[z <= 0.0]  # descending
    p, q = zplus.size, zminus.size

    def kernel(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        a, b = zplus[i], zminus[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            # a == b only for ties at the median (both zero).
            return np.where(a == b, np.sign(p - 1 - i - j), (a + b) / (a - b))

    left = np.zeros(p, dtype=np.int64)
    right = np.full(p, q - 1, dtype=np.int64)
    left_total = 0
    right_total = p * q
    median_index = right_total // 2
    while right_total - left_total > p:
        rows = np.flatnonzero(left <= right)
        row_medians = kernel(rows, (left[rows] + right[rows]) // 2)
        weights = (right[rows] - left[rows] + 1).astype(np.float64)
        order = np.argsort(row_medians, kind="mergesort")
        cumulative = np.cumsum(weights[order])
        guess = row_medians[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])]

        # P: for each row, index of the last kernel value > guess.
        # Q: for each row, count of kernel values >= guess.
        p_bounds = _row_counts(kernel, p, q, guess, strict=True) - 1
        q_bounds = _row_counts(kernel, p, q, guess, strict=False)
        sum_p = int(np.sum(p_bounds + 1))
        sum_q = int(np.sum(q_bounds))
        if median_index <= sum_p - 1:
            right = p_bounds.copy()
            right_total = sum_p
        elif median_index > sum_q - 1:
            left = q_bounds.copy()
            left_total = sum_q
        else:
            return float(guess)

    widths = np.maximum(right - left + 1, 0)
    rows = np.repeat(np.arange(p), widths)
    columns = np.arange(widths.sum()) - np.repeat(np.cumsum(widths) - widths, widths) + np.repeat(left, widths)
    remaining_values = np.sort(kernel(rows, columns))[::-1]
    return float(remaining_values[median_index - left_total])


def _row_counts(kernel, p: int, q: int, guess: float, *, strict: bool) -> np.ndarray:
    """Per row i, the number of columns j with kernel(i, j) > guess (strict) or >= guess.

    Every kernel row is non-increasing in j, so the hits form a prefix whose
    length a binary search over all rows at once finds with the exact kernel.
    """
    rows = np.arange(p)
    lo = np.zeros(p, dtype=np.int64)  # hits known for columns < lo
    hi = np.full(p, q, dtype=np.int64)  # misses known for columns >= hi
    while True:
        active = rows[lo < hi]
        if active.size == 0:
            return lo
        mid = (lo[active] + hi[active]) // 2
        values = kernel(active, mid)
        hit = values > guess if strict else values >= guess
        lo[active[hit]] = mid[hit] + 1
        hi[active[~hit]] = mid[~hit]


def adjusted_box_fences(values: np.ndarray, *, coef: float = 1.5, a: float = -4.0, b: float = 3.0) -> tuple[float, float, float]:
    """Lower fence, median and upper fence as in ``robustbase::adjboxStats``."""
    stats = fivenum(values)
    iqr = stats[3] - stats[1]
    mc = medcouple(values)
    if mc >= 0:
        low = stats[1] - coef * np.exp(a * mc) * iqr
        high = stats[3] + coef * np.exp(b * mc) * iqr
    else:
        low = stats[1] - coef * np.exp(-b * mc) * iqr
        high = stats[3] + coef * np.exp(-a * mc) * iqr
    return float(low), float(stats[2]), float(high)


def _pchip_zero_slope(x: np.ndarray, knots: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """IRon ``pchip_val`` for control points with zero slopes.

    With zero end slopes the Fritsch-Carlson correction leaves all slopes at
    zero, so each segment is ``y0 + (y1 - y0)(3t^2 - 2t^3)``; outside the
    knots the value is held at the nearest end point (linear extrapolation
    with slope zero).
    """
    out = np.empty_like(x, dtype=np.float64)
    out[x <= knots[0]] = levels[0]
    out[x >= knots[-1]] = levels[-1]
    for k in range(knots.size - 1):
        x0, x1 = knots[k], knots[k + 1]
        inside = (x > x0) & (x < x1) if k else (x > x0) & (x < x1)
        if x1 <= x0:
            continue
        t = (x[inside] - x0) / (x1 - x0)
        out[inside] = levels[k] + (levels[k + 1] - levels[k]) * (3 * t**2 - 2 * t**3)
    for k, knot in enumerate(knots):
        out[x == knot] = levels[k]
    return out


_FENCE_CACHE: dict[tuple[int, str], tuple[float, float, float]] = {}


def _cached_fences(values: np.ndarray) -> tuple[float, float, float]:
    """``adjusted_box_fences`` memoized on the content of ``values``.

    The evaluator, paired bootstrap and null references score many predictions
    against the same teacher traces; the fences depend only on those.
    """
    key = (values.size, hashlib.sha1(np.ascontiguousarray(values).tobytes()).hexdigest())
    if key not in _FENCE_CACHE:
        if len(_FENCE_CACHE) >= 64:
            _FENCE_CACHE.pop(next(iter(_FENCE_CACHE)))
        _FENCE_CACHE[key] = adjusted_box_fences(values)
    return _FENCE_CACHE[key]


def iron_relevance(target: np.ndarray, *, reference: np.ndarray | None = None) -> np.ndarray:
    """IRon default relevance ``phi(y, phi.control(y))``.

    The control points are derived from ``reference`` (default: ``target``
    itself, as IRon's ``sera`` does with ``phi.control(trues)``).
    """
    y = np.asarray(target, dtype=np.float64)
    ref = y if reference is None else np.asarray(reference, dtype=np.float64)
    low, median, high = _cached_fences(ref.reshape(-1))
    knots = np.array([low, median, high])
    levels = np.array([1.0, 0.0, 1.0])
    return _pchip_zero_slope(y.reshape(-1), knots, levels).reshape(y.shape)


def iron_sera(error: np.ndarray, relevance: np.ndarray, *, step: float = 0.001) -> float:
    """IRon ``sera``: trapezoidal area of SER_t = sum of squared errors with phi >= t."""
    err2 = np.asarray(error, dtype=np.float64).reshape(-1) ** 2
    phi = np.asarray(relevance, dtype=np.float64).reshape(-1)
    thresholds = np.arange(0.0, 1.0 + step / 2, step)
    order = np.argsort(phi)
    phi_sorted = phi[order]
    tail_sums = np.concatenate([np.cumsum(err2[order][::-1])[::-1], [0.0]])
    start = np.searchsorted(phi_sorted, thresholds, side="left")
    ser = tail_sums[start]
    return float(np.sum(step * (ser[:-1] + ser[1:]) / 2.0))


def iron_threshold_weights(relevance: np.ndarray, *, step: float = 0.001) -> np.ndarray:
    """Per-point weight A(phi) such that ``iron_sera(e, phi) == sum(e**2 * A(phi))``.

    On the threshold grid t_k = k * step, a point with relevance phi enters
    SER_t for all t_k <= phi. The trapezoidal area of that step indicator is
    ``step * (m - 0.5)`` with ``m`` the number of grid thresholds <= phi, and
    exactly one for phi >= 1.
    """
    phi = np.asarray(relevance, dtype=np.float64)
    count = np.floor(phi / step + 1e-9) + 1.0
    weights = step * (count - 0.5)
    weights = np.where(phi >= 1.0, 1.0, weights)
    return np.clip(weights, 0.0, 1.0)
