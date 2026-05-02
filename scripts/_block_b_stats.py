"""Pure-numpy statistical helpers for Block B aggregation.

Provides the four primitives V4Evaluation.md §4 mandates without a scipy
dependency: paired Wilcoxon signed-rank, Holm-Bonferroni, bootstrap CI,
and pairwise win-rate matrix.

Edge cases (per the implementation plan):
- Wilcoxon with all paired diffs zero → status='ties_only', p=1.0
- Wilcoxon with n_nonzero < 5 → status='too_few', p=1.0
- Bootstrap with all-equal values → CI = (mean, mean, mean)
- Win-rate uses strict inequality; ties go to neither (W[i,j] + W[j,i] <= 1)

Self-test: ``python scripts/_block_b_stats.py --self-test``.
"""

from __future__ import annotations

import math
import sys
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank (paired)
# ---------------------------------------------------------------------------


def wilcoxon_signed_rank(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, str]:
    """Two-sided paired Wilcoxon signed-rank test (normal approximation).

    Implements mid-rank tie correction. Normal approximation is accurate to
    ~10⁻³ on p-values for n_nonzero >= 20 (we have n=50 in Block B).

    Returns
    -------
    (z, p, status)
        z       : standardized statistic; sign indicates which side (positive
                  = x typically larger than y)
        p       : two-sided p-value
        status  : 'ok' | 'ties_only' | 'too_few'
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y shapes differ: {x.shape} vs {y.shape}")

    diffs = x - y
    nonzero = diffs[diffs != 0.0]
    n = nonzero.size
    if n == 0:
        return 0.0, 1.0, "ties_only"
    if n < 5:
        return float("nan"), 1.0, "too_few"

    abs_d = np.abs(nonzero)
    # Mid-ranks for ties.
    order = np.argsort(abs_d, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    sorted_abs = abs_d[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_abs[j] == sorted_abs[i]:
            j += 1
        # average rank for tied group is mean of (i+1, ..., j) (1-indexed)
        avg = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg
        i = j

    signs = np.sign(nonzero)
    W_plus = float(np.sum(ranks[signs > 0]))
    W_minus = float(np.sum(ranks[signs < 0]))

    expected = n * (n + 1) / 4.0
    # Tie-corrected variance (per Wilcoxon original, with mid-ranks).
    _, tie_counts = np.unique(sorted_abs, return_counts=True)
    tie_term = float(np.sum(tie_counts * (tie_counts - 1) * (tie_counts + 1)))
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if variance <= 0:
        return 0.0, 1.0, "ties_only"

    # z based on the smaller-tail W (continuity correction +/- 0.5)
    W = min(W_plus, W_minus)
    z = (W - expected) / math.sqrt(variance)
    # Two-sided p via standard normal
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    p = max(0.0, min(1.0, p))
    # Sign convention: positive z when x typically > y
    z_signed = (W_plus - W_minus) / math.sqrt(variance) / 2.0 * 2.0  # keep magnitude
    # Actually the paired standardized statistic is (W_plus - expected)/sqrt(var)
    z_signed = (W_plus - expected) / math.sqrt(variance)
    return float(z_signed), float(p), "ok"


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------


def holm_bonferroni(
    p_values: Iterable[float], alpha: float = 0.05
) -> np.ndarray:
    """Holm sequential Bonferroni. Returns boolean array of rejections.

    Skips NaN p-values (left as False).
    """
    p = np.asarray(list(p_values), dtype=float)
    n = p.size
    rejected = np.zeros(n, dtype=bool)
    if n == 0:
        return rejected
    # Sort by p ascending (NaNs last)
    valid = ~np.isnan(p)
    order = np.argsort(np.where(valid, p, np.inf), kind="mergesort")
    m = int(valid.sum())
    for k, idx in enumerate(order):
        if not valid[idx]:
            break
        thresh = alpha / (m - k)
        if p[idx] <= thresh:
            rejected[idx] = True
        else:
            break  # Holm step-down: stop on first non-reject
    return rejected


def holm_adjusted(
    p_values: Iterable[float],
) -> np.ndarray:
    """Holm-adjusted p-values (each clamped to <=1, monotonic by sorted order).

    Useful for reporting alongside the raw p-values.
    """
    p = np.asarray(list(p_values), dtype=float)
    n = p.size
    out = np.full(n, np.nan)
    if n == 0:
        return out
    valid = ~np.isnan(p)
    order = np.argsort(np.where(valid, p, np.inf), kind="mergesort")
    m = int(valid.sum())
    running_max = 0.0
    for k, idx in enumerate(order):
        if not valid[idx]:
            break
        adj = (m - k) * p[idx]
        adj = min(1.0, max(running_max, adj))
        running_max = adj
        out[idx] = adj
    return out


# ---------------------------------------------------------------------------
# Bootstrap CI on the mean
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean. Deterministic via rng_seed.

    Returns (mean, ci_low, ci_high).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if np.allclose(arr, arr[0]):
        return mean, mean, mean

    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = 1.0 - ci
    low = float(np.percentile(means, 100.0 * alpha / 2.0))
    high = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return mean, low, high


# ---------------------------------------------------------------------------
# Win-rate matrix
# ---------------------------------------------------------------------------


def win_rate_matrix(
    method_profits: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Pairwise win-rate matrix. W[i][j] = (#seeds where i > j) / n.

    Strict inequality; ties (i == j) go to neither, so
    W[i][j] + W[j][i] <= 1.
    """
    methods = list(method_profits.keys())
    arrs = {m: np.asarray(method_profits[m], dtype=float) for m in methods}
    n_seeds = next(iter(arrs.values())).size
    out: dict[str, dict[str, float]] = {m: {} for m in methods}
    for i in methods:
        for j in methods:
            if i == j:
                out[i][j] = float("nan")
                continue
            wins = int(np.sum(arrs[i] > arrs[j]))
            out[i][j] = wins / n_seeds
    return out


def significance_marker(p: float) -> str:
    """Marker per V4Evaluation.md §4.1."""
    if math.isnan(p):
        return "?"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> int:
    rng = np.random.default_rng(0)
    failures = 0

    # Test 1 — Wilcoxon: x consistently larger than y → significant
    x = np.array([10.0, 12.0, 11.0, 13.0, 14.0, 12.5, 11.8, 13.1, 12.9, 14.2,
                  13.5, 12.7, 13.8, 14.1, 12.3, 13.0, 12.8, 13.2, 13.6, 14.0,
                  12.9, 13.7, 14.3, 13.4, 12.6])
    y = x - 1.5  # consistent shift
    z, p, status = wilcoxon_signed_rank(x, y)
    if status != "ok" or p >= 0.05 or z <= 0:
        print(f"  T1 FAIL  z={z:.3f} p={p:.3e} status={status}")
        failures += 1
    else:
        print(f"  T1 PASS  consistent shift detected: z={z:.2f} p={p:.2e}")

    # Test 2 — Wilcoxon: x and y identical → ties_only
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z, p, status = wilcoxon_signed_rank(x, y)
    if status != "ties_only" or p != 1.0:
        print(f"  T2 FAIL  expected ties_only,p=1.0 got status={status} p={p}")
        failures += 1
    else:
        print(f"  T2 PASS  identical inputs -> ties_only")

    # Test 3 — Wilcoxon: too few non-zero diffs
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 6.0])  # only 1 nonzero diff
    z, p, status = wilcoxon_signed_rank(x, y)
    if status != "too_few":
        print(f"  T3 FAIL  expected too_few got status={status}")
        failures += 1
    else:
        print(f"  T3 PASS  n<5 nonzero diffs -> too_few")

    # Test 4 — Holm: 3 p-values, only smallest two should reject
    p_vals = [0.001, 0.020, 0.040]
    rej = holm_bonferroni(p_vals, alpha=0.05)
    # Holm: smallest gets alpha/3 = 0.0167 → reject (0.001 < 0.0167)
    # Next: alpha/2 = 0.025 → reject (0.020 < 0.025)
    # Next: alpha/1 = 0.05 → reject (0.040 < 0.05) — but wait, Holm step-down
    # only continues if all previous rejected; this should reject all 3.
    # Actually plain Bonferroni at alpha/3 would not reject 0.040. Holm rejects.
    if not (rej[0] and rej[1] and rej[2]):
        print(f"  T4 FAIL  Holm rejects: {rej}")
        failures += 1
    else:
        print(f"  T4 PASS  Holm rejects all three at α=0.05")

    # Test 5 — Holm: stop on first non-reject
    p_vals = [0.001, 0.040, 0.045]
    rej = holm_bonferroni(p_vals, alpha=0.05)
    # 0.001 < 0.0167 ✓
    # 0.040 vs 0.025 ✗ stop
    # 0.045 not rejected
    if not (rej[0] and not rej[1] and not rej[2]):
        print(f"  T5 FAIL  Holm step-down: {rej}")
        failures += 1
    else:
        print(f"  T5 PASS  Holm stops after first non-reject")

    # Test 6 — Bootstrap CI: deterministic and contains the mean
    arr = rng.normal(100.0, 5.0, size=50)
    m1, l1, h1 = bootstrap_ci(arr, n_resamples=1000, rng_seed=42)
    m2, l2, h2 = bootstrap_ci(arr, n_resamples=1000, rng_seed=42)
    if abs(m1 - m2) > 1e-9 or abs(l1 - l2) > 1e-9 or abs(h1 - h2) > 1e-9:
        print(f"  T6 FAIL  bootstrap not deterministic: ({m1},{l1},{h1}) vs ({m2},{l2},{h2})")
        failures += 1
    elif not (l1 <= m1 <= h1):
        print(f"  T6 FAIL  bootstrap CI does not contain mean: [{l1}, {m1}, {h1}]")
        failures += 1
    else:
        print(f"  T6 PASS  bootstrap CI={l1:.2f}, mean={m1:.2f}, hi={h1:.2f}, deterministic")

    # Test 7 — Bootstrap with all-equal values
    arr = np.full(50, 42.0)
    m, l, h = bootstrap_ci(arr)
    if not (m == 42.0 and l == 42.0 and h == 42.0):
        print(f"  T7 FAIL  all-equal bootstrap: {(m, l, h)}")
        failures += 1
    else:
        print(f"  T7 PASS  all-equal -> CI=(42, 42, 42)")

    # Test 8 — Win-rate matrix
    profits = {
        "A": np.array([10.0, 20.0, 30.0, 40.0]),
        "B": np.array([15.0, 18.0, 25.0, 50.0]),
    }
    wm = win_rate_matrix(profits)
    # A>B at indices 1,2 → A wins 2/4 = 0.5; B>A at 0,3 → 0.5
    if not (wm["A"]["B"] == 0.5 and wm["B"]["A"] == 0.5):
        print(f"  T8 FAIL  win-rate: {wm}")
        failures += 1
    else:
        print(f"  T8 PASS  win-rate symmetric on no-tie data")

    # Test 9 — Significance markers
    cases = [(1e-4, "***"), (5e-3, "**"), (3e-2, "*"), (0.5, "ns")]
    for p, expected in cases:
        got = significance_marker(p)
        if got != expected:
            print(f"  T9 FAIL  marker(p={p}): got {got}, want {expected}")
            failures += 1
    if not failures:
        print(f"  T9 PASS  significance markers")

    print()
    if failures:
        print(f"FAILED: {failures} test(s)")
        return 1
    print("All self-tests PASS.")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print("Use --self-test to run the test suite.")
