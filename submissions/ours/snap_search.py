"""Grid-aware hard macro snapping.

The routing-congestion proxy is grid based.  This pass tries small legal moves
that align hard macro centers or edges to routing-grid lines, then keeps the
placement only if the official proxy improves.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


def refine_with_snap_search(
    baseline: torch.Tensor,
    benchmark: Benchmark,
    *,
    load_plc: Callable[[Benchmark], object | None],
    is_valid: Callable[[torch.Tensor, Benchmark], bool],
) -> torch.Tensor:
    plc = load_plc(benchmark)
    if plc is None:
        return baseline

    try:
        helper_dir = Path(__file__).resolve().parent
        if str(helper_dir) not in sys.path:
            sys.path.insert(0, str(helper_dir))
        from fast_proxy import build_fast_proxy
        from fast_search import _rank_hot_macros, _single_move_valid
    except Exception:
        return baseline

    scorer = build_fast_proxy(benchmark, plc)
    if scorer is None:
        return baseline

    try:
        current = baseline.detach().clone().float()
        best_fast = float(scorer.score(current)["proxy_cost"])
    except Exception:
        return baseline

    rounds = max(1, _env_int("OURS_SNAP_ROUNDS", 3))
    macro_count = max(1, _env_int("OURS_SNAP_MACROS", 600))
    max_trials = max(0, _env_int("OURS_SNAP_TRIALS", 5000))
    if max_trials <= 0:
        return baseline

    gap = _env_float("OURS_SNAP_GAP", _env_float("OURS_GAP", 0.005))
    eps = _env_float("OURS_SNAP_EPS", 1e-6)
    trials = 0
    accepted = 0

    for _ in range(rounds):
        try:
            maps = scorer.score(current, maps=True)
            pressure = np.asarray(maps["v_routing_cong"], dtype=np.float64) + np.asarray(
                maps["h_routing_cong"],
                dtype=np.float64,
            )
            ranked = _rank_hot_macros(current, benchmark, scorer, pressure)
        except Exception:
            break
        if not ranked:
            break

        improved = False
        for idx in ranked[:macro_count]:
            for x, y in _snap_points(current, benchmark, scorer, int(idx)):
                if trials >= max_trials:
                    break
                trials += 1
                if not _single_move_valid(current, benchmark, int(idx), x, y, gap=gap):
                    continue
                candidate = current.clone()
                candidate[int(idx), 0] = float(x)
                candidate[int(idx), 1] = float(y)
                try:
                    score = scorer.score(candidate)
                except Exception:
                    continue
                if int(score.get("overlap_count", 1)) != 0:
                    continue
                cost = float(score["proxy_cost"])
                if cost + eps < best_fast:
                    current = candidate
                    best_fast = cost
                    accepted += 1
                    improved = True
            if trials >= max_trials:
                break
        if not improved or trials >= max_trials:
            break

    if accepted == 0 or torch.allclose(current, baseline, atol=1e-7, rtol=0.0):
        return baseline
    if not is_valid(current, benchmark):
        return baseline

    try:
        base_exact = compute_proxy_cost(baseline.detach().float(), benchmark, plc)
        cand_exact = compute_proxy_cost(current.detach().float(), benchmark, plc)
    except Exception:
        return baseline
    if int(cand_exact.get("overlap_count", 1)) != 0:
        return baseline
    if float(cand_exact["proxy_cost"]) + eps < float(base_exact["proxy_cost"]):
        _debug(
            f"accepted exact {float(base_exact['proxy_cost']):.6f}->"
            f"{float(cand_exact['proxy_cost']):.6f} fast={best_fast:.6f} "
            f"accepted={accepted} trials={trials}"
        )
        return current.detach().clone().float()
    _debug(
        f"rejected exact {float(base_exact['proxy_cost']):.6f}->"
        f"{float(cand_exact['proxy_cost']):.6f} fast={best_fast:.6f} "
        f"accepted={accepted} trials={trials}"
    )
    return baseline


def _snap_points(
    placement: torch.Tensor,
    benchmark: Benchmark,
    scorer,
    idx: int,
) -> list[tuple[float, float]]:
    pos = placement.detach().cpu().numpy().astype(np.float64, copy=False)
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64, copy=False)
    x0 = float(pos[idx, 0])
    y0 = float(pos[idx, 1])
    w = float(sizes[idx, 0])
    h = float(sizes[idx, 1])
    hw = 0.5 * w
    hh = 0.5 * h
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    xs = _axis_snaps(x0, hw, cw - hw, scorer.grid_width)
    ys = _axis_snaps(y0, hh, ch - hh, scorer.grid_height)
    out: list[tuple[float, float]] = []
    for x in xs:
        out.append((x, y0))
    for y in ys:
        out.append((x0, y))
    for x in xs:
        for y in ys:
            out.append((x, y))

    dedup: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in out:
        if abs(x - x0) < 1e-9 and abs(y - y0) < 1e-9:
            continue
        key = (int(round(x * 10000.0)), int(round(y * 10000.0)))
        if key in seen:
            continue
        seen.add(key)
        dedup.append((float(x), float(y)))
    return dedup


def _axis_snaps(value: float, lo: float, hi: float, grid: float) -> list[float]:
    if grid <= 0.0:
        return []
    candidates = []
    for raw in (
        round(value / grid) * grid,
        (round(value / grid) + 0.5) * grid,
        round((value - lo) / grid) * grid + lo,
        round((value + lo) / grid) * grid - lo,
    ):
        candidates.append(float(np.clip(raw, lo, hi)))
    candidates.extend(
        [
            float(np.clip(value - 0.25 * grid, lo, hi)),
            float(np.clip(value + 0.25 * grid, lo, hi)),
            float(np.clip(value - 0.50 * grid, lo, hi)),
            float(np.clip(value + 0.50 * grid, lo, hi)),
        ]
    )
    out = []
    seen = set()
    for x in candidates:
        key = int(round(x * 10000.0))
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def _env(name: str, default: str | int | float) -> str:
    return os.environ.get(name, str(default))


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, default))
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, default))
    except ValueError:
        return float(default)


def _env_bool(name: str, default: str) -> bool:
    return _env(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _debug(message: str) -> None:
    if _env_bool("OURS_SNAP_DEBUG", "0"):
        print(f"[OURS_SNAP] {message}", flush=True)
