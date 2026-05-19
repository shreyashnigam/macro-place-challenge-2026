"""Fast-proxy guided local search for congestion relief."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


def refine_with_fast_search(
    baseline: torch.Tensor,
    benchmark: Benchmark,
    *,
    load_plc: Callable[[Benchmark], object | None],
    legalize_hard: Callable[..., torch.Tensor] | None = None,
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
    except Exception:
        return baseline

    scorer = build_fast_proxy(benchmark, plc)
    if scorer is None:
        return baseline

    try:
        best = baseline.detach().clone().float()
        best_fast = float(scorer.score(best)["proxy_cost"])
    except Exception:
        return baseline

    rounds = _env_int("OURS_FAST_SEARCH_ROUNDS", 2)
    if rounds <= 0:
        return baseline

    macro_count = _env_int("OURS_FAST_SEARCH_MACROS", 10)
    max_trials = _env_int("OURS_FAST_SEARCH_TRIALS", 96)
    eps = _env_float("OURS_FAST_SEARCH_EPS", 1e-5)
    gap = _env_float("OURS_FAST_SEARCH_GAP", _env_float("OURS_GAP", 0.005))
    rng = np.random.default_rng(_env_int("OURS_FAST_SEARCH_SEED", 20260519))

    _debug(f"start fast={best_fast:.6f}")
    if _env_bool("OURS_FAST_SEARCH_LNS", "0") and legalize_hard is not None:
        lns_best, lns_fast = _overlap_lns_refine(
            best,
            benchmark,
            scorer,
            legalize_hard=legalize_hard,
            is_valid=is_valid,
            start_fast=best_fast,
        )
        if lns_fast + eps < best_fast:
            best = lns_best
            best_fast = lns_fast
            _debug(f"lns improved fast={best_fast:.6f}")

    for round_idx in range(rounds):
        try:
            score_with_maps = scorer.score(best, maps=True)
            v_map = np.asarray(score_with_maps["v_routing_cong"], dtype=np.float64)
            h_map = np.asarray(score_with_maps["h_routing_cong"], dtype=np.float64)
        except Exception:
            break
        pressure = v_map + h_map
        ranked = _rank_hot_macros(best, benchmark, scorer, pressure)
        if not ranked:
            break

        round_best = best
        round_best_fast = best_fast
        trials = 0
        for macro_idx in ranked[:macro_count]:
            proposals = _proposal_points(best, benchmark, scorer, pressure, macro_idx, rng)
            for x, y in proposals:
                if trials >= max_trials:
                    break
                trials += 1
                if not _single_move_valid(best, benchmark, macro_idx, x, y, gap=gap):
                    continue
                candidate = best.clone()
                candidate[macro_idx, 0] = float(x)
                candidate[macro_idx, 1] = float(y)
                try:
                    fast = scorer.score(candidate)
                except Exception:
                    continue
                if int(fast.get("overlap_count", 1)) != 0:
                    continue
                cost = float(fast["proxy_cost"])
                if cost + eps < round_best_fast:
                    round_best = candidate
                    round_best_fast = cost
            if trials >= max_trials:
                break

        swap_pool = ranked[: max(2, _env_int("OURS_FAST_SEARCH_SWAP_POOL", 36))]
        swap_attempts = _env_int("OURS_FAST_SEARCH_SWAP_ATTEMPTS", 0)
        for _ in range(max(0, swap_attempts)):
            if trials >= max_trials or len(swap_pool) < 2:
                break
            ia, ib = rng.choice(len(swap_pool), size=2, replace=False)
            a = int(swap_pool[int(ia)])
            b = int(swap_pool[int(ib)])
            if a == b:
                continue
            trials += 1
            candidate = best.clone()
            tmp = candidate[a].clone()
            candidate[a] = candidate[b]
            candidate[b] = tmp
            try:
                fast = scorer.score(candidate)
            except Exception:
                continue
            if int(fast.get("overlap_count", 1)) != 0:
                continue
            cost = float(fast["proxy_cost"])
            if cost + eps < round_best_fast:
                round_best = candidate
                round_best_fast = cost

        if round_best is best or round_best_fast + eps >= best_fast:
            _debug(f"round={round_idx} no_improve trials={trials} fast={best_fast:.6f}")
            break
        best = round_best
        best_fast = round_best_fast
        _debug(f"round={round_idx} improved fast={best_fast:.6f} trials={trials}")

    if torch.allclose(best, baseline, atol=1e-7, rtol=0.0):
        return baseline
    if not is_valid(best, benchmark):
        return baseline

    try:
        base_exact = compute_proxy_cost(baseline, benchmark, plc)
        best_exact = compute_proxy_cost(best, benchmark, plc)
    except Exception:
        return baseline
    if int(best_exact.get("overlap_count", 1)) != 0:
        return baseline
    if float(best_exact["proxy_cost"]) < float(base_exact["proxy_cost"]) - eps:
        _debug(
            "accepted exact "
            f"{float(base_exact['proxy_cost']):.6f}->{float(best_exact['proxy_cost']):.6f} "
            f"fast={best_fast:.6f}"
        )
        return best.detach().clone().float()
    _debug(
        "rejected exact "
        f"{float(base_exact['proxy_cost']):.6f}->{float(best_exact['proxy_cost']):.6f} "
        f"fast={best_fast:.6f}"
    )
    return baseline


def _overlap_lns_refine(
    baseline: torch.Tensor,
    benchmark: Benchmark,
    scorer,
    *,
    legalize_hard: Callable[..., torch.Tensor],
    is_valid: Callable[[torch.Tensor, Benchmark], bool],
    start_fast: float,
) -> tuple[torch.Tensor, float]:
    n_hard = int(benchmark.num_hard_macros)
    if n_hard <= 1:
        return baseline, start_fast
    movable = (~benchmark.macro_fixed[:n_hard]).detach().cpu().numpy().astype(bool, copy=False)
    movable_idx = np.where(movable)[0]
    if movable_idx.size == 0:
        return baseline, start_fast

    rng = np.random.default_rng(_env_int("OURS_FAST_SEARCH_LNS_SEED", 99173))
    iters = _env_int("OURS_FAST_SEARCH_LNS_ITERS", 2500)
    legal_period = max(1, _env_int("OURS_FAST_SEARCH_LNS_LEGAL_PERIOD", 160))
    rank_period = max(1, _env_int("OURS_FAST_SEARCH_LNS_RANK_PERIOD", 80))
    temp0 = _env_float("OURS_FAST_SEARCH_LNS_TEMP0", 0.010)
    temp1 = _env_float("OURS_FAST_SEARCH_LNS_TEMP1", 0.0005)
    lam0 = _env_float("OURS_FAST_SEARCH_LNS_LAM0", 0.0005)
    lam1 = _env_float("OURS_FAST_SEARCH_LNS_LAM1", 0.030)
    gap = _env_float("OURS_FAST_SEARCH_GAP", _env_float("OURS_GAP", 0.005))

    current = baseline.detach().clone().float()
    current_score = scorer.score(current)
    current_pen = _penalized(current_score, lam0)
    best_valid = baseline.detach().clone().float()
    best_valid_fast = float(start_fast)
    ranked = list(movable_idx)
    sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64, copy=False)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    for it in range(max(1, iters)):
        frac = float(it) / max(1.0, float(iters - 1))
        temp = temp0 * ((temp1 / max(temp0, 1e-12)) ** frac)
        lam = lam0 * ((lam1 / max(lam0, 1e-12)) ** frac)

        if it % rank_period == 0:
            try:
                maps = scorer.score(current, maps=True)
                pressure = np.asarray(maps["v_routing_cong"], dtype=np.float64) + np.asarray(
                    maps["h_routing_cong"], dtype=np.float64
                )
                ranked = _rank_hot_macros(current, benchmark, scorer, pressure)
                if not ranked:
                    ranked = list(movable_idx)
            except Exception:
                ranked = list(movable_idx)

        proposal = current.clone()
        move_kind = float(rng.random())
        if move_kind < 0.12 and len(ranked) >= 2:
            ia = int(ranked[int(rng.integers(0, min(len(ranked), 32)))])
            ib = int(ranked[int(rng.integers(0, min(len(ranked), 32)))])
            if ia != ib and movable[ia] and movable[ib]:
                tmp = proposal[ia].clone()
                proposal[ia] = proposal[ib]
                proposal[ib] = tmp
        else:
            idx = int(ranked[int(rng.integers(0, min(len(ranked), max(1, _env_int("OURS_FAST_SEARCH_LNS_TOP", 48)))))])
            if not movable[idx]:
                continue
            hw = 0.5 * float(sizes[idx, 0])
            hh = 0.5 * float(sizes[idx, 1])
            if move_kind < 0.50:
                sigma = (0.14 * (1.0 - frac) + 0.015 * frac) * max(cw, ch)
                x = float(proposal[idx, 0]) + float(rng.normal(0.0, sigma))
                y = float(proposal[idx, 1]) + float(rng.normal(0.0, sigma))
            else:
                x = float(rng.uniform(hw, max(hw, cw - hw)))
                y = float(rng.uniform(hh, max(hh, ch - hh)))
            proposal[idx, 0] = _clip(x, hw, cw - hw)
            proposal[idx, 1] = _clip(y, hh, ch - hh)

        try:
            prop_score = scorer.score(proposal)
        except Exception:
            continue
        prop_pen = _penalized(prop_score, lam)
        delta = prop_pen - current_pen
        if delta <= 0.0 or float(rng.random()) < math.exp(-delta / max(temp, 1e-12)):
            current = proposal
            current_score = prop_score
            current_pen = prop_pen

        if (it + 1) % legal_period == 0 or it == iters - 1:
            try:
                legal = legalize_hard(
                    current,
                    benchmark,
                    gap=gap,
                    max_rounds=_env_int("OURS_FAST_SEARCH_LNS_LEGAL_ROUNDS", 500),
                )
            except TypeError:
                legal = legalize_hard(current, benchmark)
            if not is_valid(legal, benchmark):
                continue
            try:
                legal_score = scorer.score(legal)
            except Exception:
                continue
            if int(legal_score.get("overlap_count", 1)) == 0 and float(legal_score["proxy_cost"]) < best_valid_fast:
                best_valid = legal.detach().clone().float()
                best_valid_fast = float(legal_score["proxy_cost"])
                _debug(f"lns legal it={it + 1} fast={best_valid_fast:.6f}")

    return best_valid, best_valid_fast


def _penalized(score: dict[str, object], lam: float) -> float:
    return float(score["proxy_cost"]) + float(lam) * float(score.get("overlap_count", 0))


def _rank_hot_macros(
    placement: torch.Tensor,
    benchmark: Benchmark,
    scorer,
    pressure: np.ndarray,
) -> list[int]:
    pos = placement[: int(benchmark.num_hard_macros)].detach().cpu().numpy().astype(np.float64, copy=False)
    sizes = benchmark.macro_sizes[: int(benchmark.num_hard_macros)].detach().cpu().numpy().astype(np.float64, copy=False)
    fixed = benchmark.macro_fixed[: int(benchmark.num_hard_macros)].detach().cpu().numpy().astype(bool, copy=False)
    route_scores = _route_participation_scores(placement, benchmark, scorer, pressure)
    ranked: list[tuple[float, int]] = []
    for idx in range(int(benchmark.num_hard_macros)):
        if fixed[idx]:
            continue
        bins = _covered_bins(scorer, pos[idx, 0], pos[idx, 1], sizes[idx, 0], sizes[idx, 1])
        if not bins:
            row, col = scorer._grid_cell(float(pos[idx, 0]), float(pos[idx, 1]))
            bins = [row * scorer.grid_cols + col]
        vals = pressure[np.asarray(bins, dtype=np.int64)]
        area = float(sizes[idx, 0] * sizes[idx, 1])
        score = (
            float(vals.max())
            + 0.35 * float(vals.mean())
            + 0.85 * float(route_scores[idx])
            + 0.02 * math.log1p(area)
        )
        ranked.append((score, idx))
    ranked.sort(reverse=True)
    return [idx for _, idx in ranked]


def _route_participation_scores(
    placement: torch.Tensor,
    benchmark: Benchmark,
    scorer,
    pressure: np.ndarray,
) -> np.ndarray:
    n_hard = int(benchmark.num_hard_macros)
    scores = np.zeros(n_hard, dtype=np.float64)
    if pressure.size == 0:
        return scores
    pos = placement.detach().cpu().numpy().astype(np.float64, copy=False)
    grid = pressure.reshape(scorer.grid_rows, scorer.grid_cols)
    integral = np.pad(grid, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    for net in scorer.nets:
        hard_owners = [int(owner) for owner in net.owners if 0 <= int(owner) < n_hard]
        if not hard_owners:
            continue
        cells = []
        for owner, offset in zip(net.owners, net.offsets):
            if int(owner) < scorer.num_macros:
                x, y = pos[int(owner)] + offset
            else:
                port_idx = int(owner) - scorer.num_macros
                if 0 <= port_idx < len(scorer.port_positions):
                    x, y = scorer.port_positions[port_idx]
                else:
                    continue
            cells.append(scorer._grid_cell(float(x), float(y)))
        if len(cells) <= 1:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = max(0, min(rows)), min(scorer.grid_rows - 1, max(rows))
        c0, c1 = max(0, min(cols)), min(scorer.grid_cols - 1, max(cols))
        area = max(1, (r1 - r0 + 1) * (c1 - c0 + 1))
        total = (
            integral[r1 + 1, c1 + 1]
            - integral[r0, c1 + 1]
            - integral[r1 + 1, c0]
            + integral[r0, c0]
        )
        route_pressure = float(total) / float(area)
        if route_pressure <= 0.0:
            continue
        inc = route_pressure * float(net.route_weight) / math.sqrt(float(len(hard_owners)))
        for owner in hard_owners:
            scores[owner] += inc
    if float(scores.max()) > 0.0:
        scores = scores / float(scores.max())
    return scores


def _proposal_points(
    placement: torch.Tensor,
    benchmark: Benchmark,
    scorer,
    pressure: np.ndarray,
    macro_idx: int,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    pos = placement.detach().cpu().numpy().astype(np.float64, copy=False)
    cur = pos[macro_idx].copy()
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64, copy=False)
    half_w = 0.5 * float(sizes[macro_idx, 0])
    half_h = 0.5 * float(sizes[macro_idx, 1])
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    hot = _hot_center(scorer, pressure, top=_env_int("OURS_FAST_SEARCH_HOT_BINS", 24))
    away = cur - hot
    norm = float(np.linalg.norm(away))
    if norm < 1e-9:
        away = rng.normal(size=2)
        norm = float(np.linalg.norm(away)) + 1e-9
    away = away / norm
    dirs = [
        away,
        -away,
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
        np.array([1.0, 1.0]) / math.sqrt(2.0),
        np.array([1.0, -1.0]) / math.sqrt(2.0),
        np.array([-1.0, 1.0]) / math.sqrt(2.0),
        np.array([-1.0, -1.0]) / math.sqrt(2.0),
    ]
    scales = [0.5, 1.0, 1.75, 2.75]
    gap = _env_float("OURS_FAST_SEARCH_GAP", _env_float("OURS_GAP", 0.005))
    proposals: list[tuple[float, float]] = []

    def add_target(tx: float, ty: float, *, legalize: bool = False) -> None:
        tx = _clip(tx, half_w, cw - half_w)
        ty = _clip(ty, half_h, ch - half_h)
        if legalize:
            legal = _legal_point_near(placement, benchmark, macro_idx, tx, ty, scorer, gap=gap)
            if legal is None:
                return
            proposals.append(legal)
        else:
            proposals.append((tx, ty))

    for scale in scales:
        for direction in dirs:
            x = cur[0] + direction[0] * scorer.grid_width * scale
            y = cur[1] + direction[1] * scorer.grid_height * scale
            add_target(x, y)

    row, col = scorer._grid_cell(float(cur[0]), float(cur[1]))
    radius = _env_int("OURS_FAST_SEARCH_COOL_RADIUS", 4)
    cool: list[tuple[float, int, int]] = []
    for rr in range(max(0, row - radius), min(scorer.grid_rows, row + radius + 1)):
        for cc in range(max(0, col - radius), min(scorer.grid_cols, col + radius + 1)):
            cool.append((float(pressure[rr * scorer.grid_cols + cc]), rr, cc))
    cool.sort()
    for _, rr, cc in cool[: _env_int("OURS_FAST_SEARCH_COOL_BINS", 8)]:
        x = (cc + 0.5) * scorer.grid_width
        y = (rr + 0.5) * scorer.grid_height
        add_target(x, y, legalize=True)

    global_cool = _env_int("OURS_FAST_SEARCH_GLOBAL_COOL_BINS", 12)
    if global_cool > 0 and pressure.size > 0:
        k = min(max(global_cool * 4, global_cool), pressure.size)
        low_order = np.argpartition(pressure, k - 1)[:k]
        low_order = low_order[np.argsort(pressure[low_order])]
        for flat in low_order[:global_cool]:
            rr = int(flat // scorer.grid_cols)
            cc = int(flat % scorer.grid_cols)
            x = (cc + 0.5) * scorer.grid_width
            y = (rr + 0.5) * scorer.grid_height
            add_target(x, y, legalize=True)

    random_targets = _env_int("OURS_FAST_SEARCH_RANDOM_TARGETS", 6)
    for _ in range(max(0, random_targets)):
        x = float(rng.uniform(half_w, max(half_w, cw - half_w)))
        y = float(rng.uniform(half_h, max(half_h, ch - half_h)))
        add_target(x, y, legalize=True)

    dedup: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in proposals:
        key = (int(round(x * 10000.0)), int(round(y * 10000.0)))
        if key in seen:
            continue
        seen.add(key)
        if abs(x - cur[0]) < 1e-9 and abs(y - cur[1]) < 1e-9:
            continue
        dedup.append((float(x), float(y)))
    return dedup


def _legal_point_near(
    placement: torch.Tensor,
    benchmark: Benchmark,
    idx: int,
    x: float,
    y: float,
    scorer,
    *,
    gap: float,
) -> tuple[float, float] | None:
    if _single_move_valid(placement, benchmark, idx, x, y, gap=gap):
        return float(x), float(y)
    max_radius = _env_int("OURS_FAST_SEARCH_LEGAL_RADIUS", 5)
    step_x = scorer.grid_width * _env_float("OURS_FAST_SEARCH_LEGAL_STEP", 0.5)
    step_y = scorer.grid_height * _env_float("OURS_FAST_SEARCH_LEGAL_STEP", 0.5)
    best: tuple[float, float] | None = None
    best_dist = float("inf")
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64, copy=False)
    half_w = 0.5 * float(sizes[idx, 0])
    half_h = 0.5 * float(sizes[idx, 1])
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    for radius in range(1, max_radius + 1):
        for dx_i in range(-radius, radius + 1):
            for dy_i in range(-radius, radius + 1):
                if abs(dx_i) != radius and abs(dy_i) != radius:
                    continue
                tx = _clip(float(x) + dx_i * step_x, half_w, cw - half_w)
                ty = _clip(float(y) + dy_i * step_y, half_h, ch - half_h)
                if not _single_move_valid(placement, benchmark, idx, tx, ty, gap=gap):
                    continue
                dist = (tx - float(x)) * (tx - float(x)) + (ty - float(y)) * (ty - float(y))
                if dist < best_dist:
                    best_dist = dist
                    best = (float(tx), float(ty))
        if best is not None:
            return best
    return None


def _covered_bins(scorer, x: float, y: float, w: float, h: float) -> list[int]:
    x_min = float(x) - 0.5 * float(w)
    x_max = float(x) + 0.5 * float(w)
    y_min = float(y) - 0.5 * float(h)
    y_max = float(y) + 0.5 * float(h)
    bl_row, bl_col = scorer._grid_cell(x_min, y_min)
    ur_row, ur_col = scorer._grid_cell(x_max, y_max)
    bins: list[int] = []
    for row in range(bl_row, ur_row + 1):
        for col in range(bl_col, ur_col + 1):
            bins.append(row * scorer.grid_cols + col)
    return bins


def _hot_center(scorer, pressure: np.ndarray, *, top: int) -> np.ndarray:
    if pressure.size == 0:
        return np.array([0.5 * scorer.canvas_width, 0.5 * scorer.canvas_height], dtype=np.float64)
    k = max(1, min(int(top), pressure.size))
    order = np.argpartition(pressure, -k)[-k:]
    weights = pressure[order].astype(np.float64)
    weights = np.maximum(weights - float(weights.min()), 1e-6)
    xs = ((order % scorer.grid_cols).astype(np.float64) + 0.5) * scorer.grid_width
    ys = ((order // scorer.grid_cols).astype(np.float64) + 0.5) * scorer.grid_height
    return np.array(
        [float(np.average(xs, weights=weights)), float(np.average(ys, weights=weights))],
        dtype=np.float64,
    )


def _single_move_valid(
    placement: torch.Tensor,
    benchmark: Benchmark,
    idx: int,
    x: float,
    y: float,
    *,
    gap: float,
) -> bool:
    n = int(benchmark.num_hard_macros)
    pos = placement[:n].detach().cpu().numpy().astype(np.float64, copy=False)
    sizes = benchmark.macro_sizes[:n].detach().cpu().numpy().astype(np.float64, copy=False)
    half_w = 0.5 * sizes[:, 0]
    half_h = 0.5 * sizes[:, 1]
    if x < half_w[idx] + gap or x > float(benchmark.canvas_width) - half_w[idx] - gap:
        return False
    if y < half_h[idx] + gap or y > float(benchmark.canvas_height) - half_h[idx] - gap:
        return False
    dx = np.abs(pos[:, 0] - float(x))
    dy = np.abs(pos[:, 1] - float(y))
    sep_x = half_w + half_w[idx] + gap
    sep_y = half_h + half_h[idx] + gap
    overlaps = (dx < sep_x) & (dy < sep_y)
    overlaps[idx] = False
    return not bool(np.any(overlaps))


def _clip(value: float, lo: float, hi: float) -> float:
    return float(min(max(float(value), float(lo)), float(hi)))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, str(default))


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return float(default)


def _env_bool(name: str, default: str) -> bool:
    return _env(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _debug(message: str) -> None:
    if _env_bool("OURS_FAST_SEARCH_DEBUG", "0"):
        print(f"[ours fast_search] {message}", file=sys.stderr)
