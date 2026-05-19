"""Native-proxy guided search over soft macro locations.

Hard macro validity is unaffected by this pass: it only moves soft macros
(standard-cell clusters in the challenge representation).  The pass is still
guarded by the official proxy evaluator before it replaces the input placement.
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


def refine_with_soft_search(
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
    except Exception:
        return baseline

    scorer = build_fast_proxy(benchmark, plc)
    if scorer is None:
        return baseline

    n_hard = int(benchmark.num_hard_macros)
    n_all = int(benchmark.num_macros)
    if n_hard >= n_all:
        return baseline

    fixed = benchmark.macro_fixed.detach().cpu().numpy().astype(bool, copy=False)
    soft_indices = np.asarray([i for i in range(n_hard, n_all) if not fixed[i]], dtype=np.int32)
    if soft_indices.size == 0:
        return baseline

    try:
        current = baseline.detach().cpu().numpy().astype(np.float64, copy=True)
        current_score = scorer.score(current, maps=True)
        best_fast = float(current_score["proxy_cost"])
    except Exception:
        return baseline

    rounds = max(1, _env_int("OURS_SOFT_SEARCH_ROUNDS", 2))
    max_trials = max(0, _env_int("OURS_SOFT_SEARCH_TRIALS", 1600))
    macro_count = max(1, _env_int("OURS_SOFT_SEARCH_MACROS", 180))
    eps = _env_float("OURS_SOFT_SEARCH_EPS", 1e-6)
    if max_trials <= 0:
        return baseline

    rng = np.random.default_rng(_env_int("OURS_SOFT_SEARCH_SEED", 20260519))
    node_nets, nets = _build_node_nets(benchmark)
    total_trials = 0
    accepted = 0
    exact_guard = _env_bool("OURS_SOFT_SEARCH_EXACT_GUARD", "1")
    exact_stride = max(0, _env_int("OURS_SOFT_SEARCH_EXACT_STRIDE", 0))
    base_exact_cost: float | None = None
    best_exact_cost: float | None = None
    best_exact: torch.Tensor | None = None
    last_exact_accept = 0
    diag = float(np.hypot(float(benchmark.canvas_width), float(benchmark.canvas_height)))

    if exact_guard:
        try:
            base_exact = compute_proxy_cost(baseline.detach().float(), benchmark, plc)
        except Exception:
            return baseline
        if int(base_exact.get("overlap_count", 1)) != 0:
            return baseline
        base_exact_cost = float(base_exact["proxy_cost"])
        best_exact_cost = base_exact_cost
        best_exact = baseline.detach().clone().float()

    def checkpoint_exact(label: str) -> None:
        nonlocal best_exact_cost, best_exact, last_exact_accept
        if not exact_guard:
            return
        candidate = torch.tensor(current, dtype=baseline.dtype)
        if not is_valid(candidate, benchmark):
            return
        try:
            cand_exact = compute_proxy_cost(candidate.detach().float(), benchmark, plc)
        except Exception:
            return
        if int(cand_exact.get("overlap_count", 1)) != 0:
            return
        cand_cost = float(cand_exact["proxy_cost"])
        if best_exact_cost is None or cand_cost < best_exact_cost - eps:
            best_exact_cost = cand_cost
            best_exact = candidate.detach().clone().float()
            _debug(
                f"checkpoint {label} exact={cand_cost:.6f} "
                f"fast={best_fast:.6f} accepted={accepted} trials={total_trials}"
            )
        last_exact_accept = accepted

    for round_idx in range(rounds):
        try:
            maps = scorer.score(current, maps=True)
        except Exception:
            break
        pressure = _pressure_map(maps, benchmark)
        ranked = _rank_soft_macros(current, benchmark, soft_indices, node_nets, pressure)
        if not ranked:
            break
        cold = _cold_centers(pressure, benchmark)
        round_improved = False

        if _env_bool("OURS_SOFT_SEARCH_BULK", "0"):
            for cand in _bulk_candidates(
                current,
                benchmark,
                ranked,
                node_nets,
                nets,
                cold,
                round_idx,
            ):
                if total_trials >= max_trials:
                    break
                total_trials += 1
                try:
                    score = scorer.score(cand)
                except Exception:
                    continue
                if int(score.get("overlap_count", 1)) != 0:
                    continue
                cost = float(score["proxy_cost"])
                if cost + eps < best_fast:
                    current = cand
                    best_fast = cost
                    accepted += 1
                    round_improved = True
                    if exact_stride > 0 and accepted - last_exact_accept >= exact_stride:
                        checkpoint_exact(f"accepted:{accepted}")

        for macro_idx in ranked[:macro_count]:
            proposals = _proposal_points(
                current,
                benchmark,
                int(macro_idx),
                node_nets,
                nets,
                cold,
                rng,
                diag,
            )
            for point in proposals:
                if total_trials >= max_trials:
                    break
                total_trials += 1
                cand = current.copy()
                cand[int(macro_idx)] = _clamp_point(point, benchmark, int(macro_idx))
                try:
                    score = scorer.score(cand)
                except Exception:
                    continue
                if int(score.get("overlap_count", 1)) != 0:
                    continue
                cost = float(score["proxy_cost"])
                if cost + eps < best_fast:
                    current = cand
                    best_fast = cost
                    accepted += 1
                    round_improved = True
                    if exact_stride > 0 and accepted - last_exact_accept >= exact_stride:
                        checkpoint_exact(f"accepted:{accepted}")

            if total_trials >= max_trials:
                break

        _debug(
            f"round={round_idx} fast={best_fast:.6f} "
            f"accepted={accepted} trials={total_trials}"
        )
        if round_improved:
            checkpoint_exact(f"round:{round_idx}")
        if not round_improved or total_trials >= max_trials:
            break

    if accepted == 0:
        return baseline

    candidate = torch.tensor(current, dtype=baseline.dtype)
    if not is_valid(candidate, benchmark):
        if exact_guard and best_exact is not None and best_exact_cost is not None:
            if base_exact_cost is not None and best_exact_cost < base_exact_cost - eps:
                return best_exact.detach().clone().float()
        return baseline

    if not exact_guard:
        _debug(
            f"accepted without exact guard fast={best_fast:.6f} "
            f"accepted={accepted} trials={total_trials}"
        )
        return candidate.detach().clone().float()

    if last_exact_accept != accepted:
        checkpoint_exact("final")
    if (
        best_exact is not None
        and best_exact_cost is not None
        and base_exact_cost is not None
        and best_exact_cost < base_exact_cost - eps
    ):
        _debug(
            "accepted exact "
            f"{base_exact_cost:.6f}->{best_exact_cost:.6f} "
            f"fast={best_fast:.6f} accepted={accepted} trials={total_trials}"
        )
        return best_exact.detach().clone().float()

    final_exact = best_exact_cost if best_exact_cost is not None else base_exact_cost
    _debug(
        "rejected exact "
        f"{base_exact_cost:.6f}->{final_exact:.6f} "
        f"fast={best_fast:.6f} accepted={accepted} trials={total_trials}"
    )
    return baseline


def _build_node_nets(benchmark: Benchmark) -> tuple[list[list[int]], list[list[int]]]:
    n_all = int(benchmark.num_macros)
    node_nets: list[list[int]] = [[] for _ in range(n_all)]
    nets: list[list[int]] = []
    for nodes_t in benchmark.net_nodes:
        nodes = [int(x) for x in nodes_t.tolist()]
        if len(nodes) <= 1:
            continue
        net_idx = len(nets)
        nets.append(nodes)
        for node in nodes:
            if 0 <= node < n_all:
                node_nets[node].append(net_idx)
    return node_nets, nets


def _pressure_map(score_with_maps: dict[str, object], benchmark: Benchmark) -> np.ndarray:
    rows = int(benchmark.grid_rows)
    cols = int(benchmark.grid_cols)
    v = np.asarray(score_with_maps["v_routing_cong"], dtype=np.float64).reshape(rows, cols)
    h = np.asarray(score_with_maps["h_routing_cong"], dtype=np.float64).reshape(rows, cols)
    return v + h


def _rank_soft_macros(
    placement: np.ndarray,
    benchmark: Benchmark,
    soft_indices: np.ndarray,
    node_nets: list[list[int]],
    pressure: np.ndarray,
) -> list[int]:
    rows, cols = pressure.shape
    bin_w = float(benchmark.canvas_width) / max(cols, 1)
    bin_h = float(benchmark.canvas_height) / max(rows, 1)
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64, copy=False)
    ranked: list[tuple[float, int]] = []
    for idx_raw in soft_indices:
        idx = int(idx_raw)
        x, y = placement[idx]
        col = min(cols - 1, max(0, int(float(x) / max(bin_w, 1e-12))))
        row = min(rows - 1, max(0, int(float(y) / max(bin_h, 1e-12))))
        area = float(sizes[idx, 0] * sizes[idx, 1])
        degree = float(len(node_nets[idx]))
        score = float(pressure[row, col]) * (1.0 + 0.001 * area) + 0.001 * degree
        ranked.append((score, idx))
    ranked.sort(reverse=True)
    return [idx for _, idx in ranked]


def _cold_centers(pressure: np.ndarray, benchmark: Benchmark) -> list[np.ndarray]:
    rows, cols = pressure.shape
    if rows <= 0 or cols <= 0:
        return []
    count = max(16, min(int(pressure.size), int(pressure.size) // 10))
    order = np.argsort(pressure.reshape(-1))[:count]
    bin_w = float(benchmark.canvas_width) / max(cols, 1)
    bin_h = float(benchmark.canvas_height) / max(rows, 1)
    centers = []
    for flat in order:
        row = int(flat) // cols
        col = int(flat) % cols
        centers.append(np.asarray([(col + 0.5) * bin_w, (row + 0.5) * bin_h], dtype=np.float64))
    return centers


def _proposal_points(
    placement: np.ndarray,
    benchmark: Benchmark,
    macro_idx: int,
    node_nets: list[list[int]],
    nets: list[list[int]],
    cold_centers: list[np.ndarray],
    rng: np.random.Generator,
    diag: float,
) -> list[np.ndarray]:
    current = placement[macro_idx].astype(np.float64, copy=True)
    proposals: list[np.ndarray] = []

    centroid = _connected_centroid(placement, benchmark, macro_idx, node_nets, nets)
    if centroid is not None:
        delta = centroid - current
        proposals.append(centroid)
        proposals.append(current + 0.35 * delta)
        proposals.append(current + 0.70 * delta)

    jitter_sigma = _env_float("OURS_SOFT_SEARCH_JITTER", 0.030) * max(diag, 1e-9)
    for _ in range(max(0, _env_int("OURS_SOFT_SEARCH_JITTERS", 2))):
        proposals.append(current + rng.normal(0.0, jitter_sigma, size=2))

    if cold_centers:
        count = min(max(1, _env_int("OURS_SOFT_SEARCH_COLD_SAMPLES", 3)), len(cold_centers))
        for raw_idx in rng.choice(len(cold_centers), size=count, replace=False):
            cold = cold_centers[int(raw_idx)]
            proposals.append(current + 0.30 * (cold - current))
            proposals.append(current + 0.65 * (cold - current))
            proposals.append(cold)

    return proposals


def _connected_centroid(
    placement: np.ndarray,
    benchmark: Benchmark,
    macro_idx: int,
    node_nets: list[list[int]],
    nets: list[list[int]],
) -> np.ndarray | None:
    n_all = int(benchmark.num_macros)
    ports = benchmark.port_positions.detach().cpu().numpy().astype(np.float64, copy=False)
    pts: list[np.ndarray] = []
    for net_idx in node_nets[macro_idx]:
        for node in nets[net_idx]:
            if node == macro_idx:
                continue
            if 0 <= node < n_all:
                pts.append(placement[node])
            else:
                port_idx = int(node) - n_all
                if 0 <= port_idx < len(ports):
                    pts.append(ports[port_idx])
    if not pts:
        return None
    return np.mean(np.asarray(pts, dtype=np.float64), axis=0)


def _bulk_candidates(
    placement: np.ndarray,
    benchmark: Benchmark,
    ranked: list[int],
    node_nets: list[list[int]],
    nets: list[list[int]],
    cold_centers: list[np.ndarray],
    round_idx: int,
) -> list[np.ndarray]:
    if not ranked:
        return []
    counts = _env_int_list("OURS_SOFT_SEARCH_BULK_COUNTS", [32, 96, 256, 512])
    alphas = _env_float_list("OURS_SOFT_SEARCH_BULK_ALPHAS", [0.18, 0.34, 0.55])
    if not counts or not alphas:
        return []
    max_count = max(1, _env_int("OURS_SOFT_SEARCH_BULK_MAX", 512))
    counts = sorted({max(1, min(int(c), len(ranked), max_count)) for c in counts})
    out: list[np.ndarray] = []

    for count in counts:
        selected = ranked[:count]
        for alpha in alphas:
            for mode in ("mixed", "cold", "centroid"):
                cand = placement.copy()
                moved = 0
                for order, idx_raw in enumerate(selected):
                    idx = int(idx_raw)
                    cur = placement[idx]
                    centroid = _connected_centroid(placement, benchmark, idx, node_nets, nets)
                    cold = None
                    if cold_centers:
                        cold = cold_centers[(order * 9973 + round_idx * 37) % len(cold_centers)]
                    if mode == "centroid":
                        if centroid is None:
                            continue
                        target = centroid
                    elif mode == "cold":
                        if cold is None:
                            continue
                        target = cold
                    else:
                        if centroid is not None and cold is not None:
                            target = 0.68 * centroid + 0.32 * cold
                        elif centroid is not None:
                            target = centroid
                        elif cold is not None:
                            target = cold
                        else:
                            continue
                    cand[idx] = _clamp_point(cur + float(alpha) * (target - cur), benchmark, idx)
                    moved += 1
                if moved > 0:
                    out.append(cand)
    return out


def _clamp_point(point: np.ndarray, benchmark: Benchmark, macro_idx: int) -> np.ndarray:
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64, copy=False)
    width = float(sizes[macro_idx, 0])
    height = float(sizes[macro_idx, 1])
    half_w = 0.5 * width
    half_h = 0.5 * height
    x = float(np.clip(float(point[0]), half_w, float(benchmark.canvas_width) - half_w))
    y = float(np.clip(float(point[1]), half_h, float(benchmark.canvas_height) - half_h))
    return np.asarray([x, y], dtype=np.float64)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: str) -> bool:
    return _env(name, default).strip().lower() not in {"0", "false", "no", "off"}


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


def _env_int_list(name: str, default: list[int]) -> list[int]:
    raw = _env(name, ",".join(str(x) for x in default)).strip()
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out or list(default)


def _env_float_list(name: str, default: list[float]) -> list[float]:
    raw = _env(name, ",".join(str(x) for x in default)).strip()
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out or list(default)


def _debug(message: str) -> None:
    if _env_bool("OURS_SOFT_SEARCH_DEBUG", "0"):
        print(f"[ours-soft-search] {message}", file=sys.stderr, flush=True)
