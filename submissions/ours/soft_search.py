"""Native-proxy guided search over soft macro locations.

Hard macro validity is unaffected by this pass: it only moves soft macros
(standard-cell clusters in the challenge representation).  The pass is still
guarded by the official proxy evaluator before it replaces the input placement.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing as mp
import os
from pathlib import Path
import sys
import math
import time
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
    if (
        _env_bool("OURS_SOFT_SEARCH_PORTFOLIO", "1")
        and os.environ.get("_OURS_SOFT_SEARCH_PORTFOLIO_ACTIVE") != "1"
    ):
        return _refine_with_soft_search_portfolio(
            baseline,
            benchmark,
            load_plc=load_plc,
            is_valid=is_valid,
        )

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

    rounds = max(1, _env_int("OURS_SOFT_SEARCH_ROUNDS", 6))
    max_trials = max(0, _env_int("OURS_SOFT_SEARCH_TRIALS", 100000))
    macro_count = max(1, _env_int("OURS_SOFT_SEARCH_MACROS", 2000))
    eps = _env_float("OURS_SOFT_SEARCH_EPS", 1e-6)
    search_started = time.monotonic()
    timeout = max(0.0, _env_float("OURS_SOFT_SEARCH_TIMEOUT", 3200.0))
    if max_trials <= 0:
        return baseline

    rng = np.random.default_rng(_env_int("OURS_SOFT_SEARCH_SEED", 20260519))
    node_nets, nets = _build_node_nets(benchmark)
    total_trials = 0
    accepted = 0
    exact_guard = _env_bool("OURS_SOFT_SEARCH_EXACT_GUARD", "1")
    exact_stride = max(0, _env_int("OURS_SOFT_SEARCH_EXACT_STRIDE", 0))
    bulk_exact_topk = max(0, _env_int("OURS_SOFT_SEARCH_BULK_EXACT_TOPK", 0))
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

    def timed_out() -> bool:
        return timeout > 0.0 and time.monotonic() - search_started >= timeout

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

    def accept_exact_candidate(cand: np.ndarray, fast_cost: float, label: str) -> bool:
        nonlocal current, best_fast, best_exact_cost, best_exact, accepted, last_exact_accept
        if not exact_guard:
            return False
        candidate = torch.tensor(cand, dtype=baseline.dtype)
        if not is_valid(candidate, benchmark):
            return False
        try:
            cand_exact = compute_proxy_cost(candidate.detach().float(), benchmark, plc)
        except Exception:
            return False
        if int(cand_exact.get("overlap_count", 1)) != 0:
            return False
        cand_cost = float(cand_exact["proxy_cost"])
        if best_exact_cost is None or cand_cost < best_exact_cost - eps:
            current = cand.copy()
            best_fast = float(fast_cost)
            best_exact_cost = cand_cost
            best_exact = candidate.detach().clone().float()
            accepted += 1
            last_exact_accept = accepted
            _debug(
                f"exact-rescue {label} exact={cand_cost:.6f} "
                f"fast={fast_cost:.6f} accepted={accepted} trials={total_trials}"
            )
            return True
        return False

    for round_idx in range(rounds):
        if timed_out():
            _debug(f"timeout before round={round_idx} trials={total_trials} accepted={accepted}")
            break
        try:
            maps = scorer.score(current, maps=True)
        except Exception:
            break
        pressure = _pressure_map(maps, benchmark)
        route_weight = _soft_route_weight(maps)
        ranked = _rank_soft_macros(
            current,
            benchmark,
            soft_indices,
            node_nets,
            nets,
            pressure,
            route_weight=route_weight,
        )
        if not ranked:
            break
        cold = _cold_centers(pressure, benchmark)
        round_improved = False

        if _env_bool("OURS_SOFT_SEARCH_BULK", "0"):
            bulk_scored: list[tuple[float, np.ndarray]] = []
            for cand in _bulk_candidates(
                current,
                benchmark,
                ranked,
                node_nets,
                nets,
                cold,
                pressure,
                route_weight,
                round_idx,
            ):
                if total_trials >= max_trials or timed_out():
                    break
                total_trials += 1
                try:
                    score = scorer.score(cand)
                except Exception:
                    continue
                if int(score.get("overlap_count", 1)) != 0:
                    continue
                cost = float(score["proxy_cost"])
                if bulk_exact_topk > 0 and exact_guard:
                    bulk_scored.append((cost, cand))
                if cost + eps < best_fast:
                    current = cand
                    best_fast = cost
                    accepted += 1
                    round_improved = True
                    if exact_stride > 0 and accepted - last_exact_accept >= exact_stride:
                        checkpoint_exact(f"accepted:{accepted}")
            if bulk_exact_topk > 0 and bulk_scored:
                bulk_scored.sort(key=lambda item: item[0])
                for exact_idx, (cost, cand) in enumerate(bulk_scored[:bulk_exact_topk]):
                    if accept_exact_candidate(cand, cost, f"bulk:{round_idx}:{exact_idx}"):
                        round_improved = True

        for macro_idx in ranked[:macro_count]:
            if timed_out():
                break
            proposals = _proposal_points(
                current,
                benchmark,
                int(macro_idx),
                node_nets,
                nets,
                pressure,
                cold,
                rng,
                diag,
                route_weight,
            )
            for point in proposals:
                if total_trials >= max_trials or timed_out():
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
        if not round_improved or total_trials >= max_trials or timed_out():
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


def _refine_with_soft_search_portfolio(
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
        base_costs = compute_proxy_cost(baseline.detach().float(), benchmark, plc)
    except Exception:
        return baseline
    if int(base_costs.get("overlap_count", 1)) != 0:
        return baseline

    best = baseline.detach().clone().float()
    best_cost = float(base_costs["proxy_cost"])
    eps = _env_float("OURS_SOFT_SEARCH_EPS", 1e-6)
    base_seed = _env_int("OURS_SOFT_SEARCH_SEED", 20260519)
    modes = _env("OURS_SOFT_SEARCH_PORTFOLIO_MODES", "bulk")
    parsed_modes = [mode.strip().lower() for mode in modes.split(",") if mode.strip()]
    if not parsed_modes:
        parsed_modes = ["bulk"]
    routes = _env("OURS_SOFT_SEARCH_PORTFOLIO_ROUTES", "auto,0.00,0.25,0.50,3.00,5.00,8.00")
    parsed_routes = [route.strip().lower() for route in routes.split(",") if route.strip()]
    if not parsed_routes:
        parsed_routes = ["auto"]
    density_weights = [
        max(0.0, float(weight))
        for weight in _env_float_list("OURS_SOFT_SEARCH_PORTFOLIO_DENSITY_WEIGHTS", [0.0, 0.50])
    ]
    if not density_weights:
        density_weights = [0.0]

    old_active = os.environ.get("_OURS_SOFT_SEARCH_PORTFOLIO_ACTIVE")
    old_bulk = os.environ.get("OURS_SOFT_SEARCH_BULK")
    old_seed = os.environ.get("OURS_SOFT_SEARCH_SEED")
    old_route = os.environ.get("OURS_SOFT_SEARCH_ROUTE_WEIGHT")
    old_density = os.environ.get("OURS_SOFT_SEARCH_DENSITY_WEIGHT")
    os.environ["_OURS_SOFT_SEARCH_PORTFOLIO_ACTIVE"] = "1"
    try:
        tasks = _portfolio_tasks(parsed_routes, parsed_modes, density_weights, base_seed)
        candidates = _portfolio_candidates(
            baseline,
            benchmark,
            tasks,
            load_plc=load_plc,
            is_valid=is_valid,
        )
        for route, mode, density_weight, candidate in candidates:
            if torch.allclose(candidate, baseline, atol=1e-7, rtol=0.0):
                continue
            if not is_valid(candidate, benchmark):
                continue
            try:
                costs = compute_proxy_cost(candidate.detach().float(), benchmark, plc)
            except Exception:
                continue
            if int(costs.get("overlap_count", 1)) != 0:
                continue
            cost = float(costs["proxy_cost"])
            if cost < best_cost - eps:
                best = candidate.detach().clone().float()
                best_cost = cost
                _debug(
                    f"portfolio route={route} mode={mode} density={density_weight:.3g} "
                    f"exact={best_cost:.6f}"
                )
    finally:
        if old_active is None:
            os.environ.pop("_OURS_SOFT_SEARCH_PORTFOLIO_ACTIVE", None)
        else:
            os.environ["_OURS_SOFT_SEARCH_PORTFOLIO_ACTIVE"] = old_active
        if old_bulk is None:
            os.environ.pop("OURS_SOFT_SEARCH_BULK", None)
        else:
            os.environ["OURS_SOFT_SEARCH_BULK"] = old_bulk
        if old_seed is None:
            os.environ.pop("OURS_SOFT_SEARCH_SEED", None)
        else:
            os.environ["OURS_SOFT_SEARCH_SEED"] = old_seed
        if old_route is None:
            os.environ.pop("OURS_SOFT_SEARCH_ROUTE_WEIGHT", None)
        else:
            os.environ["OURS_SOFT_SEARCH_ROUTE_WEIGHT"] = old_route
        if old_density is None:
            os.environ.pop("OURS_SOFT_SEARCH_DENSITY_WEIGHT", None)
        else:
            os.environ["OURS_SOFT_SEARCH_DENSITY_WEIGHT"] = old_density

    return best


def _portfolio_tasks(
    parsed_routes: list[str],
    parsed_modes: list[str],
    density_weights: list[float],
    base_seed: int,
) -> list[tuple[str, str, float, int]]:
    tasks: list[tuple[str, str, float, int]] = []
    diversify_routes = _env_bool("OURS_SOFT_SEARCH_PORTFOLIO_DIVERSE_SEEDS", "0")
    for route_idx, route in enumerate(parsed_routes):
        if route not in {"auto", "adaptive", "default"}:
            try:
                float(route)
            except ValueError:
                continue
        for mode_idx, mode in enumerate(parsed_modes):
            if mode not in {"single", "solo", "greedy", "bulk", "batch"}:
                continue
            for density_idx, density_weight in enumerate(density_weights):
                route_offset = 104729 * route_idx if diversify_routes else 0
                density_offset = 9176 * density_idx
                seed = base_seed + 1009 * mode_idx + density_offset + route_offset
                tasks.append((route, mode, float(density_weight), seed))
    return tasks


def _portfolio_candidates(
    baseline: torch.Tensor,
    benchmark: Benchmark,
    tasks: list[tuple[str, str, float, int]],
    *,
    load_plc: Callable[[Benchmark], object | None],
    is_valid: Callable[[torch.Tensor, Benchmark], bool],
) -> list[tuple[str, str, float, torch.Tensor]]:
    workers = max(1, _env_int("OURS_SOFT_SEARCH_PORTFOLIO_WORKERS", min(14, len(tasks))))
    if workers <= 1 or len(tasks) <= 1:
        return [
            (
                route,
                mode,
                density_weight,
                _run_portfolio_candidate(
                    baseline,
                    benchmark,
                    route,
                    mode,
                    density_weight,
                    seed,
                    load_plc=load_plc,
                    is_valid=is_valid,
                ),
            )
            for route, mode, density_weight, seed in tasks
        ]

    max_workers = min(workers, len(tasks))
    results: list[tuple[str, str, float, torch.Tensor]] = []
    timed_out = False
    failed_parallel = False
    pool: ProcessPoolExecutor | None = None
    try:
        baseline_np = baseline.detach().cpu().float().numpy().copy()
        benchmark_name = str(benchmark.name)
        pool_kwargs = {}
        start_method = _env("OURS_SOFT_SEARCH_PORTFOLIO_START", "fork").strip()
        if start_method:
            try:
                pool_kwargs["mp_context"] = mp.get_context(start_method)
            except ValueError:
                _debug(f"unsupported multiprocessing start={start_method!r}; using default")
        pool = ProcessPoolExecutor(max_workers=max_workers, **pool_kwargs)
        future_meta = {
            pool.submit(
                _run_portfolio_candidate_worker,
                baseline_np,
                benchmark_name,
                route,
                mode,
                density_weight,
                seed,
            ): (route, mode, density_weight)
            for route, mode, density_weight, seed in tasks
        }
        pending = set(future_meta)
        timeout = max(0.0, _env_float("OURS_SOFT_SEARCH_PORTFOLIO_TIMEOUT", 3000.0))
        deadline = time.monotonic() + timeout if timeout > 0.0 else None
        while pending:
            wait_timeout = None
            if deadline is not None:
                wait_timeout = max(0.0, deadline - time.monotonic())
            done, pending = wait(pending, timeout=wait_timeout, return_when=FIRST_COMPLETED)
            if not done:
                timed_out = True
                _debug(f"portfolio collection timeout results={len(results)} tasks={len(tasks)}")
                break
            for future in done:
                route, mode, density_weight = future_meta[future]
                try:
                    results.append((route, mode, density_weight, future.result()))
                except Exception as exc:
                    _debug(
                        f"portfolio worker route={route} mode={mode} "
                        f"density={density_weight:.3g} failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
    except Exception as exc:
        _debug(f"portfolio parallel failed: {type(exc).__name__}: {exc}")
        failed_parallel = True
    finally:
        if pool is not None:
            if timed_out:
                for future in pending:
                    future.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
            else:
                pool.shutdown(wait=True, cancel_futures=True)
    if results:
        if len(results) != len(tasks):
            _debug(f"portfolio partial results={len(results)} tasks={len(tasks)}; keeping partial")
        return results
    if timed_out:
        _debug(f"portfolio timed out with no completed results tasks={len(tasks)}")
        return []
    if not failed_parallel:
        return []
    _debug(f"portfolio falling back to serial tasks={len(tasks)}")
    return [
        (
            route,
            mode,
            density_weight,
            _run_portfolio_candidate(
                baseline,
                benchmark,
                route,
                mode,
                density_weight,
                seed,
                load_plc=load_plc,
                is_valid=is_valid,
            ),
        )
        for route, mode, density_weight, seed in tasks
    ]


def _run_portfolio_candidate_worker(
    baseline_array: np.ndarray,
    benchmark_name: str,
    route: str,
    mode: str,
    density_weight: float,
    seed: int,
) -> torch.Tensor:
    loaded = _load_benchmark_and_plc_for_worker(benchmark_name)
    if loaded is None:
        return torch.tensor(baseline_array, dtype=torch.float32)
    benchmark, plc = loaded

    def load_plc(_: Benchmark):
        return plc

    return _run_portfolio_candidate(
        torch.tensor(baseline_array, dtype=torch.float32),
        benchmark,
        route,
        mode,
        density_weight,
        seed,
        load_plc=load_plc,
        is_valid=_is_valid_for_worker,
    )


def _run_portfolio_candidate(
    baseline: torch.Tensor,
    benchmark: Benchmark,
    route: str,
    mode: str,
    density_weight: float,
    seed: int,
    *,
    load_plc: Callable[[Benchmark], object | None],
    is_valid: Callable[[torch.Tensor, Benchmark], bool],
) -> torch.Tensor:
    old_active = os.environ.get("_OURS_SOFT_SEARCH_PORTFOLIO_ACTIVE")
    old_bulk = os.environ.get("OURS_SOFT_SEARCH_BULK")
    old_seed = os.environ.get("OURS_SOFT_SEARCH_SEED")
    old_route = os.environ.get("OURS_SOFT_SEARCH_ROUTE_WEIGHT")
    old_density = os.environ.get("OURS_SOFT_SEARCH_DENSITY_WEIGHT")
    os.environ["_OURS_SOFT_SEARCH_PORTFOLIO_ACTIVE"] = "1"
    try:
        if route in {"auto", "adaptive", "default"}:
            os.environ.pop("OURS_SOFT_SEARCH_ROUTE_WEIGHT", None)
        else:
            os.environ["OURS_SOFT_SEARCH_ROUTE_WEIGHT"] = str(max(0.0, float(route)))
        os.environ["OURS_SOFT_SEARCH_DENSITY_WEIGHT"] = str(max(0.0, float(density_weight)))
        if mode in {"single", "solo", "greedy"}:
            os.environ["OURS_SOFT_SEARCH_BULK"] = "0"
        else:
            os.environ["OURS_SOFT_SEARCH_BULK"] = "1"
        os.environ["OURS_SOFT_SEARCH_SEED"] = str(seed)
        return refine_with_soft_search(
            baseline,
            benchmark,
            load_plc=load_plc,
            is_valid=is_valid,
        ).detach().cpu().float()
    finally:
        if old_active is None:
            os.environ.pop("_OURS_SOFT_SEARCH_PORTFOLIO_ACTIVE", None)
        else:
            os.environ["_OURS_SOFT_SEARCH_PORTFOLIO_ACTIVE"] = old_active
        if old_bulk is None:
            os.environ.pop("OURS_SOFT_SEARCH_BULK", None)
        else:
            os.environ["OURS_SOFT_SEARCH_BULK"] = old_bulk
        if old_seed is None:
            os.environ.pop("OURS_SOFT_SEARCH_SEED", None)
        else:
            os.environ["OURS_SOFT_SEARCH_SEED"] = old_seed
        if old_route is None:
            os.environ.pop("OURS_SOFT_SEARCH_ROUTE_WEIGHT", None)
        else:
            os.environ["OURS_SOFT_SEARCH_ROUTE_WEIGHT"] = old_route
        if old_density is None:
            os.environ.pop("OURS_SOFT_SEARCH_DENSITY_WEIGHT", None)
        else:
            os.environ["OURS_SOFT_SEARCH_DENSITY_WEIGHT"] = old_density


def _load_benchmark_and_plc_for_worker(benchmark_name: str):
    try:
        from macro_place.loader import load_benchmark, load_benchmark_from_dir
    except Exception:
        return None

    name = str(benchmark_name)
    ibm_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if (ibm_dir / "netlist.pb.txt").exists():
        try:
            return load_benchmark_from_dir(str(ibm_dir))
        except Exception:
            return None

    aliases = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "mempool_tile_ng45": "mempool_tile",
        "nvdla_ng45": "nvdla",
    }
    base_name = aliases.get(name, name.replace("_ng45", "").replace("_asap7", ""))
    ng45 = (
        Path("external/MacroPlacement/Flows/NanGate45")
        / base_name
        / "netlist"
        / "output_CT_Grouping"
    )
    if (ng45 / "netlist.pb.txt").exists():
        try:
            return load_benchmark(
                str(ng45 / "netlist.pb.txt"),
                str(ng45 / "initial.plc"),
                name=name,
            )
        except Exception:
            return None
    return None


def _load_plc_for_worker(benchmark: Benchmark):
    loaded = _load_benchmark_and_plc_for_worker(str(benchmark.name))
    if loaded is None:
        return None
    return loaded[1]


def _is_valid_for_worker(placement: torch.Tensor, benchmark: Benchmark) -> bool:
    try:
        from macro_place.utils import validate_placement

        return bool(validate_placement(placement, benchmark)[0])
    except Exception:
        return False


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
    pressure = v + h
    density_weight = _env_float("OURS_SOFT_SEARCH_DENSITY_WEIGHT", 0.0)
    if density_weight > 0.0 and "density_map" in score_with_maps:
        density = np.asarray(score_with_maps["density_map"], dtype=np.float64).reshape(rows, cols)
        pressure = pressure + density_weight * density
    return pressure


def _rank_soft_macros(
    placement: np.ndarray,
    benchmark: Benchmark,
    soft_indices: np.ndarray,
    node_nets: list[list[int]],
    nets: list[list[int]],
    pressure: np.ndarray,
    route_weight: float,
) -> list[int]:
    rows, cols = pressure.shape
    bin_w = float(benchmark.canvas_width) / max(cols, 1)
    bin_h = float(benchmark.canvas_height) / max(rows, 1)
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64, copy=False)
    route_scores = (
        _soft_route_participation_scores(
            placement,
            benchmark,
            soft_indices,
            nets,
            pressure,
        )
        if route_weight > 0.0
        else {}
    )
    ranked: list[tuple[float, int]] = []
    for idx_raw in soft_indices:
        idx = int(idx_raw)
        x, y = placement[idx]
        col = min(cols - 1, max(0, int(float(x) / max(bin_w, 1e-12))))
        row = min(rows - 1, max(0, int(float(y) / max(bin_h, 1e-12))))
        area = float(sizes[idx, 0] * sizes[idx, 1])
        degree = float(len(node_nets[idx]))
        score = (
            float(pressure[row, col]) * (1.0 + 0.001 * area)
            + 0.001 * degree
            + route_weight * float(route_scores.get(idx, 0.0))
        )
        ranked.append((score, idx))
    ranked.sort(reverse=True)
    return [idx for _, idx in ranked]


def _soft_route_weight(score_with_maps: dict[str, object]) -> float:
    raw = os.environ.get("OURS_SOFT_SEARCH_ROUTE_WEIGHT")
    if raw is not None:
        try:
            return max(0.0, float(raw))
        except ValueError:
            return 0.0
    proxy = float(score_with_maps.get("proxy_cost", 0.0))
    congestion = float(score_with_maps.get("congestion_cost", 0.0))
    if (
        proxy >= _env_float("OURS_SOFT_SEARCH_ROUTE_MIN_PROXY", 1.20)
        or congestion >= _env_float("OURS_SOFT_SEARCH_ROUTE_MIN_CONG", 1.45)
    ):
        return _env_float("OURS_SOFT_SEARCH_ROUTE_AUTO_WEIGHT", 3.00)
    return 0.0


def _soft_route_participation_scores(
    placement: np.ndarray,
    benchmark: Benchmark,
    soft_indices: np.ndarray,
    nets: list[list[int]],
    pressure: np.ndarray,
) -> dict[int, float]:
    if pressure.size == 0 or soft_indices.size == 0:
        return {}
    rows, cols = pressure.shape
    bin_w = float(benchmark.canvas_width) / max(cols, 1)
    bin_h = float(benchmark.canvas_height) / max(rows, 1)
    n_all = int(benchmark.num_macros)
    soft_set = {int(idx) for idx in soft_indices.tolist()}
    ports = benchmark.port_positions.detach().cpu().numpy().astype(np.float64, copy=False)
    integral = np.pad(pressure, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    scores: dict[int, float] = {idx: 0.0 for idx in soft_set}

    for nodes in nets:
        soft_nodes = [idx for idx in nodes if idx in soft_set]
        if not soft_nodes:
            continue
        cells: list[tuple[int, int]] = []
        for node in nodes:
            if 0 <= node < n_all:
                x, y = placement[node]
            else:
                port_idx = int(node) - n_all
                if not (0 <= port_idx < len(ports)):
                    continue
                x, y = ports[port_idx]
            col = min(cols - 1, max(0, int(float(x) / max(bin_w, 1e-12))))
            row = min(rows - 1, max(0, int(float(y) / max(bin_h, 1e-12))))
            cells.append((row, col))
        if len(cells) <= 1:
            continue
        r0 = min(row for row, _ in cells)
        r1 = max(row for row, _ in cells)
        c0 = min(col for _, col in cells)
        c1 = max(col for _, col in cells)
        area = max(1, (r1 - r0 + 1) * (c1 - c0 + 1))
        total = (
            integral[r1 + 1, c1 + 1]
            - integral[r0, c1 + 1]
            - integral[r1 + 1, c0]
            + integral[r0, c0]
        )
        participation = float(total) / float(area)
        if participation <= 0.0:
            continue
        inc = participation / max(1.0, math.sqrt(float(len(soft_nodes))))
        for idx in soft_nodes:
            scores[idx] += inc

    max_score = max(scores.values(), default=0.0)
    if max_score > 0.0:
        for idx in list(scores):
            scores[idx] /= max_score
    return scores


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
    pressure: np.ndarray,
    cold_centers: list[np.ndarray],
    rng: np.random.Generator,
    diag: float,
    route_weight: float,
) -> list[np.ndarray]:
    current = placement[macro_idx].astype(np.float64, copy=True)
    proposals: list[np.ndarray] = []

    centroid = _connected_centroid(placement, benchmark, macro_idx, node_nets, nets)
    if centroid is not None:
        delta = centroid - current
        proposals.append(centroid)
        proposals.append(current + 0.35 * delta)
        proposals.append(current + 0.70 * delta)

    if route_weight > 0.0:
        for target in _route_relief_targets(
            placement,
            benchmark,
            macro_idx,
            node_nets,
            nets,
            pressure,
            limit=_env_int("OURS_SOFT_SEARCH_ROUTE_TARGETS", 0),
        ):
            delta = target - current
            proposals.append(current + 0.35 * delta)
            proposals.append(current + 0.70 * delta)
            proposals.append(target)

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
    pressure: np.ndarray,
    route_weight: float,
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
    route_cache: dict[int, np.ndarray | None] = {}

    def route_target(idx: int) -> np.ndarray | None:
        if idx not in route_cache:
            targets = _route_relief_targets(
                placement,
                benchmark,
                idx,
                node_nets,
                nets,
                pressure,
                limit=1,
            )
            route_cache[idx] = targets[0] if targets else None
        return route_cache[idx]

    modes = ["mixed", "cold", "centroid"]
    if route_weight > 0.0 and _env_bool("OURS_SOFT_SEARCH_BULK_ROUTE", "0"):
        modes.insert(0, "route")

    for count in counts:
        selected = ranked[:count]
        for alpha in alphas:
            for mode in modes:
                cand = placement.copy()
                moved = 0
                for order, idx_raw in enumerate(selected):
                    idx = int(idx_raw)
                    cur = placement[idx]
                    centroid = _connected_centroid(placement, benchmark, idx, node_nets, nets)
                    cold = None
                    if cold_centers:
                        cold = cold_centers[(order * 9973 + round_idx * 37) % len(cold_centers)]
                    route = route_target(idx) if route_weight > 0.0 else None
                    if mode == "route":
                        if route is None:
                            continue
                        target = route
                    elif mode == "centroid":
                        if centroid is None:
                            continue
                        target = centroid
                    elif mode == "cold":
                        if cold is None:
                            continue
                        target = cold
                    else:
                        if route is not None and centroid is not None and cold is not None:
                            target = 0.52 * centroid + 0.33 * route + 0.15 * cold
                        elif route is not None and centroid is not None:
                            target = 0.62 * centroid + 0.38 * route
                        elif route is not None and cold is not None:
                            target = 0.62 * route + 0.38 * cold
                        elif centroid is not None and cold is not None:
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


def _route_relief_targets(
    placement: np.ndarray,
    benchmark: Benchmark,
    macro_idx: int,
    node_nets: list[list[int]],
    nets: list[list[int]],
    pressure: np.ndarray,
    *,
    limit: int,
) -> list[np.ndarray]:
    if limit <= 0 or pressure.size == 0 or not node_nets[macro_idx]:
        return []
    rows, cols = pressure.shape
    if rows <= 0 or cols <= 0:
        return []

    bin_w = float(benchmark.canvas_width) / max(cols, 1)
    bin_h = float(benchmark.canvas_height) / max(rows, 1)
    diag = max(float(np.hypot(float(benchmark.canvas_width), float(benchmark.canvas_height))), 1e-9)
    flat_pressure = pressure.reshape(-1)
    pressure_scale = max(float(np.percentile(flat_pressure, 90)), float(flat_pressure.mean()), 1e-9)
    integral = np.pad(pressure, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

    other_cells_by_net: list[list[tuple[int, int]]] = []
    n_all = int(benchmark.num_macros)
    ports = benchmark.port_positions.detach().cpu().numpy().astype(np.float64, copy=False)
    for net_idx in node_nets[macro_idx]:
        cells: list[tuple[int, int]] = []
        for node in nets[net_idx]:
            if int(node) == macro_idx:
                continue
            if 0 <= int(node) < n_all:
                x, y = placement[int(node)]
            else:
                port_idx = int(node) - n_all
                if not (0 <= port_idx < len(ports)):
                    continue
                x, y = ports[port_idx]
            cells.append(_point_cell(float(x), float(y), bin_w, bin_h, rows, cols))
        if cells:
            other_cells_by_net.append(cells)
    if not other_cells_by_net:
        return []

    current = placement[macro_idx].astype(np.float64, copy=False)
    centroid = _connected_centroid(placement, benchmark, macro_idx, node_nets, nets)
    if centroid is None:
        centroid = current

    candidate_flats: set[int] = set()
    low_pool = min(int(pressure.size), max(limit * _env_int("OURS_SOFT_SEARCH_ROUTE_POOL", 36), limit))
    if low_pool > 0:
        order = np.argpartition(flat_pressure, low_pool - 1)[:low_pool]
        order = order[np.argsort(flat_pressure[order])]
        candidate_flats.update(int(flat) for flat in order)

    radius = max(0, _env_int("OURS_SOFT_SEARCH_ROUTE_RADIUS", 5))
    for point in (current, centroid):
        row, col = _point_cell(float(point[0]), float(point[1]), bin_w, bin_h, rows, cols)
        for rr in range(max(0, row - radius), min(rows, row + radius + 1)):
            for cc in range(max(0, col - radius), min(cols, col + radius + 1)):
                candidate_flats.add(rr * cols + cc)

    scored: list[tuple[float, int]] = []
    dist_weight = _env_float("OURS_SOFT_SEARCH_ROUTE_DIST", 0.18)
    move_weight = _env_float("OURS_SOFT_SEARCH_ROUTE_MOVE", 0.04)
    span_weight = _env_float("OURS_SOFT_SEARCH_ROUTE_SPAN", 0.10)
    for flat in candidate_flats:
        rr = int(flat // cols)
        cc = int(flat % cols)
        target = np.asarray([(cc + 0.5) * bin_w, (rr + 0.5) * bin_h], dtype=np.float64)
        route_cost = 0.0
        for cells in other_cells_by_net:
            rows_all = [rr]
            cols_all = [cc]
            rows_all.extend(row for row, _ in cells)
            cols_all.extend(col for _, col in cells)
            r0 = max(0, min(rows_all))
            r1 = min(rows - 1, max(rows_all))
            c0 = max(0, min(cols_all))
            c1 = min(cols - 1, max(cols_all))
            area = max(1, (r1 - r0 + 1) * (c1 - c0 + 1))
            avg_pressure = (
                integral[r1 + 1, c1 + 1]
                - integral[r0, c1 + 1]
                - integral[r1 + 1, c0]
                + integral[r0, c0]
            ) / float(area)
            span = ((r1 - r0 + 1) / max(rows, 1)) + ((c1 - c0 + 1) / max(cols, 1))
            route_cost += float(avg_pressure) + pressure_scale * span_weight * float(span)
        route_cost /= max(1, len(other_cells_by_net))
        centroid_dist = float(np.linalg.norm(target - centroid)) / diag
        move_dist = float(np.linalg.norm(target - current)) / diag
        score = (
            route_cost
            + 0.20 * float(flat_pressure[flat])
            + pressure_scale * (dist_weight * centroid_dist + move_weight * move_dist)
        )
        scored.append((score, int(flat)))

    scored.sort(key=lambda item: item[0])
    targets: list[np.ndarray] = []
    seen: set[int] = set()
    for _, flat in scored:
        if flat in seen:
            continue
        seen.add(flat)
        rr = int(flat // cols)
        cc = int(flat % cols)
        targets.append(np.asarray([(cc + 0.5) * bin_w, (rr + 0.5) * bin_h], dtype=np.float64))
        if len(targets) >= limit:
            break
    return targets


def _point_cell(
    x: float,
    y: float,
    bin_w: float,
    bin_h: float,
    rows: int,
    cols: int,
) -> tuple[int, int]:
    col = min(cols - 1, max(0, int(float(x) / max(bin_w, 1e-12))))
    row = min(rows - 1, max(0, int(float(y) / max(bin_h, 1e-12))))
    return row, col


def _clamp_point(point: np.ndarray, benchmark: Benchmark, macro_idx: int) -> np.ndarray:
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64, copy=False)
    width = float(sizes[macro_idx, 0])
    height = float(sizes[macro_idx, 1])
    half_w = 0.5 * width
    half_h = 0.5 * height
    margin = max(0.0, _env_float("OURS_BOUNDARY_MARGIN", 1e-4))
    x_lo = min(half_w + margin, float(benchmark.canvas_width) - half_w - margin)
    y_lo = min(half_h + margin, float(benchmark.canvas_height) - half_h - margin)
    x_hi = float(benchmark.canvas_width) - half_w - margin
    y_hi = float(benchmark.canvas_height) - half_h - margin
    x = float(np.clip(float(point[0]), x_lo, x_hi))
    y = float(np.clip(float(point[1]), y_lo, y_hi))
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
