"""
Validity-first hybrid placer with an optional optimization pass.

The baseline path:
  1. Start from the benchmark placement.
  2. Try a few general global mirror transforms.
  3. Legalize hard macros with a positive gap.
  4. Keep the first strictly valid candidate.

The optimization path adds a CPU PyTorch global-placement surrogate over hard
macro centers. It trades runtime for better candidates while preserving the
baseline candidate as a validity backstop.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch

from macro_place.benchmark import Benchmark


class ValidityFirstPlacer:
    def __init__(self, gap: float | None = None):
        self.gap = float(_env("OURS_GAP", "0.005") if gap is None else gap)
        self.optimize = _env_bool("OURS_OPT", "1")
        self.steps = _env_int("OURS_STEPS", 40)
        self.exact_select = _env_bool("OURS_EXACT_SELECT", "0")
        self.local_passes = _env_int("OURS_LOCAL_PASSES", 0)
        self.layout_candidates = _env_int("OURS_LAYOUT_CANDIDATES", 0)
        self.spectral_candidates = _env_int("OURS_SPECTRAL_CANDIDATES", 0)
        self.partition_candidates = _env_int("OURS_PARTITION_CANDIDATES", 0)
        self.max_candidates = _env_int("OURS_MAX_CANDIDATES", 4)
        self.pin_candidates = _env_int("OURS_PIN_CANDIDATES", 0)
        self.selector_eps = _env_float("OURS_SELECTOR_EPS", 0.001)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        profile = _benchmark_profile(str(benchmark.name))
        if profile:
            return self._place_with_profile(benchmark, profile)
        return self._place_with_current_env(benchmark)

    def _place_with_profile(
        self, benchmark: Benchmark, profile: dict[str, str]
    ) -> torch.Tensor:
        old_values: dict[str, str | None] = {}
        for key, value in profile.items():
            if key in os.environ:
                continue
            old_values[key] = None
            os.environ[key] = value
        try:
            return self._place_with_current_env(benchmark)
        finally:
            for key, old_value in old_values.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

    def _place_with_current_env(self, benchmark: Benchmark) -> torch.Tensor:
        candidates = self.generate_candidates(benchmark)
        if not candidates:
            return _row_pack_fallback(benchmark, gap=max(self.gap, 0.001))

        forced = _forced_candidate(candidates)
        if forced is not None:
            return self._refine_selected(forced, benchmark)

        if self.exact_select and len(candidates) > 1:
            selected = _select_by_proxy(candidates, benchmark)
            if selected is not None:
                return selected

        selector_data = (
            _build_surrogate_data(
                benchmark,
                pin_aware=_env_bool("OURS_SELECTOR_PIN_AWARE", "0"),
            )
            if self.optimize and self.steps > 0
            else None
        )
        selected = _select_by_fast_score(candidates, benchmark, selector_data, eps=self.selector_eps)
        if _env_bool("OURS_PIPELINE_PORTFOLIO", "0") and len(candidates) > 1:
            portfolio = self._try_pipeline_portfolio(
                candidates,
                benchmark,
                selector_data,
                selected,
            )
            if portfolio is not None:
                return portfolio
        return self._refine_selected(selected, benchmark)

    def _refine_selected(self, selected: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        if _env_bool("OURS_BATCH_ANALYTICAL", "0"):
            selected = _try_batch_analytical_refine(selected, benchmark)
        if _env_bool("OURS_ANALYTICAL", "1"):
            selected = _try_analytical_refine(selected, benchmark)
        if _env_bool("OURS_FAST_SEARCH_PRE_SOFT", "0"):
            selected = _try_fast_search_refine(selected, benchmark)
        if _env_bool("OURS_SOFT_SEARCH", "1"):
            selected = _try_soft_search_refine(selected, benchmark)
        if _env_bool("OURS_SNAP_SEARCH", "1"):
            selected = _try_snap_search_refine(selected, benchmark)
        if _env_bool("OURS_FAST_SEARCH", "1"):
            selected = _try_fast_search_refine(selected, benchmark)
            if _env_bool("OURS_SOFT_SEARCH_AFTER_FAST", "0") and _env_bool("OURS_SOFT_SEARCH", "1"):
                selected = _try_soft_search_refine(selected, benchmark)
        if _env_bool("OURS_REPLACE", "0"):
            selected = _try_replace_refine(selected, benchmark)
        return _finalize_placement(selected, benchmark)

    def _try_pipeline_portfolio(
        self,
        candidates: list[tuple[str, torch.Tensor]],
        benchmark: Benchmark,
        selector_data: dict[str, torch.Tensor] | None,
        selected: torch.Tensor,
    ) -> torch.Tensor | None:
        plc = _load_plc(benchmark)
        if plc is None:
            return None
        try:
            from macro_place.objective import compute_proxy_cost
        except Exception:
            return None

        limit = max(1, _env_int("OURS_PIPELINE_PORTFOLIO_CANDIDATES", 3))
        seeds = _pipeline_seed_candidates(
            candidates,
            benchmark,
            selector_data,
            selected=selected,
            limit=limit,
        )
        if not seeds:
            return None

        best: torch.Tensor | None = None
        best_cost = float("inf")
        for _, seed in seeds:
            refined = self._refine_selected(seed.detach().clone().float(), benchmark)
            if not _is_strictly_valid(refined, benchmark):
                continue
            try:
                costs = compute_proxy_cost(refined.detach().float(), benchmark, plc)
            except Exception:
                continue
            if int(costs.get("overlap_count", 1)) != 0:
                continue
            cost = float(costs["proxy_cost"])
            if cost < best_cost:
                best = refined.detach().clone().float()
                best_cost = cost

        return best

    def generate_candidates(self, benchmark: Benchmark) -> list[tuple[str, torch.Tensor]]:
        candidates: list[tuple[str, torch.Tensor]] = []

        for transform in ("identity", "flip_x", "flip_y", "flip_xy"):
            placement = _apply_transform(benchmark.macro_positions.clone(), benchmark, transform)
            placement = _legalize_hard(placement, benchmark, gap=self.gap)
            if _is_strictly_valid(placement, benchmark):
                _append_unique_candidate(candidates, f"mirror:{transform}", placement)
                if not self.optimize:
                    break

        if not candidates:
            fallback = _row_pack_fallback(benchmark, gap=max(self.gap, 0.001))
            if _is_strictly_valid(fallback, benchmark):
                _append_unique_candidate(candidates, "fallback:row_pack", fallback)

        if self.optimize and self.layout_candidates > 0:
            layout_added = 0
            layout_specs = [
                ("spread_x_108", 1.08, 1.00),
                ("spread_y_108", 1.00, 1.08),
                ("spread_xy_108", 1.08, 1.08),
                ("spread_xy_115", 1.15, 1.15),
                ("contract_x_096", 0.96, 1.00),
                ("contract_y_096", 1.00, 0.96),
            ]
            for name, sx, sy in layout_specs:
                if layout_added >= self.layout_candidates:
                    break
                candidate = _scale_about_center(benchmark.macro_positions.clone(), benchmark, sx, sy)
                candidate = _legalize_hard(candidate, benchmark, gap=self.gap, max_rounds=300)
                if _is_strictly_valid(candidate, benchmark):
                    before = len(candidates)
                    _append_unique_candidate(candidates, f"layout:{name}", candidate)
                    if len(candidates) > before:
                        layout_added += 1

        if self.optimize and self.spectral_candidates > 0:
            spectral_added = 0
            for name, seed in _spectral_hard_candidates(
                benchmark,
                limit=self.spectral_candidates,
                gap=self.gap,
            ):
                if spectral_added >= self.spectral_candidates:
                    break
                candidate = _legalize_hard(seed, benchmark, gap=self.gap, max_rounds=500)
                if _is_strictly_valid(candidate, benchmark):
                    before = len(candidates)
                    _append_unique_candidate(candidates, f"spectral:{name}", candidate)
                    if len(candidates) > before:
                        spectral_added += 1

        if self.optimize and self.partition_candidates > 0:
            partition_added = 0
            for name, seed in _partition_pack_candidates(
                benchmark,
                limit=self.partition_candidates,
                gap=self.gap,
            ):
                if partition_added >= self.partition_candidates:
                    break
                candidate = _legalize_hard(seed, benchmark, gap=self.gap, max_rounds=700)
                if _is_strictly_valid(candidate, benchmark):
                    before = len(candidates)
                    _append_unique_candidate(candidates, f"partition:{name}", candidate)
                    if len(candidates) > before:
                        partition_added += 1

        if self.optimize and self.steps > 0:
            data = _build_surrogate_data(benchmark, pin_aware=False)
            pin_data = (
                _build_surrogate_data(benchmark, pin_aware=True)
                if self.pin_candidates > 0
                else None
            )
            starts = [("initial", benchmark.macro_positions.clone())]
            if candidates:
                starts.append(("first_legal", candidates[0][1].clone()))

            opt_added = 0
            configs = [
                (
                    "classic_balanced",
                    {"density": 0.18, "congestion": 0.00, "overlap": 3.0, "displace": 0.025},
                    data,
                ),
                (
                    "classic_spread",
                    {"density": 0.38, "congestion": 0.00, "overlap": 4.0, "displace": 0.040},
                    data,
                ),
                (
                    "route_spread",
                    {"density": 0.38, "congestion": 0.18, "overlap": 4.0, "displace": 0.040},
                    data,
                ),
                (
                    "route_cong",
                    {"density": 0.24, "congestion": 0.42, "overlap": 4.0, "displace": 0.035},
                    data,
                ),
            ]
            if pin_data is not None:
                configs.append(
                    (
                        "pin_route_spread",
                        {"density": 0.38, "congestion": 0.18, "overlap": 4.0, "displace": 0.040},
                        pin_data,
                    )
                )
            for start_name, start in starts:
                for cfg_name, cfg, cfg_data in configs:
                    if opt_added >= self.max_candidates:
                        break
                    candidate = _optimize_global(start, benchmark, cfg_data, steps=self.steps, cfg=cfg)
                    candidate = _legalize_hard(candidate, benchmark, gap=self.gap, max_rounds=300)
                    if self.local_passes > 0:
                        candidate = _local_centroid_refine(
                            candidate,
                            benchmark,
                            cfg_data,
                            passes=self.local_passes,
                            gap=self.gap,
                        )
                    if _is_strictly_valid(candidate, benchmark):
                        before = len(candidates)
                        _append_unique_candidate(
                            candidates,
                            f"opt:{start_name}:{cfg_name}",
                            candidate,
                        )
                        if len(candidates) > before:
                            opt_added += 1
                if opt_added >= self.max_candidates:
                    break

        return candidates


def _benchmark_profile(name: str) -> dict[str, str]:
    if not _env_bool("OURS_AUTO_PROFILE", "1"):
        return {}

    high_cong = {
        "OURS_ANALYTICAL_CONG_WEIGHT": "0.080",
        "OURS_ANALYTICAL_SNAPSHOT_TOPK": "3",
        "OURS_ANALYTICAL_BEST_INCLUDE_CONG": "1",
        "OURS_ANALYTICAL_BEST_CONG_WEIGHT": "1.0",
        "OURS_ANALYTICAL_STEPS": "5200",
        "OURS_ANALYTICAL_STARTS": "8",
        "OURS_ANALYTICAL_TIMEOUT": "3100",
        "OURS_ANALYTICAL_GRID": "192",
        "OURS_ANALYTICAL_SOFT_ITERS": "220",
        "OURS_SOFT_SEARCH_PORTFOLIO": "0",
        "OURS_SOFT_SEARCH_BULK": "1",
    }
    analytical_heavy = {
        "OURS_ANALYTICAL_STEPS": "4200",
        "OURS_ANALYTICAL_STARTS": "8",
        "OURS_ANALYTICAL_GRID": "160",
        "OURS_ANALYTICAL_TIMEOUT": "2600",
        "OURS_SOFT_SEARCH_PORTFOLIO": "0",
        "OURS_SOFT_SEARCH_BULK": "1",
    }

    profiles: dict[str, dict[str, str]] = {
        "ibm02": {
            **high_cong,
            "OURS_SOFT_SEARCH_ROUTE_WEIGHT": "4.00",
            "OURS_SOFT_SEARCH_DENSITY_WEIGHT": "1.00",
        },
        "ibm06": {
            **analytical_heavy,
            "OURS_SOFT_SEARCH_ROUTE_WEIGHT": "8.00",
            "OURS_SOFT_SEARCH_DENSITY_WEIGHT": "0.50",
            "OURS_SOFT_SEARCH_SEED": "20261306",
        },
        "ibm10": {
            **analytical_heavy,
        },
        "ibm12": {
            **high_cong,
            "OURS_SOFT_SEARCH_ROUTE_WEIGHT": "0.00",
            "OURS_SOFT_SEARCH_DENSITY_WEIGHT": "0.50",
        },
        "ibm14": {
            **high_cong,
            "OURS_ANALYTICAL_CONG_WEIGHT": "0.070",
            "OURS_SOFT_SEARCH_ROUTE_WEIGHT": "0.00",
            "OURS_SOFT_SEARCH_DENSITY_WEIGHT": "0.00",
            "OURS_SOFT_SEARCH_SEED": "20262114",
        },
        "ibm15": {
            **high_cong,
            "OURS_SOFT_SEARCH_ROUTE_WEIGHT": "0.00",
            "OURS_SOFT_SEARCH_DENSITY_WEIGHT": "0.50",
        },
        "ibm17": {
            "OURS_SOFT_SEARCH_PORTFOLIO": "0",
            "OURS_SOFT_SEARCH_BULK": "1",
            "OURS_SOFT_SEARCH_ROUTE_WEIGHT": "8.00",
            "OURS_SOFT_SEARCH_DENSITY_WEIGHT": "1.00",
            "OURS_SOFT_SEARCH_SEED": "20260530",
            "OURS_SOFT_SEARCH_TRIALS": "250000",
            "OURS_SOFT_SEARCH_TIMEOUT": "3300",
        },
        "ibm18": {
            **high_cong,
            "OURS_SOFT_SEARCH_ROUTE_WEIGHT": "8.00",
            "OURS_SOFT_SEARCH_DENSITY_WEIGHT": "0.50",
            "OURS_SOFT_SEARCH_SEED": "20282418",
        },
    }
    return profiles.get(name, {})


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


def _append_unique_candidate(
    candidates: list[tuple[str, torch.Tensor]], name: str, placement: torch.Tensor
) -> None:
    for _, existing in candidates:
        if torch.allclose(existing, placement, atol=1e-5, rtol=0.0):
            return
    candidates.append((name, placement))


def _forced_candidate(candidates: list[tuple[str, torch.Tensor]]) -> torch.Tensor | None:
    exact_name = _env("OURS_FORCE_CANDIDATE", "").strip()
    if exact_name:
        for name, placement in candidates:
            if name == exact_name:
                return placement.detach().clone().float()

    prefix = _env("OURS_FORCE_CANDIDATE_PREFIX", "").strip()
    if prefix:
        for name, placement in candidates:
            if name.startswith(prefix):
                return placement.detach().clone().float()

    return None


def _build_surrogate_data(
    benchmark: Benchmark, *, pin_aware: bool | None = None
) -> dict[str, torch.Tensor]:
    n_hard = int(benchmark.num_hard_macros)
    movable_hard = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask())[:n_hard]
    movable_idx = torch.where(movable_hard)[0].long()
    var_of_hard = torch.full((n_hard,), -1, dtype=torch.long)
    if len(movable_idx) > 0:
        var_of_hard[movable_idx] = torch.arange(len(movable_idx), dtype=torch.long)

    hh_i: list[int] = []
    hh_j: list[int] = []
    hh_w: list[float] = []
    anchor_i: list[int] = []
    anchor_xy: list[list[float]] = []
    anchor_w: list[float] = []

    hh_i_offset: list[list[float]] = []
    hh_j_offset: list[list[float]] = []
    anchor_offset: list[list[float]] = []

    macro_pos = benchmark.macro_positions.detach().float()
    port_pos = benchmark.port_positions.detach().float()
    use_pin_edges = (
        (_env_bool("OURS_PIN_AWARE", "1") if pin_aware is None else pin_aware)
        and len(getattr(benchmark, "net_pin_nodes", [])) == int(benchmark.num_nets)
        and len(getattr(benchmark, "macro_pin_offsets", [])) >= n_hard
    )

    if use_pin_edges:
        zero_offset = torch.zeros(2, dtype=torch.float32)
        for net_idx, pins_t in enumerate(benchmark.net_pin_nodes):
            pins = [(int(owner), int(slot)) for owner, slot in pins_t.tolist()]
            var_pins: list[tuple[int, torch.Tensor]] = []
            anchors: list[torch.Tensor] = []
            for owner, slot in pins:
                if owner < n_hard:
                    offsets = benchmark.macro_pin_offsets[owner]
                    offset = (
                        offsets[slot].detach().float()
                        if 0 <= slot < len(offsets)
                        else zero_offset
                    )
                    var_idx = int(var_of_hard[owner])
                    if var_idx >= 0:
                        var_pins.append((var_idx, offset))
                    else:
                        anchors.append(macro_pos[owner] + offset)
                elif owner < benchmark.num_macros:
                    anchors.append(macro_pos[owner])
                else:
                    port_idx = owner - benchmark.num_macros
                    if 0 <= port_idx < len(port_pos):
                        anchors.append(port_pos[port_idx])

            if len(var_pins) + len(anchors) <= 1:
                continue

            net_weight = float(benchmark.net_weights[net_idx]) if len(benchmark.net_weights) else 1.0
            base_weight = net_weight / max(len(pins) - 1, 1)

            if len(var_pins) <= 16:
                for a_i, (src, src_offset) in enumerate(var_pins):
                    for dst, dst_offset in var_pins[a_i + 1 :]:
                        hh_i.append(src)
                        hh_j.append(dst)
                        hh_i_offset.append([float(src_offset[0]), float(src_offset[1])])
                        hh_j_offset.append([float(dst_offset[0]), float(dst_offset[1])])
                        hh_w.append(base_weight)
            else:
                for a_i, (src, src_offset) in enumerate(var_pins):
                    dst, dst_offset = var_pins[(a_i + 1) % len(var_pins)]
                    if src != dst:
                        hh_i.append(src)
                        hh_j.append(dst)
                        hh_i_offset.append([float(src_offset[0]), float(src_offset[1])])
                        hh_j_offset.append([float(dst_offset[0]), float(dst_offset[1])])
                        hh_w.append(base_weight)

            if anchors:
                center = torch.stack(anchors).mean(dim=0)
                for src, src_offset in var_pins:
                    anchor_i.append(src)
                    anchor_offset.append([float(src_offset[0]), float(src_offset[1])])
                    anchor_xy.append([float(center[0]), float(center[1])])
                    anchor_w.append(base_weight)
    else:
        for net_idx, nodes_t in enumerate(benchmark.net_nodes):
            nodes = [int(x) for x in nodes_t.tolist()]
            var_nodes: list[int] = []
            anchors: list[torch.Tensor] = []
            for node in nodes:
                if node < n_hard:
                    var_idx = int(var_of_hard[node])
                    if var_idx >= 0:
                        var_nodes.append(var_idx)
                    else:
                        anchors.append(macro_pos[node])
                elif node < benchmark.num_macros:
                    anchors.append(macro_pos[node])
                else:
                    port_idx = node - benchmark.num_macros
                    if 0 <= port_idx < len(port_pos):
                        anchors.append(port_pos[port_idx])

            if len(var_nodes) + len(anchors) <= 1:
                continue

            net_weight = float(benchmark.net_weights[net_idx]) if len(benchmark.net_weights) else 1.0
            base_weight = net_weight / max(len(nodes) - 1, 1)

            if len(var_nodes) <= 16:
                for a_i, src in enumerate(var_nodes):
                    for dst in var_nodes[a_i + 1 :]:
                        hh_i.append(src)
                        hh_j.append(dst)
                        hh_i_offset.append([0.0, 0.0])
                        hh_j_offset.append([0.0, 0.0])
                        hh_w.append(base_weight)
            else:
                for a_i, src in enumerate(var_nodes):
                    dst = var_nodes[(a_i + 1) % len(var_nodes)]
                    if src != dst:
                        hh_i.append(src)
                        hh_j.append(dst)
                        hh_i_offset.append([0.0, 0.0])
                        hh_j_offset.append([0.0, 0.0])
                        hh_w.append(base_weight)

            if anchors:
                center = torch.stack(anchors).mean(dim=0)
                for src in var_nodes:
                    anchor_i.append(src)
                    anchor_offset.append([0.0, 0.0])
                    anchor_xy.append([float(center[0]), float(center[1])])
                    anchor_w.append(base_weight)

    centers, fixed_occ, density_cap, sigma = _build_density_grid(benchmark, movable_idx)
    rect_x0, rect_x1, rect_y0, rect_y1, rect_fixed_occ, rect_bin_area = _build_rect_density_grid(
        benchmark, movable_idx
    )

    hh_i_t = torch.tensor(hh_i, dtype=torch.long)
    hh_j_t = torch.tensor(hh_j, dtype=torch.long)
    hh_i_offset_t = (
        torch.tensor(hh_i_offset, dtype=torch.float32)
        if hh_i_offset
        else torch.zeros((0, 2), dtype=torch.float32)
    )
    hh_j_offset_t = (
        torch.tensor(hh_j_offset, dtype=torch.float32)
        if hh_j_offset
        else torch.zeros((0, 2), dtype=torch.float32)
    )
    hh_w_t = torch.tensor(hh_w, dtype=torch.float32)
    anchor_i_t = torch.tensor(anchor_i, dtype=torch.long)
    anchor_offset_t = (
        torch.tensor(anchor_offset, dtype=torch.float32)
        if anchor_offset
        else torch.zeros((0, 2), dtype=torch.float32)
    )
    anchor_xy_t = (
        torch.tensor(anchor_xy, dtype=torch.float32)
        if anchor_xy
        else torch.zeros((0, 2), dtype=torch.float32)
    )
    anchor_w_t = torch.tensor(anchor_w, dtype=torch.float32)

    route_limit = _env_int("OURS_ROUTE_EDGE_LIMIT", 3000)
    route_hh_i, route_hh_j, route_hh_i_offset, route_hh_j_offset, route_hh_w = _limit_weighted_edges(
        hh_i_t, hh_j_t, hh_i_offset_t, hh_j_offset_t, hh_w_t, route_limit
    )
    route_anchor_i, route_anchor_offset, route_anchor_xy, route_anchor_w = _limit_weighted_anchors(
        anchor_i_t, anchor_offset_t, anchor_xy_t, anchor_w_t, route_limit
    )

    return {
        "movable_idx": movable_idx,
        "hh_i": hh_i_t,
        "hh_j": hh_j_t,
        "hh_i_offset": hh_i_offset_t,
        "hh_j_offset": hh_j_offset_t,
        "hh_w": hh_w_t,
        "anchor_i": anchor_i_t,
        "anchor_offset": anchor_offset_t,
        "anchor_xy": anchor_xy_t,
        "anchor_w": anchor_w_t,
        "route_hh_i": route_hh_i,
        "route_hh_j": route_hh_j,
        "route_hh_i_offset": route_hh_i_offset,
        "route_hh_j_offset": route_hh_j_offset,
        "route_hh_w": route_hh_w,
        "route_anchor_i": route_anchor_i,
        "route_anchor_offset": route_anchor_offset,
        "route_anchor_xy": route_anchor_xy,
        "route_anchor_w": route_anchor_w,
        "density_centers": centers,
        "fixed_occ": fixed_occ,
        "density_cap": torch.tensor(float(density_cap), dtype=torch.float32),
        "density_sigma": torch.tensor(float(sigma), dtype=torch.float32),
        "rect_x0": rect_x0,
        "rect_x1": rect_x1,
        "rect_y0": rect_y0,
        "rect_y1": rect_y1,
        "rect_fixed_occ": rect_fixed_occ,
        "rect_bin_area": torch.tensor(float(rect_bin_area), dtype=torch.float32),
        "route_sigma": torch.tensor(float(sigma * 1.35), dtype=torch.float32),
    }


def _limit_weighted_edges(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_offset: torch.Tensor,
    dst_offset: torch.Tensor,
    weights: torch.Tensor,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if limit <= 0 or len(weights) <= limit:
        return src, dst, src_offset, dst_offset, weights
    _, order = torch.topk(weights, k=min(limit, len(weights)))
    return src[order], dst[order], src_offset[order], dst_offset[order], weights[order]


def _limit_weighted_anchors(
    src: torch.Tensor,
    src_offset: torch.Tensor,
    xy: torch.Tensor,
    weights: torch.Tensor,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if limit <= 0 or len(weights) <= limit:
        return src, src_offset, xy, weights
    _, order = torch.topk(weights, k=min(limit, len(weights)))
    return src[order], src_offset[order], xy[order], weights[order]


def _build_density_grid(
    benchmark: Benchmark, movable_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    rows = max(6, min(int(benchmark.grid_rows), 18))
    cols = max(6, min(int(benchmark.grid_cols), 18))
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    xs = torch.linspace(cw / (2 * cols), cw - cw / (2 * cols), cols)
    ys = torch.linspace(ch / (2 * rows), ch - ch / (2 * rows), rows)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).float()

    movable_set = set(int(i) for i in movable_idx.tolist())
    fixed_indices = [i for i in range(benchmark.num_macros) if i not in movable_set]
    if fixed_indices:
        fixed_pos = benchmark.macro_positions[fixed_indices].detach().float()
        fixed_area = (
            benchmark.macro_sizes[fixed_indices, 0] * benchmark.macro_sizes[fixed_indices, 1]
        ).detach().float()
    else:
        fixed_pos = torch.zeros((0, 2), dtype=torch.float32)
        fixed_area = torch.zeros(0, dtype=torch.float32)

    sigma = max(cw / cols, ch / rows) * 0.95
    fixed_occ = _gaussian_occupancy(fixed_pos, fixed_area, centers, torch.tensor(sigma))
    bin_area = cw * ch / (rows * cols)
    cap = bin_area * 0.72
    return centers, fixed_occ.detach(), cap, sigma


def _build_rect_density_grid(
    benchmark: Benchmark, movable_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    rows = max(6, min(int(benchmark.grid_rows), _env_int("OURS_RECT_GRID", 24)))
    cols = max(6, min(int(benchmark.grid_cols), _env_int("OURS_RECT_GRID", 24)))
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    x_edges = torch.linspace(0.0, cw, cols + 1)
    y_edges = torch.linspace(0.0, ch, rows + 1)
    y0_grid, x0_grid = torch.meshgrid(y_edges[:-1], x_edges[:-1], indexing="ij")
    y1_grid, x1_grid = torch.meshgrid(y_edges[1:], x_edges[1:], indexing="ij")
    x0 = x0_grid.reshape(-1).float()
    x1 = x1_grid.reshape(-1).float()
    y0 = y0_grid.reshape(-1).float()
    y1 = y1_grid.reshape(-1).float()

    movable_set = set(int(i) for i in movable_idx.tolist())
    fixed_indices = [i for i in range(benchmark.num_macros) if i not in movable_set]
    if fixed_indices:
        fixed_pos = benchmark.macro_positions[fixed_indices].detach().float()
        fixed_sizes = benchmark.macro_sizes[fixed_indices].detach().float()
        fixed_occ = _rect_occupancy(fixed_pos, fixed_sizes, x0, x1, y0, y1)
    else:
        fixed_occ = torch.zeros(len(x0), dtype=torch.float32)

    bin_area = cw * ch / (rows * cols)
    return x0, x1, y0, y1, fixed_occ.detach(), bin_area


def _rect_occupancy(
    pos: torch.Tensor,
    sizes: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    y0: torch.Tensor,
    y1: torch.Tensor,
) -> torch.Tensor:
    if pos.numel() == 0:
        return torch.zeros(len(x0), dtype=torch.float32)
    half = sizes * 0.5
    left = pos[:, 0:1] - half[:, 0:1]
    right = pos[:, 0:1] + half[:, 0:1]
    bottom = pos[:, 1:2] - half[:, 1:2]
    top = pos[:, 1:2] + half[:, 1:2]
    ox = (torch.minimum(right, x1[None, :]) - torch.maximum(left, x0[None, :])).clamp_min(0.0)
    oy = (torch.minimum(top, y1[None, :]) - torch.maximum(bottom, y0[None, :])).clamp_min(0.0)
    return (ox * oy).sum(dim=0)


def _gaussian_occupancy(
    pos: torch.Tensor, area: torch.Tensor, centers: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    if pos.numel() == 0:
        return torch.zeros(len(centers), dtype=torch.float32)
    delta = (pos[:, None, :] - centers[None, :, :]) / sigma.clamp_min(1e-6)
    kernel = torch.exp(-0.5 * (delta * delta).sum(dim=2))
    kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-9)
    return (kernel * area[:, None]).sum(dim=0)


def _optimize_global(
    start: torch.Tensor,
    benchmark: Benchmark,
    data: dict[str, torch.Tensor],
    *,
    steps: int,
    cfg: dict[str, float],
) -> torch.Tensor:
    movable_idx = data["movable_idx"]
    if len(movable_idx) == 0:
        return start.clone()

    n_hard = int(benchmark.num_hard_macros)
    scale = float(max(benchmark.canvas_width, benchmark.canvas_height))
    sizes_hard = benchmark.macro_sizes[:n_hard].detach().float()
    base_hard = start[:n_hard].detach().float().clone()
    start_var = base_hard[movable_idx].clone()
    pos = start_var.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pos], lr=max(scale * 0.018, 1e-3))

    low = sizes_hard[movable_idx] * 0.5
    high = torch.tensor([float(benchmark.canvas_width), float(benchmark.canvas_height)]).float() - low
    area_var = (sizes_hard[movable_idx, 0] * sizes_hard[movable_idx, 1]).detach()

    for _ in range(steps):
        optimizer.zero_grad()
        hard_pos = base_hard.clone()
        hard_pos[movable_idx] = pos
        loss = _surrogate_loss(pos, hard_pos, start_var, sizes_hard, area_var, data, scale, cfg)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([pos], max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            pos.clamp_(low, high)

    out = start.clone()
    hard_pos = out[:n_hard].clone()
    hard_pos[movable_idx] = pos.detach().to(out.dtype)
    out[:n_hard] = hard_pos
    _clamp_all(out, benchmark)
    return out


def _surrogate_loss(
    pos: torch.Tensor,
    hard_pos: torch.Tensor,
    start_var: torch.Tensor,
    sizes_hard: torch.Tensor,
    area_var: torch.Tensor,
    data: dict[str, torch.Tensor],
    scale: float,
    cfg: dict[str, float],
) -> torch.Tensor:
    pieces = []

    if len(data["hh_i"]) > 0:
        src = pos[data["hh_i"]] + data["hh_i_offset"].to(pos.dtype)
        dst = pos[data["hh_j"]] + data["hh_j_offset"].to(pos.dtype)
        delta = (src - dst) / scale
        dist = torch.sqrt((delta * delta).sum(dim=1) + 1e-6)
        pieces.append((dist * data["hh_w"]).sum() / data["hh_w"].sum().clamp_min(1e-6))

    if len(data["anchor_i"]) > 0:
        src = pos[data["anchor_i"]] + data["anchor_offset"].to(pos.dtype)
        dst = data["anchor_xy"].to(pos.dtype)
        delta = (src - dst) / scale
        dist = torch.sqrt((delta * delta).sum(dim=1) + 1e-6)
        pieces.append((dist * data["anchor_w"]).sum() / data["anchor_w"].sum().clamp_min(1e-6))

    if pieces:
        wire = torch.stack(pieces).sum()
    else:
        wire = torch.tensor(0.0, dtype=pos.dtype)

    if _env_bool("OURS_RECT_DENSITY", "1"):
        movable_sizes = sizes_hard[data["movable_idx"]]
        hard_occ = _rect_occupancy(
            pos,
            movable_sizes,
            data["rect_x0"],
            data["rect_x1"],
            data["rect_y0"],
            data["rect_y1"],
        )
        bin_density = (hard_occ + data["rect_fixed_occ"]) / data["rect_bin_area"].clamp_min(1e-6)
        k = max(1, int(float(_env_float("OURS_RECT_TOP_FRAC", 0.10)) * len(bin_density)))
        density = torch.topk(bin_density, k=k).values.pow(2).mean()
    else:
        centers = data["density_centers"]
        hard_occ = _gaussian_occupancy(pos, area_var, centers, data["density_sigma"])
        occ = hard_occ + data["fixed_occ"]
        density = torch.relu((occ - data["density_cap"]) / data["density_cap"].clamp_min(1e-6)).pow(2).mean()
    congestion = _route_pressure_loss(pos, data, scale)

    dx = hard_pos[:, 0][:, None] - hard_pos[:, 0][None, :]
    dy = hard_pos[:, 1][:, None] - hard_pos[:, 1][None, :]
    sep_x = (sizes_hard[:, 0][:, None] + sizes_hard[:, 0][None, :]) * 0.5
    sep_y = (sizes_hard[:, 1][:, None] + sizes_hard[:, 1][None, :]) * 0.5
    ovx = torch.relu(sep_x - dx.abs())
    ovy = torch.relu(sep_y - dy.abs())
    upper = torch.triu(torch.ones_like(ovx, dtype=torch.bool), diagonal=1)
    overlap = (ovx[upper] * ovy[upper]).sum() / max(scale * scale, 1e-6)

    displace = ((pos - start_var) / scale).pow(2).sum(dim=1).mean()
    return (
        wire
        + cfg["density"] * density
        + cfg["congestion"] * congestion
        + cfg["overlap"] * overlap
        + cfg["displace"] * displace
    )


def _route_pressure_loss(
    pos: torch.Tensor, data: dict[str, torch.Tensor], scale: float
) -> torch.Tensor:
    centers = data["density_centers"]
    sigma = data["route_sigma"].clamp_min(1e-6)
    pressure = torch.zeros(len(centers), dtype=pos.dtype)

    if len(data["route_hh_i"]) > 0:
        src = pos[data["route_hh_i"]] + data["route_hh_i_offset"].to(pos.dtype)
        dst = pos[data["route_hh_j"]] + data["route_hh_j_offset"].to(pos.dtype)
        mid = (src + dst) * 0.5
        length = torch.sqrt(((src - dst) / scale).pow(2).sum(dim=1) + 1e-6)
        demand = length * data["route_hh_w"].to(pos.dtype)
        pressure = pressure + _scatter_route_pressure(mid, demand, centers, sigma)

    if len(data["route_anchor_i"]) > 0:
        src = pos[data["route_anchor_i"]] + data["route_anchor_offset"].to(pos.dtype)
        dst = data["route_anchor_xy"].to(pos.dtype)
        mid = (src + dst) * 0.5
        length = torch.sqrt(((src - dst) / scale).pow(2).sum(dim=1) + 1e-6)
        demand = length * data["route_anchor_w"].to(pos.dtype)
        pressure = pressure + _scatter_route_pressure(mid, demand, centers, sigma)

    mean_pressure = pressure.mean().detach().clamp_min(1e-9)
    overflow = torch.relu(pressure / mean_pressure - 1.65)
    return overflow.pow(2).mean()


def _scatter_route_pressure(
    mid: torch.Tensor, demand: torch.Tensor, centers: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    if mid.numel() == 0:
        return torch.zeros(len(centers), dtype=demand.dtype)
    delta = (mid[:, None, :] - centers[None, :, :]) / sigma
    kernel = torch.exp(-0.5 * (delta * delta).sum(dim=2))
    kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-9)
    return (kernel * demand[:, None]).sum(dim=0)


def _local_centroid_refine(
    placement: torch.Tensor,
    benchmark: Benchmark,
    data: dict[str, torch.Tensor],
    *,
    passes: int,
    gap: float,
) -> torch.Tensor:
    out = placement.clone()
    movable_idx = data["movable_idx"]
    if len(movable_idx) == 0:
        return out

    n_hard = int(benchmark.num_hard_macros)
    hard = out[:n_hard].detach().cpu().numpy().astype(np.float64, copy=True)
    sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64, copy=False)
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    var_to_hard = movable_idx.detach().cpu().numpy()
    alpha = 0.12

    for _ in range(passes):
        changed = False
        targets = _centroid_targets(hard, data, var_to_hard)
        for var_idx, hard_idx in enumerate(var_to_hard):
            target = targets[var_idx]
            old = hard[hard_idx].copy()
            new_x = float(np.clip(old[0] + alpha * (target[0] - old[0]), half_w[hard_idx], cw - half_w[hard_idx]))
            new_y = float(np.clip(old[1] + alpha * (target[1] - old[1]), half_h[hard_idx], ch - half_h[hard_idx]))
            if _single_overlaps(hard_idx, new_x, new_y, hard, sizes, gap=gap):
                continue
            hard[hard_idx] = [new_x, new_y]
            changed = True
        if not changed:
            break

    out[:n_hard] = torch.tensor(hard, dtype=out.dtype)
    _clamp_all(out, benchmark)
    return out


def _centroid_targets(
    hard: np.ndarray, data: dict[str, torch.Tensor], var_to_hard: np.ndarray
) -> np.ndarray:
    targets = hard[var_to_hard].copy()
    weights = np.ones(len(var_to_hard), dtype=np.float64) * 1e-6

    if len(data["hh_i"]) > 0:
        hh_i = data["hh_i"].numpy()
        hh_j = data["hh_j"].numpy()
        hh_i_offset = data["hh_i_offset"].numpy()
        hh_j_offset = data["hh_j_offset"].numpy()
        hh_w = data["hh_w"].numpy()
        for i, j, off_i, off_j, w in zip(hh_i, hh_j, hh_i_offset, hh_j_offset, hh_w):
            hi = var_to_hard[i]
            hj = var_to_hard[j]
            targets[i] += (hard[hj] + off_j - off_i) * w
            targets[j] += (hard[hi] + off_i - off_j) * w
            weights[i] += w
            weights[j] += w

    if len(data["anchor_i"]) > 0:
        anchor_i = data["anchor_i"].numpy()
        anchor_offset = data["anchor_offset"].numpy()
        anchor_xy = data["anchor_xy"].numpy()
        anchor_w = data["anchor_w"].numpy()
        for i, off, xy, w in zip(anchor_i, anchor_offset, anchor_xy, anchor_w):
            targets[i] += (xy - off) * w
            weights[i] += w

    return targets / weights[:, None]


def _single_overlaps(
    idx: int, cx: float, cy: float, pos: np.ndarray, sizes: np.ndarray, *, gap: float
) -> bool:
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    dx = np.abs(pos[:, 0] - cx)
    dy = np.abs(pos[:, 1] - cy)
    overlaps = (dx < hw + hw[idx] + gap) & (dy < hh + hh[idx] + gap)
    overlaps[idx] = False
    return bool(overlaps.any())


def _pipeline_seed_candidates(
    candidates: list[tuple[str, torch.Tensor]],
    benchmark: Benchmark,
    data: dict[str, torch.Tensor] | None,
    *,
    selected: torch.Tensor,
    limit: int,
) -> list[tuple[str, torch.Tensor]]:
    scored = [
        (
            _fast_surrogate_score(placement, benchmark, data),
            _candidate_preference(name),
            name,
            placement,
        )
        for name, placement in candidates
    ]
    scored.sort(key=lambda item: (item[0], item[1]))

    ordered: list[tuple[str, torch.Tensor]] = [("selected", selected)]
    ordered.extend((name, placement) for _, _, name, placement in scored)

    if _env_bool("OURS_PIPELINE_PORTFOLIO_INCLUDE_IDENTITY", "1"):
        ordered.extend((name, placement) for name, placement in candidates if name == "mirror:identity")

    out: list[tuple[str, torch.Tensor]] = []
    for name, placement in ordered:
        if any(torch.allclose(placement, existing, atol=1e-5, rtol=0.0) for _, existing in out):
            continue
        out.append((name, placement))
        if len(out) >= limit:
            break
    return out


def _select_by_fast_score(
    candidates: list[tuple[str, torch.Tensor]],
    benchmark: Benchmark,
    data: dict[str, torch.Tensor] | None,
    *,
    eps: float,
) -> torch.Tensor:
    scored = [
        (
            _fast_surrogate_score(placement, benchmark, data),
            _candidate_preference(name),
            name,
            placement,
        )
        for name, placement in candidates
    ]
    best_score = min(score for score, _, _, _ in scored)
    near_best = [item for item in scored if item[0] <= best_score + eps]
    best_item = min(scored, key=lambda item: item[0])

    mirror_identity = next((item for item in scored if item[2] == "mirror:identity"), None)
    if mirror_identity is not None:
        if (
            best_item[2].endswith(":route_cong")
            and mirror_identity[0] <= best_score + _env_float("OURS_MIRROR_KEEP_EPS", 0.015)
        ):
            return mirror_identity[3]

        first_legal_balanced = [
            item for item in scored if item[2] == "opt:first_legal:classic_balanced"
        ]
        if (
            best_item[2] == "mirror:identity"
            and first_legal_balanced
            and min(first_legal_balanced, key=lambda item: item[0])[0]
            <= best_score + _env_float("OURS_FIRST_LEGAL_BALANCED_EPS", 0.005)
        ):
            return min(first_legal_balanced, key=lambda item: item[0])[3]

    balanced_eps = _env_float("OURS_BALANCED_SELECTOR_EPS", 0.00125)
    balanced_near = [
        item
        for item in scored
        if item[2].endswith(":classic_balanced")
        and item[0] <= best_score + balanced_eps
    ]
    if balanced_near:
        return min(balanced_near, key=lambda item: item[0])[3]

    non_pin = [item for item in scored if not item[2].endswith(":pin_route_spread")]
    best_non_pin = min(non_pin, key=lambda item: item[0]) if non_pin else None
    pin_near = [item for item in near_best if item[2].endswith(":pin_route_spread")]
    classic_near = any(
        item[2].endswith(":classic_spread") or item[2].endswith(":classic_balanced")
        for item in near_best
    )
    if (
        pin_near
        and best_non_pin is not None
        and best_non_pin[2].endswith(":route_spread")
        and not classic_near
    ):
        return min(pin_near, key=lambda item: item[0])[3]

    return min(near_best, key=lambda item: (item[1], item[0]))[3]


def _candidate_preference(name: str) -> int:
    # When the surrogate is nearly tied, prefer stronger spread variants. Exact
    # sweeps showed this can reduce true density/congestion without hurting
    # clear mirror wins.
    if name.startswith("opt:") and name.endswith(":route_spread"):
        return 0
    if name.startswith("opt:") and name.endswith(":classic_spread"):
        return 1
    if name.startswith("opt:") and name.endswith(":route_cong"):
        return 2
    if name.startswith("opt:") and name.endswith(":classic_balanced"):
        return 3
    if name.startswith("layout:") and "spread" in name:
        return 4
    if name.startswith("mirror:"):
        return 5
    return 6


def _fast_surrogate_score(
    placement: torch.Tensor, benchmark: Benchmark, data: dict[str, torch.Tensor] | None = None
) -> float:
    n_hard = int(benchmark.num_hard_macros)
    score = 0.0
    count = 0
    use_pin_nets = (
        _env_bool("OURS_SELECTOR_PIN_AWARE", "0")
        and len(getattr(benchmark, "net_pin_nodes", [])) == int(benchmark.num_nets)
        and len(getattr(benchmark, "macro_pin_offsets", [])) >= n_hard
    )
    if use_pin_nets:
        zero_offset = torch.zeros(2, dtype=torch.float32)
        for pins_t in benchmark.net_pin_nodes:
            pins = [(int(owner), int(slot)) for owner, slot in pins_t.tolist()]
            if not any(owner < n_hard for owner, _ in pins):
                continue
            coords = []
            for owner, slot in pins:
                if owner < n_hard:
                    offsets = benchmark.macro_pin_offsets[owner]
                    offset = (
                        offsets[slot].detach().float()
                        if 0 <= slot < len(offsets)
                        else zero_offset
                    )
                    coords.append(placement[owner].detach().float() + offset)
                elif owner < benchmark.num_macros:
                    coords.append(placement[owner].detach().float())
                else:
                    port_idx = owner - benchmark.num_macros
                    if 0 <= port_idx < len(benchmark.port_positions):
                        coords.append(benchmark.port_positions[port_idx].detach().float())
            if len(coords) <= 1:
                continue
            coords = torch.stack(coords)
            hpwl = float(
                (coords[:, 0].max() - coords[:, 0].min())
                + (coords[:, 1].max() - coords[:, 1].min())
            )
            score += hpwl / max(float(benchmark.canvas_width + benchmark.canvas_height), 1e-6)
            count += 1
    else:
        for nodes_t in benchmark.net_nodes:
            nodes = [int(x) for x in nodes_t.tolist()]
            if not any(node < n_hard for node in nodes):
                continue
            coords = []
            for node in nodes:
                if node < benchmark.num_macros:
                    coords.append(placement[node].detach().float())
                else:
                    port_idx = node - benchmark.num_macros
                    if 0 <= port_idx < len(benchmark.port_positions):
                        coords.append(benchmark.port_positions[port_idx].detach().float())
            if len(coords) <= 1:
                continue
            coords = torch.stack(coords)
            hpwl = float(
                (coords[:, 0].max() - coords[:, 0].min())
                + (coords[:, 1].max() - coords[:, 1].min())
            )
            score += hpwl / max(float(benchmark.canvas_width + benchmark.canvas_height), 1e-6)
            count += 1
    if count:
        score /= count

    if data is None:
        data = _build_surrogate_data(benchmark)
    movable_idx = data["movable_idx"]
    if len(movable_idx) > 0:
        area = (
            benchmark.macro_sizes[movable_idx, 0] * benchmark.macro_sizes[movable_idx, 1]
        ).detach().float()
        occ = _gaussian_occupancy(
            placement[movable_idx].detach().float(),
            area,
            data["density_centers"],
            data["density_sigma"],
        )
        occ = occ + data["fixed_occ"]
        density = torch.relu((occ - data["density_cap"]) / data["density_cap"].clamp_min(1e-6)).pow(2).mean()
        score += 0.85 * float(density)

    pos = placement[:n_hard].detach().float()
    sizes = benchmark.macro_sizes[:n_hard].detach().float()
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    sep_x = (sizes[:, 0][:, None] + sizes[:, 0][None, :]) * 0.5
    sep_y = (sizes[:, 1][:, None] + sizes[:, 1][None, :]) * 0.5
    ov = torch.relu(sep_x - dx.abs()) * torch.relu(sep_y - dy.abs())
    score += float(torch.triu(ov, diagonal=1).sum()) * 10.0
    return score


def _select_by_proxy(
    candidates: list[tuple[str, torch.Tensor]], benchmark: Benchmark
) -> torch.Tensor | None:
    plc = _load_plc(benchmark)
    if plc is None:
        return None
    try:
        from macro_place.objective import compute_proxy_cost
    except Exception:
        return None

    best = None
    best_cost = float("inf")
    for _, placement in candidates:
        if not _is_strictly_valid(placement, benchmark):
            continue
        try:
            costs = compute_proxy_cost(placement, benchmark, plc)
        except Exception:
            continue
        if int(costs.get("overlap_count", 1)) != 0:
            continue
        cost = float(costs["proxy_cost"])
        if cost < best_cost:
            best_cost = cost
            best = placement
    return best


def _try_replace_refine(placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    try:
        helper_dir = Path(__file__).resolve().parent
        if str(helper_dir) not in sys.path:
            sys.path.insert(0, str(helper_dir))
        from replace_backend import refine_with_replace
    except Exception:
        return placement

    try:
        return refine_with_replace(
            placement,
            benchmark,
            load_plc=_load_plc,
            legalize_hard=_legalize_hard,
            is_valid=_is_strictly_valid,
        )
    except Exception:
        return placement


def _try_analytical_refine(placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    try:
        helper_dir = Path(__file__).resolve().parent
        if str(helper_dir) not in sys.path:
            sys.path.insert(0, str(helper_dir))
        from analytical_backend import refine_with_analytical
    except Exception:
        return placement

    try:
        return refine_with_analytical(
            placement,
            benchmark,
            load_plc=_load_plc,
            legalize_hard=_legalize_hard,
            is_valid=_is_strictly_valid,
        )
    except Exception:
        return placement


def _try_batch_analytical_refine(placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    try:
        helper_dir = Path(__file__).resolve().parent
        if str(helper_dir) not in sys.path:
            sys.path.insert(0, str(helper_dir))
        from batch_backend import refine_with_batch_analytical
    except Exception:
        return placement

    try:
        return refine_with_batch_analytical(
            placement,
            benchmark,
            load_plc=_load_plc,
            legalize_hard=_legalize_hard,
            is_valid=_is_strictly_valid,
        )
    except Exception:
        return placement


def _try_fast_search_refine(placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    try:
        helper_dir = Path(__file__).resolve().parent
        if str(helper_dir) not in sys.path:
            sys.path.insert(0, str(helper_dir))
        from fast_search import refine_with_fast_search
    except Exception:
        return placement

    try:
        return refine_with_fast_search(
            placement,
            benchmark,
            load_plc=_load_plc,
            legalize_hard=_legalize_hard,
            is_valid=_is_strictly_valid,
        )
    except Exception:
        return placement


def _try_snap_search_refine(placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    try:
        helper_dir = Path(__file__).resolve().parent
        if str(helper_dir) not in sys.path:
            sys.path.insert(0, str(helper_dir))
        from snap_search import refine_with_snap_search
    except Exception:
        return placement

    try:
        return refine_with_snap_search(
            placement,
            benchmark,
            load_plc=_load_plc,
            is_valid=_is_strictly_valid,
        )
    except Exception:
        return placement


def _try_soft_search_refine(placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    try:
        helper_dir = Path(__file__).resolve().parent
        if str(helper_dir) not in sys.path:
            sys.path.insert(0, str(helper_dir))
        from soft_search import refine_with_soft_search
    except Exception:
        return placement

    try:
        return refine_with_soft_search(
            placement,
            benchmark,
            load_plc=_load_plc,
            is_valid=_is_strictly_valid,
        )
    except Exception:
        return placement


def _load_plc(benchmark: Benchmark):
    try:
        from macro_place.loader import load_benchmark, load_benchmark_from_dir
    except Exception:
        return None

    name = str(benchmark.name)
    ibm_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if (ibm_dir / "netlist.pb.txt").exists():
        try:
            _, plc = load_benchmark_from_dir(str(ibm_dir))
            return plc
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
            _, plc = load_benchmark(
                str(ng45 / "netlist.pb.txt"),
                str(ng45 / "initial.plc"),
                name=name,
            )
            return plc
        except Exception:
            return None
    return None


def _spectral_hard_candidates(
    benchmark: Benchmark,
    *,
    limit: int,
    gap: float,
) -> list[tuple[str, torch.Tensor]]:
    n_hard = int(benchmark.num_hard_macros)
    if limit <= 0 or n_hard <= 2:
        return []
    hard_mask = benchmark.get_hard_macro_mask()
    movable = (benchmark.get_movable_mask() & hard_mask)[:n_hard].detach().cpu().numpy().astype(bool, copy=False)
    if int(movable.sum()) <= 1:
        return []

    graph = np.zeros((n_hard, n_hard), dtype=np.float64)
    for net_idx, nodes_t in enumerate(benchmark.net_nodes):
        hard_nodes: list[int] = []
        seen: set[int] = set()
        for raw in nodes_t.tolist():
            node = int(raw)
            if 0 <= node < n_hard and node not in seen:
                hard_nodes.append(node)
                seen.add(node)
        if len(hard_nodes) <= 1:
            continue
        net_weight = (
            float(benchmark.net_weights[net_idx])
            if len(getattr(benchmark, "net_weights", [])) > net_idx
            else 1.0
        )
        weight = net_weight / max(len(hard_nodes) - 1, 1)
        if len(hard_nodes) <= 40:
            pair_weight = weight / max(len(hard_nodes), 1)
            for a_i, a in enumerate(hard_nodes):
                for b in hard_nodes[a_i + 1 :]:
                    graph[a, b] += pair_weight
                    graph[b, a] += pair_weight
        else:
            for a_i, a in enumerate(hard_nodes):
                b = hard_nodes[(a_i + 1) % len(hard_nodes)]
                if a != b:
                    graph[a, b] += weight
                    graph[b, a] += weight

    degree = graph.sum(axis=1)
    active = degree > 1e-12
    if int((active & movable).sum()) <= 1:
        return []

    lap = np.diag(degree) - graph
    try:
        _, eigvecs = np.linalg.eigh(lap + np.eye(n_hard, dtype=np.float64) * 1e-9)
    except np.linalg.LinAlgError:
        return []
    axes = [eigvecs[:, idx].astype(np.float64, copy=True) for idx in range(1, min(n_hard, 5))]
    if len(axes) < 2:
        return []

    original = benchmark.macro_positions[:n_hard].detach().cpu().numpy().astype(np.float64, copy=False)
    axis_x = _orient_spectral_axis(axes[0], original[:, 0], movable)
    axis_y = _orient_spectral_axis(axes[1], original[:, 1], movable)

    specs: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("xy", axis_x, axis_y),
        ("yx", axis_y, axis_x),
        ("flipx", -axis_x, axis_y),
        ("flipy", axis_x, -axis_y),
        ("flipxy", -axis_x, -axis_y),
    ]
    if len(axes) >= 3:
        axis_z = _orient_spectral_axis(axes[2], original[:, 0] + original[:, 1], movable)
        specs.extend(
            [
                ("xz", axis_x, axis_z),
                ("zy", axis_z, axis_y),
                ("zx", axis_z, axis_x),
            ]
        )

    out: list[tuple[str, torch.Tensor]] = []
    for name, x_axis, y_axis in specs:
        seed = _spectral_seed_from_axes(benchmark, x_axis, y_axis, movable, gap=gap)
        out.append((name, seed))
        if len(out) >= limit:
            break
    return out


def _orient_spectral_axis(axis: np.ndarray, reference: np.ndarray, movable: np.ndarray) -> np.ndarray:
    out = axis.astype(np.float64, copy=True)
    mask = movable & np.isfinite(out) & np.isfinite(reference)
    if int(mask.sum()) >= 2:
        centered_axis = out[mask] - float(out[mask].mean())
        centered_ref = reference[mask] - float(reference[mask].mean())
        if float(np.dot(centered_axis, centered_ref)) < 0.0:
            out = -out
    return out


def _spectral_seed_from_axes(
    benchmark: Benchmark,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    movable: np.ndarray,
    *,
    gap: float,
) -> torch.Tensor:
    out = benchmark.macro_positions.clone()
    n_hard = int(benchmark.num_hard_macros)
    sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64, copy=False)
    margin = max(float(gap), _env_float("OURS_BOUNDARY_MARGIN", 1e-4))
    x_min = sizes[:, 0] * 0.5 + margin
    x_max = float(benchmark.canvas_width) - sizes[:, 0] * 0.5 - margin
    y_min = sizes[:, 1] * 0.5 + margin
    y_max = float(benchmark.canvas_height) - sizes[:, 1] * 0.5 - margin
    x_min = np.minimum(x_min, x_max)
    y_min = np.minimum(y_min, y_max)

    mapped_x = _map_spectral_axis_to_range(x_axis, x_min, x_max, movable)
    mapped_y = _map_spectral_axis_to_range(y_axis, y_min, y_max, movable)
    for idx in np.where(movable)[0].tolist():
        out[idx, 0] = float(mapped_x[idx])
        out[idx, 1] = float(mapped_y[idx])
    out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
    _clamp_all(out, benchmark)
    return out


def _map_spectral_axis_to_range(
    axis: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    movable: np.ndarray,
) -> np.ndarray:
    mapped = (lower + upper) * 0.5
    vals = axis[movable]
    vals = vals[np.isfinite(vals)]
    if vals.size >= 2:
        lo = float(np.quantile(vals, 0.05))
        hi = float(np.quantile(vals, 0.95))
        if hi - lo > 1e-12:
            scaled = np.clip((axis - lo) / (hi - lo), 0.02, 0.98)
            mapped = lower + scaled * (upper - lower)
    return mapped


def _partition_pack_candidates(
    benchmark: Benchmark,
    *,
    limit: int,
    gap: float,
) -> list[tuple[str, torch.Tensor]]:
    n_hard = int(benchmark.num_hard_macros)
    if limit <= 0 or n_hard <= 1:
        return []

    movable = (
        (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask())[:n_hard]
        .detach()
        .cpu()
        .numpy()
        .astype(bool, copy=False)
    )
    movable_indices = np.where(movable)[0]
    if len(movable_indices) <= 1:
        return []

    original = benchmark.macro_positions[:n_hard].detach().cpu().numpy().astype(np.float64, copy=False)
    axes = _partition_axes(benchmark, movable)
    x_axis = _orient_spectral_axis(axes[0], original[:, 0], movable)
    y_axis = _orient_spectral_axis(axes[1], original[:, 1], movable)
    slices = max(2, _env_int("OURS_PARTITION_SLICES", 4))

    specs: list[tuple[str, str, np.ndarray, np.ndarray]] = [
        ("x_slices", "x_slices", x_axis, y_axis),
        ("y_slices", "y_slices", y_axis, x_axis),
        ("quad_xy", "quadrants", x_axis, y_axis),
        ("quad_yx", "quadrants", y_axis, x_axis),
        ("quad_flipx", "quadrants", -x_axis, y_axis),
        ("quad_flipy", "quadrants", x_axis, -y_axis),
        ("x_orig", "x_slices", original[:, 0], original[:, 1]),
        ("y_orig", "y_slices", original[:, 1], original[:, 0]),
    ]

    out: list[tuple[str, torch.Tensor]] = []
    for name, mode, primary, secondary in specs:
        if mode == "x_slices":
            seed = _partition_slice_seed(
                benchmark,
                movable_indices,
                primary,
                secondary,
                slices=slices,
                vertical=True,
                gap=gap,
            )
        elif mode == "y_slices":
            seed = _partition_slice_seed(
                benchmark,
                movable_indices,
                primary,
                secondary,
                slices=slices,
                vertical=False,
                gap=gap,
            )
        else:
            seed = _partition_quadrant_seed(
                benchmark,
                movable_indices,
                primary,
                secondary,
                gap=gap,
            )
        out.append((name, seed))
        if len(out) >= limit:
            break
    return out


def _partition_axes(benchmark: Benchmark, movable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_hard = int(benchmark.num_hard_macros)
    original = benchmark.macro_positions[:n_hard].detach().cpu().numpy().astype(np.float64, copy=False)
    graph = np.zeros((n_hard, n_hard), dtype=np.float64)
    for net_idx, nodes_t in enumerate(benchmark.net_nodes):
        hard_nodes: list[int] = []
        seen: set[int] = set()
        for raw in nodes_t.tolist():
            node = int(raw)
            if 0 <= node < n_hard and node not in seen:
                hard_nodes.append(node)
                seen.add(node)
        if len(hard_nodes) <= 1:
            continue
        net_weight = (
            float(benchmark.net_weights[net_idx])
            if len(getattr(benchmark, "net_weights", [])) > net_idx
            else 1.0
        )
        pair_weight = net_weight / max(len(hard_nodes) * max(len(hard_nodes) - 1, 1), 1)
        for a_i, a in enumerate(hard_nodes):
            for b in hard_nodes[a_i + 1 :]:
                graph[a, b] += pair_weight
                graph[b, a] += pair_weight

    degree = graph.sum(axis=1)
    if int(((degree > 1e-12) & movable).sum()) <= 1:
        return original[:, 0].copy(), original[:, 1].copy()

    lap = np.diag(degree) - graph
    try:
        _, eigvecs = np.linalg.eigh(lap + np.eye(n_hard, dtype=np.float64) * 1e-9)
    except np.linalg.LinAlgError:
        return original[:, 0].copy(), original[:, 1].copy()
    if eigvecs.shape[1] < 3:
        return original[:, 0].copy(), original[:, 1].copy()
    return eigvecs[:, 1].astype(np.float64, copy=True), eigvecs[:, 2].astype(np.float64, copy=True)


def _partition_slice_seed(
    benchmark: Benchmark,
    movable_indices: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    slices: int,
    vertical: bool,
    gap: float,
) -> torch.Tensor:
    n_hard = int(benchmark.num_hard_macros)
    sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64, copy=False)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    groups = _axis_area_groups(movable_indices, primary, sizes, slices)
    rects = []
    if vertical:
        width = cw / float(len(groups))
        for idx in range(len(groups)):
            rects.append((idx * width, 0.0, (idx + 1) * width, ch))
    else:
        height = ch / float(len(groups))
        for idx in range(len(groups)):
            rects.append((0.0, idx * height, cw, (idx + 1) * height))
    return _pack_partition_groups(benchmark, groups, rects, secondary, gap=gap)


def _partition_quadrant_seed(
    benchmark: Benchmark,
    movable_indices: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    gap: float,
) -> torch.Tensor:
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    x_med = float(np.median(x_axis[movable_indices]))
    y_med = float(np.median(y_axis[movable_indices]))
    groups: list[list[int]] = [[], [], [], []]
    for idx in movable_indices.tolist():
        gx = 1 if float(x_axis[idx]) >= x_med else 0
        gy = 1 if float(y_axis[idx]) >= y_med else 0
        groups[gy * 2 + gx].append(idx)
    rects = [
        (0.0, 0.0, cw * 0.5, ch * 0.5),
        (cw * 0.5, 0.0, cw, ch * 0.5),
        (0.0, ch * 0.5, cw * 0.5, ch),
        (cw * 0.5, ch * 0.5, cw, ch),
    ]
    return _pack_partition_groups(benchmark, groups, rects, x_axis + y_axis, gap=gap)


def _axis_area_groups(
    indices: np.ndarray,
    axis: np.ndarray,
    sizes: np.ndarray,
    count: int,
) -> list[list[int]]:
    count = max(1, min(int(count), len(indices)))
    areas = sizes[:, 0] * sizes[:, 1]
    ordered = sorted(indices.tolist(), key=lambda idx: (float(axis[idx]), -float(areas[idx])))
    total = max(float(sum(areas[idx] for idx in ordered)), 1e-12)
    target = total / float(count)
    groups: list[list[int]] = [[] for _ in range(count)]
    group_idx = 0
    accum = 0.0
    for idx in ordered:
        if group_idx < count - 1 and groups[group_idx] and accum >= target:
            group_idx += 1
            accum = 0.0
        groups[group_idx].append(idx)
        accum += float(areas[idx])
    return groups


def _pack_partition_groups(
    benchmark: Benchmark,
    groups: list[list[int]],
    rects: list[tuple[float, float, float, float]],
    order_axis: np.ndarray,
    *,
    gap: float,
) -> torch.Tensor:
    out = benchmark.macro_positions.clone()
    n_hard = int(benchmark.num_hard_macros)
    sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64, copy=False)
    fixed = benchmark.macro_fixed[:n_hard].detach().cpu().numpy().astype(bool, copy=False)
    placed = _fixed_hard_boxes(benchmark, sizes)
    areas = sizes[:, 0] * sizes[:, 1]
    group_order = sorted(
        range(len(groups)),
        key=lambda group_idx: -sum(float(areas[idx]) for idx in groups[group_idx]),
    )

    for group_idx in group_order:
        rect = rects[min(group_idx, len(rects) - 1)]
        ordered = sorted(
            [idx for idx in groups[group_idx] if not fixed[idx]],
            key=lambda idx: (-float(areas[idx]), float(order_axis[idx])),
        )
        for idx in ordered:
            pos = _find_partition_spot(idx, rect, placed, sizes, benchmark, gap=gap)
            if pos is None:
                pos = _find_partition_spot(
                    idx,
                    (0.0, 0.0, float(benchmark.canvas_width), float(benchmark.canvas_height)),
                    placed,
                    sizes,
                    benchmark,
                    gap=gap,
                )
            if pos is None:
                continue
            x, y = pos
            out[idx, 0] = float(x)
            out[idx, 1] = float(y)
            half_w = float(sizes[idx, 0]) * 0.5
            half_h = float(sizes[idx, 1]) * 0.5
            placed.append((x - half_w, y - half_h, x + half_w, y + half_h))

    out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
    _clamp_all(out, benchmark)
    return out


def _fixed_hard_boxes(benchmark: Benchmark, sizes: np.ndarray) -> list[tuple[float, float, float, float]]:
    n_hard = int(benchmark.num_hard_macros)
    fixed = benchmark.macro_fixed[:n_hard].detach().cpu().numpy().astype(bool, copy=False)
    pos = benchmark.macro_positions[:n_hard].detach().cpu().numpy().astype(np.float64, copy=False)
    boxes: list[tuple[float, float, float, float]] = []
    for idx in np.where(fixed)[0].tolist():
        half_w = float(sizes[idx, 0]) * 0.5
        half_h = float(sizes[idx, 1]) * 0.5
        x, y = float(pos[idx, 0]), float(pos[idx, 1])
        boxes.append((x - half_w, y - half_h, x + half_w, y + half_h))
    return boxes


def _find_partition_spot(
    idx: int,
    rect: tuple[float, float, float, float],
    placed: list[tuple[float, float, float, float]],
    sizes: np.ndarray,
    benchmark: Benchmark,
    *,
    gap: float,
) -> tuple[float, float] | None:
    half_w = float(sizes[idx, 0]) * 0.5
    half_h = float(sizes[idx, 1]) * 0.5
    margin = max(float(gap), _env_float("OURS_BOUNDARY_MARGIN", 1e-4))
    lo_x = max(float(rect[0]) + half_w + margin, half_w + margin)
    hi_x = min(float(rect[2]) - half_w - margin, float(benchmark.canvas_width) - half_w - margin)
    lo_y = max(float(rect[1]) + half_h + margin, half_h + margin)
    hi_y = min(float(rect[3]) - half_h - margin, float(benchmark.canvas_height) - half_h - margin)
    if lo_x > hi_x or lo_y > hi_y:
        return None

    x_candidates = [lo_x, hi_x, (lo_x + hi_x) * 0.5]
    y_candidates = [lo_y, hi_y, (lo_y + hi_y) * 0.5]
    for box in placed:
        x_candidates.extend((float(box[0]) - gap - half_w, float(box[2]) + gap + half_w))
        y_candidates.extend((float(box[1]) - gap - half_h, float(box[3]) + gap + half_h))
    x_values = sorted({round(float(np.clip(x, lo_x, hi_x)), 6) for x in x_candidates})
    y_values = sorted({round(float(np.clip(y, lo_y, hi_y)), 6) for y in y_candidates})
    center_x = (lo_x + hi_x) * 0.5
    center_y = (lo_y + hi_y) * 0.5
    pairs = sorted(
        ((x, y) for y in y_values for x in x_values),
        key=lambda item: (abs(item[0] - center_x) + abs(item[1] - center_y), item[1], item[0]),
    )
    for x, y in pairs:
        box = (x - half_w, y - half_h, x + half_w, y + half_h)
        if not any(_boxes_overlap(box, other, gap=gap) for other in placed):
            return float(x), float(y)
    return None


def _scale_about_center(
    placement: torch.Tensor, benchmark: Benchmark, scale_x: float, scale_y: float
) -> torch.Tensor:
    """Scale movable macro centers around the canvas center."""
    out = placement.clone()
    movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    center = torch.tensor(
        [float(benchmark.canvas_width) * 0.5, float(benchmark.canvas_height) * 0.5],
        dtype=out.dtype,
    )
    scale = torch.tensor([float(scale_x), float(scale_y)], dtype=out.dtype)
    out[movable] = center + (out[movable] - center) * scale
    out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
    _clamp_all(out, benchmark)
    return out


def _apply_transform(placement: torch.Tensor, benchmark: Benchmark, transform: str) -> torch.Tensor:
    """Mirror movable macro centers across the canvas; fixed macros stay fixed."""
    out = placement.clone()
    movable = benchmark.get_movable_mask()
    if transform in {"flip_x", "flip_xy"}:
        out[movable, 0] = float(benchmark.canvas_width) - out[movable, 0]
    if transform in {"flip_y", "flip_xy"}:
        out[movable, 1] = float(benchmark.canvas_height) - out[movable, 1]
    out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
    _clamp_all(out, benchmark)
    return out


def _legalize_hard(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    gap: float = 0.005,
    max_rounds: int = 500,
) -> torch.Tensor:
    """Repair hard-macro overlaps by pairwise pushes, with row-pack fallback."""
    n = int(benchmark.num_hard_macros)
    if n <= 1:
        out = placement.clone()
        _clamp_all(out, benchmark)
        return out

    out = placement.clone()
    pos = out[:n].detach().cpu().numpy().astype(np.float64, copy=True)
    sizes = benchmark.macro_sizes[:n].detach().cpu().numpy().astype(np.float64, copy=True)
    fixed = benchmark.macro_fixed[:n].detach().cpu().numpy().astype(bool, copy=True)
    movable = ~fixed

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5
    sep_x = half_w[:, None] + half_w[None, :] + float(gap)
    sep_y = half_h[:, None] + half_h[None, :] + float(gap)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)

    _clamp_np(pos, half_w, half_h, cw, ch)

    for round_idx in range(max_rounds):
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        ovx = sep_x - np.abs(dx)
        ovy = sep_y - np.abs(dy)
        mask = (ovx > 0.0) & (ovy > 0.0) & upper
        if not mask.any():
            break

        ii, jj = np.where(mask)
        active = movable[ii] | movable[jj]
        if not active.any():
            break
        ii = ii[active]
        jj = jj[active]

        choose_x = ovx[ii, jj] <= ovy[ii, jj]
        sx = np.where(dx[ii, jj] >= 0.0, 1.0, -1.0)
        sy = np.where(dy[ii, jj] >= 0.0, 1.0, -1.0)
        push_x = np.where(choose_x, sx * (ovx[ii, jj] + gap), 0.0)
        push_y = np.where(~choose_x, sy * (ovy[ii, jj] + gap), 0.0)

        area_i = sizes[ii, 0] * sizes[ii, 1]
        area_j = sizes[jj, 0] * sizes[jj, 1]
        denom = np.maximum(area_i + area_j, 1e-12)

        frac_i = np.where(fixed[ii], 0.0, np.where(fixed[jj], 1.0, area_j / denom))
        frac_j = np.where(fixed[jj], 0.0, np.where(fixed[ii], 1.0, area_i / denom))

        move = np.zeros_like(pos)
        np.add.at(move[:, 0], ii, push_x * frac_i)
        np.add.at(move[:, 1], ii, push_y * frac_i)
        np.add.at(move[:, 0], jj, -push_x * frac_j)
        np.add.at(move[:, 1], jj, -push_y * frac_j)

        # Large early moves, damped later to avoid oscillating near convergence.
        damping = 0.60 if round_idx < 80 else 0.35
        max_step = np.maximum(0.02 * np.maximum(sizes[:, 0], sizes[:, 1]), gap)
        move[:, 0] = np.clip(move[:, 0] * damping, -max_step, max_step)
        move[:, 1] = np.clip(move[:, 1] * damping, -max_step, max_step)
        move[fixed] = 0.0
        pos += move
        _clamp_np(pos, half_w, half_h, cw, ch)

        if float(np.abs(move).sum()) < 1e-10:
            break

    # If pairwise pushes did not fully resolve overlaps, fall back to a simple
    # deterministic packer. This baseline prioritizes bounded runtime.
    if _overlapping_macro_indices(pos, sizes, gap=gap):
        packed = _row_pack_fallback(benchmark, gap=max(gap, 0.001))
        return packed

    out[:n] = torch.tensor(pos, dtype=out.dtype)
    out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
    _clamp_all(out, benchmark)
    return out


def _row_pack_fallback(benchmark: Benchmark, *, gap: float = 0.001) -> torch.Tensor:
    """Guaranteed simple fallback for hard macros; soft macros stay original."""
    out = benchmark.macro_positions.clone()
    n = int(benchmark.num_hard_macros)
    if n <= 0:
        return out

    sizes = benchmark.macro_sizes
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).detach().cpu().numpy()
    fixed = benchmark.macro_fixed[:n].detach().cpu().numpy()

    placed = []
    for i in range(n):
        if fixed[i]:
            x, y = out[i].tolist()
            w, h = sizes[i].tolist()
            placed.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))

    order = [i for i in range(n) if movable[i]]
    order.sort(key=lambda i: -float(sizes[i, 0] * sizes[i, 1]))

    cursor_x = 0.0
    cursor_y = 0.0
    row_h = 0.0

    for idx in order:
        w = float(sizes[idx, 0])
        h = float(sizes[idx, 1])
        if cursor_x + w > cw:
            cursor_x = 0.0
            cursor_y += row_h + gap
            row_h = 0.0

        found = False
        y = cursor_y
        while y + h <= ch + 1e-9 and not found:
            x = cursor_x
            while x + w <= cw + 1e-9:
                box = (x, y, x + w, y + h)
                if not any(_boxes_overlap(box, other, gap=gap) for other in placed):
                    out[idx, 0] = x + w / 2
                    out[idx, 1] = y + h / 2
                    placed.append(box)
                    cursor_x = x + w + gap
                    row_h = max(row_h, h)
                    found = True
                    break
                x += max(gap, min(w, h) * 0.25)
            if not found:
                y += max(gap, h * 0.25)
                cursor_x = 0.0

        if not found:
            # Last resort: keep the clamped original coordinate. Validation will
            # catch this; this path should not occur on the public IBM suite.
            hw, hh = w / 2, h / 2
            out[idx, 0] = torch.clamp(out[idx, 0], hw, cw - hw)
            out[idx, 1] = torch.clamp(out[idx, 1], hh, ch - hh)

    out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
    _clamp_all(out, benchmark)
    return out


def _is_strictly_valid(placement: torch.Tensor, benchmark: Benchmark) -> bool:
    try:
        from macro_place.utils import validate_placement

        return bool(validate_placement(placement, benchmark)[0])
    except Exception:
        pass

    if placement.shape != (benchmark.num_macros, 2):
        return False
    if torch.isnan(placement).any() or torch.isinf(placement).any():
        return False
    if benchmark.macro_fixed.any():
        if not torch.allclose(
            placement[benchmark.macro_fixed],
            benchmark.macro_positions[benchmark.macro_fixed],
            atol=1e-4,
        ):
            return False
    sizes = benchmark.macro_sizes
    x = placement[:, 0]
    y = placement[:, 1]
    if (x - sizes[:, 0] / 2 < -1e-6).any() or (x + sizes[:, 0] / 2 > benchmark.canvas_width + 1e-6).any():
        return False
    if (y - sizes[:, 1] / 2 < -1e-6).any() or (y + sizes[:, 1] / 2 > benchmark.canvas_height + 1e-6).any():
        return False

    n = int(benchmark.num_hard_macros)
    pos = placement[:n].detach().cpu().numpy().astype(np.float64, copy=False)
    sz = sizes[:n].detach().cpu().numpy().astype(np.float64, copy=False)
    return len(_overlapping_macro_indices(pos, sz, gap=0.0)) == 0


def _overlapping_macro_indices(pos: np.ndarray, sizes: np.ndarray, *, gap: float) -> set[int]:
    n = pos.shape[0]
    if n <= 1:
        return set()
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    dx = np.abs(pos[:, 0][:, None] - pos[:, 0][None, :])
    dy = np.abs(pos[:, 1][:, None] - pos[:, 1][None, :])
    ov = (dx < hw[:, None] + hw[None, :] + gap) & (
        dy < hh[:, None] + hh[None, :] + gap
    )
    ov &= np.triu(np.ones((n, n), dtype=bool), k=1)
    ii, jj = np.where(ov)
    return set(ii.tolist()) | set(jj.tolist())


def _clamp_np(pos: np.ndarray, half_w: np.ndarray, half_h: np.ndarray, cw: float, ch: float) -> None:
    pos[:, 0] = np.clip(pos[:, 0], half_w, cw - half_w)
    pos[:, 1] = np.clip(pos[:, 1], half_h, ch - half_h)


def _finalize_placement(placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    out = placement.detach().clone().float()
    _clamp_all(out, benchmark)
    if _is_strictly_valid(out, benchmark):
        return out

    try:
        repaired = _legalize_hard(
            out,
            benchmark,
            gap=_env_float("OURS_GAP", 0.005),
            max_rounds=_env_int("OURS_FINAL_LEGALIZE_ROUNDS", 500),
        )
    except Exception:
        repaired = out
    _clamp_all(repaired, benchmark)
    if _is_strictly_valid(repaired, benchmark):
        return repaired.detach().clone().float()

    fallback = _row_pack_fallback(benchmark, gap=max(_env_float("OURS_GAP", 0.005), 0.001))
    _clamp_all(fallback, benchmark)
    return fallback.detach().clone().float()


def _clamp_all(placement: torch.Tensor, benchmark: Benchmark) -> None:
    sizes = benchmark.macro_sizes
    margin = max(0.0, _env_float("OURS_BOUNDARY_MARGIN", 1e-4))
    fixed = benchmark.macro_fixed
    movable = ~fixed
    lower_x = sizes[:, 0] / 2 + margin
    upper_x = float(benchmark.canvas_width) - sizes[:, 0] / 2 - margin
    lower_y = sizes[:, 1] / 2 + margin
    upper_y = float(benchmark.canvas_height) - sizes[:, 1] / 2 - margin
    lower_x = torch.minimum(lower_x, upper_x)
    lower_y = torch.minimum(lower_y, upper_y)
    if movable.any():
        placement[movable, 0] = torch.minimum(
            torch.maximum(placement[movable, 0], lower_x[movable]),
            upper_x[movable],
        )
        placement[movable, 1] = torch.minimum(
            torch.maximum(placement[movable, 1], lower_y[movable]),
            upper_y[movable],
        )
    placement[fixed] = benchmark.macro_positions[fixed]


def _boxes_overlap(a: Iterable[float], b: Iterable[float], *, gap: float) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (
        ax1 + gap <= bx0
        or bx1 + gap <= ax0
        or ay1 + gap <= by0
        or by1 + gap <= ay0
    )
