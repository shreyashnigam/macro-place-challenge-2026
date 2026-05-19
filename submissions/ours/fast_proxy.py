"""Fast proxy scorer built from the official PlacementCost formulas.

The goal is not to define a new objective.  This module pre-extracts the
netlist/grid metadata from a PlacementCost object, then evaluates the same
public proxy components directly on candidate tensors:

    proxy = wirelength + 0.5 * density + 0.5 * congestion

It is used as a cheap candidate selector/search guide.  Promising placements
are still guarded by the official scorer before replacing a baseline.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import math
import os
import platform
import subprocess
import tempfile
from typing import Iterable

import numpy as np
import torch

from macro_place.benchmark import Benchmark


@dataclass(frozen=True)
class _Net:
    owners: np.ndarray
    offsets: np.ndarray
    source_owner: int
    source_offset: np.ndarray
    hpwl_weight: float
    route_weight: float


class FastProxy:
    def __init__(self, benchmark: Benchmark, plc) -> None:
        self.name = str(benchmark.name)
        self.num_macros = int(benchmark.num_macros)
        self.num_hard = int(benchmark.num_hard_macros)
        self.canvas_width = float(benchmark.canvas_width)
        self.canvas_height = float(benchmark.canvas_height)
        self.grid_cols = int(benchmark.grid_cols)
        self.grid_rows = int(benchmark.grid_rows)
        self.grid_width = self.canvas_width / max(float(self.grid_cols), 1.0)
        self.grid_height = self.canvas_height / max(float(self.grid_rows), 1.0)
        self.grid_size = self.grid_cols * self.grid_rows
        self.grid_area = self.grid_width * self.grid_height
        self.hroutes_per_micron = float(getattr(plc, "hroutes_per_micron", benchmark.hroutes_per_micron))
        self.vroutes_per_micron = float(getattr(plc, "vroutes_per_micron", benchmark.vroutes_per_micron))
        self.grid_h_routes = self.grid_height * max(self.hroutes_per_micron, 1e-12)
        self.grid_v_routes = self.grid_width * max(self.vroutes_per_micron, 1e-12)
        self.smooth_range = int(math.floor(float(getattr(plc, "smooth_range", 0.0))))
        self.hrouting_alloc = float(getattr(plc, "hrouting_alloc", 0.0))
        self.vrouting_alloc = float(getattr(plc, "vrouting_alloc", 0.0))
        self.net_count = max(float(getattr(plc, "net_cnt", 0.0)), 1.0)

        self.sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64, copy=True)
        self.port_positions = benchmark.port_positions.detach().cpu().numpy().astype(np.float64, copy=True)
        self.nets = self._extract_nets(benchmark, plc)
        self._native = None
        self._native_arrays = self._build_native_arrays()
        if _env_bool("OURS_FAST_PROXY_NATIVE", "1"):
            self._native = _load_native()

    def score(self, placement: torch.Tensor | np.ndarray, *, maps: bool = False) -> dict[str, object]:
        pos = _as_numpy(placement)
        if not maps and self._native is not None:
            native = self._score_native(pos)
            if native is not None:
                return native
        wirelength = self.wirelength_cost(pos)
        density = self.density_cost(pos)
        congestion_result = self.congestion_cost(pos, maps=maps)
        if maps:
            congestion, v_map, h_map, extras = congestion_result
        else:
            congestion, v_map, h_map = congestion_result
            extras = {}
        overlaps = self.overlap_count(pos)
        proxy = wirelength + 0.5 * density + 0.5 * congestion
        result: dict[str, object] = {
            "proxy_cost": float(proxy),
            "wirelength_cost": float(wirelength),
            "density_cost": float(density),
            "congestion_cost": float(congestion),
            "overlap_count": int(overlaps),
        }
        if maps:
            result["v_routing_cong"] = v_map
            result["h_routing_cong"] = h_map
            result.update(extras)
        return result

    def _score_native(self, pos: np.ndarray) -> dict[str, object] | None:
        arrays = self._native_arrays
        if arrays is None or self._native is None:
            return None
        pos_c = np.ascontiguousarray(pos, dtype=np.float64)
        out = np.zeros(5, dtype=np.float64)
        try:
            rc = self._native.score_proxy_native(
                ctypes.c_int(self.num_macros),
                ctypes.c_int(self.num_hard),
                ctypes.c_int(len(self.port_positions)),
                ctypes.c_int(len(self.nets)),
                ctypes.c_int(self.grid_rows),
                ctypes.c_int(self.grid_cols),
                ctypes.c_double(self.canvas_width),
                ctypes.c_double(self.canvas_height),
                ctypes.c_double(self.hroutes_per_micron),
                ctypes.c_double(self.vroutes_per_micron),
                ctypes.c_int(self.smooth_range),
                ctypes.c_double(self.hrouting_alloc),
                ctypes.c_double(self.vrouting_alloc),
                ctypes.c_double(self.net_count),
                _ptr_double(pos_c),
                _ptr_double(self.sizes),
                _ptr_double(self.port_positions),
                _ptr_int(arrays["starts"]),
                _ptr_int(arrays["owners"]),
                _ptr_double(arrays["offsets"]),
                _ptr_int(arrays["source_owner"]),
                _ptr_double(arrays["source_offsets"]),
                _ptr_double(arrays["hpwl"]),
                _ptr_double(arrays["route"]),
                _ptr_double(out),
            )
        except Exception:
            return None
        if int(rc) != 0:
            return None
        return {
            "proxy_cost": float(out[0]),
            "wirelength_cost": float(out[1]),
            "density_cost": float(out[2]),
            "congestion_cost": float(out[3]),
            "overlap_count": int(round(float(out[4]))),
        }

    def wirelength_cost(self, pos: np.ndarray) -> float:
        total = 0.0
        for net in self.nets:
            pts = self._net_pin_positions(pos, net.owners, net.offsets)
            if pts.shape[0] == 0:
                continue
            hpwl = (float(np.max(pts[:, 0])) - float(np.min(pts[:, 0]))) + (
                float(np.max(pts[:, 1])) - float(np.min(pts[:, 1]))
            )
            total += net.hpwl_weight * hpwl
        return total / max((self.canvas_width + self.canvas_height) * self.net_count, 1e-12)

    def density_cost(self, pos: np.ndarray) -> float:
        occupied = np.zeros(self.grid_size, dtype=np.float64)
        for idx in range(self.num_macros):
            self._add_rect_to_grid(
                occupied,
                float(pos[idx, 0]),
                float(pos[idx, 1]),
                float(self.sizes[idx, 0]),
                float(self.sizes[idx, 1]),
                scale=1.0,
            )
        densities = occupied / max(self.grid_area, 1e-12)
        return 0.5 * _abu_nonzero_top(densities, 0.10, self.grid_size)

    def congestion_cost(self, pos: np.ndarray, *, maps: bool = False):
        h_route = np.zeros(self.grid_size, dtype=np.float64)
        v_route = np.zeros(self.grid_size, dtype=np.float64)
        h_macro = np.zeros(self.grid_size, dtype=np.float64)
        v_macro = np.zeros(self.grid_size, dtype=np.float64)

        for net in self.nets:
            source = self._pin_grid(pos, net.source_owner, net.source_offset)
            gcells = {source}
            pts = self._net_pin_positions(pos, net.owners, net.offsets)
            for x, y in pts:
                gcells.add(self._grid_cell(float(x), float(y)))
            if len(gcells) == 2:
                self._two_pin_route(h_route, v_route, source, gcells, net.route_weight)
            elif len(gcells) == 3:
                self._three_pin_route(h_route, v_route, list(gcells), net.route_weight)
            elif len(gcells) > 3:
                for gcell in gcells:
                    if gcell != source:
                        self._two_pin_route(h_route, v_route, source, {source, gcell}, net.route_weight)

        for idx in range(self.num_hard):
            self._macro_route_over_grid(
                v_macro,
                h_macro,
                float(pos[idx, 0]),
                float(pos[idx, 1]),
                float(self.sizes[idx, 0]),
                float(self.sizes[idx, 1]),
            )

        v_route = v_route / max(self.grid_v_routes, 1e-12)
        h_route = h_route / max(self.grid_h_routes, 1e-12)
        v_macro = v_macro / max(self.grid_v_routes, 1e-12)
        h_macro = h_macro / max(self.grid_h_routes, 1e-12)
        v_route, h_route = self._smooth(v_route, h_route)
        v_total = v_route + v_macro
        h_total = h_route + h_macro
        cost = _abu_all_top(np.concatenate([v_total, h_total]), 0.05)
        if maps:
            extras = {
                "v_net_cong": v_route,
                "h_net_cong": h_route,
                "v_macro_cong": v_macro,
                "h_macro_cong": h_macro,
                "net_congestion_cost": _abu_all_top(np.concatenate([v_route, h_route]), 0.05),
                "macro_congestion_cost": _abu_all_top(np.concatenate([v_macro, h_macro]), 0.05),
            }
            return float(cost), v_total, h_total, extras
        return float(cost), None, None

    def overlap_count(self, pos: np.ndarray) -> int:
        count = 0
        sizes = self.sizes[: self.num_hard]
        for i in range(self.num_hard):
            xi, yi = pos[i]
            wi, hi = sizes[i]
            for j in range(i + 1, self.num_hard):
                xj, yj = pos[j]
                wj, hj = sizes[j]
                if abs(float(xi - xj)) < 0.5 * float(wi + wj) and abs(float(yi - yj)) < 0.5 * float(hi + hj):
                    count += 1
        return count

    def hot_bins(self, placement: torch.Tensor | np.ndarray, *, top: int = 16) -> list[tuple[int, int, float]]:
        pos = _as_numpy(placement)
        _, v_map, h_map = self.congestion_cost(pos, maps=True)
        assert v_map is not None and h_map is not None
        combined = v_map + h_map
        if combined.size == 0:
            return []
        k = max(1, min(int(top), combined.size))
        order = np.argpartition(combined, -k)[-k:]
        order = order[np.argsort(combined[order])[::-1]]
        return [(int(idx // self.grid_cols), int(idx % self.grid_cols), float(combined[idx])) for idx in order]

    def _extract_nets(self, benchmark: Benchmark, plc) -> list[_Net]:
        name_to_owner: dict[str, int] = {}
        for bench_idx, plc_idx in enumerate(benchmark.hard_macro_indices):
            name_to_owner[plc.modules_w_pins[plc_idx].get_name()] = bench_idx
        for offset, plc_idx in enumerate(benchmark.soft_macro_indices):
            name_to_owner[plc.modules_w_pins[plc_idx].get_name()] = self.num_hard + offset
        for offset, plc_idx in enumerate(plc.port_indices):
            name_to_owner[plc.modules_w_pins[plc_idx].get_name()] = self.num_macros + offset

        nets: list[_Net] = []
        for driver_name, sinks in getattr(plc, "nets", {}).items():
            driver_idx = plc.mod_name_to_indices.get(driver_name)
            if driver_idx is None:
                continue
            source = self._owner_offset_for_pin(plc, driver_idx, name_to_owner)
            if source is None:
                continue
            owners = [source[0]]
            offsets = [source[1]]
            for sink_name in sinks:
                sink_idx = plc.mod_name_to_indices.get(sink_name)
                if sink_idx is None:
                    continue
                pin = self._owner_offset_for_pin(plc, sink_idx, name_to_owner)
                if pin is None:
                    continue
                owners.append(pin[0])
                offsets.append(pin[1])
            if len(owners) <= 1:
                continue
            driver = plc.modules_w_pins[driver_idx]
            hpwl_weight = _get_weight(driver, default=1.0)
            route_weight = hpwl_weight if driver.get_type() == "MACRO_PIN" and hpwl_weight > 1.0 else 1.0
            nets.append(
                _Net(
                    owners=np.asarray(owners, dtype=np.int32),
                    offsets=np.asarray(offsets, dtype=np.float64),
                    source_owner=int(source[0]),
                    source_offset=np.asarray(source[1], dtype=np.float64),
                    hpwl_weight=float(hpwl_weight),
                    route_weight=float(route_weight),
                )
            )
        return nets

    def _build_native_arrays(self) -> dict[str, np.ndarray] | None:
        try:
            starts = [0]
            owners: list[int] = []
            offsets: list[float] = []
            source_owner: list[int] = []
            source_offsets: list[float] = []
            hpwl: list[float] = []
            route: list[float] = []
            for net in self.nets:
                owners.extend(int(x) for x in net.owners.tolist())
                offsets.extend(float(x) for x in net.offsets.reshape(-1).tolist())
                starts.append(len(owners))
                source_owner.append(int(net.source_owner))
                source_offsets.extend(float(x) for x in net.source_offset.reshape(-1).tolist())
                hpwl.append(float(net.hpwl_weight))
                route.append(float(net.route_weight))
            return {
                "starts": np.ascontiguousarray(np.asarray(starts, dtype=np.int32)),
                "owners": np.ascontiguousarray(np.asarray(owners, dtype=np.int32)),
                "offsets": np.ascontiguousarray(np.asarray(offsets, dtype=np.float64)),
                "source_owner": np.ascontiguousarray(np.asarray(source_owner, dtype=np.int32)),
                "source_offsets": np.ascontiguousarray(np.asarray(source_offsets, dtype=np.float64)),
                "hpwl": np.ascontiguousarray(np.asarray(hpwl, dtype=np.float64)),
                "route": np.ascontiguousarray(np.asarray(route, dtype=np.float64)),
            }
        except Exception:
            return None

    def _owner_offset_for_pin(self, plc, pin_idx: int, name_to_owner: dict[str, int]) -> tuple[int, tuple[float, float]] | None:
        pin = plc.modules_w_pins[pin_idx]
        pin_type = pin.get_type()
        if pin_type == "PORT":
            owner = name_to_owner.get(pin.get_name())
            if owner is None:
                return None
            return int(owner), (0.0, 0.0)
        if pin_type != "MACRO_PIN":
            return None
        ref_idx = plc.get_ref_node_id(pin_idx)
        if ref_idx < 0:
            return None
        ref_name = plc.modules_w_pins[ref_idx].get_name()
        owner = name_to_owner.get(ref_name)
        if owner is None:
            return None
        return int(owner), _get_offset(pin)

    def _net_pin_positions(self, pos: np.ndarray, owners: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        pts = np.empty((owners.shape[0], 2), dtype=np.float64)
        macro_mask = owners < self.num_macros
        if np.any(macro_mask):
            pts[macro_mask] = pos[owners[macro_mask]] + offsets[macro_mask]
        if np.any(~macro_mask):
            port_idx = owners[~macro_mask] - self.num_macros
            valid = (port_idx >= 0) & (port_idx < len(self.port_positions))
            port_pts = np.zeros((port_idx.shape[0], 2), dtype=np.float64)
            if np.any(valid):
                port_pts[valid] = self.port_positions[port_idx[valid]]
            pts[~macro_mask] = port_pts
        return pts

    def _pin_grid(self, pos: np.ndarray, owner: int, offset: np.ndarray) -> tuple[int, int]:
        if owner < self.num_macros:
            x, y = pos[owner] + offset
        else:
            port_idx = owner - self.num_macros
            if 0 <= port_idx < len(self.port_positions):
                x, y = self.port_positions[port_idx]
            else:
                x, y = 0.0, 0.0
        return self._grid_cell(float(x), float(y))

    def _grid_cell(self, x: float, y: float) -> tuple[int, int]:
        row = math.floor(y / max(self.grid_height, 1e-12))
        col = math.floor(x / max(self.grid_width, 1e-12))
        row = max(0, min(int(row), self.grid_rows - 1))
        col = max(0, min(int(col), self.grid_cols - 1))
        return row, col

    def _add_rect_to_grid(
        self,
        grid: np.ndarray,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        scale: float,
    ) -> None:
        x_min = x - 0.5 * w
        x_max = x + 0.5 * w
        y_min = y - 0.5 * h
        y_max = y + 0.5 * h
        bl_row, bl_col = self._grid_cell(x_min, y_min)
        ur_row, ur_col = self._grid_cell(x_max, y_max)
        for row in range(bl_row, ur_row + 1):
            gy0 = row * self.grid_height
            gy1 = (row + 1) * self.grid_height
            oy = min(y_max, gy1) - max(y_min, gy0)
            if oy <= 0.0:
                continue
            base = row * self.grid_cols
            for col in range(bl_col, ur_col + 1):
                gx0 = col * self.grid_width
                gx1 = (col + 1) * self.grid_width
                ox = min(x_max, gx1) - max(x_min, gx0)
                if ox > 0.0:
                    grid[base + col] += scale * ox * oy

    def _two_pin_route(
        self,
        h_route: np.ndarray,
        v_route: np.ndarray,
        source: tuple[int, int],
        gcells: Iterable[tuple[int, int]],
        weight: float,
    ) -> None:
        cells = list(gcells)
        sink = cells[1] if cells[0] == source else cells[0]
        row_min = min(sink[0], source[0])
        row_max = max(sink[0], source[0])
        col_min = min(sink[1], source[1])
        col_max = max(sink[1], source[1])
        for col in range(col_min, col_max):
            h_route[source[0] * self.grid_cols + col] += weight
        for row in range(row_min, row_max):
            v_route[row * self.grid_cols + sink[1]] += weight

    def _three_pin_route(
        self,
        h_route: np.ndarray,
        v_route: np.ndarray,
        gcells: list[tuple[int, int]],
        weight: float,
    ) -> None:
        temp = list(gcells)
        temp.sort(key=lambda x: (x[1], x[0]))
        y1, x1 = temp[0]
        y2, x2 = temp[1]
        y3, x3 = temp[2]
        if x1 < x2 and x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
            self._l_route(h_route, v_route, temp, weight)
        elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
            for col in range(x1, x2):
                h_route[y1 * self.grid_cols + col] += weight
            for row in range(y1, max(y2, y3)):
                v_route[row * self.grid_cols + x2] += weight
        elif y2 == y3:
            for col in range(x1, x2):
                h_route[y1 * self.grid_cols + col] += weight
            for col in range(x2, x3):
                h_route[y2 * self.grid_cols + col] += weight
            for row in range(min(y2, y1), max(y2, y1)):
                v_route[row * self.grid_cols + x2] += weight
        else:
            self._t_route(h_route, v_route, temp, weight)

    def _l_route(
        self,
        h_route: np.ndarray,
        v_route: np.ndarray,
        gcells: list[tuple[int, int]],
        weight: float,
    ) -> None:
        gcells.sort(key=lambda x: (x[1], x[0]))
        y1, x1 = gcells[0]
        y2, x2 = gcells[1]
        y3, x3 = gcells[2]
        for col in range(x1, x2):
            h_route[y1 * self.grid_cols + col] += weight
        for col in range(x2, x3):
            h_route[y2 * self.grid_cols + col] += weight
        for row in range(min(y1, y2), max(y1, y2)):
            v_route[row * self.grid_cols + x2] += weight
        for row in range(min(y2, y3), max(y2, y3)):
            v_route[row * self.grid_cols + x3] += weight

    def _t_route(
        self,
        h_route: np.ndarray,
        v_route: np.ndarray,
        gcells: list[tuple[int, int]],
        weight: float,
    ) -> None:
        gcells.sort()
        y1, x1 = gcells[0]
        y2, x2 = gcells[1]
        y3, x3 = gcells[2]
        xmin = min(x1, x2, x3)
        xmax = max(x1, x2, x3)
        for col in range(xmin, xmax):
            h_route[y2 * self.grid_cols + col] += weight
        for row in range(min(y1, y2), max(y1, y2)):
            v_route[row * self.grid_cols + x1] += weight
        for row in range(min(y2, y3), max(y2, y3)):
            v_route[row * self.grid_cols + x3] += weight

    def _macro_route_over_grid(
        self,
        v_macro: np.ndarray,
        h_macro: np.ndarray,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> None:
        x_min = x - 0.5 * w
        x_max = x + 0.5 * w
        y_min = y - 0.5 * h
        y_max = y + 0.5 * h
        bl_row, bl_col = self._grid_cell(x_min, y_min)
        ur_row, ur_col = self._grid_cell(x_max, y_max)
        partial_v = False
        partial_h = False
        overlaps: list[tuple[int, int, float, float]] = []
        for row in range(bl_row, ur_row + 1):
            gy0 = row * self.grid_height
            gy1 = (row + 1) * self.grid_height
            for col in range(bl_col, ur_col + 1):
                gx0 = col * self.grid_width
                gx1 = (col + 1) * self.grid_width
                x_dist = max(0.0, min(x_max, gx1) - max(x_min, gx0))
                y_dist = max(0.0, min(y_max, gy1) - max(y_min, gy0))
                if x_dist <= 0.0 or y_dist <= 0.0:
                    continue
                if ur_row != bl_row and (
                    (row == bl_row and abs(y_dist - self.grid_height) > 1e-5)
                    or (row == ur_row and abs(y_dist - self.grid_height) > 1e-5)
                ):
                    partial_v = True
                if ur_col != bl_col and (
                    (col == bl_col and abs(x_dist - self.grid_width) > 1e-5)
                    or (col == ur_col and abs(x_dist - self.grid_width) > 1e-5)
                ):
                    partial_h = True
                idx = row * self.grid_cols + col
                v_macro[idx] += x_dist * self.vrouting_alloc
                h_macro[idx] += y_dist * self.hrouting_alloc
                overlaps.append((row, col, x_dist, y_dist))
        if partial_v:
            for row, col, x_dist, _ in overlaps:
                if row == ur_row:
                    v_macro[row * self.grid_cols + col] -= x_dist * self.vrouting_alloc
        if partial_h:
            for row, col, _, y_dist in overlaps:
                if col == ur_col:
                    h_macro[row * self.grid_cols + col] -= y_dist * self.hrouting_alloc

    def _smooth(self, v_route: np.ndarray, h_route: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.smooth_range <= 0:
            return v_route, h_route
        v_grid = v_route.reshape(self.grid_rows, self.grid_cols)
        h_grid = h_route.reshape(self.grid_rows, self.grid_cols)
        v_temp = np.zeros_like(v_grid)
        h_temp = np.zeros_like(h_grid)
        sr = self.smooth_range
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                lp = max(0, col - sr)
                rp = min(self.grid_cols - 1, col + sr)
                v_temp[row, lp : rp + 1] += v_grid[row, col] / float(rp - lp + 1)
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                lp = max(0, row - sr)
                up = min(self.grid_rows - 1, row + sr)
                h_temp[lp : up + 1, col] += h_grid[row, col] / float(up - lp + 1)
        return v_temp.reshape(-1), h_temp.reshape(-1)


def build_fast_proxy(benchmark: Benchmark, plc) -> FastProxy | None:
    try:
        return FastProxy(benchmark, plc)
    except Exception:
        return None


def _get_weight(node, default: float) -> float:
    try:
        return float(node.get_weight())
    except Exception:
        return float(default)


def _get_offset(pin) -> tuple[float, float]:
    try:
        x, y = pin.get_offset()
        return float(x), float(y)
    except Exception:
        return float(getattr(pin, "x_offset", 0.0)), float(getattr(pin, "y_offset", 0.0))


def _as_numpy(placement: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(placement, torch.Tensor):
        return placement.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(placement, dtype=np.float64)


def _abu_nonzero_top(values: np.ndarray, frac: float, total_count: int) -> float:
    nonzero = values[values != 0.0]
    if total_count < 10:
        return float(np.mean(nonzero)) if nonzero.size else 0.0
    cnt = int(math.floor(total_count * frac))
    if cnt <= 0:
        return float(np.max(values)) if values.size else 0.0
    if nonzero.size == 0:
        return 0.0
    k = min(cnt, nonzero.size)
    top = np.partition(nonzero, -k)[-k:]
    return float(np.sum(top) / float(cnt))


def _abu_all_top(values: np.ndarray, frac: float) -> float:
    if values.size == 0:
        return 0.0
    cnt = int(math.floor(values.size * frac))
    if cnt <= 0:
        return float(np.max(values))
    k = min(cnt, values.size)
    top = np.partition(values, -k)[-k:]
    return float(np.mean(top))


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _ptr_double(arr: np.ndarray):
    return np.ascontiguousarray(arr, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _ptr_int(arr: np.ndarray):
    return np.ascontiguousarray(arr, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))


_NATIVE_LIB = None
_NATIVE_TRIED = False


def _load_native():
    global _NATIVE_LIB, _NATIVE_TRIED
    if _NATIVE_TRIED:
        return _NATIVE_LIB
    _NATIVE_TRIED = True
    src = os.path.join(os.path.dirname(__file__), "fast_proxy_native.cpp")
    if not os.path.exists(src):
        return None
    try:
        with open(src, "rb") as f:
            digest = hashlib.sha1(f.read()).hexdigest()[:16]
        suffix = "dylib" if platform.system() == "Darwin" else "so"
        out = os.path.join(tempfile.gettempdir(), f"ours_fast_proxy_native_{digest}.{suffix}")
        if not os.path.exists(out):
            cmd = ["g++", "-O3", "-std=c++17", "-fPIC", src, "-o", out]
            if platform.system() == "Darwin":
                cmd.insert(3, "-dynamiclib")
            else:
                cmd.insert(3, "-shared")
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=30,
            )
        lib = ctypes.CDLL(out)
        lib.score_proxy_native.restype = ctypes.c_int
        lib.score_proxy_native.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        _NATIVE_LIB = lib
    except Exception:
        _NATIVE_LIB = None
    return _NATIVE_LIB
