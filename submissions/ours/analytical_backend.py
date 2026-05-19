"""Optional analytical-placement candidate generation for the ours placer.

This backend is deliberately guarded by exact proxy scoring.  It can generate
large global moves from a smooth wirelength+density objective, but it only
replaces the incoming placement when the official proxy evaluator improves.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
from typing import Callable

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


@dataclass(frozen=True)
class _Config:
    seed: int
    perturb: float
    lambda0: float
    anchor_weight: float
    gamma_scale: float
    step_scale: float
    target_density_pad: float
    lambda_growth: float = 1.02
    anchor_decay: float = 0.995
    target_overflow: float = 0.08
    lambda_max: float = 1000.0
    congestion_weight: float = 0.0


def refine_with_analytical(
    baseline: torch.Tensor,
    benchmark: Benchmark,
    *,
    load_plc: Callable[[Benchmark], object | None],
    legalize_hard: Callable[..., torch.Tensor],
    is_valid: Callable[[torch.Tensor, Benchmark], bool],
) -> torch.Tensor:
    """Return a true-proxy-improving analytical candidate, or ``baseline``."""

    plc = load_plc(benchmark)
    if plc is None:
        _debug("no plc found")
        return baseline

    try:
        best = baseline.detach().clone().float()
        best_cost = float(compute_proxy_cost(best, benchmark, plc)["proxy_cost"])
    except Exception:
        _debug("could not score baseline")
        return baseline

    fast_scorer = _build_fast_scorer(benchmark, plc) if _env_bool("OURS_ANALYTICAL_FAST_SELECT", "0") else None
    if fast_scorer is not None:
        try:
            best_select_cost = float(fast_scorer.score(best)["proxy_cost"])
        except Exception:
            fast_scorer = None
            best_select_cost = best_cost
    else:
        best_select_cost = best_cost

    steps = _env_int("OURS_ANALYTICAL_STEPS", 800)
    if steps <= 0:
        return baseline
    grid = _env_int("OURS_ANALYTICAL_GRID", 96)
    starts = max(1, _env_int("OURS_ANALYTICAL_STARTS", 1))
    timeout = _env_float("OURS_ANALYTICAL_TIMEOUT", 240.0)
    start_time = time.monotonic()

    best_name = "baseline"
    for name, config in _configs()[:starts]:
        if time.monotonic() - start_time > timeout and name != "base":
            break
        try:
            candidate = _run_global_place(
                benchmark,
                config,
                steps=steps,
                grid=grid,
            )
            gap = _env_float("OURS_ANALYTICAL_GAP", 0.005)
            candidate = _spiral_legalize_hard(candidate, benchmark, gap=gap)
            if not is_valid(candidate, benchmark):
                candidate = legalize_hard(
                    candidate,
                    benchmark,
                    gap=gap,
                    max_rounds=_env_int("OURS_ANALYTICAL_LEGALIZE_ROUNDS", 500),
                )
            if not is_valid(candidate, benchmark):
                _debug(f"{name}: invalid after legalization")
                continue
            if _env_bool("OURS_ANALYTICAL_SOFT_REFINE", "0"):
                candidate = _refine_soft(candidate, benchmark, config, grid=grid)
            costs = (
                fast_scorer.score(candidate)
                if fast_scorer is not None
                else compute_proxy_cost(candidate, benchmark, plc)
            )
            if int(costs.get("overlap_count", 1)) != 0:
                _debug(f"{name}: overlap_count={costs.get('overlap_count')}")
                continue
            cost = float(costs["proxy_cost"])
            label = "fast" if fast_scorer is not None else "proxy"
            _debug(f"{name}: {label}={cost:.6f} best_select={best_select_cost:.6f}")
            if cost < best_select_cost:
                best_select_cost = cost
                best = candidate.detach().clone().float()
                best_name = name
        except Exception as exc:
            _debug(f"{name}: failed: {type(exc).__name__}")
            continue

    if fast_scorer is not None and not torch.allclose(best, baseline, atol=1e-7, rtol=0.0):
        try:
            exact = compute_proxy_cost(best, benchmark, plc)
        except Exception:
            return baseline
        if int(exact.get("overlap_count", 1)) != 0:
            return baseline
        exact_cost = float(exact["proxy_cost"])
        _debug(
            f"fast-selected={best_name} exact={exact_cost:.6f} "
            f"baseline={best_cost:.6f} fast={best_select_cost:.6f}"
        )
        if exact_cost < best_cost:
            return best
        _debug("fast-selected candidate rejected by exact score")
        return baseline

    if fast_scorer is None:
        best_cost = best_select_cost
    _debug(f"selected={best_name} proxy={best_cost:.6f}")
    return best


def _build_fast_scorer(benchmark: Benchmark, plc):
    try:
        from fast_proxy import build_fast_proxy
    except Exception:
        return None
    try:
        return build_fast_proxy(benchmark, plc)
    except Exception:
        return None


def _configs() -> list[tuple[str, _Config]]:
    raw = os.environ.get("OURS_ANALYTICAL_CONFIGS", "").strip()
    if raw:
        parsed: list[tuple[str, _Config]] = []
        for i, spec in enumerate(raw.split(";")):
            parts = [p.strip() for p in spec.split(",") if p.strip()]
            if len(parts) < 7:
                continue
            try:
                cfg = _Config(
                    seed=int(parts[0]),
                    perturb=float(parts[1]),
                    lambda0=float(parts[2]),
                    anchor_weight=float(parts[3]),
                    gamma_scale=float(parts[4]),
                    step_scale=float(parts[5]),
                    target_density_pad=float(parts[6]),
                    lambda_growth=float(parts[7]) if len(parts) > 7 else 1.02,
                    anchor_decay=float(parts[8]) if len(parts) > 8 else 0.995,
                    target_overflow=float(parts[9]) if len(parts) > 9 else 0.08,
                    lambda_max=float(parts[10]) if len(parts) > 10 else 1000.0,
                    congestion_weight=float(parts[11]) if len(parts) > 11 else 0.0,
                )
            except ValueError:
                continue
            parsed.append((f"custom{i}", cfg))
        if parsed:
            return parsed

    return [
        ("base", _Config(17, 0.000, 0.010, 0.150, 4.00, 0.040, 0.000)),
        (
            "cong_light",
            _Config(17, 0.000, 0.010, 0.150, 4.00, 0.040, 0.000, 1.02, 0.995, 0.08, 1000.0, 0.010),
        ),
        (
            "route_relief",
            _Config(1259, 0.015, 0.0305, 0.050, 4.00, 0.0394, -0.030, 1.05, 0.998, 0.08, 1000.0),
        ),
        ("loose", _Config(23, 0.020, 0.010, 0.150, 4.00, 0.040, 0.000)),
        ("free", _Config(59, 0.030, 0.010, 0.000, 4.00, 0.040, 0.000)),
        ("low_lambda", _Config(41, 0.050, 0.005, 0.150, 4.00, 0.040, 0.000)),
        ("high_lambda", _Config(83, 0.050, 0.020, 0.150, 4.00, 0.040, 0.000)),
        ("slow", _Config(101, 0.030, 0.010, 0.150, 2.00, 0.020, 0.000)),
    ]


def _run_global_place(
    benchmark: Benchmark,
    config: _Config,
    *,
    steps: int,
    grid: int,
) -> torch.Tensor:
    torch.manual_seed(config.seed)

    device = _device()
    n_all = int(benchmark.num_macros)
    n_hard = int(benchmark.num_hard_macros)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    diag = math.hypot(cw, ch)

    sizes = benchmark.macro_sizes.detach().float().to(device)
    fixed = benchmark.macro_fixed.detach().to(device)
    movable = (~fixed).float().unsqueeze(1)
    anchor = benchmark.macro_positions.detach().float().to(device)
    pos = anchor.clone()

    if config.perturb > 0.0:
        noise = torch.randn_like(pos)
        pos = pos + float(config.perturb) * diag * noise * movable

    lower = 0.5 * sizes
    upper = torch.stack(
        [
            torch.full((n_all,), cw, device=device, dtype=torch.float32) - lower[:, 0],
            torch.full((n_all,), ch, device=device, dtype=torch.float32) - lower[:, 1],
        ],
        dim=1,
    )
    pos = _clamp(pos, lower, upper, anchor, fixed)
    y = pos.clone()

    pin_owner, pin_offset, pin_net, net_weight = _build_pin_tensors(benchmark, device)
    n_nets = int(net_weight.numel())
    ports = benchmark.port_positions.detach().float().to(device)
    if n_nets == 0:
        return benchmark.macro_positions.clone()

    nx, ny = _grid_shape(cw, ch, grid)
    bin_w = cw / float(nx)
    bin_h = ch / float(ny)
    target_density = _target_density(sizes, cw, ch, config.target_density_pad)
    gamma_start = max(diag * 0.02 * float(config.gamma_scale), 1e-6)
    gamma_end = max(diag * 0.002, 1e-6)

    wl_scale, den_scale = _initial_grad_scales(
        pos,
        sizes,
        ports,
        pin_owner,
        pin_offset,
        pin_net,
        net_weight,
        n_all,
        n_nets,
        cw,
        ch,
        nx,
        ny,
        target_density,
        gamma_start,
        movable,
    )
    density_lambda = float(config.lambda0) * wl_scale / den_scale
    anchor_lambda = float(config.anchor_weight) * wl_scale
    cong_weight = _env_float("OURS_ANALYTICAL_CONG_WEIGHT", config.congestion_weight)
    if cong_weight > 0.0:
        cong_scale = _initial_congestion_grad_scale(
            pos,
            ports,
            pin_owner,
            pin_offset,
            pin_net,
            net_weight,
            n_all,
            n_nets,
            cw,
            ch,
            int(benchmark.grid_cols),
            int(benchmark.grid_rows),
            gamma_start,
            movable,
        )
        congestion_lambda = cong_weight * wl_scale / cong_scale
    else:
        congestion_lambda = 0.0

    a = 1.0
    best_pos = pos.clone()
    best_score = float("inf")
    total_area = float((sizes[:n_hard, 0] * sizes[:n_hard, 1]).sum().item())

    for it in range(int(steps)):
        frac = float(it) / max(1.0, float(steps - 1))
        gamma = gamma_start * ((gamma_end / gamma_start) ** frac)
        req = y.detach().clone().requires_grad_(True)
        pin_pos = _pin_positions(req, ports, pin_owner, pin_offset, n_all)
        wl = _lse_wirelength(pin_pos, pin_net, n_nets, net_weight, gamma)
        den_energy, density = _density_energy(req, sizes, cw, ch, nx, ny, target_density)
        if congestion_lambda > 0.0:
            cong_energy = _congestion_energy(
                pin_pos,
                pin_net,
                n_nets,
                net_weight,
                cw,
                ch,
                int(benchmark.grid_cols),
                int(benchmark.grid_rows),
                gamma,
            )
        else:
            cong_energy = req.sum() * 0.0
        disp = (req - anchor) * movable
        anchor_loss = (disp * disp).sum()
        loss = (
            wl
            + density_lambda * den_energy
            + congestion_lambda * cong_energy
            + anchor_lambda * anchor_loss
        )
        (grad,) = torch.autograd.grad(loss, req)
        grad = grad * movable
        grad_norm = float(grad.norm().item()) + 1e-12
        step = float(config.step_scale) * diag / grad_norm

        pos_new = _clamp(y - step * grad, lower, upper, anchor, fixed)
        a_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * a * a))
        beta = (a - 1.0) / a_new
        y = _clamp(pos_new + beta * (pos_new - pos), lower, upper, anchor, fixed)
        pos = pos_new
        a = a_new

        with torch.no_grad():
            overflow = torch.clamp(density - target_density, min=0.0).sum()
            overflow = float((overflow * bin_w * bin_h / max(total_area, 1e-9)).item())
            smooth_score = float(wl.item()) * (
                1.0 + 5.0 * max(0.0, overflow - float(config.target_overflow))
            )
            if smooth_score < best_score:
                best_score = smooth_score
                best_pos = pos.detach().clone()

        if it > 5:
            density_lambda = min(
                density_lambda * float(config.lambda_growth),
                float(config.lambda_max),
            )
            anchor_lambda *= float(config.anchor_decay)

    result = benchmark.macro_positions.clone()
    result[:] = best_pos.detach().cpu()
    return result.float()


def _spiral_legalize_hard(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    gap: float,
) -> torch.Tensor:
    n_hard = int(benchmark.num_hard_macros)
    if n_hard <= 1:
        return placement

    out = placement.detach().clone().float()
    pos = out[:n_hard].cpu().numpy().astype(np.float64).copy()
    anchor = pos.copy()
    sizes = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed[:n_hard].cpu().numpy().astype(bool)
    movable = ~fixed
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    hw = 0.5 * sizes[:, 0]
    hh = 0.5 * sizes[:, 1]

    for i in range(n_hard):
        if fixed[i]:
            pos[i] = benchmark.macro_positions[i].cpu().numpy().astype(np.float64)
        pos[i, 0] = np.clip(pos[i, 0], hw[i] + gap, cw - hw[i] - gap)
        pos[i, 1] = np.clip(pos[i, 1], hh[i] + gap, ch - hh[i] - gap)

    sep_x = hw[:, None] + hw[None, :] + float(gap)
    sep_y = hh[:, None] + hh[None, :] + float(gap)
    order = sorted(range(n_hard), key=lambda i: -float(sizes[i, 0] * sizes[i, 1]))
    placed = np.zeros(n_hard, dtype=bool)
    legal = pos.copy()

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue
        if not _has_overlap_with_placed(legal[idx], idx, legal, placed, sep_x, sep_y):
            placed[idx] = True
            continue

        step = max(float(sizes[idx, 0]), float(sizes[idx, 1])) * 0.20
        best = legal[idx].copy()
        best_dist = float("inf")
        for radius in range(1, _env_int("OURS_ANALYTICAL_SPIRAL_RADIUS", 360)):
            found = False
            for dx_i in range(-radius, radius + 1):
                for dy_i in range(-radius, radius + 1):
                    if abs(dx_i) != radius and abs(dy_i) != radius:
                        continue
                    cand = np.array(
                        [
                            np.clip(legal[idx, 0] + dx_i * step, hw[idx] + gap, cw - hw[idx] - gap),
                            np.clip(legal[idx, 1] + dy_i * step, hh[idx] + gap, ch - hh[idx] - gap),
                        ],
                        dtype=np.float64,
                    )
                    if _has_overlap_with_placed(cand, idx, legal, placed, sep_x, sep_y):
                        continue
                    dist = float(np.sum((cand - anchor[idx]) ** 2))
                    if dist < best_dist:
                        best_dist = dist
                        best = cand
                        found = True
            if found:
                break
        legal[idx] = best
        placed[idx] = True

    out[:n_hard] = torch.tensor(legal, dtype=torch.float32)
    if bool(benchmark.macro_fixed.any()):
        out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
    return out


def _has_overlap_with_placed(
    point: np.ndarray,
    idx: int,
    legal: np.ndarray,
    placed: np.ndarray,
    sep_x: np.ndarray,
    sep_y: np.ndarray,
) -> bool:
    if not bool(placed.any()):
        return False
    dx = np.abs(float(point[0]) - legal[:, 0])
    dy = np.abs(float(point[1]) - legal[:, 1])
    overlaps = (dx < sep_x[idx]) & (dy < sep_y[idx]) & placed
    overlaps[idx] = False
    return bool(overlaps.any())


def _refine_soft(
    placement: torch.Tensor,
    benchmark: Benchmark,
    config: _Config,
    *,
    grid: int,
) -> torch.Tensor:
    n_all = int(benchmark.num_macros)
    n_hard = int(benchmark.num_hard_macros)
    if n_all <= n_hard:
        return placement

    iters = _env_int("OURS_ANALYTICAL_SOFT_ITERS", 120)
    if iters <= 0:
        return placement

    device = _device()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    diag = math.hypot(cw, ch)

    pos = placement.detach().clone().float().to(device)
    anchor = benchmark.macro_positions.detach().float().to(device)
    sizes = benchmark.macro_sizes.detach().float().to(device)
    fixed = benchmark.macro_fixed.detach().to(device)
    soft_mov = (~fixed).float().unsqueeze(1)
    soft_mov[:n_hard] = 0.0
    if float(soft_mov.sum().item()) <= 0.0:
        return placement

    lower = 0.5 * sizes
    upper = torch.stack(
        [
            torch.full((n_all,), cw, device=device, dtype=torch.float32) - lower[:, 0],
            torch.full((n_all,), ch, device=device, dtype=torch.float32) - lower[:, 1],
        ],
        dim=1,
    )

    pin_owner, pin_offset, pin_net, net_weight = _build_pin_tensors(benchmark, device)
    n_nets = int(net_weight.numel())
    if n_nets == 0:
        return placement
    ports = benchmark.port_positions.detach().float().to(device)
    nx, ny = _grid_shape(cw, ch, grid)
    target_density = _target_density(sizes, cw, ch, config.target_density_pad)
    gamma = max(diag * 0.002, 1e-6)

    req = pos.detach().clone().requires_grad_(True)
    pin_pos = _pin_positions(req, ports, pin_owner, pin_offset, n_all)
    wl_only = _lse_wirelength(pin_pos, pin_net, n_nets, net_weight, gamma)
    (wl_grad,) = torch.autograd.grad(wl_only, req)

    req = pos.detach().clone().requires_grad_(True)
    den_only, rho0 = _density_energy(req, sizes, cw, ch, nx, ny, target_density)
    (den_grad,) = torch.autograd.grad(den_only, req)

    mask = soft_mov.squeeze(1) > 0
    wl_norm = float(wl_grad[mask].norm().item()) + 1e-12
    den_norm = float(den_grad[mask].norm().item()) + 1e-12
    density_lambda = 0.2 * wl_norm / den_norm
    base_overflow = float(
        (
            torch.clamp(rho0 - target_density, min=0.0).sum()
            * (cw / float(nx))
            * (ch / float(ny))
        ).item()
    )

    y = pos.clone()
    a = 1.0
    best = pos.clone()
    best_wl = float("inf")
    anchor_lambda = _env_float("OURS_ANALYTICAL_SOFT_ANCHOR", 0.001)

    for _ in range(int(iters)):
        req = y.detach().clone().requires_grad_(True)
        pin_pos = _pin_positions(req, ports, pin_owner, pin_offset, n_all)
        wl = _lse_wirelength(pin_pos, pin_net, n_nets, net_weight, gamma)
        den_energy, density = _density_energy(req, sizes, cw, ch, nx, ny, target_density)
        disp = (req - anchor) * soft_mov
        loss = wl + density_lambda * den_energy + anchor_lambda * (disp * disp).sum()
        (grad,) = torch.autograd.grad(loss, req)
        grad = grad * soft_mov
        grad_norm = float(grad.norm().item()) + 1e-12
        step = 0.015 * diag / grad_norm

        pos_new = _clamp(y - step * grad, lower, upper, pos, soft_mov.squeeze(1) <= 0)
        a_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * a * a))
        beta = (a - 1.0) / a_new
        y = _clamp(pos_new + beta * (pos_new - pos), lower, upper, pos, soft_mov.squeeze(1) <= 0)
        pos = pos_new
        a = a_new

        with torch.no_grad():
            overflow = float(
                (
                    torch.clamp(density - target_density, min=0.0).sum()
                    * (cw / float(nx))
                    * (ch / float(ny))
                ).item()
            )
            cur_wl = float(wl.item())
            if overflow <= base_overflow * 1.15 and cur_wl < best_wl:
                best_wl = cur_wl
                best = pos.detach().clone()

    result = placement.detach().clone().float()
    result[:] = best.detach().cpu()
    result[:n_hard] = placement[:n_hard]
    if bool(benchmark.macro_fixed.any()):
        result[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
    return result


def _initial_grad_scales(
    pos: torch.Tensor,
    sizes: torch.Tensor,
    ports: torch.Tensor,
    pin_owner: torch.Tensor,
    pin_offset: torch.Tensor,
    pin_net: torch.Tensor,
    net_weight: torch.Tensor,
    n_all: int,
    n_nets: int,
    cw: float,
    ch: float,
    nx: int,
    ny: int,
    target_density: float,
    gamma: float,
    movable: torch.Tensor,
) -> tuple[float, float]:
    req = pos.detach().clone().requires_grad_(True)
    pin_pos = _pin_positions(req, ports, pin_owner, pin_offset, n_all)
    wl = _lse_wirelength(pin_pos, pin_net, n_nets, net_weight, gamma)
    (wl_grad,) = torch.autograd.grad(wl, req)

    req = pos.detach().clone().requires_grad_(True)
    den, _ = _density_energy(req, sizes, cw, ch, nx, ny, target_density)
    (den_grad,) = torch.autograd.grad(den, req)

    mask = movable.squeeze(1) > 0
    return (
        float(wl_grad[mask].norm().item()) + 1e-12,
        float(den_grad[mask].norm().item()) + 1e-12,
    )


def _initial_congestion_grad_scale(
    pos: torch.Tensor,
    ports: torch.Tensor,
    pin_owner: torch.Tensor,
    pin_offset: torch.Tensor,
    pin_net: torch.Tensor,
    net_weight: torch.Tensor,
    n_all: int,
    n_nets: int,
    cw: float,
    ch: float,
    nx: int,
    ny: int,
    gamma: float,
    movable: torch.Tensor,
) -> float:
    req = pos.detach().clone().requires_grad_(True)
    pin_pos = _pin_positions(req, ports, pin_owner, pin_offset, n_all)
    cong = _congestion_energy(pin_pos, pin_net, n_nets, net_weight, cw, ch, nx, ny, gamma)
    (grad,) = torch.autograd.grad(cong, req)
    mask = movable.squeeze(1) > 0
    return float(grad[mask].norm().item()) + 1e-12


def _build_pin_tensors(
    benchmark: Benchmark,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    owners: list[int] = []
    offsets: list[tuple[float, float]] = []
    nets: list[int] = []
    weights: list[float] = []

    source = benchmark.net_pin_nodes if benchmark.net_pin_nodes else []
    if source and _env_bool("OURS_ANALYTICAL_PIN_LEVEL", "0"):
        for net_idx, pins in enumerate(source):
            if pins.numel() < 4:
                continue
            net_id = len(weights)
            weights.append(float(benchmark.net_weights[net_idx]) if benchmark.net_weights.numel() else 1.0)
            for owner, slot in pins.tolist():
                owners.append(int(owner))
                offsets.append(_pin_offset(benchmark, int(owner), int(slot)))
                nets.append(net_id)
    else:
        for net_idx, nodes in enumerate(benchmark.net_nodes):
            if nodes.numel() < 2:
                continue
            net_id = len(weights)
            k = int(nodes.numel())
            weights.append(2.0 / max(k, 2))
            for owner in nodes.tolist():
                owners.append(int(owner))
                offsets.append((0.0, 0.0))
                nets.append(net_id)

    if not owners:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty((0, 2), dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

    return (
        torch.tensor(owners, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.float32, device=device),
        torch.tensor(nets, dtype=torch.long, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
    )


def _pin_offset(benchmark: Benchmark, owner: int, slot: int) -> tuple[float, float]:
    if 0 <= owner < int(benchmark.num_hard_macros):
        if owner < len(benchmark.macro_pin_offsets):
            pins = benchmark.macro_pin_offsets[owner]
            if 0 <= slot < int(pins.shape[0]):
                x, y = pins[slot].tolist()
                return float(x), float(y)
    return 0.0, 0.0


def _pin_positions(
    pos: torch.Tensor,
    ports: torch.Tensor,
    pin_owner: torch.Tensor,
    pin_offset: torch.Tensor,
    n_all: int,
) -> torch.Tensor:
    is_port = pin_owner >= n_all
    macro_idx = torch.where(is_port, torch.zeros_like(pin_owner), pin_owner)
    port_idx = torch.where(is_port, pin_owner - n_all, torch.zeros_like(pin_owner))
    macro_pos = pos[macro_idx] + pin_offset
    if ports.numel() == 0:
        port_pos = torch.zeros_like(macro_pos)
    else:
        port_idx = torch.clamp(port_idx, min=0, max=max(ports.shape[0] - 1, 0))
        port_pos = ports[port_idx]
    return torch.where(is_port.unsqueeze(1), port_pos, macro_pos)


def _lse_wirelength(
    pin_pos: torch.Tensor,
    pin_net: torch.Tensor,
    n_nets: int,
    net_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    if pin_pos.numel() == 0:
        return pin_pos.sum() * 0.0
    inv_gamma = 1.0 / max(float(gamma), 1e-6)
    x = pin_pos[:, 0] * inv_gamma
    y = pin_pos[:, 1] * inv_gamma
    with torch.no_grad():
        neg_inf = torch.full((n_nets,), -1e30, device=pin_pos.device, dtype=pin_pos.dtype)
        x_max = neg_inf.clone().scatter_reduce(0, pin_net, x.detach(), reduce="amax", include_self=True)
        x_min = neg_inf.clone().scatter_reduce(0, pin_net, (-x).detach(), reduce="amax", include_self=True)
        y_max = neg_inf.clone().scatter_reduce(0, pin_net, y.detach(), reduce="amax", include_self=True)
        y_min = neg_inf.clone().scatter_reduce(0, pin_net, (-y).detach(), reduce="amax", include_self=True)

    sxp = torch.zeros(n_nets, device=pin_pos.device, dtype=pin_pos.dtype).scatter_add(
        0, pin_net, torch.exp(x - x_max[pin_net])
    )
    sxn = torch.zeros(n_nets, device=pin_pos.device, dtype=pin_pos.dtype).scatter_add(
        0, pin_net, torch.exp(-x - x_min[pin_net])
    )
    syp = torch.zeros(n_nets, device=pin_pos.device, dtype=pin_pos.dtype).scatter_add(
        0, pin_net, torch.exp(y - y_max[pin_net])
    )
    syn = torch.zeros(n_nets, device=pin_pos.device, dtype=pin_pos.dtype).scatter_add(
        0, pin_net, torch.exp(-y - y_min[pin_net])
    )

    wl_x = float(gamma) * (torch.log(sxp + 1e-20) + x_max + torch.log(sxn + 1e-20) + x_min)
    wl_y = float(gamma) * (torch.log(syp + 1e-20) + y_max + torch.log(syn + 1e-20) + y_min)
    return (net_weight * (wl_x + wl_y)).sum()


def _smooth_net_bounds(
    pin_pos: torch.Tensor,
    pin_net: torch.Tensor,
    n_nets: int,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inv_gamma = 1.0 / max(float(gamma), 1e-6)
    x = pin_pos[:, 0] * inv_gamma
    y = pin_pos[:, 1] * inv_gamma
    with torch.no_grad():
        neg_inf = torch.full((n_nets,), -1e30, device=pin_pos.device, dtype=pin_pos.dtype)
        x_max = neg_inf.clone().scatter_reduce(0, pin_net, x.detach(), reduce="amax", include_self=True)
        x_min = neg_inf.clone().scatter_reduce(0, pin_net, (-x).detach(), reduce="amax", include_self=True)
        y_max = neg_inf.clone().scatter_reduce(0, pin_net, y.detach(), reduce="amax", include_self=True)
        y_min = neg_inf.clone().scatter_reduce(0, pin_net, (-y).detach(), reduce="amax", include_self=True)

    sxp = torch.zeros(n_nets, device=pin_pos.device, dtype=pin_pos.dtype).scatter_add(
        0, pin_net, torch.exp(x - x_max[pin_net])
    )
    sxn = torch.zeros(n_nets, device=pin_pos.device, dtype=pin_pos.dtype).scatter_add(
        0, pin_net, torch.exp(-x - x_min[pin_net])
    )
    syp = torch.zeros(n_nets, device=pin_pos.device, dtype=pin_pos.dtype).scatter_add(
        0, pin_net, torch.exp(y - y_max[pin_net])
    )
    syn = torch.zeros(n_nets, device=pin_pos.device, dtype=pin_pos.dtype).scatter_add(
        0, pin_net, torch.exp(-y - y_min[pin_net])
    )

    x_hi = float(gamma) * (torch.log(sxp + 1e-20) + x_max)
    x_lo = -float(gamma) * (torch.log(sxn + 1e-20) + x_min)
    y_hi = float(gamma) * (torch.log(syp + 1e-20) + y_max)
    y_lo = -float(gamma) * (torch.log(syn + 1e-20) + y_min)
    return x_lo, x_hi, y_lo, y_hi


def _congestion_energy(
    pin_pos: torch.Tensor,
    pin_net: torch.Tensor,
    n_nets: int,
    net_weight: torch.Tensor,
    cw: float,
    ch: float,
    nx: int,
    ny: int,
    gamma: float,
) -> torch.Tensor:
    if pin_pos.numel() == 0 or n_nets == 0 or nx <= 0 or ny <= 0:
        return pin_pos.sum() * 0.0
    x_lo, x_hi, y_lo, y_hi = _smooth_net_bounds(pin_pos, pin_net, n_nets, gamma)
    bin_w = cw / float(nx)
    bin_h = ch / float(ny)
    grid_x = (torch.arange(nx, device=pin_pos.device, dtype=pin_pos.dtype) + 0.5) * bin_w
    grid_y = (torch.arange(ny, device=pin_pos.device, dtype=pin_pos.dtype) + 0.5) * bin_h
    sigma = max(bin_w, bin_h) * _env_float("OURS_ANALYTICAL_CONG_SIGMA", 0.75)

    in_x = torch.sigmoid((x_hi[:, None] - grid_x[None, :]) / sigma) * torch.sigmoid(
        (grid_x[None, :] - x_lo[:, None]) / sigma
    )
    in_y = torch.sigmoid((y_hi[:, None] - grid_y[None, :]) / sigma) * torch.sigmoid(
        (grid_y[None, :] - y_lo[:, None]) / sigma
    )
    width = (x_hi - x_lo).clamp_min(1e-6)
    height = (y_hi - y_lo).clamp_min(1e-6)
    if _env_bool("OURS_ANALYTICAL_CONG_RUDY", "1"):
        demand_weight = net_weight * (width + height) / (width * height)
    else:
        demand_weight = net_weight
    demand = torch.einsum("n,nr,nc->rc", demand_weight, in_y, in_x)
    flat = demand.reshape(-1)
    top_k = max(1, int(flat.numel() * _env_float("OURS_ANALYTICAL_CONG_TOP_PCT", 0.05)))
    values, _ = torch.topk(flat, top_k)
    return values.mean() / max(float(n_nets), 1.0)


def _density_energy(
    pos: torch.Tensor,
    sizes: torch.Tensor,
    cw: float,
    ch: float,
    nx: int,
    ny: int,
    target_density: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    density = _rect_density(pos, sizes, cw, ch, nx, ny)
    overflow = density - float(target_density)
    overflow = overflow - overflow.mean()
    psi = _poisson(overflow, cw / float(nx), ch / float(ny))
    return 0.5 * (overflow * psi).sum() * (cw / float(nx)) * (ch / float(ny)), density


def _rect_density(
    pos: torch.Tensor,
    sizes: torch.Tensor,
    cw: float,
    ch: float,
    nx: int,
    ny: int,
) -> torch.Tensor:
    bin_w = cw / float(nx)
    bin_h = ch / float(ny)
    x0 = torch.arange(nx, device=pos.device, dtype=pos.dtype) * bin_w
    y0 = torch.arange(ny, device=pos.device, dtype=pos.dtype) * bin_h
    lx = pos[:, 0:1] - 0.5 * sizes[:, 0:1]
    ux = pos[:, 0:1] + 0.5 * sizes[:, 0:1]
    ly = pos[:, 1:2] - 0.5 * sizes[:, 1:2]
    uy = pos[:, 1:2] + 0.5 * sizes[:, 1:2]
    ox = torch.clamp(torch.minimum(ux, x0 + bin_w) - torch.maximum(lx, x0), min=0.0)
    oy = torch.clamp(torch.minimum(uy, y0 + bin_h) - torch.maximum(ly, y0), min=0.0)
    return torch.einsum("mi,mj->ij", oy, ox) / max(bin_w * bin_h, 1e-9)


def _poisson(rho: torch.Tensor, bin_w: float, bin_h: float) -> torch.Tensor:
    ny, nx = rho.shape
    rho_hat = torch.fft.rfft2(rho)
    ky = torch.fft.fftfreq(ny, d=bin_h, dtype=rho.dtype, device=rho.device) * (2.0 * math.pi)
    kx = torch.fft.rfftfreq(nx, d=bin_w, dtype=rho.dtype, device=rho.device) * (2.0 * math.pi)
    k2 = ky.unsqueeze(1).square() + kx.unsqueeze(0).square()
    psi_hat = rho_hat / torch.where(k2 == 0.0, torch.ones_like(k2), k2)
    psi_hat[0, 0] = 0.0
    return torch.fft.irfft2(psi_hat, s=(ny, nx))


def _target_density(sizes: torch.Tensor, cw: float, ch: float, pad: float) -> float:
    area = float((sizes[:, 0] * sizes[:, 1]).sum().item())
    return min(1.0, max(0.05, area / max(cw * ch, 1e-9) + float(pad)))


def _grid_shape(cw: float, ch: float, target: int) -> tuple[int, int]:
    target = max(32, int(target))
    nx = max(32, int(round(target * math.sqrt(cw / max(ch, 1e-9)))))
    ny = max(32, int(round(target * math.sqrt(ch / max(cw, 1e-9)))))
    return nx, ny


def _clamp(
    pos: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    anchor: torch.Tensor,
    fixed: torch.Tensor,
) -> torch.Tensor:
    clamped = torch.minimum(torch.maximum(pos, lower), upper)
    return torch.where(fixed.unsqueeze(1), anchor, clamped)


def _device() -> torch.device:
    if _env_bool("OURS_ANALYTICAL_GPU", "1") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return float(default)


def _debug(message: str) -> None:
    if _env_bool("OURS_ANALYTICAL_DEBUG", "0"):
        print(f"[OURS_ANALYTICAL] {message}", flush=True)
