"""Batched analytical candidate generation.

This is an opt-in global search path.  It runs several diverse smooth global
placement starts in one torch optimization, then legalizes and selects the best
candidate with the native proxy scorer before the official exact guard.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from typing import Callable

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


def refine_with_batch_analytical(
    baseline: torch.Tensor,
    benchmark: Benchmark,
    *,
    load_plc: Callable[[Benchmark], object | None],
    legalize_hard: Callable[..., torch.Tensor],
    is_valid: Callable[[torch.Tensor, Benchmark], bool],
) -> torch.Tensor:
    plc = load_plc(benchmark)
    if plc is None:
        return baseline

    try:
        helper_dir = Path(__file__).resolve().parent
        if str(helper_dir) not in sys.path:
            sys.path.insert(0, str(helper_dir))
        from analytical_backend import (
            _build_pin_tensors,
            _grid_shape,
            _initial_grad_scales,
            _spiral_legalize_hard,
            _target_density,
        )
        from fast_proxy import build_fast_proxy
    except Exception as exc:
        _debug(f"import failed: {type(exc).__name__}")
        return baseline

    scorer = build_fast_proxy(benchmark, plc)
    if scorer is None:
        return baseline

    try:
        base_exact = compute_proxy_cost(baseline.detach().float(), benchmark, plc)
        if int(base_exact.get("overlap_count", 1)) != 0:
            return baseline
        base_cost = float(base_exact["proxy_cost"])
    except Exception:
        return baseline

    k_count = max(1, _env_int("OURS_BATCH_K", 6))
    steps = max(0, _env_int("OURS_BATCH_STEPS", 220))
    if steps <= 0:
        return baseline

    device = _device()
    n_all = int(benchmark.num_macros)
    n_hard = int(benchmark.num_hard_macros)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    diag = math.hypot(cw, ch)

    sizes = benchmark.macro_sizes.detach().float().to(device)
    fixed = benchmark.macro_fixed.detach().to(device)
    movable = (~fixed).float().unsqueeze(0).unsqueeze(2)
    anchor = baseline.detach().float().to(device)
    lower = 0.5 * sizes
    upper = torch.stack(
        [
            torch.full((n_all,), cw, device=device, dtype=torch.float32) - lower[:, 0],
            torch.full((n_all,), ch, device=device, dtype=torch.float32) - lower[:, 1],
        ],
        dim=1,
    )

    starts = _initial_batch(anchor, benchmark, k_count, device=device)
    pos = starts.detach().clone()
    y = pos.clone()
    accel = 1.0

    pin_owner, pin_offset, pin_net, net_weight = _build_pin_tensors(benchmark, device)
    n_nets = int(net_weight.numel())
    if n_nets == 0:
        return baseline
    ports = benchmark.port_positions.detach().float().to(device)
    grid = _env_int("OURS_BATCH_GRID", 72)
    nx, ny = _grid_shape(cw, ch, grid)
    target_density = _target_density(
        sizes,
        cw,
        ch,
        _env_float("OURS_BATCH_TARGET_DENSITY_PAD", 0.0),
    )
    gamma_start = max(diag * _env_float("OURS_BATCH_GAMMA_START", 0.08), 1e-6)
    gamma_end = max(diag * _env_float("OURS_BATCH_GAMMA_END", 0.004), 1e-6)

    wl_scale, den_scale = _initial_grad_scales(
        starts[0],
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
        (~fixed).float().unsqueeze(1),
    )
    density_lambda = _env_float("OURS_BATCH_DENSITY", 0.020) * wl_scale / den_scale
    anchor_lambda0 = _env_float("OURS_BATCH_ANCHOR", 0.020) * wl_scale
    overlap_lambda = _env_float("OURS_BATCH_OVERLAP", 40.0) * wl_scale
    snapshots: list[torch.Tensor] = []
    snap_period = max(1, _env_int("OURS_BATCH_SNAPSHOT_PERIOD", max(40, steps // 4)))

    for it in range(steps):
        frac = float(it) / max(1.0, float(steps - 1))
        gamma = gamma_start * ((gamma_end / gamma_start) ** frac)
        anchor_lambda = anchor_lambda0 * (
            _env_float("OURS_BATCH_ANCHOR_DECAY", 0.985) ** float(it)
        )
        req = y.detach().clone().requires_grad_(True)
        pin_pos = _pin_positions_batch(req, ports, pin_owner, pin_offset, n_all)
        wl = _lse_wirelength_batch(pin_pos, pin_net, n_nets, net_weight, gamma)
        den_energy, _ = _density_energy_batch(req, sizes, cw, ch, nx, ny, target_density)
        disp = (req - anchor.unsqueeze(0)) * movable
        ov = _hard_overlap_energy_batch(req, sizes, n_hard, cw, ch)
        total = (
            wl
            + density_lambda * den_energy
            + anchor_lambda * (disp * disp).sum(dim=(1, 2))
            + overlap_lambda * ov
        ).mean()
        if not torch.isfinite(total):
            break
        (grad,) = torch.autograd.grad(total, req)
        with torch.no_grad():
            grad = grad * movable
            norm = grad.reshape(k_count, -1).norm(dim=1).clamp_min(1e-12)
            step = _env_float("OURS_BATCH_STEP_SCALE", 0.040) * diag / norm
            pos_new = req - step.view(k_count, 1, 1) * grad
            pos_new = torch.minimum(torch.maximum(pos_new, lower.unsqueeze(0)), upper.unsqueeze(0))
            pos_new = torch.where(fixed.view(1, n_all, 1), anchor.view(1, n_all, 2), pos_new)
            accel_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * accel * accel))
            beta = (accel - 1.0) / accel_new
            y = pos_new + beta * (pos_new - pos)
            y = torch.minimum(torch.maximum(y, lower.unsqueeze(0)), upper.unsqueeze(0))
            y = torch.where(fixed.view(1, n_all, 1), anchor.view(1, n_all, 2), y)
            pos = pos_new
            accel = accel_new
            if it > 5:
                density_lambda = min(
                    density_lambda * _env_float("OURS_BATCH_DENSITY_GROWTH", 1.02),
                    _env_float("OURS_BATCH_DENSITY_MAX", 1000.0),
                )

        if (it + 1) % snap_period == 0 or it == steps - 1:
            snapshots.append(pos.detach().cpu().clone())

    if not snapshots:
        snapshots.append(pos.detach().cpu().clone())

    best = baseline.detach().clone().float()
    best_fast = float(scorer.score(best)["proxy_cost"])
    ranked_candidates: list[tuple[float, int, int, torch.Tensor]] = []
    overlap_penalty = _env_float("OURS_BATCH_PREFILTER_OVERLAP_PENALTY", 0.03)
    for snap_idx, snap in enumerate(snapshots):
        for k in range(snap.shape[0]):
            candidate = snap[k].detach().clone().float()
            try:
                score = scorer.score(candidate)
            except Exception:
                continue
            penalized = float(score["proxy_cost"]) + overlap_penalty * float(
                score.get("overlap_count", 0)
            )
            ranked_candidates.append((penalized, snap_idx, k, candidate))
    ranked_candidates.sort(key=lambda item: item[0])

    eval_topk = max(1, _env_int("OURS_BATCH_EVAL_TOPK", 8))
    checked = 0
    for _, _, _, candidate_in in ranked_candidates[:eval_topk]:
        candidate = candidate_in.detach().clone().float()
        candidate = _spiral_legalize_hard(
            candidate,
            benchmark,
            gap=_env_float("OURS_BATCH_GAP", _env_float("OURS_GAP", 0.005)),
        )
        try:
            candidate = legalize_hard(
                candidate,
                benchmark,
                gap=_env_float("OURS_BATCH_GAP", _env_float("OURS_GAP", 0.005)),
                max_rounds=_env_int("OURS_BATCH_LEGALIZE_ROUNDS", 500),
            )
        except TypeError:
            candidate = legalize_hard(candidate, benchmark)
        if not is_valid(candidate, benchmark):
            continue
        checked += 1
        try:
            score = scorer.score(candidate)
        except Exception:
            continue
        if int(score.get("overlap_count", 1)) != 0:
            continue
        cost = float(score["proxy_cost"])
        if cost < best_fast:
            best_fast = cost
            best = candidate

    _debug(
        f"prefiltered={len(ranked_candidates)} eval_topk={eval_topk} "
        f"checked={checked} best_fast={best_fast:.6f} base_exact={base_cost:.6f}"
    )
    if torch.allclose(best, baseline, atol=1e-7, rtol=0.0):
        return baseline

    try:
        best_exact = compute_proxy_cost(best.detach().float(), benchmark, plc)
    except Exception:
        return baseline
    if int(best_exact.get("overlap_count", 1)) != 0:
        return baseline
    eps = _env_float("OURS_BATCH_EPS", 1e-6)
    if float(best_exact["proxy_cost"]) + eps < base_cost:
        _debug(f"accepted exact {base_cost:.6f}->{float(best_exact['proxy_cost']):.6f}")
        return best.detach().clone().float()
    _debug(f"rejected exact {base_cost:.6f}->{float(best_exact['proxy_cost']):.6f}")
    return baseline


def _initial_batch(
    anchor: torch.Tensor,
    benchmark: Benchmark,
    k_count: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    starts = anchor.detach().clone().repeat(k_count, 1, 1).to(device)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    center = torch.tensor([0.5 * cw, 0.5 * ch], dtype=torch.float32, device=device)
    fixed = benchmark.macro_fixed.detach().to(device)
    movable = ~fixed
    sizes = benchmark.macro_sizes.detach().float().to(device)
    lower = 0.5 * sizes
    upper = torch.stack(
        [
            torch.full((int(benchmark.num_macros),), cw, device=device) - lower[:, 0],
            torch.full((int(benchmark.num_macros),), ch, device=device) - lower[:, 1],
        ],
        dim=1,
    )
    torch.manual_seed(_env_int("OURS_BATCH_SEED", 20260519))
    diag = math.hypot(cw, ch)

    for k in range(k_count):
        cur = starts[k]
        mode = k % 8
        if mode in {1, 3, 5, 7}:
            cur[movable, 0] = cw - cur[movable, 0]
        if mode in {2, 3, 6, 7}:
            cur[movable, 1] = ch - cur[movable, 1]
        if mode in {4, 5}:
            scale = torch.tensor([1.10, 1.00], device=device)
            cur[movable] = center + (cur[movable] - center) * scale
        if mode in {6, 7}:
            scale = torch.tensor([1.00, 1.10], device=device)
            cur[movable] = center + (cur[movable] - center) * scale
        sigma = _env_float("OURS_BATCH_PERTURB", 0.020) * diag * (0.4 + 0.2 * (k % 5))
        if sigma > 0.0 and k > 0:
            cur[movable] += torch.randn_like(cur[movable]) * sigma
        clamped = torch.minimum(torch.maximum(cur, lower), upper)
        starts[k] = torch.where(fixed.unsqueeze(1), anchor, clamped)
    return starts


def _hard_overlap_energy(
    pos: torch.Tensor,
    sizes: torch.Tensor,
    n_hard: int,
    cw: float,
    ch: float,
) -> torch.Tensor:
    if n_hard <= 1:
        return pos.sum() * 0.0
    hard = pos[:n_hard]
    hs = sizes[:n_hard]
    dx = hard[:, 0:1] - hard[:, 0].unsqueeze(0)
    dy = hard[:, 1:2] - hard[:, 1].unsqueeze(0)
    sep_x = 0.5 * (hs[:, 0:1] + hs[:, 0].unsqueeze(0))
    sep_y = 0.5 * (hs[:, 1:2] + hs[:, 1].unsqueeze(0))
    ov = torch.relu(sep_x - dx.abs()) * torch.relu(sep_y - dy.abs())
    upper = torch.triu(torch.ones_like(ov, dtype=torch.bool), diagonal=1)
    return ov[upper].sum() / max(cw * ch, 1e-9)


def _pin_positions_batch(
    pos: torch.Tensor,
    ports: torch.Tensor,
    pin_owner: torch.Tensor,
    pin_offset: torch.Tensor,
    n_all: int,
) -> torch.Tensor:
    is_port = pin_owner >= n_all
    macro_idx = torch.where(is_port, torch.zeros_like(pin_owner), pin_owner)
    port_idx = torch.where(is_port, pin_owner - n_all, torch.zeros_like(pin_owner))
    macro_pos = pos[:, macro_idx, :] + pin_offset.unsqueeze(0)
    if ports.numel() == 0:
        port_pos = torch.zeros_like(macro_pos)
    else:
        port_idx = torch.clamp(port_idx, min=0, max=max(ports.shape[0] - 1, 0))
        port_pos = ports[port_idx].unsqueeze(0).expand(pos.shape[0], -1, -1)
    return torch.where(is_port.view(1, -1, 1), port_pos, macro_pos)


def _lse_wirelength_batch(
    pin_pos: torch.Tensor,
    pin_net: torch.Tensor,
    n_nets: int,
    net_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    if pin_pos.numel() == 0:
        return pin_pos.sum(dim=(1, 2)) * 0.0
    k_count = int(pin_pos.shape[0])
    p_count = int(pin_pos.shape[1])
    inv_gamma = 1.0 / max(float(gamma), 1e-6)
    offsets = torch.arange(k_count, device=pin_pos.device, dtype=torch.long).view(k_count, 1)
    flat_net = (pin_net.view(1, p_count) + offsets * int(n_nets)).reshape(-1)
    flat_size = k_count * int(n_nets)
    x = (pin_pos[:, :, 0] * inv_gamma).reshape(-1)
    y = (pin_pos[:, :, 1] * inv_gamma).reshape(-1)
    with torch.no_grad():
        neg_inf = torch.full((flat_size,), -1e30, device=pin_pos.device, dtype=pin_pos.dtype)
        x_max = neg_inf.clone().scatter_reduce(0, flat_net, x.detach(), reduce="amax", include_self=True)
        x_min = neg_inf.clone().scatter_reduce(0, flat_net, (-x).detach(), reduce="amax", include_self=True)
        y_max = neg_inf.clone().scatter_reduce(0, flat_net, y.detach(), reduce="amax", include_self=True)
        y_min = neg_inf.clone().scatter_reduce(0, flat_net, (-y).detach(), reduce="amax", include_self=True)

    sxp = torch.zeros(flat_size, device=pin_pos.device, dtype=pin_pos.dtype).scatter_add(
        0, flat_net, torch.exp(x - x_max[flat_net])
    )
    sxn = torch.zeros(flat_size, device=pin_pos.device, dtype=pin_pos.dtype).scatter_add(
        0, flat_net, torch.exp(-x - x_min[flat_net])
    )
    syp = torch.zeros(flat_size, device=pin_pos.device, dtype=pin_pos.dtype).scatter_add(
        0, flat_net, torch.exp(y - y_max[flat_net])
    )
    syn = torch.zeros(flat_size, device=pin_pos.device, dtype=pin_pos.dtype).scatter_add(
        0, flat_net, torch.exp(-y - y_min[flat_net])
    )

    wl_x = float(gamma) * (torch.log(sxp + 1e-20) + x_max + torch.log(sxn + 1e-20) + x_min)
    wl_y = float(gamma) * (torch.log(syp + 1e-20) + y_max + torch.log(syn + 1e-20) + y_min)
    weights = net_weight.repeat(k_count)
    return (weights * (wl_x + wl_y)).reshape(k_count, int(n_nets)).sum(dim=1)


def _density_energy_batch(
    pos: torch.Tensor,
    sizes: torch.Tensor,
    cw: float,
    ch: float,
    nx: int,
    ny: int,
    target_density: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    density = _rect_density_batch(pos, sizes, cw, ch, nx, ny)
    overflow = density - float(target_density)
    overflow = overflow - overflow.mean(dim=(1, 2), keepdim=True)
    psi = _poisson_batch(overflow, cw / float(nx), ch / float(ny))
    energy = 0.5 * (overflow * psi).sum(dim=(1, 2)) * (cw / float(nx)) * (ch / float(ny))
    return energy, density


def _rect_density_batch(
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
    lx = pos[:, :, 0:1] - 0.5 * sizes[:, 0].view(1, -1, 1)
    ux = pos[:, :, 0:1] + 0.5 * sizes[:, 0].view(1, -1, 1)
    ly = pos[:, :, 1:2] - 0.5 * sizes[:, 1].view(1, -1, 1)
    uy = pos[:, :, 1:2] + 0.5 * sizes[:, 1].view(1, -1, 1)
    ox = torch.clamp(torch.minimum(ux, x0.view(1, 1, nx) + bin_w) - torch.maximum(lx, x0.view(1, 1, nx)), min=0.0)
    oy = torch.clamp(torch.minimum(uy, y0.view(1, 1, ny) + bin_h) - torch.maximum(ly, y0.view(1, 1, ny)), min=0.0)
    return torch.einsum("kmy,kmx->kyx", oy, ox) / max(bin_w * bin_h, 1e-9)


def _poisson_batch(rho: torch.Tensor, bin_w: float, bin_h: float) -> torch.Tensor:
    _, ny, nx = rho.shape
    rho_hat = torch.fft.rfft2(rho)
    ky = torch.fft.fftfreq(ny, d=bin_h, dtype=rho.dtype, device=rho.device) * (2.0 * math.pi)
    kx = torch.fft.rfftfreq(nx, d=bin_w, dtype=rho.dtype, device=rho.device) * (2.0 * math.pi)
    k2 = ky.unsqueeze(1).square() + kx.unsqueeze(0).square()
    psi_hat = rho_hat / torch.where(k2 == 0.0, torch.ones_like(k2), k2)
    psi_hat[:, 0, 0] = 0.0
    return torch.fft.irfft2(psi_hat, s=(ny, nx))


def _hard_overlap_energy_batch(
    pos: torch.Tensor,
    sizes: torch.Tensor,
    n_hard: int,
    cw: float,
    ch: float,
) -> torch.Tensor:
    if n_hard <= 1:
        return pos.sum(dim=(1, 2)) * 0.0
    hard = pos[:, :n_hard, :]
    hs = sizes[:n_hard]
    dx = hard[:, :, 0:1] - hard[:, :, 0].unsqueeze(1)
    dy = hard[:, :, 1:2] - hard[:, :, 1].unsqueeze(1)
    sep_x = 0.5 * (hs[:, 0:1] + hs[:, 0].unsqueeze(0))
    sep_y = 0.5 * (hs[:, 1:2] + hs[:, 1].unsqueeze(0))
    ov = torch.relu(sep_x.unsqueeze(0) - dx.abs()) * torch.relu(sep_y.unsqueeze(0) - dy.abs())
    upper = torch.triu(torch.ones((n_hard, n_hard), device=pos.device, dtype=torch.bool), diagonal=1)
    return ov[:, upper].sum(dim=1) / max(cw * ch, 1e-9)


def _device() -> torch.device:
    if _env_bool("OURS_BATCH_GPU", "1") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _env(name: str, default: str | int | float) -> str:
    return os.environ.get(name, str(default))


def _env_bool(name: str, default: str) -> bool:
    return _env(name, default).strip().lower() not in {"0", "false", "no", "off"}


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


def _debug(message: str) -> None:
    if _env_bool("OURS_BATCH_DEBUG", "0"):
        print(f"[OURS_BATCH] {message}", flush=True)
