"""Optional RePlAce-backed candidate generation for the ours placer.

This module is intentionally opt-in.  It exports the challenge benchmark to a
Bookshelf handoff, runs the bundled Linux RePlAce binary, imports any placement
files it produced, legalizes hard macros, and keeps a candidate only if the
official proxy evaluator says it improves on the input placement.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class _Export:
    name: str
    directory: Path
    metadata_path: Path


@dataclass(frozen=True)
class _Config:
    density: float
    pcofmax: float
    extra_args: tuple[str, ...] = ()

    def args(self) -> list[str]:
        return [
            "-den",
            _fmt(self.density),
            "-pcofmax",
            _fmt(self.pcofmax),
            *self.extra_args,
        ]


def refine_with_replace(
    baseline: torch.Tensor,
    benchmark: Benchmark,
    *,
    load_plc: Callable[[Benchmark], object | None],
    legalize_hard: Callable[..., torch.Tensor],
    is_valid: Callable[[torch.Tensor, Benchmark], bool],
) -> torch.Tensor:
    """Return a true-proxy-improving RePlAce candidate, or ``baseline``."""

    plc = load_plc(benchmark)
    if plc is None:
        return baseline

    binary = Path(os.environ.get(
        "OURS_REPLACE_BINARY",
        "external/MacroPlacement/Flows/util/RePlAceFlow/RePlAce-static",
    ))
    if not binary.is_absolute():
        binary = (Path.cwd() / binary).resolve()
    if not binary.exists():
        return baseline

    try:
        best = baseline.detach().clone().float()
        best_cost = float(compute_proxy_cost(best, benchmark, plc)["proxy_cost"])
    except Exception:
        return baseline

    work_root = Path(os.environ.get("OURS_REPLACE_WORKDIR", tempfile.gettempdir()))
    run_root = work_root / "macro_place_ours_replace" / _safe_name(str(benchmark.name)) / str(os.getpid())
    bs_name = _safe_name(str(benchmark.name))

    try:
        export = _write_bookshelf(
            benchmark,
            plc,
            run_root / "ETC" / bs_name,
            bs_name=bs_name,
            scale=_env_int("OURS_REPLACE_SCALE", 1000),
            initial_placement=best,
        )
    except Exception:
        return baseline

    configs = _configs()
    timeout = _env_float("OURS_REPLACE_TIMEOUT", 180.0)
    stop_after_first = _env_bool("OURS_REPLACE_STOP_AFTER_FIRST", "1")
    max_candidates = _env_int("OURS_REPLACE_MAX_CANDIDATES", 4)
    tried = 0

    for config in configs:
        if tried >= max_candidates:
            break
        try:
            pl_paths = _run_replace(
                export,
                config,
                binary=binary,
                timeout=timeout,
                stop_after_first=stop_after_first,
            )
        except Exception:
            continue

        for pl_path in pl_paths:
            if tried >= max_candidates:
                break
            tried += 1
            try:
                candidate = _import_bookshelf(pl_path, export.metadata_path, benchmark)
                candidate = legalize_hard(
                    candidate,
                    benchmark,
                    gap=_env_float("OURS_REPLACE_GAP", 0.005),
                    max_rounds=_env_int("OURS_REPLACE_LEGALIZE_ROUNDS", 800),
                )
                if not is_valid(candidate, benchmark):
                    continue
                cost = float(compute_proxy_cost(candidate, benchmark, plc)["proxy_cost"])
            except Exception:
                continue
            if cost < best_cost:
                best = candidate.detach().clone().float()
                best_cost = cost

    return best


def _configs() -> list[_Config]:
    raw = os.environ.get("OURS_REPLACE_CONFIGS", "").strip()
    if raw:
        out: list[_Config] = []
        for spec in raw.split(";"):
            parts = [part.strip() for part in spec.split(",") if part.strip()]
            if len(parts) < 2:
                continue
            extra = tuple(parts[2:])
            out.append(_Config(float(parts[0]), float(parts[1]), extra))
        if out:
            return out
    return [
        _Config(0.72, 1.03),
        _Config(0.80, 1.03),
        _Config(0.80, 1.03, ("-bin", "64")),
    ]


def _write_bookshelf(
    benchmark: Benchmark,
    plc,
    output_dir: Path,
    *,
    bs_name: str,
    scale: int,
    initial_placement: torch.Tensor,
) -> _Export:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(benchmark, scale)
    row_h = rows[0]["height"]
    nodes, name_to_bs = _nodes(
        benchmark,
        plc,
        scale,
        soft_cell_height=row_h,
        initial_placement=initial_placement,
    )
    nets = _nets(plc, name_to_bs, scale)

    _write_text(output_dir / f"{bs_name}.aux", _aux(bs_name))
    _write_text(output_dir / f"{bs_name}.nodes", _nodes_text(nodes))
    _write_text(output_dir / f"{bs_name}.nets", _nets_text(nets))
    _write_text(output_dir / f"{bs_name}.wts", _wts_text(nets))
    _write_text(output_dir / f"{bs_name}.pl", _pl_text(nodes))
    _write_text(output_dir / f"{bs_name}.scl", _scl_text(rows))
    _write_text(output_dir / f"{bs_name}.shapes", "UCLA shapes 1.0\n\nNumNonRectangularNodes : 0\n")
    _write_text(output_dir / f"{bs_name}.route", _route_text(benchmark, scale))

    metadata = {
        "scale": int(scale),
        "num_macros": int(benchmark.num_macros),
        "nodes": nodes,
    }
    metadata_path = output_dir / f"{bs_name}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n")
    return _Export(name=bs_name, directory=output_dir, metadata_path=metadata_path)


def _nodes(
    benchmark: Benchmark,
    plc,
    scale: int,
    *,
    soft_cell_height: int,
    initial_placement: torch.Tensor,
) -> tuple[list[dict], dict[str, str]]:
    nodes: list[dict] = []
    name_to_bs: dict[str, str] = {}
    canvas_w = _q(benchmark.canvas_width, scale)
    canvas_h = _q(benchmark.canvas_height, scale)
    plc_indices = list(benchmark.hard_macro_indices) + list(benchmark.soft_macro_indices)
    if len(plc_indices) != int(benchmark.num_macros):
        raise ValueError("benchmark/plc macro index mismatch")

    for bench_idx, plc_idx in enumerate(plc_indices):
        node = plc.modules_w_pins[int(plc_idx)]
        original_name = node.get_name()
        bs_name = f"m{bench_idx}"
        name_to_bs[original_name] = bs_name
        w, h = benchmark.macro_sizes[bench_idx].tolist()
        width = max(1, _q(w, scale))
        height = max(1, _q(h, scale))
        if bench_idx >= benchmark.num_hard_macros and not bool(benchmark.macro_fixed[bench_idx]):
            area = max(1, width * height)
            height = max(1, int(soft_cell_height))
            width = max(1, int(round(area / height)))
        cx, cy = initial_placement[bench_idx].tolist()
        llx = _clamp_int(_q(cx, scale) - width // 2, 0, max(0, canvas_w - width))
        lly = _clamp_int(_q(cy, scale) - height // 2, 0, max(0, canvas_h - height))
        fixed = bool(benchmark.macro_fixed[bench_idx])
        nodes.append(
            {
                "bookshelf_name": bs_name,
                "bench_idx": int(bench_idx),
                "width": int(width),
                "height": int(height),
                "llx": int(llx),
                "lly": int(lly),
                "terminal": fixed,
                "terminal_ni": False,
                "orientation": _orientation(node),
            }
        )

    for port_offset, plc_idx in enumerate(getattr(plc, "port_indices", [])):
        port = plc.modules_w_pins[int(plc_idx)]
        original_name = port.get_name()
        bs_name = f"p{port_offset}"
        name_to_bs[original_name] = bs_name
        x, y = port.get_pos()
        nodes.append(
            {
                "bookshelf_name": bs_name,
                "bench_idx": int(benchmark.num_macros) + int(port_offset),
                "width": 1,
                "height": 1,
                "llx": _q(x, scale),
                "lly": _q(y, scale),
                "terminal": True,
                "terminal_ni": True,
                "orientation": "N",
            }
        )
    return nodes, name_to_bs


def _nets(plc, name_to_bs: Mapping[str, str], scale: int) -> list[tuple[str, list[tuple[str, str, int, int]]]]:
    pin_to_owner = _pin_owner_map(plc, scale)
    out: list[tuple[str, list[tuple[str, str, int, int]]]] = []
    for net_idx, (driver, sinks) in enumerate(getattr(plc, "nets", {}).items()):
        pins: list[tuple[str, str, int, int]] = []
        for raw_pin in [driver, *list(sinks)]:
            owner, ox, oy = _resolve_pin(raw_pin, name_to_bs, pin_to_owner)
            if owner is None:
                continue
            bs_name = name_to_bs.get(owner)
            if bs_name is None:
                continue
            pins.append((bs_name, "O" if not pins else "I", ox, oy))
        if len(pins) >= 2:
            out.append((f"net{net_idx}", pins))
    return out


def _pin_owner_map(plc, scale: int) -> dict[str, tuple[str, int, int]]:
    out: dict[str, tuple[str, int, int]] = {}
    for pin_idx in getattr(plc, "hard_macro_pin_indices", []):
        pin = plc.modules_w_pins[int(pin_idx)]
        if not hasattr(pin, "get_name") or not hasattr(pin, "get_macro_name"):
            continue
        macro_name = pin.get_macro_name()
        if not macro_name:
            continue
        ox, oy = pin.get_offset() if hasattr(pin, "get_offset") else (pin.x_offset, pin.y_offset)
        out[pin.get_name()] = (macro_name, _q(ox, scale), _q(oy, scale))
    return out


def _resolve_pin(
    pin_name: str,
    name_to_bs: Mapping[str, str],
    pin_to_owner: Mapping[str, tuple[str, int, int]],
) -> tuple[str | None, int, int]:
    if pin_name in pin_to_owner:
        return pin_to_owner[pin_name]
    if pin_name in name_to_bs:
        return pin_name, 0, 0
    parent = pin_name.split("/")[0]
    if parent in name_to_bs:
        return parent, 0, 0
    return None, 0, 0


def _run_replace(
    export: _Export,
    config: _Config,
    *,
    binary: Path,
    timeout: float,
    stop_after_first: bool,
) -> list[Path]:
    cwd = export.directory.parent.parent
    before = set(_experiment_dirs(cwd, export.name))
    cmd = [str(binary), "-bmflag", "etc", "-bmname", export.name, *config.args()]
    log_dir = cwd / "replace_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{export.name}_{_fmt(config.density)}_{_fmt(config.pcofmax)}.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True)
        first_seen: float | None = None
        while proc.poll() is None:
            if stop_after_first and _fresh_pl_ready(cwd, export.name, before):
                if first_seen is None:
                    first_seen = time.monotonic()
                elif time.monotonic() - first_seen >= 0.05:
                    proc.kill()
                    break
            else:
                first_seen = None
            if time.monotonic() - started >= float(timeout):
                proc.kill()
                break
            time.sleep(0.05)
        proc.wait()

    new_dirs = [path for path in _experiment_dirs(cwd, export.name) if path not in before]
    return _pl_paths(new_dirs, export.name)


def _import_bookshelf(pl_path: Path, metadata_path: Path, benchmark: Benchmark) -> torch.Tensor:
    metadata = json.loads(metadata_path.read_text())
    scale = float(metadata["scale"])
    nodes = {str(node["bookshelf_name"]): node for node in metadata["nodes"]}
    entries = _read_pl(pl_path)
    placement = benchmark.macro_positions.clone().float()
    imported = torch.zeros((benchmark.num_macros,), dtype=torch.bool)
    for name, (llx, lly) in entries.items():
        node = nodes.get(name)
        if node is None:
            continue
        bench_idx = int(node["bench_idx"])
        if bench_idx < 0 or bench_idx >= benchmark.num_macros:
            continue
        placement[bench_idx, 0] = float((llx + 0.5 * float(node["width"])) / scale)
        placement[bench_idx, 1] = float((lly + 0.5 * float(node["height"])) / scale)
        imported[bench_idx] = True
    if not bool(imported.all()):
        missing = torch.where(~imported)[0].tolist()
        raise ValueError(f"missing imported macro {missing[0]}")
    if benchmark.macro_fixed.any():
        placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
    _clamp(placement, benchmark)
    return placement


def _read_pl(path: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("UCLA"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            out[parts[0]] = (float(parts[1]), float(parts[2]))
        except ValueError:
            continue
    return out


def _rows(benchmark: Benchmark, scale: int) -> list[dict]:
    rows = max(1, int(benchmark.grid_rows))
    canvas_w = _q(benchmark.canvas_width, scale)
    canvas_h = _q(benchmark.canvas_height, scale)
    row_h = max(1, int(math.ceil(canvas_h / rows)))
    out = []
    y = 0
    for _ in range(rows):
        out.append({"coordinate": y, "height": row_h, "num_sites": canvas_w})
        y += row_h
    return out


def _aux(bs_name: str) -> str:
    return (
        "RowBasedPlacement : "
        f"{bs_name}.nodes {bs_name}.nets {bs_name}.wts {bs_name}.pl "
        f"{bs_name}.scl {bs_name}.shapes {bs_name}.route\n"
    )


def _nodes_text(nodes: Sequence[dict]) -> str:
    terminals = sum(1 for node in nodes if node["terminal"])
    lines = ["UCLA nodes 1.0", "", f"NumNodes      : {len(nodes)}", f"NumTerminals  : {terminals}", ""]
    for node in nodes:
        suffix = " terminal_NI" if node["terminal_ni"] else " terminal" if node["terminal"] else ""
        lines.append(f"{node['bookshelf_name']}\t{node['width']}\t{node['height']}{suffix}")
    return "\n".join(lines) + "\n"


def _nets_text(nets: Sequence[tuple[str, Sequence[tuple[str, str, int, int]]]]) -> str:
    pin_count = sum(len(pins) for _, pins in nets)
    lines = ["UCLA nets 1.0", "", f"NumNets : {len(nets)}", f"NumPins : {pin_count}", ""]
    for name, pins in nets:
        lines.append(f"NetDegree : {len(pins)} {name}")
        for bs_name, direction, ox, oy in pins:
            lines.append(f"\t{bs_name}\t{direction}\t: {ox} {oy}")
    return "\n".join(lines) + "\n"


def _wts_text(nets: Sequence[tuple[str, Sequence[tuple[str, str, int, int]]]]) -> str:
    return "UCLA wts 1.0\n\n" + "\n".join(f"{name} 1" for name, _ in nets) + "\n"


def _pl_text(nodes: Sequence[dict]) -> str:
    lines = ["UCLA pl 1.0", ""]
    for node in nodes:
        suffix = " /FIXED_NI" if node["terminal_ni"] else " /FIXED" if node["terminal"] else ""
        lines.append(f"{node['bookshelf_name']}\t{node['llx']}\t{node['lly']}\t: {node['orientation']}{suffix}")
    return "\n".join(lines) + "\n"


def _scl_text(rows: Sequence[dict]) -> str:
    lines = ["UCLA scl 1.0", "", f"NumRows : {len(rows)}", ""]
    for row in rows:
        lines.extend(
            [
                "CoreRow Horizontal",
                f"  Coordinate    : {row['coordinate']}",
                f"  Height        : {row['height']}",
                "  Sitewidth     : 1",
                "  Sitespacing   : 1",
                "  Siteorient    : N",
                "  Sitesymmetry  : Y",
                f"  SubrowOrigin  : 0 NumSites : {row['num_sites']}",
                "End",
            ]
        )
    return "\n".join(lines) + "\n"


def _route_text(benchmark: Benchmark, scale: int) -> str:
    rows = max(1, int(benchmark.grid_rows))
    cols = max(1, int(benchmark.grid_cols))
    tile_w = max(1, int(math.ceil(_q(benchmark.canvas_width, scale) / cols)))
    tile_h = max(1, int(math.ceil(_q(benchmark.canvas_height, scale) / rows)))
    h_cap = max(1, int(round(float(benchmark.hroutes_per_micron) * tile_w / scale)))
    v_cap = max(1, int(round(float(benchmark.vroutes_per_micron) * tile_h / scale)))
    return "\n".join(
        [
            "UCLA route 1.0",
            "",
            f"Grid  : {cols} {rows}",
            f"VerticalCapacity   : {v_cap}",
            f"HorizontalCapacity : {h_cap}",
            "MinWireWidth       : 1",
            "MinWireSpacing     : 0",
            "ViaSpacing         : 0",
            "GridOrigin         : 0 0",
            f"TileSize           : {tile_w} {tile_h}",
            "BlockagePorosity   : 0",
        ]
    ) + "\n"


def _experiment_dirs(cwd: Path, bs_name: str) -> list[Path]:
    root = cwd / "outputs" / "ETC" / bs_name
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir() and path.name.startswith("experiment")])


def _pl_paths(dirs: Iterable[Path], bs_name: str) -> list[Path]:
    out: list[Path] = []
    seen = set()
    for directory in dirs:
        paths = [directory / f"{bs_name}.eplace-mGP2D.pl"]
        paths.extend(sorted(directory.glob("*.pl")))
        for path in paths:
            if not path.exists():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(path)
    return out


def _fresh_pl_ready(cwd: Path, bs_name: str, before: set[Path]) -> bool:
    new_dirs = [path for path in _experiment_dirs(cwd, bs_name) if path not in before]
    for pl_path in _pl_paths(new_dirs, bs_name):
        try:
            if pl_path.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def _clamp(placement: torch.Tensor, benchmark: Benchmark) -> None:
    sizes = benchmark.macro_sizes.to(dtype=placement.dtype)
    placement[:, 0].clamp_(sizes[:, 0] * 0.5, float(benchmark.canvas_width) - sizes[:, 0] * 0.5)
    placement[:, 1].clamp_(sizes[:, 1] * 0.5, float(benchmark.canvas_height) - sizes[:, 1] * 0.5)
    if benchmark.macro_fixed.any():
        placement[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _q(value: float, scale: int) -> int:
    return int(round(float(value) * int(scale)))


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(value), int(hi)))


def _orientation(node) -> str:
    orient = node.get_orientation() if hasattr(node, "get_orientation") else None
    if not orient or orient in {"-", "R0"}:
        return "N"
    return str(orient)


def _safe_name(name: str) -> str:
    safe = _SAFE_NAME_RE.sub("_", name.strip()).strip("_")
    return safe or "benchmark"


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"


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
