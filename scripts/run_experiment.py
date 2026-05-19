"""Run and record macro-placement experiments.

This wraps the official local evaluator. It does not change scoring; it just
keeps structured records so we can compare iterations without scraping logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

from macro_place.evaluate import BENCHMARKS, _load_placer, evaluate_benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement


SMOKE_BENCHMARKS = ["ibm08", "ibm15"]
QUICK_BENCHMARKS = ["ibm01", "ibm03", "ibm08", "ibm10", "ibm15", "ibm17"]
QUICK_LITE_BENCHMARKS = ["ibm01", "ibm08", "ibm15"]


def _parse_benchmarks(args: argparse.Namespace) -> list[str]:
    if args.benchmarks:
        return [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    if args.smoke:
        return SMOKE_BENCHMARKS
    if args.quick_lite:
        return QUICK_LITE_BENCHMARKS
    if args.quick:
        return QUICK_BENCHMARKS
    if args.all:
        return list(BENCHMARKS)
    return [args.benchmark]


def _safe_label(label: str) -> str:
    keep = []
    for ch in label:
        keep.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(keep).strip("_") or "experiment"


def _summarize(results: list[dict], target: float) -> dict:
    avg_proxy = statistics.fmean(r["proxy_cost"] for r in results)
    total_runtime = sum(r["runtime"] for r in results)
    total_overlaps = sum(r["overlaps"] for r in results)
    return {
        "count": len(results),
        "avg_proxy": avg_proxy,
        "avg_wirelength": statistics.fmean(r["wirelength"] for r in results),
        "avg_density": statistics.fmean(r["density"] for r in results),
        "avg_congestion": statistics.fmean(r["congestion"] for r in results),
        "total_runtime": total_runtime,
        "avg_runtime": total_runtime / len(results),
        "total_overlaps": total_overlaps,
        "target": target,
        "gap_to_target": avg_proxy - target,
        "pct_over_target": ((avg_proxy / target) - 1.0) * 100.0,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "name",
        "proxy_cost",
        "wirelength",
        "density",
        "congestion",
        "overlaps",
        "runtime",
        "valid",
        "sa_baseline",
        "replace_baseline",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_candidate_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "benchmark",
        "candidate",
        "proxy_cost",
        "wirelength",
        "density",
        "congestion",
        "overlaps",
        "valid",
        "generation_runtime",
        "eval_runtime",
        "winner",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _get_named_candidates(placer, benchmark) -> list[tuple[str, object]]:
    if hasattr(placer, "generate_candidates"):
        raw_candidates = placer.generate_candidates(benchmark)
    else:
        raw_candidates = [("place", placer.place(benchmark))]

    candidates = []
    used_names: dict[str, int] = {}
    for idx, item in enumerate(raw_candidates):
        if isinstance(item, tuple) and len(item) == 2:
            name, placement = item
            name = str(name)
        else:
            name = f"candidate:{idx}"
            placement = item
        count = used_names.get(name, 0)
        used_names[name] = count + 1
        if count:
            name = f"{name}#{count + 1}"
        candidates.append((name, placement))
    return candidates


def evaluate_candidate_benchmark(placer, name: str, testcase_root: str) -> tuple[dict, list[dict]]:
    benchmark, plc = load_benchmark_from_dir(f"{testcase_root}/{name}")

    generation_start = time.time()
    candidates = _get_named_candidates(placer, benchmark)
    generation_runtime = time.time() - generation_start

    rows = []
    best_row = None
    best_proxy = float("inf")

    for candidate_name, placement in candidates:
        eval_start = time.time()
        is_valid, _ = validate_placement(placement, benchmark)
        costs = compute_proxy_cost(placement, benchmark, plc)
        eval_runtime = time.time() - eval_start
        row = {
            "benchmark": name,
            "candidate": candidate_name,
            "proxy_cost": costs["proxy_cost"],
            "wirelength": costs["wirelength_cost"],
            "density": costs["density_cost"],
            "congestion": costs["congestion_cost"],
            "overlaps": costs["overlap_count"],
            "valid": is_valid,
            "generation_runtime": generation_runtime,
            "eval_runtime": eval_runtime,
            "winner": False,
        }
        rows.append(row)
        if is_valid and int(costs["overlap_count"]) == 0 and float(costs["proxy_cost"]) < best_proxy:
            best_proxy = float(costs["proxy_cost"])
            best_row = row

    if best_row is None:
        raise RuntimeError(f"No valid candidate for {name}")

    best_row["winner"] = True
    result = {
        "name": name,
        "proxy_cost": best_row["proxy_cost"],
        "wirelength": best_row["wirelength"],
        "density": best_row["density"],
        "congestion": best_row["congestion"],
        "overlaps": best_row["overlaps"],
        "runtime": generation_runtime,
        "valid": best_row["valid"],
        "sa_baseline": None,
        "replace_baseline": None,
        "winner": best_row["candidate"],
        "candidate_count": len(candidates),
        "candidate_eval_runtime": sum(float(row["eval_runtime"]) for row in rows),
    }
    return result, rows


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _print_table(results: list[dict], summary: dict) -> None:
    print()
    has_winner = any("winner" in r for r in results)
    if has_winner:
        print(
            f"{'bench':>7}  {'proxy':>8}  {'wl':>7}  {'den':>7}  {'cong':>7}  "
            f"{'ov':>3}  {'valid':>5}  {'rt':>7}  winner"
        )
        print("-" * 89)
    else:
        print(
            f"{'bench':>7}  {'proxy':>8}  {'wl':>7}  {'den':>7}  {'cong':>7}  "
            f"{'ov':>3}  {'valid':>5}  {'rt':>7}"
        )
        print("-" * 63)
    for r in results:
        line = (
            f"{r['name']:>7}  {r['proxy_cost']:>8.4f}  {r['wirelength']:>7.3f}  "
            f"{r['density']:>7.3f}  {r['congestion']:>7.3f}  {r['overlaps']:>3}  "
            f"{str(bool(r['valid'])):>5}  {r['runtime']:>6.2f}s"
        )
        if has_winner:
            line += f"  {r.get('winner', '')}"
        print(line)
    print("-" * (89 if has_winner else 63))
    print(
        f"{'AVG':>7}  {summary['avg_proxy']:>8.4f}  {summary['avg_wirelength']:>7.3f}  "
        f"{summary['avg_density']:>7.3f}  {summary['avg_congestion']:>7.3f}  "
        f"{summary['total_overlaps']:>3}  {'':>5}  {summary['total_runtime']:>6.2f}s"
    )
    print(
        f"target={summary['target']:.4f}  gap={summary['gap_to_target']:+.4f}  "
        f"over_target={summary['pct_over_target']:+.1f}%"
    )


def _metadata(
    *,
    args: argparse.Namespace,
    placer,
    placer_path: Path,
    benchmarks: list[str],
    timestamp: str,
    started: float,
    env: dict[str, str],
    summary: dict,
    partial: bool,
) -> dict:
    return {
        "label": args.label,
        "placer": str(placer_path),
        "placer_class": type(placer).__name__,
        "benchmarks": benchmarks,
        "created_at": timestamp,
        "wall_runtime": time.time() - started,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "env": env,
        "candidate_mode": args.candidates,
        "partial": partial,
        "summary": summary,
    }


def _write_run_record(
    run_dir: Path,
    *,
    metadata: dict,
    results: list[dict],
    candidate_rows: list[dict],
) -> None:
    _write_csv(run_dir / "results.csv", results)
    if metadata.get("candidate_mode"):
        _write_candidate_csv(run_dir / "candidates.csv", candidate_rows)
    with (run_dir / "results.json").open("w") as f:
        json.dump(
            _json_safe(
                {
                    "metadata": metadata,
                    "results": results,
                    "candidates": candidate_rows if metadata.get("candidate_mode") else [],
                }
            ),
            f,
            indent=2,
        )


def _print_result_line(result: dict) -> None:
    print(
        f"{result['name']}: proxy={float(result['proxy_cost']):.4f} "
        f"wl={float(result['wirelength']):.3f} "
        f"den={float(result['density']):.3f} "
        f"cong={float(result['congestion']):.3f} "
        f"ov={int(result['overlaps'])} "
        f"rt={float(result['runtime']):.2f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a macro-placement experiment run.")
    parser.add_argument("placer", help="Path to a placer .py file.")
    parser.add_argument("--label", default="experiment", help="Short label for output files.")
    parser.add_argument("--target", type=float, default=0.9, help="Target average proxy score.")
    parser.add_argument("--out-dir", default="experiments", help="Directory for run records.")
    parser.add_argument("--benchmark", "-b", default="ibm01", help="Single benchmark name.")
    parser.add_argument("--benchmarks", help="Comma-separated benchmark names.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the fastest prelim subset: ibm08, ibm15.",
    )
    parser.add_argument(
        "--quick-lite",
        action="store_true",
        help="Run a smaller dev subset: ibm01, ibm08, ibm15.",
    )
    parser.add_argument("--quick", action="store_true", help="Run the fixed quick subset.")
    parser.add_argument("--all", "-a", action="store_true", help="Run all IBM benchmarks.")
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="Exact-score every named candidate and select the best true proxy candidate.",
    )
    args = parser.parse_args()

    benchmarks = _parse_benchmarks(args)
    placer_path = Path(args.placer)
    placer = _load_placer(placer_path)
    testcase_root = Path("external/MacroPlacement/Testcases/ICCAD04")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / f"{timestamp}_{_safe_label(args.label)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    env = {key: value for key, value in os.environ.items() if key.startswith("OURS_")}
    started = time.time()
    results = []
    candidate_rows = []
    for name in benchmarks:
        print(f"{name}...", flush=True)
        if args.candidates:
            result, rows = evaluate_candidate_benchmark(placer, name, str(testcase_root))
            candidate_rows.extend(rows)
        else:
            result = evaluate_benchmark(placer, name, str(testcase_root))
            result.pop("placement", None)
            result.pop("benchmark", None)
            result.pop("plc", None)
        results.append(result)
        _print_result_line(result)

        partial_summary = _summarize(results, args.target)
        partial_metadata = _metadata(
            args=args,
            placer=placer,
            placer_path=placer_path,
            benchmarks=benchmarks,
            timestamp=timestamp,
            started=started,
            env=env,
            summary=partial_summary,
            partial=len(results) < len(benchmarks),
        )
        _write_run_record(
            run_dir,
            metadata=partial_metadata,
            results=results,
            candidate_rows=candidate_rows,
        )

    summary = _summarize(results, args.target)
    metadata = _metadata(
        args=args,
        placer=placer,
        placer_path=placer_path,
        benchmarks=benchmarks,
        timestamp=timestamp,
        started=started,
        env=env,
        summary=summary,
        partial=False,
    )
    _write_run_record(
        run_dir,
        metadata=metadata,
        results=results,
        candidate_rows=candidate_rows,
    )

    _print_table(results, summary)
    print(f"\nrecorded: {run_dir}")


if __name__ == "__main__":
    main()
