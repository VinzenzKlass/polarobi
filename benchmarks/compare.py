"""
A/B benchmark: model building with polarobi at two checkouts, on the same machine.

Separate runs on separate machines differ by more than most changes, so this compares a base and
a head checkout in one go: both run this directory's `build_model.py`, polarobi only, in
alternating order over several rounds, so that both see the same machine and the same drift.
Each round pairs one base and one head run; a change counts as real when the head is faster, or
slower, in significantly more rounds than chance would give (an exact sign test).

With `perf` on the path, each run also records hardware counters, and with `strace` the memory
syscalls of one extra run per side. Both cover the whole process: imports, data generation and
the warm-up run as well as the timed builds.

Each checkout needs its own environment, from `uv sync --group bench` in its directory, or one
given with `--base-python` and `--head-python`. `--base-preload` and `--head-preload` swap in
another allocator through LD_PRELOAD; both sides can be the same checkout, to compare allocators
alone. benchmarks/docker/run.py runs all of this in containers.

    git worktree add ../base master
    (cd ../base && uv sync --group bench)
    uv run --group bench python benchmarks/compare.py --base ../base --head .
"""

import argparse
import csv
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

_here = Path(__file__).parent
_results = _here / "results"
_sides = ("base", "head")
_events = [
    "instructions",
    "cycles",
    "branch-misses",
    "L1-dcache-load-misses",
    "page-faults",
    "context-switches",
]
_syscalls = ["mmap", "munmap", "mremap", "brk", "madvise", "mprotect"]


def _python(checkout: Path) -> Path:
    scripts = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    return checkout.resolve() / ".venv" / scripts


def _benchmark(
    python: Path, sizes: list[int], repeats: int, out: Path, cpus: str | None
) -> list[str]:
    command = [
        str(python),
        str(_here / "build_model.py"),
        "--builders",
        "polarobi",
        "--sizes",
        *map(str, sizes),
        "--repeats",
        str(repeats),
        "--out",
        str(out),
    ]
    return [*command, "--cpus", cpus] if cpus else command


def _totals(path: Path) -> dict[int, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {result["rows"]: result["total_s"] for result in report["results"]}


def _counters(path: Path) -> dict[str, float]:
    """The counters from `perf stat -x ,` output: value, unit, event, ..."""
    counters: dict[str, float] = {}
    for row in csv.reader(path.read_text(encoding="utf-8").splitlines()):
        if len(row) > 2 and (event := row[2].split(":")[0]) in _events:
            try:
                counters[event] = float(row[0])
            except ValueError:  # <not supported> or <not counted>
                pass
    return counters


def _calls(path: Path) -> dict[str, int]:
    """The calls per syscall from `strace -c` output, whose last column is the syscall."""
    calls: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[-1] in _syscalls and parts[3].isdigit():
            calls[parts[-1]] = int(parts[3])
    return calls


def _sign_test(wins: int, rounds: int) -> float:
    """Two-sided p-value of `wins` out of `rounds` under a fair coin."""
    tail = sum(math.comb(rounds, i) for i in range(min(wins, rounds - wins) + 1))
    return min(1.0, 2 * tail / 2**rounds)


def _change(base: float, head: float) -> str:
    return f"{(head / base - 1) * 100:+.1f}%" if base else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare polarobi at two checkouts.")
    parser.add_argument("--base", type=Path, required=True, help="Checkout to compare against.")
    parser.add_argument("--head", type=Path, required=True, help="Checkout to measure.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3, help="Timed builds per run.")
    parser.add_argument("--cpus", help="Passed on to build_model.py, e.g. 0,1,2,3.")
    parser.add_argument("--label", default=platform.machine(), help="Machine name for the report.")
    parser.add_argument("--out", type=Path, default=_results / "compare.json")
    parser.add_argument("--summary", type=Path, help="Markdown report to append to.")
    for side in _sides:
        parser.add_argument(
            f"--{side}-python", type=Path, help=f"Python to run, the {side}'s .venv by default."
        )
        parser.add_argument(
            f"--{side}-preload", help="Shared library to preload, such as an allocator (Linux)."
        )
    args = parser.parse_args()

    checkouts: dict[str, Path] = {"base": args.base, "head": args.head}
    pythons = {side: getattr(args, f"{side}_python") or _python(checkouts[side]) for side in _sides}
    envs: dict[str, dict[str, str]] = {}
    for side in _sides:
        envs[side] = dict(os.environ)
        if preload := getattr(args, f"{side}_preload"):
            envs[side]["LD_PRELOAD"] = preload
    work = args.out.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    perf, strace = shutil.which("perf"), shutil.which("strace")

    # Per side: totals per size, one per round, and counters per event, one per round.
    totals: dict[str, dict[int, list[float]]] = {side: {} for side in _sides}
    counters: dict[str, dict[str, list[float]]] = {side: {} for side in _sides}
    # One untimed run per side first: the very first run pays for cold file and disk caches.
    for side in _sides:
        print(f"Warm-up: {side}", flush=True)
        command = _benchmark(pythons[side], args.sizes[:1], 1, work / f"{side}-warmup.json", None)
        subprocess.run(command, check=True, env=envs[side])
    for round in range(args.rounds):
        for side in _sides if round % 2 == 0 else reversed(_sides):
            out = work / f"{side}-{round}.json"
            command = _benchmark(pythons[side], args.sizes, args.repeats, out, args.cpus)
            stat = work / f"{side}-{round}.perf"
            if perf:
                events = ",".join(_events)
                command = [perf, "stat", "-x", ",", "-o", str(stat), "-e", events, "--", *command]
            print(f"Round {round + 1}/{args.rounds}: {side}", flush=True)
            subprocess.run(command, check=True, env=envs[side])
            for size, total in _totals(out).items():
                totals[side].setdefault(size, []).append(total)
            if perf:
                for event, value in _counters(stat).items():
                    counters[side].setdefault(event, []).append(value)

    # strace slows the run down many times over, so one run per side at the smallest size.
    calls: dict[str, dict[str, int]] = {}
    if strace:
        for side in _sides:
            trace = work / f"{side}.strace"
            command = _benchmark(
                pythons[side], args.sizes[:1], 1, work / f"{side}-strace.json", args.cpus
            )
            print(f"strace: {side}", flush=True)
            subprocess.run(
                [strace, "-c", "-f", "-e", "trace=memory", "-o", str(trace), *command],
                check=True,
                env=envs[side],
            )
            calls[side] = _calls(trace)

    timings: list[dict[str, object]] = []
    lines = [
        f"## polarobi build time: head vs base on {args.label}",
        "",
        f"{args.rounds} rounds in alternating order, median of {args.repeats} builds per run.",
        "",
        "| rows | base | head | change | head faster in | p | verdict |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for size in args.sizes:
        base, head = totals["base"][size], totals["head"][size]
        ratios = [h / b for b, h in zip(base, head, strict=True)]
        wins = sum(ratio < 1 for ratio in ratios)
        p = _sign_test(wins, len(ratios))
        verdict = "no clear change"
        if p < 0.05:
            verdict = "faster" if wins > len(ratios) / 2 else "slower"
        median = statistics.median(ratios)
        timings.append(
            {
                "rows": size,
                "base_s": base,
                "head_s": head,
                "median_ratio": median,
                "head_faster_rounds": wins,
                "p": p,
                "verdict": verdict,
            }
        )
        lines.append(
            f"| {size:,} | {statistics.median(base) * 1e3:.1f} ms "
            f"| {statistics.median(head) * 1e3:.1f} ms | {(median - 1) * 100:+.1f}% "
            f"| {wins}/{len(ratios)} | {p:.3f} | {verdict} |"
        )

    if perf and counters["base"]:
        lines += [
            "",
            "### Hardware counters, whole process, median per run",
            "",
            "| event | base | head | change |",
            "|---|---:|---:|---:|",
        ]
        for event in _events:
            if event in counters["base"] and event in counters["head"]:
                base = statistics.median(counters["base"][event])
                head = statistics.median(counters["head"][event])
                lines.append(f"| {event} | {base:,.0f} | {head:,.0f} | {_change(base, head)} |")

    if calls:
        lines += [
            "",
            f"### Memory syscalls, whole process, one run at {args.sizes[0]:,} rows",
            "",
            "| syscall | base | head | change |",
            "|---|---:|---:|---:|",
        ]
        for syscall in _syscalls:
            base, head = calls["base"].get(syscall, 0), calls["head"].get(syscall, 0)
            if base or head:
                lines.append(f"| {syscall} | {base:,} | {head:,} | {_change(base, head)} |")

    report = {
        "label": args.label,
        "settings": {"sizes": args.sizes, "rounds": args.rounds, "repeats": args.repeats},
        "timings": timings,
        "counters": counters,
        "syscalls": calls,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as summary:
            summary.write(text + "\n")


if __name__ == "__main__":
    main()
