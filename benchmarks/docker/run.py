"""
benchmarks/compare.py in Docker, over several Linux distributions and allocators.

Builds one image per distribution, then runs one container per distribution and allocator. Each
container compares the working tree, uncommitted changes included, against a base ref on the
same machine, both under the same allocator. With `--vs-system`, it compares allocators instead:
the working tree under each allocator against itself under the system allocator (glibc malloc).

Builds are cached in a Docker volume per distribution. Results go to
`benchmarks/results/docker/<distribution>-<allocators>/`, and all reports together to
`benchmarks/results/docker/summary.md`. Options this script does not know are passed on to
compare.py.

    uv run python benchmarks/docker/run.py
    uv run python benchmarks/docker/run.py --images debian:13 --allocators system mimalloc
    uv run python benchmarks/docker/run.py --vs-system --sizes 1000000 --rounds 6

Timings in containers are only comparable within one run on one machine. Hardware counters need
perf access to the host's PMU, which Docker Desktop's VM does not offer; software counters and
syscalls work everywhere. Images match the host's architecture: emulated ones are too slow to time.
Allocators are preloaded into the Python process, which covers Python, numpy, pandas, Gurobi and
polarobi's own allocations, but not polars', which links its own allocator.
"""

import argparse
import re
import shlex
import shutil
import subprocess
from pathlib import Path

_here = Path(__file__).parent
_repo = _here.parents[1]
_results = _repo / "benchmarks" / "results" / "docker"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmarks/compare.py in Docker.", epilog="Other options go to compare.py."
    )
    parser.add_argument("--images", nargs="+", default=["debian:13", "ubuntu:24.04", "fedora:43"])
    parser.add_argument(
        "--allocators",
        nargs="+",
        default=["system", "jemalloc", "mimalloc", "tcmalloc"],
        help="system, jemalloc, mimalloc, tcmalloc, or a library path inside the image.",
    )
    parser.add_argument("--base", default="master", help="Ref to compare the working tree against.")
    parser.add_argument(
        "--vs-system",
        action="store_true",
        help="Compare each allocator against the system allocator, on the working tree alone.",
    )
    args, compare = parser.parse_known_args()
    if not shutil.which("docker"):
        raise SystemExit("docker is not on the PATH: install Docker, or run this where it is.")

    runs: list[tuple[str, str, str, str]] = []
    for image in args.images:
        for allocator in args.allocators:
            if args.vs_system:
                if allocator != "system":
                    runs.append((image, "-", "system", allocator))
            else:
                runs.append((image, args.base, allocator, allocator))

    for image in args.images:
        print(f"Building the image for {image}", flush=True)
        tag = f"polarobi-bench:{_slug(image)}"
        subprocess.run(
            ["docker", "build", "--build-arg", f"BASE={image}", "-t", tag, str(_here)], check=True
        )

    reports: list[str] = []
    for image, base, base_allocator, head_allocator in runs:
        if base == "-":
            label = f"{image}, {head_allocator} vs {base_allocator}"
        else:
            label = f"{image}, {base} vs working tree, {head_allocator}"
        results = _results / _slug(f"{image}-{base_allocator}-vs-{head_allocator}")
        results.mkdir(parents=True, exist_ok=True)
        (results / "summary.md").unlink(missing_ok=True)
        print(f"Running {label}", flush=True)
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                # perf and strace: perf_event_open and ptrace are blocked by default.
                "--cap-add",
                "PERFMON",
                "--cap-add",
                "SYS_PTRACE",
                "--security-opt",
                "seccomp=unconfined",
                "-e",
                f"COMPARE_ARGS={shlex.join(compare)}",
                "-v",
                f"{_repo}:/repo:ro",
                "-v",
                f"{results}:/results",
                "-v",
                f"polarobi-bench-{_slug(image)}:/cache",
                f"polarobi-bench:{_slug(image)}",
                base,
                base_allocator,
                head_allocator,
                label,
            ],
            check=True,
        )
        reports.append((results / "summary.md").read_text(encoding="utf-8"))

    summary = _results / "summary.md"
    summary.write_text("\n".join(reports), encoding="utf-8")
    print(f"All reports written to {summary}.")


if __name__ == "__main__":
    main()
